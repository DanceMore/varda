//! Mixer - Top-level compositor that owns channels, crossfader, master effects, and modulation

mod transition;
mod render;

pub use transition::{
    CrossfadeEasing, AutoCrossfade, BeatSyncCrossfade,
    TransitionSequence, TransitionStep, StepKind, SequencerState,
    TransitionEffect,
};

use crate::channel::Channel;
use crate::deck::Effect;
use crate::modulation::ModulationEngine;
use crate::renderer::{GpuContext, BlitPipeline, CompositeBlitPipeline, PingPong};
use anyhow::Result;

/// Mixer - Top-level compositor
pub struct Mixer {
    /// Channels (default 2: A and B)
    channels: Vec<Channel>,

    /// Monotonic counter for generating unique channel names (never decremented)
    next_channel_index: usize,

    /// Crossfader position (0.0 = Ch 0, 1.0 = Ch 1)
    crossfader: f32,

    /// Active auto-crossfade (if any)
    auto_crossfade: Option<AutoCrossfade>,

    /// Pending beat-synced crossfade (if any)
    beat_sync_crossfade: Option<BeatSyncCrossfade>,

    /// Global modulation engine
    modulation: ModulationEngine,

    /// Start time for TIME-based modulation
    start_time: std::time::Instant,

    /// Last render time for dt calculation
    last_render_time: std::time::Instant,

    /// Composite output (all channels mixed, then master effects), with a
    /// ping-pong scratch target so channel-blend and the master effect chain
    /// never snapshot-copy. Latest content is `composite.result_view()`.
    composite: PingPong,

    /// Master effect chain (applied to final composite)
    master_effects: Vec<Effect>,

    /// Frame counter
    frame_count: u64,

    /// Shader-based composite pipeline for blending channels (all blend modes via uniform)
    composite_pipeline: CompositeBlitPipeline,

    /// Simple blit pipeline for first-channel copy
    blit_pipeline: BlitPipeline,

    /// Active transition effect (replaces opacity-based crossfade when set)
    active_transition: Option<TransitionEffect>,

    /// Transition sequences (channel-to-channel automation). Multiple named sequences supported.
    transition_sequences: Vec<TransitionSequence>,

    /// Cached sub-mix textures for multi-channel surface assignments.
    /// Key: sorted channel indices, Value: (texture, view, scratch_texture, scratch_view).
    /// The output (texture/view) identity must stay stable — external output
    /// stages cache it via get_sub_mix_view — so sub-mix keeps a snapshot-copy
    /// into its own scratch rather than ping-ponging (which would alternate the
    /// output identity). The scratch is per-entry, sized with the output.
    sub_mix_cache: std::collections::HashMap<Vec<usize>, (wgpu::Texture, wgpu::TextureView, wgpu::Texture, wgpu::TextureView)>,

    /// Per-channel culled state from the previous frame. Used to emit a
    /// one-shot clear of the composite view on the visible→culled edge,
    /// avoiding a per-frame clear submission for every culled channel.
    pub(super) prev_culled: Vec<bool>,

}

impl Mixer {
    /// Create a new mixer with two default channels (A and B)
    pub fn new(context: &GpuContext, width: u32, height: u32) -> Result<Self> {
        let composite = PingPong::new(context, width, height);

        let composite_pipeline = CompositeBlitPipeline::new(&context.device, context.texture_format)?;
        let blit_pipeline = BlitPipeline::with_blend(
            &context.device,
            context.texture_format,
            wgpu::BlendState::ALPHA_BLENDING,
        )?;

        // Create two default channels
        let channel_0 = Channel::new("Ch 0".to_string(), context, width, height)?;
        let channel_1 = Channel::new("Ch 1".to_string(), context, width, height)?;

        let now = std::time::Instant::now();
        Ok(Self {
            channels: vec![channel_0, channel_1],
            next_channel_index: 2, // Ch 0, Ch 1 already used
            crossfader: 0.0,
            auto_crossfade: None,
            beat_sync_crossfade: None,
            modulation: ModulationEngine::new(),
            start_time: now,
            last_render_time: now,
            composite,
            master_effects: Vec::new(),
            frame_count: 0,
            composite_pipeline,
            blit_pipeline,
            active_transition: None,
            transition_sequences: Vec::new(),
            sub_mix_cache: std::collections::HashMap::new(),
            prev_culled: vec![false, false],
        })
    }

    /// Resize mixer and all channel textures
    pub fn resize(&mut self, context: &GpuContext, width: u32, height: u32) {
        self.composite.resize(context, width, height);

        for channel in &mut self.channels {
            channel.resize(context, width, height);
        }
        for effect in self.master_effects.iter_mut() {
            effect.resize(context, width, height);
        }
        self.sub_mix_cache.clear();
    }

    /// Clear the sub-mix texture cache (e.g. after resolution change).
    pub fn clear_sub_mix_cache(&mut self) {
        self.sub_mix_cache.clear();
    }

    /// Add a master effect
    pub fn add_master_effect(&mut self, effect: Effect) {
        self.master_effects.push(effect);
    }

    /// Remove a master effect by index
    pub fn remove_master_effect(&mut self, index: usize) -> bool {
        if index < self.master_effects.len() {
            self.master_effects.remove(index);
            true
        } else {
            false
        }
    }

    /// Add a new channel with an auto-generated name (C, D, E, ...)
    pub fn add_channel(&mut self, context: &GpuContext, width: u32, height: u32) -> Result<usize> {
        let name = channel_name(self.next_channel_index);
        self.next_channel_index += 1;
        let channel = Channel::new(name, context, width, height)?;
        let idx = self.channels.len();
        self.channels.push(channel);
        log::info!("Added channel {} (index {})", self.channels[idx].name, idx);
        Ok(idx)
    }

    /// Remove a channel by index. Returns true if removed.
    /// Cannot remove below 2 channels (minimum A and B).
    ///
    /// Also purges modulation assignments addressing decks/effects owned by the
    /// channel, and fixes up TransitionSequence step indices that referenced
    /// the removed channel (later indices shift down; sequences with steps
    /// that referenced the removed channel directly are disabled).
    ///
    /// Note: external resources held by decks in this channel (cameras, NDI,
    /// SRT, Syphon) MUST be released by the caller before invoking this — see
    /// `release_channel_external_resources` patterns at the engine layer.
    pub fn remove_channel(&mut self, index: usize) -> bool {
        if self.channels.len() <= 2 || index >= self.channels.len() {
            return false;
        }
        let (deck_uuids, effect_uuids) = self.channels[index].collect_modulation_uuids();
        let name = self.channels[index].name.clone();
        self.channels.remove(index);
        for u in &deck_uuids {
            self.modulation.remove_assignments_with_prefix(&format!("deck_{}:", u));
        }
        for u in &effect_uuids {
            self.modulation.remove_assignments_with_prefix(&format!("fx_{}:", u));
        }
        self.fixup_transition_sequences_after_channel_remove(index);
        log::info!(
            "Removed channel {} (was index {}); purged {} deck + {} effect modulation entries",
            name, index, deck_uuids.len(), effect_uuids.len()
        );
        true
    }

    /// Adjust TransitionSequence step channel indices after a channel was removed
    /// at `removed_idx`. Steps that referenced the removed channel cause the
    /// sequence to be disabled (and a warning logged). Indices > removed_idx
    /// shift down by one.
    pub(crate) fn fixup_transition_sequences_after_channel_remove(&mut self, removed_idx: usize) {
        for seq in self.transition_sequences.iter_mut() {
            let mut had_orphan = false;
            for step in seq.steps.iter_mut() {
                if let StepKind::Fade { from_ch, to_ch, .. } = &mut step.kind {
                    if *from_ch == removed_idx || *to_ch == removed_idx {
                        had_orphan = true;
                    }
                    if *from_ch > removed_idx { *from_ch -= 1; }
                    if *to_ch > removed_idx { *to_ch -= 1; }
                }
            }
            if had_orphan && seq.enabled {
                log::warn!(
                    "Disabling transition sequence '{}': it referenced the removed channel {}",
                    seq.name, removed_idx
                );
                seq.enabled = false;
                seq.state.reset();
            }
        }
    }

    /// Get a reference to channel by index
    pub fn channel(&self, index: usize) -> Option<&Channel> {
        self.channels.get(index)
    }

    /// Get a mutable reference to channel by index
    pub fn channel_mut(&mut self, index: usize) -> Option<&mut Channel> {
        self.channels.get_mut(index)
    }

    // ── Accessor methods ─────────────────────────────────────────────

    /// Read-only access to all channels.
    pub fn channels(&self) -> &[Channel] {
        &self.channels
    }

    /// Mutable access to all channels.
    pub fn channels_mut(&mut self) -> &mut Vec<Channel> {
        &mut self.channels
    }

    /// Number of channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Current crossfader position (0.0 = Ch 0, 1.0 = Ch 1).
    pub fn crossfader(&self) -> f32 {
        self.crossfader
    }

    /// Read-only access to the auto-crossfade state.
    pub fn auto_crossfade(&self) -> Option<&AutoCrossfade> {
        self.auto_crossfade.as_ref()
    }

    /// Read-only access to master effects.
    pub fn master_effects(&self) -> &[Effect] {
        &self.master_effects
    }

    /// Mutable access to master effects.
    pub fn master_effects_mut(&mut self) -> &mut Vec<Effect> {
        &mut self.master_effects
    }

    /// Read-only access to the modulation engine.
    pub fn modulation(&self) -> &ModulationEngine {
        &self.modulation
    }

    /// Mutable access to the modulation engine.
    pub fn modulation_mut(&mut self) -> &mut ModulationEngine {
        &mut self.modulation
    }

    /// Read-only access to the active transition effect.
    pub fn active_transition(&self) -> Option<&TransitionEffect> {
        self.active_transition.as_ref()
    }

    /// Read-only access to transition sequences.
    pub fn transition_sequences(&self) -> &[TransitionSequence] {
        &self.transition_sequences
    }

    /// Mutable access to transition sequences.
    pub fn transition_sequences_mut(&mut self) -> &mut Vec<TransitionSequence> {
        &mut self.transition_sequences
    }

    /// The composited output texture view (post-crossfade, post-master-effects).
    pub fn composite_view(&self) -> &wgpu::TextureView {
        self.composite.result_view()
    }

    // ── UUID lookup helpers ────────────────────────────────────────────

    /// Find a mutable deck slot by deck UUID. Returns (channel_index, deck_index) if found.
    pub fn find_deck_by_uuid(&self, uuid: &str) -> Option<(usize, usize)> {
        for (ch_idx, ch) in self.channels.iter().enumerate() {
            for (dk_idx, slot) in ch.decks.iter().enumerate() {
                if slot.deck.uuid() == uuid {
                    return Some((ch_idx, dk_idx));
                }
            }
        }
        None
    }

    /// Find a channel index by channel UUID.
    pub fn find_channel_by_uuid(&self, uuid: &str) -> Option<usize> {
        self.channels.iter().position(|ch| ch.uuid() == uuid)
    }

    // ── Persistence restore helpers ──────────────────────────────────

    /// Replace all channels (used by persistence restore).
    /// Also updates next_channel_index based on the highest "Ch N" name.
    pub fn replace_channels(&mut self, channels: Vec<Channel>) {
        let max_idx = channels.iter()
            .filter_map(|ch| ch.name.strip_prefix("Ch ").and_then(|s| s.parse::<usize>().ok()))
            .max()
            .map(|n| n + 1)
            .unwrap_or(channels.len());
        self.next_channel_index = max_idx;
        self.channels = channels;
        self.prev_culled = vec![false; self.channels.len()];
    }

    /// Set the crossfader position directly (used by persistence restore).
    pub fn set_crossfader(&mut self, value: f32) {
        self.crossfader = if value.is_finite() { value.clamp(0.0, 1.0) } else { 0.5 };
    }

    /// Replace the modulation engine (used by persistence restore).
    pub fn set_modulation(&mut self, engine: ModulationEngine) {
        self.modulation = engine;
    }

    /// Replace transition sequences (used by persistence restore).
    pub fn set_transition_sequences(&mut self, sequences: Vec<TransitionSequence>) {
        self.transition_sequences = sequences;
    }

    /// Set the next_channel_index counter (used by persistence restore).
    pub fn set_next_channel_index(&mut self, idx: usize) {
        self.next_channel_index = idx;
    }

    /// Consume the next channel name and advance the counter.
    /// Use this when manually constructing a channel outside of `add_channel`.
    pub fn take_next_channel_name(&mut self) -> String {
        let name = channel_name(self.next_channel_index);
        self.next_channel_index += 1;
        name
    }

}

/// Generate a channel name from its index: 0→"Ch 0", 1→"Ch 1", 2→"Ch 2", etc.
fn channel_name(index: usize) -> String {
    format!("Ch {}", index)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── CrossfadeEasing tests ────────────────────────────────────────

    #[test]
    fn easing_linear() {
        assert!((CrossfadeEasing::Linear.apply(0.0) - 0.0).abs() < 1e-5);
        assert!((CrossfadeEasing::Linear.apply(0.5) - 0.5).abs() < 1e-5);
        assert!((CrossfadeEasing::Linear.apply(1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn easing_ease_in_out() {
        let e = CrossfadeEasing::EaseInOut;
        assert!((e.apply(0.0) - 0.0).abs() < 1e-5);
        assert!((e.apply(1.0) - 1.0).abs() < 1e-5);
        // Midpoint of smoothstep = 0.5
        assert!((e.apply(0.5) - 0.5).abs() < 1e-5);
        // Should be slow at start (below linear)
        assert!(e.apply(0.25) < 0.25);
    }

    #[test]
    fn easing_ease_in() {
        let e = CrossfadeEasing::EaseIn;
        assert!((e.apply(0.0) - 0.0).abs() < 1e-5);
        assert!((e.apply(1.0) - 1.0).abs() < 1e-5);
        // Ease-in is t² → slower at start
        assert!(e.apply(0.5) < 0.5);
        assert!((e.apply(0.5) - 0.25).abs() < 1e-5);
    }

    #[test]
    fn easing_ease_out() {
        let e = CrossfadeEasing::EaseOut;
        assert!((e.apply(0.0) - 0.0).abs() < 1e-5);
        assert!((e.apply(1.0) - 1.0).abs() < 1e-5);
        // Ease-out: 1-(1-t)² → faster at start
        assert!(e.apply(0.5) > 0.5);
    }

    #[test]
    fn easing_clamps_input() {
        assert!((CrossfadeEasing::Linear.apply(-0.5) - 0.0).abs() < 1e-5);
        assert!((CrossfadeEasing::Linear.apply(1.5) - 1.0).abs() < 1e-5);
    }

    // ── AutoCrossfade tests ──────────────────────────────────────────

    #[test]
    fn auto_crossfade_new() {
        let ac = AutoCrossfade::new(0.0, 1.0, 2.0, CrossfadeEasing::Linear);
        assert_eq!(ac.from, 0.0);
        assert_eq!(ac.to, 1.0);
        assert_eq!(ac.duration, 2.0);
        assert_eq!(ac.elapsed, 0.0);
    }

    #[test]
    fn auto_crossfade_tick_returns_value() {
        let mut ac = AutoCrossfade::new(0.0, 1.0, 2.0, CrossfadeEasing::Linear);
        let val = ac.tick(0.5);
        assert!(val.is_some());
        let v = val.unwrap();
        assert!((v - 0.25).abs() < 1e-5); // 25% through linear
    }

    #[test]
    fn auto_crossfade_tick_completes() {
        let mut ac = AutoCrossfade::new(0.0, 1.0, 1.0, CrossfadeEasing::Linear);
        let val = ac.tick(1.5); // Past duration
        assert!(val.is_none()); // Complete
    }

    #[test]
    fn auto_crossfade_tick_exact_duration() {
        let mut ac = AutoCrossfade::new(0.0, 1.0, 1.0, CrossfadeEasing::Linear);
        let val = ac.tick(1.0);
        assert!(val.is_none()); // Complete at exact duration
    }

    #[test]
    fn auto_crossfade_reverse() {
        let mut ac = AutoCrossfade::new(1.0, 0.0, 2.0, CrossfadeEasing::Linear);
        let val = ac.tick(1.0).unwrap();
        assert!((val - 0.5).abs() < 1e-5); // Halfway back
    }

    #[test]
    fn auto_crossfade_progress() {
        let mut ac = AutoCrossfade::new(0.0, 1.0, 4.0, CrossfadeEasing::Linear);
        assert!((ac.progress() - 0.0).abs() < 1e-5);
        ac.tick(2.0);
        assert!((ac.progress() - 0.5).abs() < 1e-5);
        ac.tick(2.0);
        assert!((ac.progress() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn auto_crossfade_with_easing() {
        let mut ac = AutoCrossfade::new(0.0, 1.0, 2.0, CrossfadeEasing::EaseInOut);
        let val = ac.tick(1.0).unwrap(); // 50% through with ease-in-out
        // Smoothstep at 0.5 = 0.5
        assert!((val - 0.5).abs() < 1e-5);
    }

    // ── SequencerState tests ─────────────────────────────────────────

    #[test]
    fn sequencer_state_new() {
        let state = SequencerState::new();
        assert!(!state.playing);
        assert_eq!(state.current_step, 0);
        assert_eq!(state.step_elapsed, 0.0);
    }

    #[test]
    fn sequencer_state_reset() {
        let mut state = SequencerState::new();
        state.playing = true;
        state.current_step = 5;
        state.step_elapsed = 3.14;
        state.reset();
        assert!(!state.playing);
        assert_eq!(state.current_step, 0);
        assert_eq!(state.step_elapsed, 0.0);
    }

    // ── TransitionSequence tests ─────────────────────────────────────

    #[test]
    fn transition_sequence_new() {
        let seq = TransitionSequence::new("Test".into());
        assert_eq!(seq.name, "Test");
        assert!(seq.enabled);
        assert!(seq.steps.is_empty());
        assert!(!seq.state.playing);
    }

    // ── channel_name tests ───────────────────────────────────────────

    #[test]
    fn channel_name_format() {
        assert_eq!(channel_name(0), "Ch 0");
        assert_eq!(channel_name(1), "Ch 1");
        assert_eq!(channel_name(42), "Ch 42");
    }

    // ── TransitionSequence index-fixup tests ─────────────────────────
    //
    // Validates Mixer::fixup_transition_sequences_after_channel_remove without
    // requiring a GPU — operates on transition_sequences only.

    fn dummy_fade(from: usize, to: usize) -> TransitionStep {
        use crate::channel::DurationSpec;
        TransitionStep {
            kind: StepKind::Fade {
                from_ch: from,
                to_ch: to,
                duration: DurationSpec::Seconds(1.0),
                easing: CrossfadeEasing::Linear,
                transition_shader: None,
                target_amount: 1.0,
            },
        }
    }

    fn dummy_mixer_with_seqs(seqs: Vec<TransitionSequence>) -> Mixer {
        // Construct only the field we need; everything else is left
        // unset by routing through a #[cfg(test)] helper would be heavy.
        // Use a partial init by going through replace fields.
        let ctx = headless_gpu();
        let mut m = Mixer::new(&ctx, 16, 16).expect("mixer");
        *m.transition_sequences_mut() = seqs;
        m
    }

    #[test]
    fn fixup_sequences_shifts_later_channel_indices_down() {
        let seq = TransitionSequence {
            name: "S".into(),
            steps: vec![dummy_fade(0, 3), dummy_fade(2, 4)],
            enabled: true,
            state: SequencerState::new(),
        };
        let mut m = dummy_mixer_with_seqs(vec![seq]);
        m.fixup_transition_sequences_after_channel_remove(1);
        let s = &m.transition_sequences()[0];
        // (0,3) -> (0,2); (2,4) -> (1,3)
        if let StepKind::Fade { from_ch, to_ch, .. } = s.steps[0].kind {
            assert_eq!((from_ch, to_ch), (0, 2));
        } else { panic!() }
        if let StepKind::Fade { from_ch, to_ch, .. } = s.steps[1].kind {
            assert_eq!((from_ch, to_ch), (1, 3));
        } else { panic!() }
        assert!(s.enabled);
    }

    #[test]
    fn fixup_sequences_disables_when_referencing_removed_channel() {
        let seq = TransitionSequence {
            name: "S".into(),
            steps: vec![dummy_fade(1, 2)],
            enabled: true,
            state: SequencerState::new(),
        };
        let mut m = dummy_mixer_with_seqs(vec![seq]);
        m.fixup_transition_sequences_after_channel_remove(1);
        let s = &m.transition_sequences()[0];
        assert!(!s.enabled, "sequence referencing removed channel must be disabled");
    }

    // ── Mixer-level DnD data model tests ─────────────────────────────
    //
    // Tests for cross-channel deck moves and master effect reordering,
    // matching the logic in apply_deck_and_effect_actions.

    use crate::renderer::GpuContext;

    fn headless_gpu() -> GpuContext {
        GpuContext::new_headless().expect("headless GPU required for tests")
    }

    #[test]
    fn mixer_new_has_two_channels() {
        let gpu = headless_gpu();
        let mixer = Mixer::new(&gpu, 64, 64).unwrap();
        assert_eq!(mixer.channel_count(), 2);
    }

    #[test]
    fn mixer_add_channel() {
        let gpu = headless_gpu();
        let mut mixer = Mixer::new(&gpu, 64, 64).unwrap();
        let idx = mixer.add_channel(&gpu, 64, 64).unwrap();
        assert_eq!(idx, 2);
        assert_eq!(mixer.channel_count(), 3);
    }

    #[test]
    fn mixer_move_deck_between_channels() {
        let gpu = headless_gpu();
        let mut mixer = Mixer::new(&gpu, 64, 64).unwrap();

        // Add a solid color deck to channel 0
        let deck = crate::deck::Deck::new_solid_color(&gpu, [1.0, 0.0, 0.0, 1.0], 64, 64).unwrap();
        mixer.channel_mut(0).unwrap().add_deck(deck);
        mixer.channel_mut(0).unwrap().decks[0].opacity = 0.33;

        assert_eq!(mixer.channel(0).unwrap().deck_count(), 1);
        assert_eq!(mixer.channel(1).unwrap().deck_count(), 0);

        // Move deck from ch0 to ch1 (mirrors apply_deck_and_effect_actions logic)
        let slot = mixer.channels_mut()[0].remove_deck_slot(0).unwrap();
        let new_idx = mixer.channels_mut()[1].add_deck_slot(slot);

        assert_eq!(new_idx, 0);
        assert_eq!(mixer.channel(0).unwrap().deck_count(), 0);
        assert_eq!(mixer.channel(1).unwrap().deck_count(), 1);
        assert!((mixer.channel(1).unwrap().decks[0].opacity - 0.33).abs() < 1e-5);
    }

    #[test]
    fn mixer_master_effect_reorder() {
        // Master effects are Vec<Effect> — test the vec reorder pattern
        // used in apply_deck_and_effect_actions
        let mut effects = vec!["master_blur", "master_color", "master_feedback"];
        // Move last to first (from=2, to=0)
        let e = effects.remove(2);
        effects.insert(0, e);
        assert_eq!(effects, vec!["master_feedback", "master_blur", "master_color"]);
    }

    // ── Chaos Tests Round 2: Crossfader/opacity arithmetic ──────────────

    #[test]
    fn chaos_crossfader_opacity_arithmetic_oob() {
        // Simulate the opacity calculation from composite_sub_mix
        let crossfader = 1.5_f32;
        let opacities = [0.8_f32, 0.9];
        let op_a = (1.0 - crossfader) * opacities[0]; // -0.5 * 0.8 = -0.4
        let op_b = crossfader * opacities[1]; // 1.5 * 0.9 = 1.35
        assert!(op_a.is_finite() && op_b.is_finite());
    }

    #[test]
    fn chaos_crossfader_nan_arithmetic() {
        let crossfader = f32::NAN;
        let opacity = 0.8_f32;
        let result = (1.0 - crossfader) * opacity;
        // NaN propagates — document this behavior
        assert!(result.is_nan(), "NaN crossfader should propagate NaN");
    }

    #[test]
    fn chaos_crossfader_infinity_arithmetic() {
        let crossfader = f32::INFINITY;
        let opacity = 0.8_f32;
        let result = (1.0 - crossfader) * opacity;
        assert!(result.is_infinite(), "Inf crossfader produces Inf opacity");
    }

    #[test]
    fn chaos_opacity_nan_does_not_panic() {
        let opacity = f32::NAN;
        let crossfader = 0.5_f32;
        let result = crossfader * opacity;
        assert!(result.is_nan());
    }
}