//! Mixer render pipeline — compositing, master effects, sub-mixes.

use crate::renderer::{GpuContext, ISFUniforms};
use anyhow::Result;
use super::{Mixer, CrossfadeEasing, AutoCrossfade};

impl Mixer {
    /// Pre-update modulation engine with latest audio data.
    pub fn update_modulation(&mut self, audio_values: &crate::modulation::AudioValues) {
        let time = self.start_time.elapsed().as_secs_f32();
        self.modulation.update(time, audio_values);
    }

    /// Render all channels and composite them via crossfader, then apply master effects.
    pub fn render(&mut self, context: &GpuContext, audio_data: &crate::audio::AudioData, audio_values: &crate::modulation::AudioValues) -> Result<()> {
        let now = std::time::Instant::now();
        // Clamp dt to a sane window: a startup spike or paused window can hand
        // us multi-second deltas that would blow up frame-rate-dependent
        // shader math. Floor at 1ms so divisions stay finite.
        let dt = (now - self.last_render_time).as_secs_f32().clamp(0.001, 0.1);
        self.last_render_time = now;

        // Tick auto-crossfade
        if let Some(auto) = &mut self.auto_crossfade {
            match auto.tick(dt) {
                Some(value) => self.crossfader = value,
                None => {
                    let target = auto.to;
                    self.crossfader = target;
                    self.auto_crossfade = None;
                    log::info!("Auto-crossfade complete, crossfader = {:.2}", target);
                }
            }
        }

        // Handle beat-synced crossfade
        if let Some(bsc) = &mut self.beat_sync_crossfade {
            if !bsc.started {
                let phase = audio_data.beat_phase();
                if phase < 0.05 && audio_data.bpm.is_some() {
                    let bpm = audio_data.bpm.unwrap_or(120.0);
                    let duration_secs = bsc.beats * 60.0 / bpm;
                    bsc.auto = Some(AutoCrossfade::new(
                        self.crossfader, bsc.to, duration_secs, CrossfadeEasing::EaseInOut,
                    ));
                    bsc.started = true;
                    log::info!("Beat-synced crossfade started: {:.1} beats at {:.0} BPM = {:.2}s",
                        bsc.beats, bpm, duration_secs);
                }
            }

            if let Some(auto) = &mut bsc.auto {
                match auto.tick(dt) {
                    Some(value) => self.crossfader = value,
                    None => {
                        let target = bsc.to;
                        self.crossfader = target;
                        self.beat_sync_crossfade = None;
                        log::info!("Beat-synced crossfade complete, crossfader = {:.2}", target);
                    }
                }
            }
        }

        // Tick transition sequence
        let bpm = audio_data.bpm.map(|b| b as f64);
        self.tick_sequence(dt, bpm);

        // Update global modulation engine
        let time = self.start_time.elapsed().as_secs_f32();
        self.modulation.update(time, audio_values);

        // Compute effective opacity per channel
        let channel_count = self.channels.len();
        let effective_opacities: Vec<f32> = if channel_count == 2 {
            vec![
                (1.0 - self.crossfader) * self.channels[0].opacity,
                self.crossfader * self.channels[1].opacity,
            ]
        } else {
            self.channels.iter().map(|ch| ch.opacity).collect()
        };

        // Always tick video frames on every channel so players stay in sync
        // even when a channel is fully faded out by the crossfader.
        for channel in self.channels.iter_mut() {
            channel.tick_video_frames(context);
        }

        // Ensure prev_culled tracks the current channel count (channels may be
        // added or removed between frames). A newly-added channel starts as
        // "not culled" so its first cull triggers a clear edge.
        if self.prev_culled.len() != self.channels.len() {
            self.prev_culled.resize(self.channels.len(), false);
        }

        let mut clear_cmds = Vec::new();
        for (ch_idx, channel) in self.channels.iter_mut().enumerate() {
            let is_culled = effective_opacities.get(ch_idx).copied().unwrap_or(0.0) < 0.001;
            let was_culled = self.prev_culled[ch_idx];
            if is_culled {
                // Reset stats so culled channels don't show stale render metrics
                channel.render_time_ms = 0.0;
                channel.active_deck_count = 0;
                // One-shot clear on the visible→culled transition only: a
                // channel that stays culled across frames keeps its already-
                // black composite_view, and a channel that re-enters visibility
                // overwrites it on its next render().
                if !was_culled {
                    clear_cmds.push(channel.clear_composite_cmd(context));
                }
                self.prev_culled[ch_idx] = true;
                continue;
            }
            self.prev_culled[ch_idx] = false;
            if let Err(e) = channel.render(context, audio_data, &self.modulation, ch_idx, time, dt) {
                log::error!("Channel {} render failed, skipping: {}", ch_idx, e);
                continue;
            }
        }
        if !clear_cmds.is_empty() {
            context.queue.submit(clear_cmds);
        }

        self.sync_transition_progress();
        self.composite_channels(context, dt)?;
        self.apply_master_effects(context, audio_data, time, dt)?;

        self.frame_count += 1;
        Ok(())
    }


    /// Per-channel opacity weights for the opacity-based composite passes.
    ///
    /// In 2-channel mode this returns the corrected pre-scaled weights for
    /// the two-pass blit→composite path. Channel B's effective crossfade
    /// weight is used as the second pass's composite opacity; channel A is
    /// pre-scaled so that after the second pass attenuates the destination
    /// by (1 - b), the final result is still the requested linear mix:
    ///     ((1-cf)·opacityA)·A + (cf·opacityB)·B.
    /// For >2 channels each channel just contributes its own opacity.
    fn composite_pass_opacities(&self) -> Vec<f32> {
        if self.channels.len() == 2 {
            let b_weight = (self.crossfader * self.channels[1].opacity).clamp(0.0, 1.0);
            let a_weight = ((1.0 - self.crossfader) * self.channels[0].opacity).clamp(0.0, 1.0);
            let a_copy_opacity = if b_weight < 1.0 {
                (a_weight / (1.0 - b_weight)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            vec![a_copy_opacity, b_weight]
        } else {
            self.channels.iter().map(|ch| ch.opacity).collect()
        }
    }

    fn composite_channels(&mut self, context: &GpuContext, dt: f32) -> Result<()> {
        let channel_count = self.channels.len();
        if channel_count == 0 {
            let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Mixer Clear Encoder"),
            });
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Mixer Clear Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.composite_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            }
            context.queue.submit(std::iter::once(encoder.finish()));
            return Ok(());
        }

        // If we have exactly 2 channels and a transition shader is active, use it
        if channel_count == 2 {
            if let Some(transition) = &self.active_transition {
                let width = self.composite_texture.width();
                let height = self.composite_texture.height();

                let uniforms = ISFUniforms {
                    time: self.start_time.elapsed().as_secs_f32(),
                    time_delta: dt,
                    frame_index: self.frame_count as u32,
                    pass_index: 0,
                    render_size: [width as f32, height as f32],
                    phase_times: [0.0; 4],
                    ..Default::default()
                };

                let params_data = transition.params.build_buffer_data();
                if let Some(buf) = transition.params.buffer() {
                    context.queue.write_buffer(buf, 0, &params_data);
                }

                transition.pipeline.render_to(
                    context,
                    &self.channels[0].composite_view,
                    &self.channels[1].composite_view,
                    &self.composite_view,
                    &uniforms,
                    transition.params.buffer(),
                );

                return Ok(());
            }
        }

        // Fallback: opacity-based crossfade
        //
        // For 2-channel mode the first channel is blitted onto a cleared-to-black
        // target using ALPHA_BLENDING.  The hardware blend applies SrcAlpha to the
        // RGB output, so if the blit shader also multiplies alpha by opacity, the
        // effective weight becomes opacity² (double-application).
        //
        // To avoid this, channel B's effective crossfade weight is used as the
        // second channel's composite opacity. Channel A is pre-scaled so that,
        // after the second pass attenuates the destination by (1 - b), the final
        // result is still the requested linear mix:
        //     ((1-cf)·opacityA)·A + (cf·opacityB)·B.
        let opacities: Vec<f32> = self.composite_pass_opacities();

        // Make sure the per-channel bind-group caches track the current
        // channel count. resize() / channel mutations clear these, but a
        // length mismatch can also occur if a future caller forgets — fix
        // it lazily here.
        if self.channel_blit_bg_cache.len() != self.channels.len() {
            self.channel_blit_bg_cache.resize_with(self.channels.len(), || None);
        }
        if self.channel_composite_bg_cache.len() != self.channels.len() {
            self.channel_composite_bg_cache.resize_with(self.channels.len(), || None);
        }

        // Submit per-channel to ensure each channel's uniform buffer writes
        // are consumed before the next channel overwrites them.
        let mut is_first = true;
        for i in 0..self.channels.len() {
            let opacity = opacities[i];
            if opacity <= 0.0 { continue; }

            if is_first {
                // First visible channel: simple blit copy
                self.blit_pipeline.set_opacity(&context.queue, opacity);
                if self.channel_blit_bg_cache[i].is_none() {
                    self.channel_blit_bg_cache[i] = Some(
                        self.blit_pipeline.create_bind_group(&context.device, &self.channels[i].composite_view),
                    );
                }
                let bind_group = self.channel_blit_bg_cache[i].as_ref().unwrap();
                let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Mixer Composite Encoder (first)"),
                });
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Mixer Composite Pass (first)"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.composite_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    self.blit_pipeline.render(&mut render_pass, bind_group);
                }
                context.queue.submit(std::iter::once(encoder.finish()));
                is_first = false;
            } else {
                // Subsequent channels: snapshot + composite shader
                let mut copy_encoder = context.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("Mixer Snapshot Copy") },
                );
                copy_encoder.copy_texture_to_texture(
                    self.composite_texture.as_image_copy(),
                    self.effect_ping_texture.as_image_copy(),
                    self.composite_texture.size(),
                );

                let blend_mode = self.channels[i].blend_mode;
                self.composite_pipeline.set_params(&context.queue, opacity, blend_mode.to_index(), [1.0, 1.0], [0.0, 0.0]);
                if self.channel_composite_bg_cache[i].is_none() {
                    self.channel_composite_bg_cache[i] = Some(
                        self.composite_pipeline.create_bind_group(
                            &context.device,
                            &self.channels[i].composite_view,
                            &self.effect_ping_view,
                        ),
                    );
                }
                let bind_group = self.channel_composite_bg_cache[i].as_ref().unwrap();
                let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Mixer Composite Encoder"),
                });
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Mixer Composite Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.composite_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    self.composite_pipeline.render(&mut render_pass, bind_group);
                }
                context.queue.submit([copy_encoder.finish(), encoder.finish()]);
            }
        }

        Ok(())
    }

    /// Prepare sub-mix textures for all unique multi-channel surface sources.
    pub fn prepare_sub_mixes(&mut self, sources: &[Vec<usize>], context: &GpuContext) {
        let needed: std::collections::HashSet<Vec<usize>> = sources.iter().cloned().collect();
        self.sub_mix_cache.retain(|k, _| needed.contains(k));

        for mut indices in sources.iter().cloned() {
            indices.sort();
            indices.dedup();
            if !self.sub_mix_cache.contains_key(&indices) {
                let width = self.composite_texture.width();
                let height = self.composite_texture.height();
                let tex = context.create_render_texture(width, height);
                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                self.sub_mix_cache.insert(indices.clone(), (tex, view));
            }
            self.composite_sub_mix(&indices, context);
        }
    }

    /// Composite a specific subset of channels into the cached sub-mix texture.
    fn composite_sub_mix(&self, indices: &[usize], context: &GpuContext) {
        let (sub_tex, sub_view) = match self.sub_mix_cache.get(indices) {
            Some(entry) => entry,
            None => return,
        };

        // Same linear-crossfade formula as composite_channels — share the helper
        // so sub-mixes get the corrected opacity-compensation math.
        let opacities: Vec<f32> = self.composite_pass_opacities();

        // Submit per-channel to ensure each channel's uniform buffer writes
        // are consumed before the next channel overwrites them.
        let mut is_first = true;
        for &ch_idx in indices {
            if ch_idx >= self.channels.len() { continue; }
            let channel = &self.channels[ch_idx];
            let opacity = opacities.get(ch_idx).copied().unwrap_or(0.0);
            if opacity <= 0.0 { continue; }

            if is_first {
                // First visible channel: simple blit copy
                self.blit_pipeline.set_opacity(&context.queue, opacity);
                let bind_group = self.blit_pipeline.create_bind_group(&context.device, &channel.composite_view);
                let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Sub-mix Composite Encoder (first)"),
                });
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Sub-mix Composite Pass (first)"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: sub_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    self.blit_pipeline.render(&mut render_pass, &bind_group);
                }
                context.queue.submit(std::iter::once(encoder.finish()));
                is_first = false;
            } else {
                // Subsequent channels: snapshot sub-mix → effect_ping, composite shader
                let mut copy_encoder = context.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("Sub-mix Snapshot Copy") },
                );
                copy_encoder.copy_texture_to_texture(
                    sub_tex.as_image_copy(),
                    self.effect_ping_texture.as_image_copy(),
                    sub_tex.size(),
                );

                let blend_mode = channel.blend_mode;
                self.composite_pipeline.set_params(&context.queue, opacity, blend_mode.to_index(), [1.0, 1.0], [0.0, 0.0]);
                let bind_group = self.composite_pipeline.create_bind_group(&context.device, &channel.composite_view, &self.effect_ping_view);
                let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Sub-mix Composite Encoder"),
                });
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Sub-mix Composite Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: sub_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    self.composite_pipeline.render(&mut render_pass, &bind_group);
                }
                context.queue.submit([copy_encoder.finish(), encoder.finish()]);
            }
        }

        if is_first {
            let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sub-mix Clear Encoder"),
            });
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Sub-mix Clear Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: sub_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            }
            context.queue.submit(std::iter::once(encoder.finish()));
        }
    }

    /// Get the sub-mix texture view for a given set of channel indices.
    pub fn get_sub_mix_view(&self, indices: &[usize]) -> Option<&wgpu::TextureView> {
        self.sub_mix_cache.get(indices).map(|(_, v)| v)
    }


    fn apply_master_effects(&mut self, context: &GpuContext, audio_data: &crate::audio::AudioData, time: f32, dt: f32) -> Result<()> {
        if self.master_effects.is_empty() {
            return Ok(());
        }

        let width = self.composite_texture.width();
        let height = self.composite_texture.height();

        let uniforms = ISFUniforms {
            time,
            time_delta: dt,
            frame_index: self.frame_count as u32,
            pass_index: 0,
            render_size: [width as f32, height as f32],
            audio_level: audio_data.level,
            audio_bass: audio_data.bass(),
            audio_mid: audio_data.mid(),
            audio_treble: audio_data.treble(),
            audio_bpm: audio_data.bpm.unwrap_or(0.0),
            audio_beat_phase: audio_data.beat_phase(),
            date: crate::deck::get_current_date(),
            phase_times: [0.0; 4],
        };

        let mut read_from_composite = true;
        let mut cmd_buffers: Vec<wgpu::CommandBuffer> = Vec::new();

        for effect in self.master_effects.iter_mut() {
            if !effect.enabled { continue; }

            let (input_view, output_view) = if read_from_composite {
                (&self.composite_view, &self.effect_ping_view)
            } else {
                (&self.effect_ping_view, &self.composite_view)
            };

            effect.apply(context, input_view, output_view, &uniforms, &mut cmd_buffers)?;
            read_from_composite = !read_from_composite;
        }

        if !read_from_composite {
            let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Master Effect Final Copy Encoder"),
            });
            encoder.copy_texture_to_texture(
                self.effect_ping_texture.as_image_copy(),
                self.composite_texture.as_image_copy(),
                self.composite_texture.size(),
            );
            cmd_buffers.push(encoder.finish());
        }

        if !cmd_buffers.is_empty() {
            context.queue.submit(cmd_buffers);
        }

        Ok(())
    }
}
