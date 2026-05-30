//! ModulationEngine — manages sources, assignments, and per-frame evaluation.

use super::{AudioValues, ModulationSource, ModulationSourceEntry, ParamModulation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A resolved modulation assignment that uses a direct index into the source value vector.
/// Used to eliminate HashMap lookups in the render hot path.
#[derive(Debug, Clone)]
struct ResolvedAssignment {
    source_idx: usize,
    amount: f32,
}

/// Modulation engine manages sources and assignments for a deck
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModulationEngine {
    /// Available modulation sources (with stable UUIDs)
    pub sources: Vec<ModulationSourceEntry>,
    /// Map from parameter name to list of modulations
    pub assignments: HashMap<String, Vec<ParamModulation>>,
    /// UUID → index cache for O(1) lookups during tick
    #[serde(skip)]
    uuid_to_idx: HashMap<String, usize>,
    #[serde(skip)]
    prev_values: Vec<f32>,
    #[serde(skip)]
    current_values: Vec<f32>,
    #[serde(skip)]
    prev_time: Option<f32>,
    /// Cached topological evaluation order. Rebuilt when sources or
    /// assignments mutate; otherwise reused across frames so the per-frame
    /// tick stops doing an O(N²·MAX_MOD_DEPTH) re-sort that produces the
    /// same result.
    #[serde(skip)]
    cached_order: Vec<usize>,
    #[serde(skip)]
    cached_order_valid: bool,
    /// Pre-resolved mod-on-mod assignments to eliminate per-frame format! and HashMap lookups.
    /// Index matches `self.sources` index.
    #[serde(skip)]
    resolved_mod_on_mod: Vec<Vec<(String, Vec<ResolvedAssignment>)>>,
    /// Version counter incremented whenever sources or assignments mutate.
    /// Used by `ShaderParams` to cache "is modulated" status.
    #[serde(default)]
    pub version: u64,
}

impl ModulationEngine {
    pub fn new() -> Self {
        Self::default()
    }

    fn rebuild_uuid_index(&mut self) {
        self.uuid_to_idx.clear();
        for (i, entry) in self.sources.iter().enumerate() {
            self.uuid_to_idx.insert(entry.uuid.clone(), i);
        }
    }

    /// Ensure uuid_to_idx is populated (needed after deserialization)
    pub fn ensure_index(&mut self) {
        if self.uuid_to_idx.len() != self.sources.len() {
            self.rebuild_uuid_index();
        }
    }

    /// Add a new source, returns its UUID
    pub fn add_source(&mut self, source: ModulationSource) -> String {
        let entry = ModulationSourceEntry::new(source);
        let uuid = entry.uuid.clone();
        self.sources.push(entry);
        self.prev_values.push(0.0);
        self.current_values.push(0.0);
        self.uuid_to_idx
            .insert(uuid.clone(), self.sources.len() - 1);
        self.invalidate_evaluation_order();
        self.version += 1;
        uuid
    }

    /// Add a source with a specific UUID (for preset loading)
    pub fn add_source_with_uuid(&mut self, uuid: String, source: ModulationSource) -> String {
        let entry = ModulationSourceEntry::with_uuid(uuid.clone(), source);
        self.sources.push(entry);
        self.prev_values.push(0.0);
        self.current_values.push(0.0);
        self.uuid_to_idx
            .insert(uuid.clone(), self.sources.len() - 1);
        self.invalidate_evaluation_order();
        self.version += 1;
        uuid
    }

    /// Remove a source by UUID
    pub fn remove_source(&mut self, uuid: &str) {
        if let Some(idx) = self.uuid_to_idx.get(uuid).copied() {
            self.sources.remove(idx);
            if idx < self.prev_values.len() {
                self.prev_values.remove(idx);
            }
            if idx < self.current_values.len() {
                self.current_values.remove(idx);
            }
            // Remove assignments referencing this source (no reindexing needed)
            for mods in self.assignments.values_mut() {
                mods.retain(|m| m.source_id != uuid);
            }
            // Remove mod-on-mod assignments targeting this source
            let mod_prefix = format!("mod:{}:", uuid);
            self.assignments.retain(|k, _| !k.starts_with(&mod_prefix));
            self.rebuild_uuid_index();
            self.invalidate_evaluation_order();
            self.version += 1;
        }
    }

    /// Remove all assignments whose key starts with the given prefix.
    /// Used to clean up orphaned assignments when a deck or effect is removed.
    pub fn remove_assignments_with_prefix(&mut self, prefix: &str) {
        let before = self.assignments.len();
        self.assignments.retain(|k, _| !k.starts_with(prefix));
        let removed = before - self.assignments.len();
        if removed > 0 {
            log::info!(
                "Removed {} orphaned modulation assignments with prefix '{}'",
                removed,
                prefix
            );
            self.invalidate_evaluation_order();
            self.version += 1;
        }
    }

    pub fn assign(
        &mut self,
        param_name: &str,
        source_id: &str,
        amount: f32,
        component: Option<usize>,
    ) {
        if !self.uuid_to_idx.contains_key(source_id) {
            self.ensure_index();
            if !self.uuid_to_idx.contains_key(source_id) {
                return;
            }
        }
        let modulation = ParamModulation {
            source_id: source_id.to_string(),
            amount,
            component,
        };
        self.assignments
            .entry(param_name.to_string())
            .or_default()
            .push(modulation);
        // Only mod-on-mod assignments alter the topology; non-mod assignments
        // don't change evaluation order but invalidating universally keeps
        // the cache invariant simple.
        self.invalidate_evaluation_order();
        self.version += 1;
    }

    pub fn assign_mod_on_mod(
        &mut self,
        target_uuid: &str,
        param_name: &str,
        modulator_uuid: &str,
        amount: f32,
    ) {
        let key = format!("mod:{}:{}", target_uuid, param_name);
        self.assign(&key, modulator_uuid, amount, None);
        // assign() already increments version
    }

    pub fn clear_mod_on_mod(&mut self, target_uuid: &str, param_name: &str) {
        let key = format!("mod:{}:{}", target_uuid, param_name);
        if self.assignments.remove(&key).is_some() {
            self.invalidate_evaluation_order();
            self.version += 1;
        }
    }

    pub fn clear_assignments(&mut self, param_name: &str) {
        if self.assignments.remove(param_name).is_some() {
            self.invalidate_evaluation_order();
            self.version += 1;
        }
    }

    pub fn trigger_adsr(&mut self, uuid: &str) {
        if let Some(&idx) = self.uuid_to_idx.get(uuid) {
            self.sources[idx].source.gate_on();
        }
    }

    pub fn release_adsr(&mut self, uuid: &str) {
        if let Some(&idx) = self.uuid_to_idx.get(uuid) {
            self.sources[idx].source.gate_off();
        }
    }

    /// Get a mutable reference to a source by UUID
    pub fn source_mut(&mut self, uuid: &str) -> Option<&mut ModulationSource> {
        self.ensure_index();
        self.uuid_to_idx
            .get(uuid)
            .copied()
            .map(|idx| &mut self.sources[idx].source)
    }

    /// Find source by UUID (returns exists check)
    pub fn has_source(&self, uuid: &str) -> bool {
        self.sources.iter().any(|e| e.uuid == uuid)
    }

    pub(crate) fn source_idx(&self, uuid: &str) -> Option<usize> {
        self.uuid_to_idx.get(uuid).copied()
    }

    fn resolve_mod_on_mod_cache(&mut self) {
        let n = self.sources.len();
        self.resolved_mod_on_mod = vec![Vec::new(); n];

        for i in 0..n {
            let uuid = &self.sources[i].uuid;
            let params = match &self.sources[i].source {
                ModulationSource::LFO { .. } => &["frequency", "phase", "amplitude"][..],
                ModulationSource::AudioBand { .. } => &["gain", "smoothing"][..],
                ModulationSource::ADSR { .. } => &["attack", "decay", "sustain", "release"][..],
                ModulationSource::StepSequencer { .. } => &["rate"][..],
            };

            for &param in params {
                let key = format!("mod:{}:{}", uuid, param);
                if let Some(mods) = self.assignments.get(&key) {
                    let mut resolved = Vec::new();
                    for m in mods {
                        if let Some(&src_idx) = self.uuid_to_idx.get(&m.source_id) {
                            resolved.push(ResolvedAssignment {
                                source_idx: src_idx,
                                amount: m.amount,
                            });
                        }
                    }
                    if !resolved.is_empty() {
                        self.resolved_mod_on_mod[i].push((param.to_string(), resolved));
                    }
                }
            }
        }
    }

    fn apply_mod_on_mod_optimized(
        &self,
        idx: usize,
        source: &ModulationSource,
    ) -> ModulationSource {
        let resolved_params = &self.resolved_mod_on_mod[idx];
        if resolved_params.is_empty() {
            return source.clone();
        }

        let mut modified = source.clone();
        for (param_name, assignments) in resolved_params {
            let mut offset = 0.0;
            for m in assignments {
                if let Some(val) = self.current_values.get(m.source_idx) {
                    offset += val * m.amount;
                }
            }

            match &mut modified {
                ModulationSource::LFO {
                    frequency,
                    phase,
                    amplitude,
                    ..
                } => match param_name.as_str() {
                    "frequency" => *frequency = (*frequency + offset).max(0.001),
                    "phase" => *phase = (*phase + offset).clamp(0.0, 1.0),
                    "amplitude" => *amplitude = (*amplitude + offset).clamp(0.0, 1.0),
                    _ => {}
                },
                ModulationSource::AudioBand {
                    gain, smoothing, ..
                } => match param_name.as_str() {
                    "gain" => *gain = (*gain + offset).max(0.0),
                    "smoothing" => *smoothing = (*smoothing + offset).clamp(0.0, 0.99),
                    _ => {}
                },
                ModulationSource::ADSR {
                    attack,
                    decay,
                    sustain,
                    release,
                    ..
                } => match param_name.as_str() {
                    "attack" => *attack = (*attack + offset).max(0.001),
                    "decay" => *decay = (*decay + offset).max(0.001),
                    "sustain" => *sustain = (*sustain + offset).clamp(0.0, 1.0),
                    "release" => *release = (*release + offset).max(0.001),
                    _ => {}
                },
                ModulationSource::StepSequencer { rate, .. } => match param_name.as_str() {
                    "rate" => *rate = (*rate + offset).max(0.01),
                    _ => {}
                },
            }
        }
        modified
    }

    /// Mark the cached evaluation order stale. Call from every mutation that
    /// can affect topology: source add/remove, assignment add/remove/clear,
    /// or any code path that reorders `self.sources`.
    pub(crate) fn invalidate_evaluation_order(&mut self) {
        self.cached_order_valid = false;
    }

    /// Topological evaluation order honoring mod-on-mod dependencies.
    /// Result is cached and reused across frames; mutators must call
    /// `invalidate_evaluation_order()` after touching `sources` or
    /// `assignments`.
    pub(crate) fn evaluation_order(&mut self) -> Vec<usize> {
        self.ensure_index();
        if self.cached_order_valid && self.cached_order.len() == self.sources.len() {
            return self.cached_order.clone();
        }

        self.resolve_mod_on_mod_cache();

        const MAX_MOD_DEPTH: usize = 4;
        let n = self.sources.len();
        self.cached_order.clear();
        if n == 0 {
            self.cached_order_valid = true;
            return Vec::new();
        }

        let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (key, mods) in &self.assignments {
            if let Some(target_uuid) = Self::parse_mod_target(key) {
                if let Some(target_idx) = self.source_idx(target_uuid) {
                    for m in mods {
                        if let Some(src_idx) = self.source_idx(&m.source_id) {
                            if src_idx != target_idx {
                                deps[target_idx].push(src_idx);
                            }
                        }
                    }
                }
            }
        }

        let mut order = Vec::with_capacity(n);
        let mut evaluated = vec![false; n];
        for _pass in 0..MAX_MOD_DEPTH {
            let mut progress = false;
            for i in 0..n {
                if evaluated[i] {
                    continue;
                }
                if deps[i].iter().all(|&d| evaluated[d]) {
                    order.push(i);
                    evaluated[i] = true;
                    progress = true;
                }
            }
            if !progress {
                break;
            }
        }
        for i in 0..n {
            if !evaluated[i] {
                order.push(i);
            }
        }
        self.cached_order = order.clone();
        self.cached_order_valid = true;
        order
    }

    /// Parse mod-on-mod key: "mod:{uuid}:{param}" → Some(uuid)
    pub(crate) fn parse_mod_target(key: &str) -> Option<&str> {
        let parts: Vec<&str> = key.splitn(3, ':').collect();
        if parts.len() >= 2 && parts[0] == "mod" {
            Some(parts[1])
        } else {
            None
        }
    }

    /// Update all source values for the current frame
    pub fn update(&mut self, time: f32, audio: &AudioValues) {
        self.ensure_index();
        let dt = self.prev_time.map_or(0.016, |prev| time - prev);
        self.prev_time = Some(time);

        while self.prev_values.len() < self.sources.len() {
            self.prev_values.push(0.0);
        }
        while self.current_values.len() < self.sources.len() {
            self.current_values.push(0.0);
        }

        let order = self.evaluation_order();
        for i in order {
            // Optimization: avoid cloning and applying mod-on-mod if no assignments exist for this source
            let value = if self.resolved_mod_on_mod[i].is_empty() {
                self.sources[i]
                    .source
                    .calculate(time, dt, audio, self.prev_values[i])
            } else {
                let mut effective = self.apply_mod_on_mod_optimized(i, &self.sources[i].source);
                let val = effective.calculate(time, dt, audio, self.prev_values[i]);

                // Copy back mutable state changes (ADSR stage progression)
                if let (
                    ModulationSource::ADSR {
                        stage,
                        stage_time,
                        current_level,
                        ..
                    },
                    ModulationSource::ADSR {
                        stage: eff_stage,
                        stage_time: eff_st,
                        current_level: eff_cl,
                        ..
                    },
                ) = (&mut self.sources[i].source, &effective)
                {
                    *stage = *eff_stage;
                    *stage_time = *eff_st;
                    *current_level = *eff_cl;
                }
                val
            };

            self.current_values[i] = value;
            self.prev_values[i] = value;
        }
    }

    /// Get the total modulation offset for a scalar parameter
    pub fn get_modulation(&self, param_name: &str) -> f32 {
        self.get_modulation_for_component(param_name, None)
    }

    /// Get the total modulation offset for a specific component (color params)
    pub fn get_modulation_for_component(&self, param_name: &str, component: Option<usize>) -> f32 {
        let Some(mods) = self.assignments.get(param_name) else {
            return 0.0;
        };
        let mut total = 0.0;
        for m in mods {
            if m.component == component {
                if let Some(&idx) = self.uuid_to_idx.get(&m.source_id) {
                    if idx < self.current_values.len() {
                        total += self.current_values[idx] * m.amount;
                    }
                }
            }
        }
        total
    }

    /// Check if a parameter has any modulations assigned
    pub fn has_modulation(&self, param_name: &str) -> bool {
        self.assignments
            .get(param_name)
            .map_or(false, |v| !v.is_empty())
    }

    /// Get number of sources
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Get current computed values for all sources (for UI visualization)
    pub fn current_values(&self) -> &[f32] {
        &self.current_values
    }

    /// Get current value for a source by UUID
    pub fn current_value_for(&self, uuid: &str) -> f32 {
        self.uuid_to_idx
            .get(uuid)
            .and_then(|&idx| self.current_values.get(idx).copied())
            .unwrap_or(0.0)
    }

    /// Find an existing source by UUID
    pub fn find_source_by_uuid(&self, uuid: &str) -> Option<&ModulationSourceEntry> {
        self.sources.iter().find(|e| e.uuid == uuid)
    }

    /// Iterate over all assignments (key → modulations).
    pub fn assignments_iter(
        &self,
    ) -> impl Iterator<Item = (&String, &Vec<super::ParamModulation>)> {
        self.assignments.iter()
    }
}
