//! Shader parameter system for ISF user inputs

use crate::isf::ISFInput;
use crate::modulation::ModulationEngine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Cached metadata for parameter rendering/modulation to avoid HashMap lookups
#[derive(Debug, Clone)]
struct ParamRenderInfo {
    min: f32,
    max: f32,
    range: f32,
}

impl Default for ParamRenderInfo {
    fn default() -> Self {
        Self {
            min: 0.0,
            max: 1.0,
            range: 1.0,
        }
    }
}

/// A resolved modulation assignment that uses a direct index into the source value vector
#[derive(Debug, Clone)]
struct ResolvedMod {
    source_idx: usize,
    amount: f32,
    component: Option<usize>,
}

/// Parameter value types matching ISF input types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(untagged)]
pub enum ParamValue {
    Float(f32),
    Bool(bool),
    Long(i32),
    Color([f32; 4]),
    Point2D([f32; 2]),
}

impl ParamValue {
    /// Create from ISF input default value
    pub fn from_isf_input(input: &ISFInput) -> Self {
        match input.input_type.as_str() {
            "float" => {
                let val = input
                    .default
                    .as_ref()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                ParamValue::Float(val)
            }
            "bool" => {
                let val = input
                    .default
                    .as_ref()
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                ParamValue::Bool(val)
            }
            "long" => {
                let val = input.default.as_ref().and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                ParamValue::Long(val)
            }
            "color" => {
                let arr = input
                    .default
                    .as_ref()
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        let mut color = [1.0f32; 4];
                        for (i, val) in arr.iter().take(4).enumerate() {
                            color[i] = val.as_f64().unwrap_or(1.0) as f32;
                        }
                        color
                    })
                    .unwrap_or([1.0, 1.0, 1.0, 1.0]);
                ParamValue::Color(arr)
            }
            "point2D" => {
                let arr = input
                    .default
                    .as_ref()
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        let mut point = [0.0f32; 2];
                        for (i, val) in arr.iter().take(2).enumerate() {
                            point[i] = val.as_f64().unwrap_or(0.0) as f32;
                        }
                        point
                    })
                    .unwrap_or([0.0, 0.0]);
                ParamValue::Point2D(arr)
            }
            _ => ParamValue::Float(0.0), // Default fallback
        }
    }

    /// Size in bytes (aligned to 4 bytes for GPU)
    pub fn byte_size(&self) -> usize {
        match self {
            ParamValue::Float(_) => 4,
            ParamValue::Bool(_) => 4, // Stored as u32
            ParamValue::Long(_) => 4,
            ParamValue::Color(_) => 16,
            ParamValue::Point2D(_) => 8,
        }
    }

    /// Write value to byte buffer
    pub fn write_bytes(&self, buffer: &mut Vec<u8>) {
        match self {
            ParamValue::Float(v) => buffer.extend_from_slice(&v.to_le_bytes()),
            ParamValue::Bool(v) => {
                buffer.extend_from_slice(&(if *v { 1u32 } else { 0u32 }).to_le_bytes())
            }
            ParamValue::Long(v) => buffer.extend_from_slice(&v.to_le_bytes()),
            ParamValue::Color(v) => {
                for f in v {
                    buffer.extend_from_slice(&f.to_le_bytes());
                }
            }
            ParamValue::Point2D(v) => {
                for f in v {
                    buffer.extend_from_slice(&f.to_le_bytes());
                }
            }
        }
    }

    /// Write value directly to a byte slice at its beginning
    pub fn write_to_buffer(&self, buffer: &mut [u8]) {
        match self {
            ParamValue::Float(v) => buffer[0..4].copy_from_slice(&v.to_le_bytes()),
            ParamValue::Bool(v) => {
                buffer[0..4].copy_from_slice(&(if *v { 1u32 } else { 0u32 }).to_le_bytes())
            }
            ParamValue::Long(v) => buffer[0..4].copy_from_slice(&v.to_le_bytes()),
            ParamValue::Color(v) => {
                for (i, f) in v.iter().enumerate() {
                    let start = i * 4;
                    buffer[start..start + 4].copy_from_slice(&f.to_le_bytes());
                }
            }
            ParamValue::Point2D(v) => {
                for (i, f) in v.iter().enumerate() {
                    let start = i * 4;
                    buffer[start..start + 4].copy_from_slice(&f.to_le_bytes());
                }
            }
        }
    }
}

/// Shader parameters - stores current values and GPU buffer
pub struct ShaderParams {
    /// Parameter names in order (for consistent buffer layout)
    pub param_order: Vec<String>,
    /// Fast name-to-index mapping for `values` vector
    #[serde(skip)]
    pub name_to_idx: HashMap<String, usize>,
    /// Current values (matched to `param_order`)
    pub values: Vec<ParamValue>,
    /// ISF input definitions (for UI metadata: min/max/label)
    pub definitions: HashMap<String, ISFInput>,
    /// GPU buffer (created on demand)
    buffer: Option<wgpu::Buffer>,
    /// Buffer needs re-upload
    dirty: bool,
    /// Cached modulation keys (e.g. "deck_uuid:param_name") to avoid per-frame allocations.
    #[serde(skip)]
    cached_mod_keys: Vec<String>,
    /// The prefix used to generate `cached_mod_keys`.
    #[serde(skip)]
    last_mod_prefix: Option<String>,
    /// Whether the cached keys are valid for the current prefix.
    #[serde(skip)]
    cached_keys_valid: bool,
    /// Per-parameter modulation status cache
    #[serde(skip)]
    is_modulated: Vec<bool>,
    /// The `ModulationEngine` version when `is_modulated` was last updated
    #[serde(skip)]
    last_mod_version: u64,
    /// Cached min/max for float parameters
    #[serde(skip)]
    render_info: Vec<ParamRenderInfo>,
    /// Resolved modulation assignments (per parameter)
    #[serde(skip)]
    resolved_mods: Vec<Vec<ResolvedMod>>,
    /// Byte offsets for each parameter in the uniform buffer.
    #[serde(skip)]
    param_offsets: Vec<usize>,
    /// Cached byte buffer of base parameter values (unmodulated).
    #[serde(skip)]
    base_data: Vec<u8>,
    /// Reusable byte buffer for GPU upload data
    #[serde(skip)]
    data_cache: Vec<u8>,
}

impl ShaderParams {
    /// Create from ISF inputs
    pub fn from_inputs(inputs: &[ISFInput]) -> Self {
        let mut param_order = Vec::new();
        let mut name_to_idx = HashMap::new();
        let mut values = Vec::new();
        let mut definitions = HashMap::new();

        for input in inputs {
            // Skip non-parameter types (image, audio, audioFFT handled separately)
            match input.input_type.as_str() {
                "float" | "bool" | "long" | "color" | "point2D" => {
                    let idx = values.len();
                    param_order.push(input.name.clone());
                    name_to_idx.insert(input.name.clone(), idx);
                    values.push(ParamValue::from_isf_input(input));
                    definitions.insert(input.name.clone(), input.clone());
                }
                _ => {} // Skip image, audio, audioFFT, event
            }
        }

        let is_modulated = vec![false; values.len()];
        let mut render_info = vec![ParamRenderInfo::default(); values.len()];
        let resolved_mods = vec![Vec::new(); values.len()];

        // Pre-calculate render info for float parameters
        for (i, name) in param_order.iter().enumerate() {
            if let Some(def) = definitions.get(name) {
                let min = def.min.unwrap_or(0.0);
                let max = def.max.unwrap_or(1.0);
                render_info[i] = ParamRenderInfo {
                    min,
                    max,
                    range: max - min,
                };
            }
        }

        Self {
            param_order,
            name_to_idx,
            values,
            definitions,
            buffer: None,
            dirty: true,
            cached_mod_keys: Vec::new(),
            last_mod_prefix: None,
            cached_keys_valid: false,
            is_modulated,
            last_mod_version: 0,
            render_info,
            resolved_mods,
            param_offsets: Vec::new(),
            base_data: Vec::new(),
            data_cache: Vec::new(),
        }
    }

    /// Check if this has any parameters
    pub fn is_empty(&self) -> bool {
        self.param_order.is_empty()
    }

    /// Get a parameter value by name
    pub fn get(&self, name: &str) -> Option<&ParamValue> {
        self.name_to_idx
            .get(name)
            .and_then(|&idx| self.values.get(idx))
    }

    /// Get a mutable reference to a parameter value by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut ParamValue> {
        if let Some(&idx) = self.name_to_idx.get(name) {
            self.dirty = true;
            return self.values.get_mut(idx);
        }
        None
    }

    /// Get a float value
    pub fn get_float(&self, name: &str) -> Option<f32> {
        match self.get(name) {
            Some(ParamValue::Float(v)) => Some(*v),
            _ => None,
        }
    }

    /// Set a float value
    pub fn set_float(&mut self, name: &str, value: f32) {
        if let Some(ParamValue::Float(v)) = self.get_mut(name) {
            *v = value;
            self.dirty = true;
        }
    }

    /// Get a bool value
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        match self.get(name) {
            Some(ParamValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    /// Set a bool value
    pub fn set_bool(&mut self, name: &str, value: bool) {
        if let Some(ParamValue::Bool(v)) = self.get_mut(name) {
            *v = value;
            self.dirty = true;
        }
    }

    /// Get a color value
    pub fn get_color(&self, name: &str) -> Option<[f32; 4]> {
        match self.get(name) {
            Some(ParamValue::Color(v)) => Some(*v),
            _ => None,
        }
    }

    /// Set a color value
    pub fn set_color(&mut self, name: &str, value: [f32; 4]) {
        if let Some(ParamValue::Color(v)) = self.get_mut(name) {
            *v = value;
            self.dirty = true;
        }
    }

    /// Get a long (enum) value
    pub fn get_long(&self, name: &str) -> Option<i32> {
        match self.get(name) {
            Some(ParamValue::Long(v)) => Some(*v),
            _ => None,
        }
    }

    /// Set a long value
    pub fn set_long(&mut self, name: &str, value: i32) {
        if let Some(ParamValue::Long(v)) = self.get_mut(name) {
            *v = value;
            self.dirty = true;
        }
    }

    /// Get a point2D value
    pub fn get_point2d(&self, name: &str) -> Option<[f32; 2]> {
        match self.get(name) {
            Some(ParamValue::Point2D(v)) => Some(*v),
            _ => None,
        }
    }

    /// Set a point2D value
    pub fn set_point2d(&mut self, name: &str, value: [f32; 2]) {
        if let Some(ParamValue::Point2D(v)) = self.get_mut(name) {
            *v = value;
            self.dirty = true;
        }
    }

    /// Calculate total buffer size (with std140 alignment)
    pub fn buffer_size(&self) -> usize {
        let mut size = 0usize;
        for value in &self.values {
            // std140 alignment rules
            let alignment = match value {
                ParamValue::Float(_) | ParamValue::Bool(_) | ParamValue::Long(_) => 4,
                ParamValue::Point2D(_) => 8,
                ParamValue::Color(_) => 16,
            };
            // Align to required alignment
            size = (size + alignment - 1) & !(alignment - 1);
            size += value.byte_size();
        }
        // Minimum 16 bytes for wgpu, align to 16
        (size.max(16) + 15) & !15
    }

    /// Build byte buffer for GPU upload (respects std140 alignment rules)
    pub fn build_buffer_data(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(self.buffer_size());
        for value in &self.values {
            // std140 alignment rules:
            // - float, bool, int: 4-byte alignment
            // - vec2: 8-byte alignment
            // - vec3, vec4: 16-byte alignment
            let alignment = match value {
                ParamValue::Float(_) | ParamValue::Bool(_) | ParamValue::Long(_) => 4,
                ParamValue::Point2D(_) => 8,
                ParamValue::Color(_) => 16,
            };
            // Pad to required alignment
            while data.len() % alignment != 0 {
                data.push(0);
            }
            value.write_bytes(&mut data);
        }
        // Pad to minimum 16 bytes
        while data.len() < 16 {
            data.push(0);
        }
        // Align to 16 bytes (uniform buffer requirement)
        while data.len() % 16 != 0 {
            data.push(0);
        }
        data
    }

    /// Create or get GPU buffer
    pub fn ensure_buffer(&mut self, device: &wgpu::Device) -> &wgpu::Buffer {
        if self.buffer.is_none() {
            let data = self.build_buffer_data();
            self.buffer = Some(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Shader Params Buffer"),
                    contents: &data,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                }),
            );
            self.dirty = false;
        }
        self.buffer
            .as_ref()
            .expect("ensure_buffer must be called before buffer() access")
    }

    /// Update GPU buffer if dirty
    pub fn update_buffer(&mut self, queue: &wgpu::Queue) {
        if self.dirty {
            if let Some(buffer) = &self.buffer {
                let data = self.build_buffer_data();
                queue.write_buffer(buffer, 0, &data);
                self.dirty = false;
            }
        }
    }

    /// Get the buffer reference (panics if not created)
    pub fn buffer(&self) -> Option<&wgpu::Buffer> {
        self.buffer.as_ref()
    }

    /// Mark as needing re-upload
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Generic set method for any parameter value
    pub fn set(&mut self, name: &str, value: ParamValue) {
        if let Some(v) = self.get_mut(name) {
            *v = value;
            self.dirty = true;
        }
    }

    /// Reset all parameters to their default values from ISF definitions
    pub fn reset_to_defaults(&mut self) {
        for (i, name) in self.param_order.iter().enumerate() {
            if let Some(definition) = self.definitions.get(name) {
                let default_value = ParamValue::from_isf_input(definition);
                self.values[i] = default_value;
            }
        }
        self.dirty = true;
    }

    /// Build a HashMap of all current parameter values (for persistence/UI)
    pub fn values_map(&self) -> HashMap<String, ParamValue> {
        self.param_order
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), self.values[i]))
            .collect()
    }

    /// Set multiple parameters from a HashMap (for persistence/REST API)
    pub fn update_from_map(&mut self, map: &HashMap<String, ParamValue>) {
        for (name, value) in map {
            self.set(name, *value);
        }
    }

    /// Pre-format modulation keys for the current prefix.
    /// This eliminates per-frame string allocations in the render path.
    fn ensure_mod_keys(&mut self, param_prefix: Option<&str>) {
        let prefix_changed = self.last_mod_prefix.as_deref() != param_prefix;

        if !self.cached_keys_valid || prefix_changed {
            self.cached_mod_keys.clear();
            for name in &self.param_order {
                let key = match param_prefix {
                    Some(prefix) => format!("{}:{}", prefix, name),
                    None => name.clone(),
                };
                self.cached_mod_keys.push(key);
            }
            self.last_mod_prefix = param_prefix.map(|s| s.to_string());
            self.cached_keys_valid = true;
        }
    }

    /// Ensure all cached vectors are correctly sized and populated.
    /// This handles the case where the struct was deserialized (skipping these fields).
    fn ensure_caches(&mut self) {
        let n = self.values.len();
        if self.is_modulated.len() != n {
            self.is_modulated = vec![false; n];
        }
        if self.resolved_mods.len() != n {
            self.resolved_mods = vec![Vec::new(); n];
        }
        if self.param_offsets.len() != n {
            self.param_offsets = vec![0; n];
        }

        if self.render_info.len() != n {
            self.render_info = vec![ParamRenderInfo::default(); n];
            for (i, name) in self.param_order.iter().enumerate() {
                if let Some(def) = self.definitions.get(name) {
                    let min = def.min.unwrap_or(0.0);
                    let max = def.max.unwrap_or(1.0);
                    self.render_info[i] = ParamRenderInfo {
                        min,
                        max,
                        range: max - min,
                    };
                }
            }
        }
    }

    /// Rebuild the base_data cache and param_offsets if the dirty flag is set.
    fn ensure_base_data(&mut self) {
        if self.dirty || self.base_data.is_empty() {
            let size = self.buffer_size();
            self.base_data.clear();
            self.base_data.reserve(size);

            for i in 0..self.values.len() {
                let value = &self.values[i];
                let alignment = match value {
                    ParamValue::Float(_) | ParamValue::Bool(_) | ParamValue::Long(_) => 4,
                    ParamValue::Point2D(_) => 8,
                    ParamValue::Color(_) => 16,
                };
                while self.base_data.len() % alignment != 0 {
                    self.base_data.push(0);
                }
                self.param_offsets[i] = self.base_data.len();
                value.write_bytes(&mut self.base_data);
            }
            while self.base_data.len() < 16 {
                self.base_data.push(0);
            }
            while self.base_data.len() % 16 != 0 {
                self.base_data.push(0);
            }

            self.dirty = false;
        }
    }

    /// Build byte buffer with modulation applied
    /// This creates a temporary modulated value for GPU upload without modifying base values
    /// `param_prefix` is used to look up modulation (e.g., "deck0" to look up "deck0:paramname")
    pub fn build_modulated_buffer_data(
        &mut self,
        modulation: &ModulationEngine,
        param_prefix: Option<&str>,
    ) -> &[u8] {
        self.ensure_caches();
        self.ensure_base_data();

        // Check if prefix changed before ensure_mod_keys updates it
        let prefix_changed = self.last_mod_prefix.as_deref() != param_prefix;
        self.ensure_mod_keys(param_prefix);

        // Refresh modulation cache if version or prefix changed.
        // We resolve all active modulations to direct source indices to eliminate
        // HashMap lookups and O(N) searches in the hot loop.
        if self.last_mod_version != modulation.version || prefix_changed {
            for (i, _name) in self.param_order.iter().enumerate() {
                let mod_key = &self.cached_mod_keys[i];
                self.is_modulated[i] = modulation.has_modulation(mod_key);

                self.resolved_mods[i].clear();
                if let Some(assignments) = modulation.assignments.get(mod_key) {
                    for m in assignments {
                        if let Some(src_idx) = modulation.source_idx(&m.source_id) {
                            self.resolved_mods[i].push(ResolvedMod {
                                source_idx: src_idx,
                                amount: m.amount,
                                component: m.component,
                            });
                        }
                    }
                }
            }
            self.last_mod_version = modulation.version;
        }

        // Fast path: if no parameters are modulated, return base_data directly.
        let any_modulated = self.is_modulated.iter().any(|&m| m);
        if !any_modulated {
            return &self.base_data;
        }

        // Copy cached base data and patch modulated parameters at their pre-calculated offsets.
        self.data_cache.clear();
        self.data_cache.extend_from_slice(&self.base_data);

        let mod_values = modulation.current_values();

        for i in 0..self.values.len() {
            if self.is_modulated[i] {
                let modulated = self.apply_resolved_modulation(i, &self.values[i], mod_values);
                let offset = self.param_offsets[i];
                modulated.write_to_buffer(&mut self.data_cache[offset..]);
            }
        }

        &self.data_cache
    }

    /// Apply pre-resolved modulation to a parameter value using direct source indexing.
    fn apply_resolved_modulation(
        &self,
        idx: usize,
        value: &ParamValue,
        mod_values: &[f32],
    ) -> ParamValue {
        let resolved = &self.resolved_mods[idx];
        if resolved.is_empty() {
            return *value;
        }

        match value {
            ParamValue::Float(base) => {
                let mut offset = 0.0;
                for m in resolved {
                    if let Some(val) = mod_values.get(m.source_idx) {
                        offset += val * m.amount;
                    }
                }

                if offset == 0.0 {
                    return *value;
                }

                let info = &self.render_info[idx];
                let modulated = (base + offset * info.range).clamp(info.min, info.max);
                ParamValue::Float(modulated)
            }
            ParamValue::Color(base) => {
                let mut result = *base;
                let mut changed = false;

                for m in resolved {
                    if let Some(val) = mod_values.get(m.source_idx) {
                        if let Some(comp_idx) = m.component {
                            if comp_idx < 4 {
                                result[comp_idx] =
                                    (result[comp_idx] + val * m.amount).clamp(0.0, 1.0);
                                changed = true;
                            }
                        } else {
                            // Scalar modulation applied to all RGB components
                            for j in 0..3 {
                                result[j] = (result[j] + val * m.amount).clamp(0.0, 1.0);
                            }
                            changed = true;
                        }
                    }
                }
                if changed {
                    ParamValue::Color(result)
                } else {
                    *value
                }
            }
            ParamValue::Point2D(base) => {
                let mut result = *base;
                let mut changed = false;

                for m in resolved {
                    if let Some(val) = mod_values.get(m.source_idx) {
                        if let Some(comp_idx) = m.component {
                            if comp_idx < 2 {
                                result[comp_idx] += val * m.amount;
                                changed = true;
                            }
                        } else {
                            result[0] += val * m.amount;
                            result[1] += val * m.amount;
                            changed = true;
                        }
                    }
                }
                if changed {
                    ParamValue::Point2D(result)
                } else {
                    *value
                }
            }
            _ => *value,
        }
    }

    /// Update GPU buffer with modulation applied
    /// `param_prefix` is used to look up modulation (e.g., "deck0" to look up "deck0:paramname")
    pub fn update_buffer_with_modulation(
        &mut self,
        queue: &wgpu::Queue,
        modulation: &ModulationEngine,
        param_prefix: Option<&str>,
    ) {
        // Clone the buffer handle (cheap Arc clone) to avoid borrow conflict with self during build_modulated_buffer_data
        let buffer = if let Some(b) = &self.buffer {
            b.clone()
        } else {
            return;
        };

        let data = self.build_modulated_buffer_data(modulation, param_prefix);
        queue.write_buffer(&buffer, 0, data);
        // Note: we don't clear dirty flag here since base values may have changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isf::ISFInput;

    fn make_float_input(name: &str, default: f64, min: f32, max: f32) -> ISFInput {
        ISFInput {
            name: name.to_string(),
            input_type: "float".to_string(),
            default: Some(serde_json::json!(default)),
            min: Some(min),
            max: Some(max),
            label: Some(name.to_string()),
            values: None,
            labels: None,
            identity: None,
        }
    }

    fn make_bool_input(name: &str, default: bool) -> ISFInput {
        ISFInput {
            name: name.to_string(),
            input_type: "bool".to_string(),
            default: Some(serde_json::json!(default)),
            min: None,
            max: None,
            label: None,
            values: None,
            labels: None,
            identity: None,
        }
    }

    fn make_color_input(name: &str) -> ISFInput {
        ISFInput {
            name: name.to_string(),
            input_type: "color".to_string(),
            default: Some(serde_json::json!([1.0, 0.0, 0.0, 1.0])),
            min: None,
            max: None,
            label: None,
            values: None,
            labels: None,
            identity: None,
        }
    }

    fn make_long_input(name: &str, default: i64) -> ISFInput {
        ISFInput {
            name: name.to_string(),
            input_type: "long".to_string(),
            default: Some(serde_json::json!(default)),
            min: None,
            max: None,
            label: None,
            values: Some(vec![
                serde_json::json!(0),
                serde_json::json!(1),
                serde_json::json!(2),
            ]),
            labels: Some(vec!["A".into(), "B".into(), "C".into()]),
            identity: None,
        }
    }

    fn make_point2d_input(name: &str) -> ISFInput {
        ISFInput {
            name: name.to_string(),
            input_type: "point2D".to_string(),
            default: Some(serde_json::json!([0.5, 0.5])),
            min: None,
            max: None,
            label: None,
            values: None,
            labels: None,
            identity: None,
        }
    }

    // ── ParamValue tests ─────────────────────────────────────────────

    #[test]
    fn param_value_from_float_input() {
        let input = make_float_input("brightness", 0.75, 0.0, 1.0);
        match ParamValue::from_isf_input(&input) {
            ParamValue::Float(v) => assert!((v - 0.75).abs() < 1e-5),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn param_value_from_bool_input() {
        let input = make_bool_input("enabled", true);
        match ParamValue::from_isf_input(&input) {
            ParamValue::Bool(v) => assert!(v),
            other => panic!("Expected Bool, got {:?}", other),
        }
    }

    #[test]
    fn param_value_from_color_input() {
        let input = make_color_input("tint");
        match ParamValue::from_isf_input(&input) {
            ParamValue::Color(c) => {
                assert!((c[0] - 1.0).abs() < 1e-5);
                assert!((c[1] - 0.0).abs() < 1e-5);
                assert!((c[2] - 0.0).abs() < 1e-5);
                assert!((c[3] - 1.0).abs() < 1e-5);
            }
            other => panic!("Expected Color, got {:?}", other),
        }
    }

    #[test]
    fn param_value_from_long_input() {
        let input = make_long_input("mode", 2);
        match ParamValue::from_isf_input(&input) {
            ParamValue::Long(v) => assert_eq!(v, 2),
            other => panic!("Expected Long, got {:?}", other),
        }
    }

    #[test]
    fn param_value_from_point2d_input() {
        let input = make_point2d_input("center");
        match ParamValue::from_isf_input(&input) {
            ParamValue::Point2D(p) => {
                assert!((p[0] - 0.5).abs() < 1e-5);
                assert!((p[1] - 0.5).abs() < 1e-5);
            }
            other => panic!("Expected Point2D, got {:?}", other),
        }
    }

    #[test]
    fn param_value_byte_sizes() {
        assert_eq!(ParamValue::Float(0.0).byte_size(), 4);
        assert_eq!(ParamValue::Bool(true).byte_size(), 4);
        assert_eq!(ParamValue::Long(0).byte_size(), 4);
        assert_eq!(ParamValue::Color([0.0; 4]).byte_size(), 16);
        assert_eq!(ParamValue::Point2D([0.0; 2]).byte_size(), 8);
    }

    #[test]
    fn param_value_write_bytes_float() {
        let mut buf = Vec::new();
        ParamValue::Float(1.0).write_bytes(&mut buf);
        assert_eq!(buf.len(), 4);
        assert_eq!(f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]), 1.0);
    }

    #[test]
    fn param_value_write_bytes_bool() {
        let mut buf = Vec::new();
        ParamValue::Bool(true).write_bytes(&mut buf);
        let val = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        assert_eq!(val, 1);

        let mut buf2 = Vec::new();
        ParamValue::Bool(false).write_bytes(&mut buf2);
        let val2 = u32::from_le_bytes([buf2[0], buf2[1], buf2[2], buf2[3]]);
        assert_eq!(val2, 0);
    }

    #[test]
    fn param_value_write_bytes_color() {
        let mut buf = Vec::new();
        ParamValue::Color([1.0, 0.5, 0.25, 0.0]).write_bytes(&mut buf);
        assert_eq!(buf.len(), 16);
    }

    #[test]
    fn param_value_write_to_buffer() {
        let mut buf = [0u8; 16];
        ParamValue::Float(1.0).write_to_buffer(&mut buf[0..4]);
        assert_eq!(f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]), 1.0);

        let mut buf = [0u8; 16];
        ParamValue::Color([1.0, 0.5, 0.25, 0.0]).write_to_buffer(&mut buf);
        assert_eq!(f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]), 1.0);
        assert_eq!(f32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]), 0.5);
    }

    // ── ShaderParams tests ───────────────────────────────────────────

    #[test]
    fn shader_params_from_inputs() {
        let inputs = vec![
            make_float_input("brightness", 0.5, 0.0, 1.0),
            make_bool_input("invert", false),
        ];
        let params = ShaderParams::from_inputs(&inputs);
        assert_eq!(params.param_order.len(), 2);
        assert!(!params.is_empty());
    }

    #[test]
    fn shader_params_skips_image_inputs() {
        let inputs = vec![
            make_float_input("brightness", 0.5, 0.0, 1.0),
            ISFInput {
                name: "inputImage".to_string(),
                input_type: "image".to_string(),
                default: None,
                min: None,
                max: None,
                label: None,
                values: None,
                labels: None,
                identity: None,
            },
        ];
        let params = ShaderParams::from_inputs(&inputs);
        assert_eq!(params.param_order.len(), 1); // image skipped
        assert_eq!(params.values.len(), 1);
    }

    #[test]
    fn shader_params_get_set_float() {
        let inputs = vec![make_float_input("brightness", 0.5, 0.0, 1.0)];
        let mut params = ShaderParams::from_inputs(&inputs);
        assert!((params.get_float("brightness").unwrap() - 0.5).abs() < 1e-5);
        params.set_float("brightness", 0.8);
        assert!((params.get_float("brightness").unwrap() - 0.8).abs() < 1e-5);
    }

    #[test]
    fn shader_params_get_set_bool() {
        let inputs = vec![make_bool_input("invert", false)];
        let mut params = ShaderParams::from_inputs(&inputs);
        assert_eq!(params.get_bool("invert"), Some(false));
        params.set_bool("invert", true);
        assert_eq!(params.get_bool("invert"), Some(true));
    }

    #[test]
    fn shader_params_get_set_color() {
        let inputs = vec![make_color_input("tint")];
        let mut params = ShaderParams::from_inputs(&inputs);
        let c = params.get_color("tint").unwrap();
        assert!((c[0] - 1.0).abs() < 1e-5);
        params.set_color("tint", [0.0, 1.0, 0.0, 1.0]);
        let c2 = params.get_color("tint").unwrap();
        assert!((c2[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn shader_params_get_set_long() {
        let inputs = vec![make_long_input("mode", 0)];
        let mut params = ShaderParams::from_inputs(&inputs);
        assert_eq!(params.get_long("mode"), Some(0));
        params.set_long("mode", 2);
        assert_eq!(params.get_long("mode"), Some(2));
    }

    #[test]
    fn shader_params_get_set_point2d() {
        let inputs = vec![make_point2d_input("center")];
        let mut params = ShaderParams::from_inputs(&inputs);
        let p = params.get_point2d("center").unwrap();
        assert!((p[0] - 0.5).abs() < 1e-5);
        params.set_point2d("center", [0.1, 0.9]);
        let p2 = params.get_point2d("center").unwrap();
        assert!((p2[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn shader_params_generic_set() {
        let inputs = vec![make_float_input("brightness", 0.5, 0.0, 1.0)];
        let mut params = ShaderParams::from_inputs(&inputs);
        params.set("brightness", ParamValue::Float(0.9));
        assert!((params.get_float("brightness").unwrap() - 0.9).abs() < 1e-5);
    }

    #[test]
    fn shader_params_set_nonexistent_noop() {
        let inputs = vec![make_float_input("brightness", 0.5, 0.0, 1.0)];
        let mut params = ShaderParams::from_inputs(&inputs);
        params.set("nonexistent", ParamValue::Float(1.0)); // should not crash
        assert!(params.get_float("nonexistent").is_none());
    }

    #[test]
    fn shader_params_buffer_size_min_16() {
        let params = ShaderParams::from_inputs(&[]);
        assert!(params.buffer_size() >= 16);
    }

    #[test]
    fn shader_params_buffer_size_aligned_to_16() {
        let inputs = vec![make_float_input("a", 0.0, 0.0, 1.0)];
        let params = ShaderParams::from_inputs(&inputs);
        assert_eq!(params.buffer_size() % 16, 0);
    }

    #[test]
    fn shader_params_build_buffer_data() {
        let inputs = vec![
            make_float_input("brightness", 0.5, 0.0, 1.0),
            make_bool_input("invert", true),
        ];
        let params = ShaderParams::from_inputs(&inputs);
        let data = params.build_buffer_data();
        assert!(data.len() >= 16);
        assert_eq!(data.len() % 16, 0);
        // First 4 bytes should be 0.5f32
        let val = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert!((val - 0.5).abs() < 1e-5);
    }

    #[test]
    fn shader_params_reset_to_defaults() {
        let inputs = vec![make_float_input("brightness", 0.5, 0.0, 1.0)];
        let mut params = ShaderParams::from_inputs(&inputs);
        params.set_float("brightness", 0.9);
        params.reset_to_defaults();
        assert!((params.get_float("brightness").unwrap() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn shader_params_empty() {
        let params = ShaderParams::from_inputs(&[]);
        assert!(params.is_empty());
    }

    #[test]
    fn shader_params_modulated_buffer_no_modulation() {
        let inputs = vec![make_float_input("brightness", 0.5, 0.0, 1.0)];
        let mut params = ShaderParams::from_inputs(&inputs);
        let engine = ModulationEngine::new();
        let base = params.build_buffer_data();
        let data = params.build_modulated_buffer_data(&engine, None);
        assert_eq!(data, base, "No modulation should produce identical buffer");
    }

    #[test]
    fn shader_params_modulated_buffer_with_modulation() {
        let inputs = vec![make_float_input("brightness", 0.5, 0.0, 1.0)];
        let mut params = ShaderParams::from_inputs(&inputs);
        let mut engine = ModulationEngine::new();
        let uuid = engine.add_source(crate::modulation::ModulationSource::LFO {
            waveform: crate::modulation::LFOWaveform::Sine,
            frequency: 1.0,
            phase: 0.0,
            amplitude: 1.0,
            bipolar: true,
        });
        engine.update(0.25, &crate::modulation::AudioValues::default());
        engine.assign("brightness", &uuid, 0.5, None);

        let base = params.build_buffer_data();
        let modulated = params.build_modulated_buffer_data(&engine, None);
        // Modulated should differ from base (LFO at t=0.25 is non-zero)
        assert_ne!(modulated, base, "Modulated buffer should differ from base");
    }

    #[test]
    fn shader_params_modulated_with_prefix() {
        let inputs = vec![make_float_input("brightness", 0.5, 0.0, 1.0)];
        let mut params = ShaderParams::from_inputs(&inputs);
        let mut engine = ModulationEngine::new();
        let uuid = engine.add_source(crate::modulation::ModulationSource::sine_lfo(1.0));
        engine.update(0.25, &crate::modulation::AudioValues::default());
        // Assign with prefix "deck0:brightness"
        engine.assign("deck0:brightness", &uuid, 0.5, None);

        let base = params.build_buffer_data();
        let modulated = params.build_modulated_buffer_data(&engine, Some("deck0"));
        assert_ne!(modulated, base, "Prefixed modulation should apply");
    }

    #[test]
    fn shader_params_std140_alignment_point2d() {
        // Point2D requires 8-byte alignment
        let inputs = vec![
            make_float_input("a", 1.0, 0.0, 1.0), // 4 bytes at offset 0
            make_point2d_input("center"),         // should align to offset 8
        ];
        let params = ShaderParams::from_inputs(&inputs);
        let data = params.build_buffer_data();
        // offset 0..4: float a
        // offset 4..8: padding (align to 8 for vec2)
        // offset 8..16: point2D center
        assert!(data.len() >= 16);
        let p0 = f32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let p1 = f32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        assert!((p0 - 0.5).abs() < 1e-5);
        assert!((p1 - 0.5).abs() < 1e-5);
    }

    #[test]
    fn shader_params_std140_alignment_color() {
        // Color requires 16-byte alignment
        let inputs = vec![
            make_float_input("a", 1.0, 0.0, 1.0), // 4 bytes at offset 0
            make_color_input("tint"),             // should align to offset 16
        ];
        let params = ShaderParams::from_inputs(&inputs);
        let data = params.build_buffer_data();
        assert!(data.len() >= 32);
        // tint starts at offset 16
        let r = f32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        assert!((r - 1.0).abs() < 1e-5); // red = 1.0
    }
}
