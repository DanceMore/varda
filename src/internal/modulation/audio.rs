//! Audio analysis values for the modulation engine.

use std::sync::Arc;

/// Audio analysis values for a single source, passed to modulation engine.
#[derive(Debug, Clone)]
pub struct AudioSourceValues {
    pub fft: Arc<[f32]>,
    pub level: f32,
    pub sample_rate: f32,
    /// Pre-calculated energy for the Bass band (20-250Hz).
    pub bass: f32,
    /// Pre-calculated energy for the Mid band (250-2000Hz).
    pub mid: f32,
    /// Pre-calculated energy for the Treble band (2000-20000Hz).
    pub treble: f32,
    /// Pre-calculated energy for the Full band (20-20000Hz).
    pub full: f32,
}

impl AudioSourceValues {
    /// Compute energy in a frequency range from the FFT data.
    /// Returns a perceptually-scaled value in roughly 0.0–1.0 range
    /// suitable for driving modulation (dB-based mapping).
    pub fn energy_in_range(&self, freq_low: f32, freq_high: f32) -> f32 {
        // Optimization: return pre-calculated standard bands if possible.
        if (freq_low - 20.0).abs() < 0.1 {
            if (freq_high - 250.0).abs() < 0.1 { return self.bass; }
            if (freq_high - 20000.0).abs() < 0.1 { return self.full; }
        } else if (freq_low - 250.0).abs() < 0.1 && (freq_high - 2000.0).abs() < 0.1 {
            return self.mid;
        } else if (freq_low - 2000.0).abs() < 0.1 && (freq_high - 20000.0).abs() < 0.1 {
            return self.treble;
        }

        crate::audio::compute_energy_from_fft(&self.fft, self.sample_rate, freq_low, freq_high)
    }
}

/// All audio source data for the current frame.
#[derive(Debug, Clone, Default)]
pub struct AudioValues {
    /// Per-source audio data, keyed by AudioSourceId.
    pub sources: std::collections::HashMap<crate::audio::AudioSourceId, AudioSourceValues>,
}

impl AudioValues {
    /// Clear all audio source data. Reuses the HashMap's capacity.
    pub fn clear(&mut self) {
        self.sources.clear();
    }

    /// Get the first/primary source's data (convenience).
    pub fn primary(&self) -> Option<&AudioSourceValues> {
        self.sources.iter().min_by_key(|(id, _)| **id).map(|(_, v)| v)
    }
}
