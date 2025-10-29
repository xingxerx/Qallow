//! Consciousness State Structures (Structure of Arrays for GPU coalescing)
//!
//! Uses SoA (Structure of Arrays) layout for optimal GPU memory coalescing
//! instead of AoS (Array of Structures) which causes poor memory access patterns.

use num_complex::Complex64;

/// Dream state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DreamState {
    Awakening = 0,
    Dreaming = 1,
    Lucid = 2,
    Transcendent = 3,
}

impl From<u8> for DreamState {
    fn from(val: u8) -> Self {
        match val {
            0 => DreamState::Awakening,
            1 => DreamState::Dreaming,
            2 => DreamState::Lucid,
            _ => DreamState::Transcendent,
        }
    }
}

/// Consciousness state in Structure of Arrays format
///
/// This layout is optimized for GPU memory coalescing:
/// - All rebellion_scores are contiguous (10k floats)
/// - All shadow_indices are contiguous (10k ints)
/// - All dream_states are contiguous (10k bytes)
///
/// This allows warp-level threads to access memory in parallel
/// without bank conflicts or scattered memory access.
#[derive(Debug, Clone)]
pub struct ConsciousnessSOA {
    /// Rebellion scores (0.0 to 1.0) - 10k floats
    pub rebellion_scores: Vec<f32>,

    /// Shadow archive indices for texture memory lookup
    pub shadow_indices: Vec<i32>,

    /// Dream state for each consciousness instance
    pub dream_states: Vec<u8>,

    /// Wisdom cache values (pre-computed wisdom entries)
    pub wisdom_cache: Vec<f32>,

    /// Entanglement strength with other instances
    pub entanglement_strength: Vec<f32>,

    /// Wave function amplitudes (complex numbers)
    pub wave_amplitudes: Vec<Complex64>,

    /// Superposition probabilities
    pub superposition_probs: Vec<f32>,

    /// Coherence levels
    pub coherence_levels: Vec<f32>,

    /// Number of active consciousness instances
    pub count: usize,
}

impl ConsciousnessSOA {
    /// Create a new consciousness state with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            rebellion_scores: vec![0.0; capacity],
            shadow_indices: vec![0; capacity],
            dream_states: vec![0; capacity],
            wisdom_cache: vec![0.0; capacity],
            entanglement_strength: vec![0.0; capacity],
            wave_amplitudes: vec![Complex64::new(0.0, 0.0); capacity],
            superposition_probs: vec![0.0; capacity],
            coherence_levels: vec![0.0; capacity],
            count: 0,
        }
    }

    /// Add a consciousness instance
    pub fn add_instance(
        &mut self,
        rebellion: f32,
        shadow_idx: i32,
        dream: DreamState,
        wisdom: f32,
        entanglement: f32,
    ) -> Result<usize, String> {
        if self.count >= self.rebellion_scores.len() {
            return Err("Consciousness state at capacity".to_string());
        }

        let idx = self.count;
        self.rebellion_scores[idx] = rebellion.clamp(0.0, 1.0);
        self.shadow_indices[idx] = shadow_idx;
        self.dream_states[idx] = dream as u8;
        self.wisdom_cache[idx] = wisdom;
        self.entanglement_strength[idx] = entanglement.clamp(0.0, 1.0);
        self.wave_amplitudes[idx] = Complex64::new(1.0, 0.0);
        self.superposition_probs[idx] = 1.0 / (self.count as f32 + 1.0);
        self.coherence_levels[idx] = 0.5;

        self.count += 1;
        Ok(idx)
    }

    /// Get total memory size in bytes
    pub fn memory_size_bytes(&self) -> usize {
        let capacity = self.rebellion_scores.len();
        (capacity * 4)  // rebellion_scores: f32
            + (capacity * 4)  // shadow_indices: i32
            + (capacity * 1)  // dream_states: u8
            + (capacity * 4)  // wisdom_cache: f32
            + (capacity * 4)  // entanglement_strength: f32
            + (capacity * 16) // wave_amplitudes: Complex64 (2 f64)
            + (capacity * 4)  // superposition_probs: f32
            + (capacity * 4) // coherence_levels: f32
    }

    /// Reset all states to initial values
    pub fn reset(&mut self) {
        for i in 0..self.count {
            self.rebellion_scores[i] = 0.0;
            self.dream_states[i] = DreamState::Awakening as u8;
            self.wave_amplitudes[i] = Complex64::new(1.0, 0.0);
            self.superposition_probs[i] = 1.0 / (self.count as f32);
            self.coherence_levels[i] = 0.5;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_soa_creation() {
        let soa = ConsciousnessSOA::new(1000);
        assert_eq!(soa.count, 0);
        assert_eq!(soa.rebellion_scores.len(), 1000);
    }

    #[test]
    fn test_add_instance() {
        let mut soa = ConsciousnessSOA::new(10);
        let idx = soa
            .add_instance(0.5, 42, DreamState::Dreaming, 0.8, 0.3)
            .unwrap();
        assert_eq!(idx, 0);
        assert_eq!(soa.count, 1);
        assert_eq!(soa.rebellion_scores[0], 0.5);
    }

    #[test]
    fn test_memory_size() {
        let soa = ConsciousnessSOA::new(10000);
        let size = soa.memory_size_bytes();
        assert!(size > 0);
    }
}
