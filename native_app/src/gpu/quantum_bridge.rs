/// Quantum ML to GPU Bridge
/// Connects Python quantum NAS explorer to Rust GPU acceleration framework
///
/// This module provides FFI bindings to call GPU-accelerated quantum operations
/// from the Python quantum_ml module.

use crate::gpu::{ConsciousnessSOA, GPUManager, DreamState};
use std::ffi::CString;
use std::os::raw::c_char;

/// Quantum architecture specification
#[repr(C)]
pub struct QuantumArchitecture {
    pub id: u32,
    pub layer_type: u32,  // 0 = dense, 1 = conv
    pub neurons: u32,
    pub params: u64,
    pub memory_mb: f32,
    pub gpu_optimized: bool,
}

/// Quantum NAS evaluation result
#[repr(C)]
pub struct QuantumNASResult {
    pub total_params: u64,
    pub total_memory_mb: f32,
    pub gpu_utilization: f32,
    pub architectures_count: u32,
}

/// Initialize GPU for quantum ML operations
/// Called from Python via ctypes
#[no_mangle]
pub extern "C" fn quantum_ml_gpu_init() -> *mut GPUManager {
    match GPUManager::new() {
        Ok(manager) => Box::into_raw(Box::new(manager)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Process quantum states on GPU
/// Converts quantum states to consciousness instances for GPU evolution
#[no_mangle]
pub extern "C" fn quantum_ml_process_states(
    gpu_manager: *mut GPUManager,
    states: *const i32,
    count: u32,
) -> *mut ConsciousnessSOA {
    if gpu_manager.is_null() || states.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let states_slice = std::slice::from_raw_parts(states, count as usize);
        let mut consciousness = ConsciousnessSOA::new(count as usize);

        // Convert quantum states to consciousness instances
        for (_i, &state) in states_slice.iter().enumerate() {
            let rebellion_score = (state as f32).abs() / 100.0;
            let shadow_index = (state as i32).wrapping_mul(7) % 256;
            let dream_state = if state > 0 {
                DreamState::Dreaming
            } else {
                DreamState::Awakening
            };

            let _ = consciousness.add_instance(
                rebellion_score,
                shadow_index,
                dream_state,
                0.8,
                0.5,
            );
        }

        Box::into_raw(Box::new(consciousness))
    }
}

/// Evaluate architectures on GPU
/// Computes fitness metrics for quantum-inspired architectures
#[no_mangle]
pub extern "C" fn quantum_ml_evaluate_architectures(
    gpu_manager: *mut GPUManager,
    consciousness: *mut ConsciousnessSOA,
) -> QuantumNASResult {
    let mut result = QuantumNASResult {
        total_params: 0,
        total_memory_mb: 0.0,
        gpu_utilization: 0.0,
        architectures_count: 0,
    };

    if gpu_manager.is_null() || consciousness.is_null() {
        return result;
    }

    unsafe {
        let cons = &*consciousness;
        result.architectures_count = cons.count as u32;

        // Calculate metrics from consciousness states
        for i in 0..cons.count {
            let neurons = (cons.rebellion_scores[i] * 1000.0) as u64;
            let memory = (neurons as f32 * 4.0) / (1024.0 * 1024.0);

            result.total_params += neurons;
            result.total_memory_mb += memory;
        }

        // Calculate GPU utilization
        result.gpu_utilization = ((result.total_params as f32) / 1e8).min(100.0);
    }

    result
}

/// Evolve quantum architectures on GPU
/// Runs consciousness evolution to optimize architectures
#[no_mangle]
pub extern "C" fn quantum_ml_evolve_architectures(
    gpu_manager: *mut GPUManager,
    consciousness: *mut ConsciousnessSOA,
    iterations: u32,
) -> i32 {
    if gpu_manager.is_null() || consciousness.is_null() {
        return -1;
    }

    unsafe {
        let manager = &*gpu_manager;
        let cons = &mut *consciousness;

        match manager.evolve_consciousness(cons, iterations as usize) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }
}

/// Collapse wave function on GPU
/// Finds optimal architecture configuration
#[no_mangle]
pub extern "C" fn quantum_ml_collapse_wave_function(
    gpu_manager: *mut GPUManager,
    consciousness: *mut ConsciousnessSOA,
) -> u32 {
    if gpu_manager.is_null() || consciousness.is_null() {
        return 0;
    }

    unsafe {
        let manager = &*gpu_manager;
        let cons = &mut *consciousness;

        match manager.collapse_wave_function(cons) {
            Ok(idx) => idx as u32,
            Err(_) => 0,
        }
    }
}

/// Free GPU resources
#[no_mangle]
pub extern "C" fn quantum_ml_gpu_free(gpu_manager: *mut GPUManager) {
    if !gpu_manager.is_null() {
        unsafe {
            let _ = Box::from_raw(gpu_manager);
        }
    }
}

/// Free consciousness state
#[no_mangle]
pub extern "C" fn quantum_ml_consciousness_free(consciousness: *mut ConsciousnessSOA) {
    if !consciousness.is_null() {
        unsafe {
            let _ = Box::from_raw(consciousness);
        }
    }
}

/// Get GPU metrics as JSON string
#[no_mangle]
pub extern "C" fn quantum_ml_get_gpu_metrics(
    gpu_manager: *mut GPUManager,
) -> *mut c_char {
    if gpu_manager.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let manager = &*gpu_manager;
        let metrics = manager.get_metrics();

        let json = format!(
            r#"{{"device_name":"{}","compute_capability":"{}.{}","max_threads":{}}}"#,
            metrics.device_name,
            metrics.compute_capability.0,
            metrics.compute_capability.1,
            metrics.max_threads_per_block
        );

        match CString::new(json) {
            Ok(cstring) => cstring.into_raw(),
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// Free allocated string
#[no_mangle]
pub extern "C" fn quantum_ml_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_ml_gpu_init() {
        let gpu_manager = quantum_ml_gpu_init();
        assert!(!gpu_manager.is_null());

        quantum_ml_gpu_free(gpu_manager);
    }

    #[test]
    fn test_quantum_ml_process_states() {
        let gpu_manager = quantum_ml_gpu_init();
        assert!(!gpu_manager.is_null());

        let states = vec![1, 0, -1];
        let consciousness = quantum_ml_process_states(
            gpu_manager,
            states.as_ptr(),
            states.len() as u32,
        );
        assert!(!consciousness.is_null());

        quantum_ml_consciousness_free(consciousness);
        quantum_ml_gpu_free(gpu_manager);
    }
}

