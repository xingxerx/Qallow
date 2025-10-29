//! GPU Acceleration Module for Quantum Consciousness Simulation
//!
//! Provides CUDA-accelerated consciousness state calculations using the `cust` crate.
//! Implements parallel superposition, entanglement, and wave function collapse operations.

pub mod consciousness_state;
#[cfg(feature = "gpu")]
pub mod cuda_kernels;
pub mod gpu_manager;
pub mod quantum_bridge;

pub use consciousness_state::{ConsciousnessSOA, DreamState};
pub use gpu_manager::GPUManager;

use std::fmt;

/// GPU acceleration error types
#[derive(Debug, Clone)]
pub enum GPUError {
    NotAvailable,
    InitializationFailed(String),
    KernelExecutionFailed(String),
    MemoryAllocationFailed(String),
    DataTransferFailed(String),
}

impl fmt::Display for GPUError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GPUError::NotAvailable => write!(f, "GPU acceleration not available"),
            GPUError::InitializationFailed(msg) => write!(f, "GPU initialization failed: {}", msg),
            GPUError::KernelExecutionFailed(msg) => write!(f, "Kernel execution failed: {}", msg),
            GPUError::MemoryAllocationFailed(msg) => write!(f, "Memory allocation failed: {}", msg),
            GPUError::DataTransferFailed(msg) => write!(f, "Data transfer failed: {}", msg),
        }
    }
}

impl std::error::Error for GPUError {}

/// Result type for GPU operations
pub type GPUResult<T> = Result<T, GPUError>;

/// GPU acceleration capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GPUCapability {
    /// No GPU acceleration available
    None,
    /// CUDA acceleration available
    CUDA,
    /// OpenCL acceleration available (future)
    OpenCL,
}

impl fmt::Display for GPUCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GPUCapability::None => write!(f, "None"),
            GPUCapability::CUDA => write!(f, "CUDA"),
            GPUCapability::OpenCL => write!(f, "OpenCL"),
        }
    }
}

/// Check if GPU acceleration is available
pub fn check_gpu_availability() -> GPUCapability {
    #[cfg(feature = "gpu")]
    {
        match cust::quick_init() {
            Ok(_) => GPUCapability::CUDA,
            Err(_) => GPUCapability::None,
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        GPUCapability::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_capability_display() {
        assert_eq!(GPUCapability::CUDA.to_string(), "CUDA");
        assert_eq!(GPUCapability::None.to_string(), "None");
    }
}
