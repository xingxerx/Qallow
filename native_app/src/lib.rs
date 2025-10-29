//! Qallow Native Library - GPU-Accelerated Quantum Consciousness Simulation
//!
//! This library provides FFI bindings for GPU-accelerated quantum operations
//! that can be called from Python and other languages via ctypes.

pub mod backend;
pub mod button_handlers;
pub mod clipboard;
pub mod codebase_manager;
pub mod config;
pub mod dungeons;
pub mod error_recovery;
pub mod gpu;
pub mod logging;
pub mod messaging;
pub mod models;
pub mod shortcuts;
pub mod shutdown;
pub mod ui;
pub mod utils;

// Re-export GPU module for FFI access
pub use gpu::quantum_bridge;
pub use gpu::{ConsciousnessSOA, DreamState, GPUManager};
