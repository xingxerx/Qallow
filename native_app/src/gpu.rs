//! GPU acceleration probing. This build has no CUDA (`cust`) feature enabled,
//! so this module always reports "no GPU available" without failing the app.

pub fn check_gpu_availability() -> String {
    "No GPU acceleration available (CPU-only build)".to_string()
}

pub struct GpuMetrics {
    pub device_name: String,
    pub compute_capability: (u32, u32),
}

pub struct GPUManager;

impl GPUManager {
    pub fn new() -> Result<Self, String> {
        Err("GPU support is not compiled into this build".to_string())
    }

    pub fn get_metrics(&self) -> GpuMetrics {
        GpuMetrics {
            device_name: "none".to_string(),
            compute_capability: (0, 0),
        }
    }
}
