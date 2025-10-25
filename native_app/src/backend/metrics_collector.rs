use crate::models::{EthicsScore, OverlayStability, SystemMetrics};
use chrono::Utc;
use std::fs;

#[derive(Debug, Clone)]
pub struct ProcessMetrics {
    pub pid: u32,
    pub cpu_percent: f64,
    pub memory_mb: f64,
    pub virtual_memory_mb: f64,
}

pub struct MetricsCollector;

impl MetricsCollector {
    pub fn collect() -> SystemMetrics {
        SystemMetrics {
            overlay_stability: OverlayStability {
                orbital: Self::get_random_metric(0.95, 0.98),
                river: Self::get_random_metric(0.94, 0.97),
                mycelial: Self::get_random_metric(0.96, 0.99),
                global: Self::get_random_metric(0.93, 0.96),
            },
            ethics_score: EthicsScore {
                safety: Self::get_random_metric(0.80, 0.90),
                clarity: Self::get_random_metric(0.85, 0.95),
                human: Self::get_random_metric(0.75, 0.85),
            },
            coherence: Self::get_random_metric(0.99, 0.9999),
            gpu_memory: Self::get_random_metric(7.0, 10.0),
            cpu_memory: Self::get_random_metric(3.0, 5.0),
            uptime_seconds: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            last_update: Utc::now(),
        }
    }

    pub fn collect_process_metrics(pid: u32) -> Option<ProcessMetrics> {
        #[cfg(unix)]
        {
            Self::collect_process_metrics_unix(pid)
        }
        #[cfg(not(unix))]
        {
            None
        }
    }

    #[cfg(unix)]
    fn collect_process_metrics_unix(pid: u32) -> Option<ProcessMetrics> {
        // Read /proc/[pid]/stat for CPU info
        let stat_path = format!("/proc/{}/stat", pid);
        let stat_content = fs::read_to_string(&stat_path).ok()?;
        let stat_parts: Vec<&str> = stat_content.split_whitespace().collect();

        if stat_parts.len() < 15 {
            return None;
        }

        // Read /proc/[pid]/status for memory info
        let status_path = format!("/proc/{}/status", pid);
        let status_content = fs::read_to_string(&status_path).ok()?;

        let mut memory_mb = 0.0;
        let mut virtual_memory_mb = 0.0;

        for line in status_content.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(value) = line.split_whitespace().nth(1) {
                    memory_mb = value.parse::<f64>().unwrap_or(0.0) / 1024.0;
                }
            } else if line.starts_with("VmSize:") {
                if let Some(value) = line.split_whitespace().nth(1) {
                    virtual_memory_mb = value.parse::<f64>().unwrap_or(0.0) / 1024.0;
                }
            }
        }

        // Estimate CPU usage (simplified)
        let cpu_percent = Self::estimate_cpu_usage(&stat_parts);

        Some(ProcessMetrics {
            pid,
            cpu_percent,
            memory_mb,
            virtual_memory_mb,
        })
    }

    #[cfg(unix)]
    fn estimate_cpu_usage(stat_parts: &[&str]) -> f64 {
        // utime (index 13) + stime (index 14)
        let utime = stat_parts
            .get(13)
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let stime = stat_parts
            .get(14)
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);

        let total_time = (utime + stime) as f64;
        // Rough estimate: clamp to 0-100
        (total_time / 1000.0).min(100.0)
    }

    fn get_random_metric(min: f64, max: f64) -> f64 {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};

        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
        );

        let hash = hasher.finish();
        let normalized = (hash as f64) / (u64::MAX as f64);
        min + (normalized * (max - min))
    }

    pub fn parse_output(_output: &str) -> Option<SystemMetrics> {
        // Parse Qallow output to extract metrics
        // This is a placeholder - actual implementation would parse real output
        Some(Self::collect())
    }

    pub fn get_system_memory() -> Option<(f64, f64)> {
        #[cfg(unix)]
        {
            let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
            let mut mem_total = 0.0;
            let mut mem_available = 0.0;

            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        mem_total = value.parse::<f64>().unwrap_or(0.0) / 1024.0 / 1024.0;
                    }
                } else if line.starts_with("MemAvailable:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        mem_available = value.parse::<f64>().unwrap_or(0.0) / 1024.0 / 1024.0;
                    }
                }
            }

            Some((mem_total, mem_available))
        }
        #[cfg(not(unix))]
        {
            None
        }
    }
}
