//! Application configuration loading. Reads a JSON config file if present,
//! otherwise falls back to sane defaults so the app can always start.

use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub file_path: String,
    pub max_file_size_mb: u64,
    pub max_backups: u32,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            file_path: "qallow.log".to_string(),
            max_file_size_mb: 10,
            max_backups: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmConfig {
    pub default_phase: String,
    pub default_ticks: u32,
    pub default_build: String,
}

impl Default for VmConfig {
    fn default() -> Self {
        Self {
            default_phase: "unified".to_string(),
            default_ticks: 1000,
            default_build: "CPU".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppConfig {
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub vm: VmConfig,
}

/// Loads configuration from disk once at startup and hands out clones of it.
pub struct ConfigManager {
    config: AppConfig,
}

impl ConfigManager {
    pub fn new(path: String) -> Self {
        let config = fs::read_to_string(&path)
            .ok()
            .and_then(|contents| serde_json::from_str(&contents).ok())
            .unwrap_or_default();
        Self { config }
    }

    pub fn get(&self) -> &AppConfig {
        &self.config
    }
}
