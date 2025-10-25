use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub app: AppSettings,
    pub logging: LoggingSettings,
    pub ui: UISettings,
    pub vm: VMSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub auto_save_interval_secs: u64,
    pub max_terminal_lines: usize,
    pub max_audit_logs: usize,
    pub enable_auto_recovery: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingSettings {
    pub level: String,
    pub file_path: String,
    pub max_file_size_mb: u64,
    pub max_backups: usize,
    pub enable_console: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UISettings {
    pub theme: String,
    pub window_width: u32,
    pub window_height: u32,
    pub font_size: u32,
    pub auto_scroll_terminal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VMSettings {
    pub default_build: String,
    pub default_phase: String,
    pub default_ticks: u32,
    pub process_timeout_secs: u64,
    pub enable_metrics_collection: bool,
    pub metrics_interval_ms: u64,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            app: AppSettings {
                auto_save_interval_secs: 30,
                max_terminal_lines: 1000,
                max_audit_logs: 500,
                enable_auto_recovery: true,
            },
            logging: LoggingSettings {
                level: "info".to_string(),
                file_path: "qallow.log".to_string(),
                max_file_size_mb: 10,
                max_backups: 5,
                enable_console: true,
            },
            ui: UISettings {
                theme: "dark".to_string(),
                window_width: 1600,
                window_height: 1000,
                font_size: 12,
                auto_scroll_terminal: true,
            },
            vm: VMSettings {
                default_build: "CPU".to_string(),
                default_phase: "Phase14".to_string(),
                default_ticks: 1000,
                process_timeout_secs: 300,
                enable_metrics_collection: true,
                metrics_interval_ms: 500,
            },
        }
    }
}

pub struct ConfigManager {
    config_file: String,
    config: AppConfig,
}

impl ConfigManager {
    pub fn new(config_file: String) -> Self {
        let config = Self::load_or_default(&config_file);
        Self {
            config_file,
            config,
        }
    }

    fn load_or_default(config_file: &str) -> AppConfig {
        if Path::new(config_file).exists() {
            match fs::read_to_string(config_file) {
                Ok(content) => match serde_json::from_str(&content) {
                    Ok(config) => {
                        eprintln!("[CONFIG] Loaded config from {}", config_file);
                        return config;
                    }
                    Err(e) => {
                        eprintln!("[CONFIG] Failed to parse config: {}", e);
                    }
                },
                Err(e) => {
                    eprintln!("[CONFIG] Failed to read config: {}", e);
                }
            }
        }

        let default_config = AppConfig::default();
        let _ = Self::save_config(config_file, &default_config);
        default_config
    }

    fn save_config(config_file: &str, config: &AppConfig) -> Result<(), String> {
        match serde_json::to_string_pretty(config) {
            Ok(json) => match fs::write(config_file, json) {
                Ok(_) => {
                    eprintln!("[CONFIG] Saved config to {}", config_file);
                    Ok(())
                }
                Err(e) => Err(format!("Failed to write config: {}", e)),
            },
            Err(e) => Err(format!("Failed to serialize config: {}", e)),
        }
    }

    pub fn get(&self) -> &AppConfig {
        &self.config
    }

    pub fn get_mut(&mut self) -> &mut AppConfig {
        &mut self.config
    }

    pub fn save(&self) -> Result<(), String> {
        Self::save_config(&self.config_file, &self.config)
    }

    pub fn reload(&mut self) -> Result<(), String> {
        self.config = Self::load_or_default(&self.config_file);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert_eq!(config.app.auto_save_interval_secs, 30);
        assert_eq!(config.ui.window_width, 1600);
    }

    #[test]
    fn test_config_manager_creation() {
        let manager = ConfigManager::new("test_config.json".to_string());
        assert_eq!(manager.get().app.auto_save_interval_secs, 30);
    }
}
