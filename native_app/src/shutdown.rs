use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::fs;
use std::path::Path;
use crate::models::AppState;
use serde_json;

/// Global shutdown flag
pub static SHUTDOWN_FLAG: AtomicBool = AtomicBool::new(false);

/// Graceful shutdown manager
pub struct ShutdownManager {
    state_file: String,
    shutdown_requested: Arc<AtomicBool>,
}

impl ShutdownManager {
    pub fn new(state_file: String) -> Self {
        Self {
            state_file,
            shutdown_requested: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Initialize signal handlers for graceful shutdown
    pub fn init_signal_handlers() {
        #[cfg(unix)]
        {
            use std::sync::atomic::AtomicBool;
            use std::sync::Arc;

            let shutdown_flag = Arc::new(AtomicBool::new(false));
            let shutdown_flag_clone = shutdown_flag.clone();

            // Handle SIGTERM
            ctrlc::set_handler(move || {
                eprintln!("\n[SHUTDOWN] Received interrupt signal, shutting down gracefully...");
                shutdown_flag_clone.store(true, Ordering::SeqCst);
                SHUTDOWN_FLAG.store(true, Ordering::SeqCst);
            })
            .expect("Error setting Ctrl-C handler");
        }
    }

    /// Check if shutdown was requested
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::SeqCst) || SHUTDOWN_FLAG.load(Ordering::SeqCst)
    }

    /// Request shutdown
    pub fn request_shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::SeqCst);
        SHUTDOWN_FLAG.store(true, Ordering::SeqCst);
    }

    /// Save application state to file
    pub fn save_state(&self, state: &AppState) -> Result<(), String> {
        match serde_json::to_string_pretty(state) {
            Ok(json) => {
                match fs::write(&self.state_file, json) {
                    Ok(_) => {
                        eprintln!("[SHUTDOWN] State saved to {}", self.state_file);
                        Ok(())
                    }
                    Err(e) => Err(format!("Failed to write state file: {}", e)),
                }
            }
            Err(e) => Err(format!("Failed to serialize state: {}", e)),
        }
    }

    /// Load application state from file
    pub fn load_state(&self) -> Result<AppState, String> {
        if !Path::new(&self.state_file).exists() {
            return Ok(AppState::new());
        }

        match fs::read_to_string(&self.state_file) {
            Ok(json) => {
                match serde_json::from_str(&json) {
                    Ok(state) => {
                        eprintln!("[SHUTDOWN] State loaded from {}", self.state_file);
                        Ok(state)
                    }
                    Err(e) => {
                        eprintln!("[SHUTDOWN] Failed to deserialize state: {}", e);
                        Ok(AppState::new())
                    }
                }
            }
            Err(e) => {
                eprintln!("[SHUTDOWN] Failed to read state file: {}", e);
                Ok(AppState::new())
            }
        }
    }

    /// Perform cleanup operations
    pub fn cleanup(&self) -> Result<(), String> {
        eprintln!("[SHUTDOWN] Performing cleanup...");
        // Add cleanup operations here
        Ok(())
    }
}

impl Drop for ShutdownManager {
    fn drop(&mut self) {
        eprintln!("[SHUTDOWN] ShutdownManager dropped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shutdown_manager_creation() {
        let manager = ShutdownManager::new("test_state.json".to_string());
        assert!(!manager.is_shutdown_requested());
    }

    #[test]
    fn test_shutdown_request() {
        let manager = ShutdownManager::new("test_state.json".to_string());
        manager.request_shutdown();
        assert!(manager.is_shutdown_requested());
    }
}

