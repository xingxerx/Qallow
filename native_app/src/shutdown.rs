//! Graceful shutdown handling: a global flag flipped by a Ctrl-C handler,
//! plus persistence of `AppState` across restarts.

use crate::models::AppState;
use std::fs;
use std::sync::atomic::AtomicBool;

/// Set to `true` by the Ctrl-C signal handler; polled from the main loop.
pub static SHUTDOWN_FLAG: AtomicBool = AtomicBool::new(false);

pub struct ShutdownManager {
    state_path: String,
}

impl ShutdownManager {
    pub fn new(path: String) -> Self {
        Self { state_path: path }
    }

    /// Installs a process-wide Ctrl-C handler that flips `SHUTDOWN_FLAG`.
    pub fn init_signal_handlers() {
        let _ = ctrlc::set_handler(|| {
            SHUTDOWN_FLAG.store(true, std::sync::atomic::Ordering::SeqCst);
        });
    }

    pub fn load_state(&self) -> Result<AppState, String> {
        let contents = fs::read_to_string(&self.state_path).map_err(|e| e.to_string())?;
        serde_json::from_str(&contents).map_err(|e| e.to_string())
    }

    pub fn save_state(&self, state: &AppState) -> Result<(), String> {
        let json = serde_json::to_string_pretty(state).map_err(|e| e.to_string())?;
        fs::write(&self.state_path, json).map_err(|e| e.to_string())
    }

    pub fn cleanup(&self) -> Result<(), String> {
        Ok(())
    }
}
