//! Tracks the on-disk location of the Qallow codebase so other modules
//! (button handlers, headless runs) can find and invoke project tooling.

use crate::logging::AppLogger;
use std::path::Path;

pub struct CodebaseManager {
    root_path: String,
    #[allow(dead_code)]
    logger: AppLogger,
}

impl CodebaseManager {
    pub fn new(path: &str, logger: AppLogger) -> Result<Self, String> {
        if !Path::new(path).exists() {
            return Err(format!("Codebase path does not exist: {}", path));
        }
        Ok(Self {
            root_path: path.to_string(),
            logger,
        })
    }

    pub fn root_path(&self) -> &str {
        &self.root_path
    }
}
