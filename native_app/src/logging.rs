//! Simple file + stdout logger with basic size-based rotation.

use chrono::Utc;
use parking_lot::Mutex;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::sync::Arc;

struct LoggerInner {
    file_path: String,
    max_file_size_mb: u64,
    max_backups: u32,
}

#[derive(Clone)]
pub struct AppLogger {
    inner: Arc<Mutex<LoggerInner>>,
}

impl AppLogger {
    pub fn new(file_path: String, max_file_size_mb: u64, max_backups: u32) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LoggerInner {
                file_path,
                max_file_size_mb,
                max_backups,
            })),
        }
    }

    /// Ensures the log directory exists. Safe to call multiple times.
    pub fn init(&self) -> Result<(), String> {
        let inner = self.inner.lock();
        if let Some(parent) = std::path::Path::new(&inner.file_path).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }

    fn rotate_if_needed(inner: &LoggerInner) {
        if inner.max_file_size_mb == 0 {
            return;
        }
        let max_bytes = inner.max_file_size_mb.saturating_mul(1024 * 1024);
        if let Ok(meta) = fs::metadata(&inner.file_path) {
            if meta.len() > max_bytes {
                for i in (1..inner.max_backups).rev() {
                    let src = format!("{}.{}", inner.file_path, i);
                    let dst = format!("{}.{}", inner.file_path, i + 1);
                    let _ = fs::rename(&src, &dst);
                }
                let _ = fs::rename(&inner.file_path, format!("{}.1", inner.file_path));
            }
        }
    }

    fn write_line(&self, level: &str, message: &str) -> Result<(), String> {
        let inner = self.inner.lock();
        Self::rotate_if_needed(&inner);
        let line = format!("[{}] [{}] {}\n", Utc::now().to_rfc3339(), level, message);
        println!("[{}] {}", level, message);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&inner.file_path)
            .map_err(|e| e.to_string())?;
        file.write_all(line.as_bytes()).map_err(|e| e.to_string())
    }

    pub fn info(&self, message: &str) -> Result<(), String> {
        self.write_line("INFO", message)
    }

    pub fn warn(&self, message: &str) -> Result<(), String> {
        self.write_line("WARN", message)
    }

    pub fn error(&self, message: &str) -> Result<(), String> {
        self.write_line("ERROR", message)
    }
}
