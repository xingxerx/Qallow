use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use chrono::Local;

pub struct AppLogger {
    log_file: String,
    max_file_size: u64,
    max_backups: usize,
}

impl AppLogger {
    pub fn new(log_file: String, max_file_size_mb: u64, max_backups: usize) -> Self {
        Self {
            log_file,
            max_file_size: max_file_size_mb * 1024 * 1024,
            max_backups,
        }
    }

    pub fn init(&self) -> Result<(), String> {
        // Create log directory if it doesn't exist
        if let Some(parent) = Path::new(&self.log_file).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create log directory: {}", e))?;
            }
        }
        Ok(())
    }

    pub fn log(&self, level: &str, message: &str) -> Result<(), String> {
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
        let log_line = format!("[{}] [{}] {}\n", timestamp, level, message);

        // Check if rotation is needed
        if self.should_rotate() {
            self.rotate_logs()?;
        }

        // Write to file
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_file)
            .map_err(|e| format!("Failed to open log file: {}", e))?;

        file.write_all(log_line.as_bytes())
            .map_err(|e| format!("Failed to write to log file: {}", e))?;

        // Also print to console
        print!("{}", log_line);

        Ok(())
    }

    pub fn info(&self, message: &str) -> Result<(), String> {
        self.log("INFO", message)
    }

    pub fn warn(&self, message: &str) -> Result<(), String> {
        self.log("WARN", message)
    }

    pub fn error(&self, message: &str) -> Result<(), String> {
        self.log("ERROR", message)
    }

    pub fn debug(&self, message: &str) -> Result<(), String> {
        self.log("DEBUG", message)
    }

    pub fn success(&self, message: &str) -> Result<(), String> {
        self.log("SUCCESS", message)
    }

    fn should_rotate(&self) -> bool {
        if let Ok(metadata) = fs::metadata(&self.log_file) {
            metadata.len() > self.max_file_size
        } else {
            false
        }
    }

    fn rotate_logs(&self) -> Result<(), String> {
        // Remove oldest backup if we have too many
        for i in (self.max_backups..1000).rev() {
            let backup_path = format!("{}.{}", self.log_file, i);
            if Path::new(&backup_path).exists() {
                fs::remove_file(&backup_path)
                    .map_err(|e| format!("Failed to remove old backup: {}", e))?;
            }
        }

        // Rotate existing backups
        for i in (1..self.max_backups).rev() {
            let old_path = format!("{}.{}", self.log_file, i);
            let new_path = format!("{}.{}", self.log_file, i + 1);
            if Path::new(&old_path).exists() {
                fs::rename(&old_path, &new_path)
                    .map_err(|e| format!("Failed to rotate backup: {}", e))?;
            }
        }

        // Rename current log to .1
        let backup_path = format!("{}.1", self.log_file);
        if Path::new(&self.log_file).exists() {
            fs::rename(&self.log_file, &backup_path)
                .map_err(|e| format!("Failed to rotate current log: {}", e))?;
        }

        Ok(())
    }

    pub fn clear(&self) -> Result<(), String> {
        fs::remove_file(&self.log_file)
            .map_err(|e| format!("Failed to clear log file: {}", e))?;
        Ok(())
    }

    pub fn read_recent(&self, lines: usize) -> Result<Vec<String>, String> {
        let content = fs::read_to_string(&self.log_file)
            .map_err(|e| format!("Failed to read log file: {}", e))?;

        Ok(content
            .lines()
            .rev()
            .take(lines)
            .map(|s| s.to_string())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logger_creation() {
        let logger = AppLogger::new("test.log".to_string(), 10, 5);
        assert_eq!(logger.max_file_size, 10 * 1024 * 1024);
    }

    #[test]
    fn test_logger_init() {
        let logger = AppLogger::new("test_logs/test.log".to_string(), 10, 5);
        assert!(logger.init().is_ok());
    }
}

