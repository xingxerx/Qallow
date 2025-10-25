use std::collections::VecDeque;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct ErrorEvent {
    pub timestamp: Instant,
    pub error_type: ErrorType,
    pub message: String,
    pub recovery_attempted: bool,
    pub recovery_successful: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorType {
    ProcessStartFailed,
    ProcessCrashed,
    ProcessTimeout,
    MemoryExceeded,
    IOError,
    ConfigError,
    Unknown,
}

pub struct ErrorRecoveryManager {
    error_history: VecDeque<ErrorEvent>,
    max_history: usize,
    retry_policy: RetryPolicy,
}

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 500,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0,
        }
    }
}

impl ErrorRecoveryManager {
    pub fn new(max_history: usize) -> Self {
        Self {
            error_history: VecDeque::with_capacity(max_history),
            max_history,
            retry_policy: RetryPolicy::default(),
        }
    }

    pub fn set_retry_policy(&mut self, policy: RetryPolicy) {
        self.retry_policy = policy;
    }

    pub fn record_error(&mut self, error_type: ErrorType, message: String) {
        let event = ErrorEvent {
            timestamp: Instant::now(),
            error_type,
            message,
            recovery_attempted: false,
            recovery_successful: false,
        };

        self.error_history.push_back(event);
        if self.error_history.len() > self.max_history {
            self.error_history.pop_front();
        }
    }

    pub fn mark_recovery_attempted(&mut self, success: bool) {
        if let Some(event) = self.error_history.back_mut() {
            event.recovery_attempted = true;
            event.recovery_successful = success;
        }
    }

    pub fn get_retry_delay(&self, attempt: u32) -> Duration {
        let delay_ms = (self.retry_policy.initial_delay_ms as f64
            * self.retry_policy.backoff_multiplier.powi(attempt as i32))
        .min(self.retry_policy.max_delay_ms as f64) as u64;
        Duration::from_millis(delay_ms)
    }

    pub fn should_retry(&self, attempt: u32) -> bool {
        attempt < self.retry_policy.max_retries
    }

    pub fn get_error_count(&self, error_type: ErrorType) -> usize {
        self.error_history
            .iter()
            .filter(|e| e.error_type == error_type)
            .count()
    }

    pub fn get_recent_errors(&self, seconds: u64) -> Vec<ErrorEvent> {
        let cutoff = Instant::now() - Duration::from_secs(seconds);
        self.error_history
            .iter()
            .filter(|e| e.timestamp > cutoff)
            .cloned()
            .collect()
    }

    pub fn get_error_history(&self) -> Vec<ErrorEvent> {
        self.error_history.iter().cloned().collect()
    }

    pub fn clear_history(&mut self) {
        self.error_history.clear();
    }

    pub fn is_critical_state(&self) -> bool {
        let recent_errors = self.get_recent_errors(60);
        let failed_recoveries = recent_errors
            .iter()
            .filter(|e| e.recovery_attempted && !e.recovery_successful)
            .count();

        failed_recoveries >= 3
    }

    pub fn suggest_recovery(&self, error_type: ErrorType) -> String {
        match error_type {
            ErrorType::ProcessStartFailed => {
                "Try checking if the binary path is correct and the file has execute permissions."
                    .to_string()
            }
            ErrorType::ProcessCrashed => {
                "The process crashed unexpectedly. Check the logs for more details.".to_string()
            }
            ErrorType::ProcessTimeout => {
                "The process took too long to complete. Try increasing the timeout or reducing the workload."
                    .to_string()
            }
            ErrorType::MemoryExceeded => {
                "The process exceeded available memory. Try reducing the problem size or increasing system memory."
                    .to_string()
            }
            ErrorType::IOError => {
                "An I/O error occurred. Check disk space and file permissions.".to_string()
            }
            ErrorType::ConfigError => {
                "Configuration error detected. Check the config file for syntax errors.".to_string()
            }
            ErrorType::Unknown => {
                "An unknown error occurred. Check the logs for more information.".to_string()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_recovery_manager_creation() {
        let manager = ErrorRecoveryManager::new(100);
        assert_eq!(manager.get_error_count(ErrorType::ProcessStartFailed), 0);
    }

    #[test]
    fn test_record_error() {
        let mut manager = ErrorRecoveryManager::new(100);
        manager.record_error(ErrorType::ProcessStartFailed, "Test error".to_string());
        assert_eq!(manager.get_error_count(ErrorType::ProcessStartFailed), 1);
    }

    #[test]
    fn test_retry_delay() {
        let manager = ErrorRecoveryManager::new(100);
        let delay0 = manager.get_retry_delay(0);
        let delay1 = manager.get_retry_delay(1);
        assert!(delay1 > delay0);
    }

    #[test]
    fn test_should_retry() {
        let manager = ErrorRecoveryManager::new(100);
        assert!(manager.should_retry(0));
        assert!(manager.should_retry(2));
        assert!(!manager.should_retry(3));
    }
}
