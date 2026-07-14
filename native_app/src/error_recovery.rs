//! Small helpers for describing/recovering from recoverable errors.

/// Wraps an error message with a suggested recovery action for display in
/// the UI or logs.
pub fn describe_recovery(context: &str, error: &str) -> String {
    format!("{} failed: {}. You may retry the action.", context, error)
}
