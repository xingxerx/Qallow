//! Keyboard shortcut definitions. Placeholder until the UI wires up
//! application-wide hotkeys.

/// Application keyboard shortcuts, keyed by action name.
pub struct Shortcuts;

impl Shortcuts {
    pub fn describe(action: &str) -> Option<&'static str> {
        match action {
            "quit" => Some("Ctrl+Q"),
            _ => None,
        }
    }
}
