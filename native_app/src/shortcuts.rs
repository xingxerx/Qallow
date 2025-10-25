use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Shortcut {
    // Application
    Quit,
    Settings,
    Help,

    // VM Control
    StartVM,
    StopVM,
    RestartVM,

    // UI Navigation
    DashboardTab,
    MetricsTab,
    TerminalTab,
    AuditLogTab,
    ControlPanelTab,

    // Terminal
    ClearTerminal,
    CopyTerminal,
    PasteTerminal,

    // General
    Refresh,
    Save,
    Undo,
    Redo,
}

#[derive(Debug, Clone)]
pub struct ShortcutBinding {
    pub shortcut: Shortcut,
    pub key_combo: String,
    pub description: String,
    pub enabled: bool,
}

pub struct ShortcutManager {
    bindings: HashMap<String, Shortcut>,
    descriptions: HashMap<Shortcut, String>,
}

impl ShortcutManager {
    pub fn new() -> Self {
        let mut manager = Self {
            bindings: HashMap::new(),
            descriptions: HashMap::new(),
        };
        manager.init_default_bindings();
        manager
    }

    fn init_default_bindings(&mut self) {
        // Application shortcuts
        self.register("ctrl+q", Shortcut::Quit, "Quit application");
        self.register("ctrl+,", Shortcut::Settings, "Open settings");
        self.register("f1", Shortcut::Help, "Show help");

        // VM Control
        self.register("ctrl+shift+s", Shortcut::StartVM, "Start VM");
        self.register("ctrl+shift+x", Shortcut::StopVM, "Stop VM");
        self.register("ctrl+shift+r", Shortcut::RestartVM, "Restart VM");

        // UI Navigation
        self.register("ctrl+1", Shortcut::DashboardTab, "Go to Dashboard");
        self.register("ctrl+2", Shortcut::MetricsTab, "Go to Metrics");
        self.register("ctrl+3", Shortcut::TerminalTab, "Go to Terminal");
        self.register("ctrl+4", Shortcut::AuditLogTab, "Go to Audit Log");
        self.register("ctrl+5", Shortcut::ControlPanelTab, "Go to Control Panel");

        // Terminal
        self.register("ctrl+l", Shortcut::ClearTerminal, "Clear terminal");
        self.register("ctrl+c", Shortcut::CopyTerminal, "Copy from terminal");
        self.register("ctrl+v", Shortcut::PasteTerminal, "Paste to terminal");

        // General
        self.register("f5", Shortcut::Refresh, "Refresh");
        self.register("ctrl+s", Shortcut::Save, "Save");
        self.register("ctrl+z", Shortcut::Undo, "Undo");
        self.register("ctrl+y", Shortcut::Redo, "Redo");
    }

    fn register(&mut self, key_combo: &str, shortcut: Shortcut, description: &str) {
        self.bindings.insert(key_combo.to_string(), shortcut);
        self.descriptions.insert(shortcut, description.to_string());
    }

    #[allow(dead_code)]
    pub fn get_shortcut(&self, key_combo: &str) -> Option<Shortcut> {
        self.bindings.get(key_combo).copied()
    }

    #[allow(dead_code)]
    pub fn get_description(&self, shortcut: Shortcut) -> Option<&str> {
        self.descriptions.get(&shortcut).map(|s| s.as_str())
    }

    #[allow(dead_code)]
    pub fn get_key_combo(&self, shortcut: Shortcut) -> Option<String> {
        self.bindings
            .iter()
            .find(|(_, &s)| s == shortcut)
            .map(|(k, _)| k.clone())
    }

    #[allow(dead_code)]
    pub fn get_all_bindings(&self) -> Vec<ShortcutBinding> {
        self.bindings
            .iter()
            .map(|(key_combo, &shortcut)| ShortcutBinding {
                shortcut,
                key_combo: key_combo.clone(),
                description: self
                    .descriptions
                    .get(&shortcut)
                    .cloned()
                    .unwrap_or_default(),
                enabled: true,
            })
            .collect()
    }

    #[allow(dead_code)]
    pub fn rebind(&mut self, key_combo: &str, shortcut: Shortcut) -> Result<(), String> {
        // Remove old binding for this shortcut
        self.bindings.retain(|_, &mut s| s != shortcut);

        // Add new binding
        self.bindings.insert(key_combo.to_string(), shortcut);
        Ok(())
    }

    #[allow(dead_code)]
    pub fn reset_to_defaults(&mut self) {
        self.bindings.clear();
        self.descriptions.clear();
        self.init_default_bindings();
    }

    #[allow(dead_code)]
    pub fn export_bindings(&self) -> String {
        let mut output = String::from("# Qallow Keyboard Shortcuts\n\n");

        let mut bindings: Vec<_> = self.get_all_bindings();
        bindings.sort_by(|a, b| a.key_combo.cmp(&b.key_combo));

        for binding in bindings {
            output.push_str(&format!(
                "{:<20} - {}\n",
                binding.key_combo, binding.description
            ));
        }

        output
    }
}

impl Default for ShortcutManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shortcut_manager_creation() {
        let manager = ShortcutManager::new();
        assert!(manager.get_shortcut("ctrl+q").is_some());
    }

    #[test]
    fn test_get_shortcut() {
        let manager = ShortcutManager::new();
        assert_eq!(manager.get_shortcut("ctrl+q"), Some(Shortcut::Quit));
    }

    #[test]
    fn test_get_description() {
        let manager = ShortcutManager::new();
        assert_eq!(
            manager.get_description(Shortcut::Quit),
            Some("Quit application")
        );
    }

    #[test]
    fn test_rebind() {
        let mut manager = ShortcutManager::new();
        assert!(manager.rebind("alt+q", Shortcut::Quit).is_ok());
        assert_eq!(manager.get_shortcut("alt+q"), Some(Shortcut::Quit));
    }

    #[test]
    fn test_export_bindings() {
        let manager = ShortcutManager::new();
        let export = manager.export_bindings();
        assert!(export.contains("Quit application"));
    }
}
