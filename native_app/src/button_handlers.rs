//! Handlers wired up to the phase/unified buttons in the main window. They
//! update `AppState`, log the action, notify the UI thread, and launch the
//! underlying `qallow` CLI binary for the requested phase.

use crate::backend::process_manager::ProcessManager;
use crate::codebase_manager::CodebaseManager;
use crate::logging::AppLogger;
use crate::messaging::UiMessage;
use crate::models::{AppState, LineType, Phase};
use fltk::app;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

pub struct ButtonHandler {
    state: Arc<Mutex<AppState>>,
    process_manager: Arc<Mutex<ProcessManager>>,
    logger: Arc<AppLogger>,
    codebase_mgr: Option<Arc<CodebaseManager>>,
    sender: Option<app::Sender<UiMessage>>,
}

impl ButtonHandler {
    pub fn new(
        state: Arc<Mutex<AppState>>,
        process_manager: Arc<Mutex<ProcessManager>>,
        logger: Arc<AppLogger>,
        codebase_mgr: Option<Arc<CodebaseManager>>,
        sender: Option<app::Sender<UiMessage>>,
    ) -> Self {
        Self {
            state,
            process_manager,
            logger,
            codebase_mgr,
            sender,
        }
    }

    /// Launches the given phase via the `qallow` CLI binary, updating shared
    /// state and logs along the way.
    pub fn on_run_phase(&self, phase: Phase) -> Result<(), String> {
        let phase_label = phase.to_str();
        let _ = self.logger.info(&format!("Running phase: {}", phase_label));

        let qallow_root = self
            .codebase_mgr
            .as_ref()
            .map(|mgr| mgr.root_path().to_string());

        if let Ok(mut state) = self.state.lock() {
            state.selected_phase = phase;
            state.set_running(true);
            state.add_terminal_line(
                format!("Starting phase {}...", phase_label),
                LineType::Info,
            );
        }

        if let Some(sender) = &self.sender {
            sender.send(UiMessage::Refresh);
        }

        let binary_name = if cfg!(windows) {
            "qallow.exe"
        } else {
            "qallow"
        };
        let candidates = [
            format!("build/{}", binary_name),
            format!("target/debug/{}", binary_name),
            format!("target/release/{}", binary_name),
        ];

        let binary_path = candidates
            .iter()
            .map(|rel| match &qallow_root {
                Some(root) => Path::new(root).join(rel),
                None => Path::new(rel).to_path_buf(),
            })
            .find(|path| path.exists());

        let binary_path = match binary_path {
            Some(path) => path,
            None => {
                let msg = format!(
                    "qallow backend binary not found (looked for {} under {}) — build it with `cargo build -p qallow_cli`",
                    candidates.join(", "),
                    qallow_root.as_deref().unwrap_or(".")
                );
                let _ = self.logger.error(&msg);
                if let Ok(mut state) = self.state.lock() {
                    state.set_running(false);
                    state.add_terminal_line(msg.clone(), LineType::Error);
                }
                if let Some(sender) = &self.sender {
                    sender.send(UiMessage::Refresh);
                }
                return Err(msg);
            }
        };

        let mut command = Command::new(&binary_path);
        if let Some(root) = &qallow_root {
            command.current_dir(root);
        }
        command.arg("run");
        match phase {
            Phase::Phase2 => {
                command.arg("--phase=2");
            }
            Phase::Phase4 => {
                command.arg("--phase=4");
            }
            Phase::Phase1 | Phase::Phase3 | Phase::Unified => {
                command.arg("unified");
            }
        }
        command.stdout(Stdio::null()).stderr(Stdio::null());

        match command.spawn() {
            Ok(child) => {
                if let Ok(mut pm) = self.process_manager.lock() {
                    pm.insert(phase_label.to_string(), child);
                }
                let _ = self
                    .logger
                    .info(&format!("Phase {} launched", phase_label));
                Ok(())
            }
            Err(e) => {
                let _ = self
                    .logger
                    .error(&format!("Failed to launch phase {}: {}", phase_label, e));
                if let Ok(mut state) = self.state.lock() {
                    state.set_running(false);
                    state.add_terminal_line(
                        format!("Failed to launch phase: {}", e),
                        LineType::Error,
                    );
                }
                Err(format!("Failed to launch phase {}: {}", phase_label, e))
            }
        }
    }
}
