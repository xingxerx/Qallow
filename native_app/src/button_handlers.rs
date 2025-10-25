use std::sync::{Arc, Mutex};
use crate::models::{AppState, BuildType, Phase, LineType, TerminalLine, AuditLog, LogLevel};
use crate::backend::process_manager::ProcessManager;
use crate::logging::AppLogger;
use chrono::Utc;

/// Handles all button click events and connects them to backend functionality
pub struct ButtonHandler {
    state: Arc<Mutex<AppState>>,
    process_manager: Arc<Mutex<ProcessManager>>,
    logger: Arc<AppLogger>,
}

impl ButtonHandler {
    pub fn new(
        state: Arc<Mutex<AppState>>,
        process_manager: Arc<Mutex<ProcessManager>>,
        logger: Arc<AppLogger>,
    ) -> Self {
        ButtonHandler {
            state,
            process_manager,
            logger,
        }
    }

    /// Handle Start VM button click
    pub fn on_start_vm(&self) -> Result<(), String> {
        let mut state = self.state.lock().map_err(|e| format!("State lock error: {}", e))?;
        let mut pm = self.process_manager.lock().map_err(|e| format!("PM lock error: {}", e))?;

        if state.vm_running {
            return Err("VM is already running".to_string());
        }

        // Start the VM with current configuration
        pm.start_vm(state.selected_build, state.selected_phase, state.phase_config.ticks)?;

        state.vm_running = true;
        state.mind_started_at = Some(Utc::now());

        // Add terminal output
        let line = TerminalLine {
            timestamp: Utc::now(),
            content: format!(
                "🚀 Starting VM with {:?} build on {:?}",
                state.selected_build, state.selected_phase
            ),
            line_type: LineType::Info,
        };
        state.terminal_output.push_back(line);

        // Add audit log
        let audit = AuditLog {
            timestamp: Utc::now(),
            level: LogLevel::Success,
            component: "ControlPanel".to_string(),
            message: "VM started successfully".to_string(),
        };
        state.audit_logs.push_back(audit);

        let _ = self.logger.info("✓ VM started successfully");
        Ok(())
    }

    /// Handle Stop VM button click
    pub fn on_stop_vm(&self) -> Result<(), String> {
        let mut state = self.state.lock().map_err(|e| format!("State lock error: {}", e))?;
        let mut pm = self.process_manager.lock().map_err(|e| format!("PM lock error: {}", e))?;

        if !state.vm_running {
            return Err("VM is not running".to_string());
        }

        // Gracefully stop the VM
        pm.try_graceful_stop(30)?;

        state.vm_running = false;

        // Add terminal output
        let line = TerminalLine {
            timestamp: Utc::now(),
            content: "⏹️ VM stopped".to_string(),
            line_type: LineType::Info,
        };
        state.terminal_output.push_back(line);

        // Add audit log
        let audit = AuditLog {
            timestamp: Utc::now(),
            level: LogLevel::Warning,
            component: "ControlPanel".to_string(),
            message: "VM stopped".to_string(),
        };
        state.audit_logs.push_back(audit);

        let _ = self.logger.info("✓ VM stopped");
        Ok(())
    }

    /// Handle Pause button click
    pub fn on_pause(&self) -> Result<(), String> {
        let mut state = self.state.lock().map_err(|e| format!("State lock error: {}", e))?;

        if !state.vm_running {
            return Err("VM is not running".to_string());
        }

        // Add terminal output
        let line = TerminalLine {
            timestamp: Utc::now(),
            content: "⏸️ VM paused".to_string(),
            line_type: LineType::Info,
        };
        state.terminal_output.push_back(line);

        let _ = self.logger.info("✓ VM paused");
        Ok(())
    }

    /// Handle Reset button click
    pub fn on_reset(&self) -> Result<(), String> {
        let mut state = self.state.lock().map_err(|e| format!("State lock error: {}", e))?;

        if state.vm_running {
            return Err("Cannot reset while VM is running".to_string());
        }

        // Reset state
        state.current_step = 0;
        state.reward = 0.0;
        state.energy = 0.0;
        state.risk = 0.0;
        state.terminal_output.clear();
        state.telemetry.clear();

        // Add terminal output
        let line = TerminalLine {
            timestamp: Utc::now(),
            content: "🔄 System reset".to_string(),
            line_type: LineType::Info,
        };
        state.terminal_output.push_back(line);

        // Add audit log
        let audit = AuditLog {
            timestamp: Utc::now(),
            level: LogLevel::Info,
            component: "ControlPanel".to_string(),
            message: "System reset".to_string(),
        };
        state.audit_logs.push_back(audit);

        let _ = self.logger.info("✓ System reset");
        Ok(())
    }

    /// Handle Build selection change
    pub fn on_build_selected(&self, build: BuildType) -> Result<(), String> {
        let mut state = self.state.lock().map_err(|e| format!("State lock error: {}", e))?;

        if state.vm_running {
            return Err("Cannot change build while VM is running".to_string());
        }

        state.selected_build = build;

        let line = TerminalLine {
            timestamp: Utc::now(),
            content: format!("📦 Build selected: {:?}", build),
            line_type: LineType::Info,
        };
        state.terminal_output.push_back(line);

        let _ = self.logger.info(&format!("✓ Build changed to {:?}", build));
        Ok(())
    }

    /// Handle Phase selection change
    pub fn on_phase_selected(&self, phase: Phase) -> Result<(), String> {
        let mut state = self.state.lock().map_err(|e| format!("State lock error: {}", e))?;

        if state.vm_running {
            return Err("Cannot change phase while VM is running".to_string());
        }

        state.selected_phase = phase;

        let line = TerminalLine {
            timestamp: Utc::now(),
            content: format!("📍 Phase selected: {:?}", phase),
            line_type: LineType::Info,
        };
        state.terminal_output.push_back(line);

        let _ = self.logger.info(&format!("✓ Phase changed to {:?}", phase));
        Ok(())
    }

    /// Handle Export Metrics button click
    pub fn on_export_metrics(&self) -> Result<String, String> {
        let state = self.state.lock().map_err(|e| format!("State lock error: {}", e))?;

        // Export metrics as JSON
        let metrics_json = serde_json::to_string_pretty(&state.metrics)
            .map_err(|e| format!("Serialization error: {}", e))?;

        let _ = self.logger.info("✓ Metrics exported");
        Ok(metrics_json)
    }

    /// Handle Save Config button click
    pub fn on_save_config(&self) -> Result<(), String> {
        let state = self.state.lock().map_err(|e| format!("State lock error: {}", e))?;

        // Save configuration to file
        let config_json = serde_json::to_string_pretty(&state.phase_config)
            .map_err(|e| format!("Serialization error: {}", e))?;

        std::fs::write("qallow_phase_config.json", config_json)
            .map_err(|e| format!("File write error: {}", e))?;

        let _ = self.logger.info("✓ Configuration saved");
        Ok(())
    }

    /// Handle View Logs button click
    pub fn on_view_logs(&self) -> Result<Vec<String>, String> {
        let state = self.state.lock().map_err(|e| format!("State lock error: {}", e))?;

        let logs: Vec<String> = state
            .audit_logs
            .iter()
            .map(|log| {
                format!(
                    "[{}] {:?} - {}: {}",
                    log.timestamp.format("%H:%M:%S"),
                    log.level,
                    log.component,
                    log.message
                )
            })
            .collect();

        Ok(logs)
    }
}

