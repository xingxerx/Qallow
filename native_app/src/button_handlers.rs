use crate::backend::process_manager::ProcessManager;
use crate::codebase_manager::CodebaseManager;
use crate::logging::AppLogger;
use crate::messaging::UiMessage;
use crate::models::{AppState, AuditLog, BuildType, LineType, LogLevel, Phase, TerminalLine};
use chrono::Utc;
use fltk::app::Sender;
use std::sync::{Arc, Mutex};
use std::thread;

/// Handles all button click events and connects them to backend functionality
pub struct ButtonHandler {
    state: Arc<Mutex<AppState>>,
    process_manager: Arc<Mutex<ProcessManager>>,
    logger: Arc<AppLogger>,
    codebase_manager: Option<Arc<CodebaseManager>>,
    ui_sender: Option<Sender<UiMessage>>,
}

impl ButtonHandler {
    pub fn new(
        state: Arc<Mutex<AppState>>,
        process_manager: Arc<Mutex<ProcessManager>>,
        logger: Arc<AppLogger>,
        codebase_manager: Option<Arc<CodebaseManager>>,
        ui_sender: Option<Sender<UiMessage>>,
    ) -> Self {
        ButtonHandler {
            state,
            process_manager,
            logger,
            codebase_manager,
            ui_sender,
        }
    }

    /// Handle Start VM button click
    pub fn on_start_vm(&self) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;
        let mut pm = self
            .process_manager
            .lock()
            .map_err(|e| format!("PM lock error: {}", e))?;

        if state.vm_running {
            return Err("VM is already running".to_string());
        }

        // Start the VM with current configuration
        pm.start_vm(
            state.selected_build,
            state.selected_phase,
            state.phase_config.ticks,
        )?;

        state.vm_running = true;
        state.mind_started_at = Some(Utc::now());
        state.current_step = 0;

        // Add terminal output
        let build_str = match state.selected_build {
            BuildType::CPU => "CPU",
            BuildType::CUDA => "CUDA",
        };
        let phase_str = match state.selected_phase {
            Phase::Phase13 => "Phase 13",
            Phase::Phase14 => "Phase 14",
            Phase::Phase15 => "Phase 15",
        };

        let line = TerminalLine {
            timestamp: Utc::now(),
            content: format!(
                "🚀 Starting Qallow VM with {} build on {} (ticks: {})",
                build_str, phase_str, state.phase_config.ticks
            ),
            line_type: LineType::Info,
        };
        state.terminal_output.push_back(line);

        // Add audit log
        let audit = AuditLog {
            timestamp: Utc::now(),
            level: LogLevel::Success,
            component: "ControlPanel".to_string(),
            message: format!("VM started with {} build on {}", build_str, phase_str),
        };
        state.audit_logs.push_back(audit);

        let _ = self.logger.info(&format!(
            "✓ VM started with {} build on {}",
            build_str, phase_str
        ));
        Ok(())
    }

    /// Handle Stop VM button click
    pub fn on_stop_vm(&self) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;
        let mut pm = self
            .process_manager
            .lock()
            .map_err(|e| format!("PM lock error: {}", e))?;

        if !state.vm_running {
            return Err("VM is not running".to_string());
        }

        // Gracefully stop the VM
        pm.try_graceful_stop(30)?;

        state.vm_running = false;

        // Calculate uptime
        let uptime = state
            .mind_started_at
            .map(|start| Utc::now().signed_duration_since(start).num_seconds())
            .unwrap_or(0);

        // Add terminal output
        let line = TerminalLine {
            timestamp: Utc::now(),
            content: format!(
                "⏹️ VM stopped gracefully (uptime: {}s, steps: {}, reward: {:.2})",
                uptime, state.current_step, state.reward
            ),
            line_type: LineType::Info,
        };
        state.terminal_output.push_back(line);

        // Add audit log
        let audit = AuditLog {
            timestamp: Utc::now(),
            level: LogLevel::Warning,
            component: "ControlPanel".to_string(),
            message: format!(
                "VM stopped after {}s with {} steps",
                uptime, state.current_step
            ),
        };
        state.audit_logs.push_back(audit);

        let _ = self.logger.info(&format!("✓ VM stopped after {}s", uptime));
        Ok(())
    }

    /// Handle Pause button click
    pub fn on_pause(&self) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;

        if !state.vm_running {
            return Err("VM is not running".to_string());
        }

        state.vm_running = false;

        // Add terminal output with current metrics
        let line = TerminalLine {
            timestamp: Utc::now(),
            content: format!(
                "⏸️ VM paused (step: {}, reward: {:.2}, energy: {:.2}, risk: {:.2})",
                state.current_step, state.reward, state.energy, state.risk
            ),
            line_type: LineType::Info,
        };
        state.terminal_output.push_back(line);

        // Add audit log
        let audit = AuditLog {
            timestamp: Utc::now(),
            level: LogLevel::Info,
            component: "ControlPanel".to_string(),
            message: format!("VM paused at step {}", state.current_step),
        };
        state.audit_logs.push_back(audit);

        let _ = self
            .logger
            .info(&format!("✓ VM paused at step {}", state.current_step));
        Ok(())
    }

    /// Handle Reset button click
    pub fn on_reset(&self) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;

        if state.vm_running {
            return Err("Cannot reset while VM is running".to_string());
        }

        // Store previous metrics for comparison
        let prev_steps = state.current_step;
        let prev_reward = state.reward;

        // Reset state
        state.current_step = 0;
        state.reward = 0.0;
        state.energy = 0.0;
        state.risk = 0.0;
        state.mind_started_at = None;
        state.telemetry.clear();

        // Add terminal output
        let line = TerminalLine {
            timestamp: Utc::now(),
            content: format!(
                "🔄 System reset (cleared {} steps, reward: {:.2})",
                prev_steps, prev_reward
            ),
            line_type: LineType::Info,
        };
        state.terminal_output.push_back(line);

        // Add audit log
        let audit = AuditLog {
            timestamp: Utc::now(),
            level: LogLevel::Info,
            component: "ControlPanel".to_string(),
            message: format!(
                "System reset - cleared {} steps and {:.2} reward",
                prev_steps, prev_reward
            ),
        };
        state.audit_logs.push_back(audit);

        let _ = self
            .logger
            .info(&format!("✓ System reset - cleared {} steps", prev_steps));
        Ok(())
    }

    /// Handle Build selection change
    pub fn on_build_selected(&self, build: BuildType) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;

        if state.vm_running {
            return Err("Cannot change build while VM is running".to_string());
        }

        let build_str = match build {
            BuildType::CPU => "CPU",
            BuildType::CUDA => "CUDA",
        };

        state.selected_build = build;

        let line = TerminalLine {
            timestamp: Utc::now(),
            content: format!(
                "📦 Build selected: {} (optimized for {})",
                build_str,
                if build == BuildType::CUDA {
                    "GPU acceleration"
                } else {
                    "CPU processing"
                }
            ),
            line_type: LineType::Info,
        };
        state.terminal_output.push_back(line);

        // Add audit log
        let audit = AuditLog {
            timestamp: Utc::now(),
            level: LogLevel::Info,
            component: "ControlPanel".to_string(),
            message: format!("Build changed to {}", build_str),
        };
        state.audit_logs.push_back(audit);

        let _ = self
            .logger
            .info(&format!("✓ Build changed to {}", build_str));
        Ok(())
    }

    /// Handle Phase selection change
    pub fn on_phase_selected(&self, phase: Phase) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;

        if state.vm_running {
            return Err("Cannot change phase while VM is running".to_string());
        }

        let phase_str = match phase {
            Phase::Phase13 => "Phase 13",
            Phase::Phase14 => "Phase 14",
            Phase::Phase15 => "Phase 15",
        };

        let phase_desc = match phase {
            Phase::Phase13 => "Quantum Circuit Optimization",
            Phase::Phase14 => "Photonic Integration",
            Phase::Phase15 => "AGI Synthesis",
        };

        state.selected_phase = phase;

        let line = TerminalLine {
            timestamp: Utc::now(),
            content: format!("📍 Phase selected: {} - {}", phase_str, phase_desc),
            line_type: LineType::Info,
        };
        state.terminal_output.push_back(line);

        // Add audit log
        let audit = AuditLog {
            timestamp: Utc::now(),
            level: LogLevel::Info,
            component: "ControlPanel".to_string(),
            message: format!("Phase changed to {} ({})", phase_str, phase_desc),
        };
        state.audit_logs.push_back(audit);

        let _ = self
            .logger
            .info(&format!("✓ Phase changed to {}", phase_str));
        Ok(())
    }

    /// Handle Export Metrics button click
    pub fn on_export_metrics(&self) -> Result<String, String> {
        let state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;

        // Create comprehensive metrics export
        let export_data = serde_json::json!({
            "timestamp": Utc::now().to_rfc3339(),
            "vm_running": state.vm_running,
            "current_step": state.current_step,
            "reward": state.reward,
            "energy": state.energy,
            "risk": state.risk,
            "selected_build": format!("{:?}", state.selected_build),
            "selected_phase": format!("{:?}", state.selected_phase),
            "metrics": state.metrics,
            "telemetry_count": state.telemetry.len(),
            "terminal_lines": state.terminal_output.len(),
            "audit_logs": state.audit_logs.len(),
        });

        let metrics_json = serde_json::to_string_pretty(&export_data)
            .map_err(|e| format!("Serialization error: {}", e))?;

        let _ = self.logger.info(&format!(
            "✓ Metrics exported ({} bytes)",
            metrics_json.len()
        ));
        Ok(metrics_json)
    }

    /// Handle Save Config button click
    pub fn on_save_config(&self) -> Result<(), String> {
        let state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;

        // Create comprehensive configuration export
        let config_export = serde_json::json!({
            "timestamp": Utc::now().to_rfc3339(),
            "phase_config": state.phase_config,
            "selected_build": format!("{:?}", state.selected_build),
            "selected_phase": format!("{:?}", state.selected_phase),
            "current_metrics": {
                "step": state.current_step,
                "reward": state.reward,
                "energy": state.energy,
                "risk": state.risk,
            },
            "vm_running": state.vm_running,
        });

        let config_json = serde_json::to_string_pretty(&config_export)
            .map_err(|e| format!("Serialization error: {}", e))?;

        std::fs::write("qallow_phase_config.json", &config_json)
            .map_err(|e| format!("File write error: {}", e))?;

        // Add audit log
        let audit = AuditLog {
            timestamp: Utc::now(),
            level: LogLevel::Success,
            component: "ControlPanel".to_string(),
            message: "Configuration saved to qallow_phase_config.json".to_string(),
        };

        // Need to drop the lock before acquiring it again
        drop(state);

        if let Ok(mut state) = self.state.lock() {
            state.audit_logs.push_back(audit);
        }

        let _ = self
            .logger
            .info("✓ Configuration saved to qallow_phase_config.json");
        Ok(())
    }

    /// Handle View Logs button click
    pub fn on_view_logs(&self) -> Result<Vec<String>, String> {
        let state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;

        let mut logs: Vec<String> = Vec::new();

        // Add header
        logs.push("═══════════════════════════════════════════════════════════════".to_string());
        logs.push(format!("📋 Audit Log - {} entries", state.audit_logs.len()));
        logs.push("═══════════════════════════════════════════════════════════════".to_string());
        logs.push("".to_string());

        // Add audit logs
        for log in state.audit_logs.iter().rev().take(50) {
            let (level_icon, level_str) = match log.level {
                LogLevel::Info => ("ℹ️", "INFO"),
                LogLevel::Success => ("✅", "SUCCESS"),
                LogLevel::Warning => ("⚠️", "WARNING"),
                LogLevel::Error => ("❌", "ERROR"),
            };

            logs.push(format!(
                "{} [{}] {} - {}: {}",
                level_icon,
                log.timestamp.format("%H:%M:%S"),
                level_str,
                log.component,
                log.message
            ));
        }

        logs.push("".to_string());
        logs.push("═══════════════════════════════════════════════════════════════".to_string());

        Ok(logs)
    }

    /// Handle Build Native App button click
    pub fn on_build_native_app(&self) -> Result<String, String> {
        let manager = self
            .codebase_manager
            .as_ref()
            .ok_or_else(|| "Codebase manager not available".to_string())?;

        let result = manager.build_native_app()?;
        let terminal_message = format!("🛠️ Native app build result: {}", result);

        {
            let mut state = self
                .state
                .lock()
                .map_err(|e| format!("State lock error: {}", e))?;

            state.terminal_output.push_back(TerminalLine {
                timestamp: Utc::now(),
                content: terminal_message.clone(),
                line_type: LineType::Info,
            });

            state.audit_logs.push_back(AuditLog {
                timestamp: Utc::now(),
                level: LogLevel::Success,
                component: "Codebase".to_string(),
                message: "Native app build executed".to_string(),
            });
        }

        let _ = self
            .logger
            .info("✓ Native app build executed via control panel");

        Ok(result)
    }

    /// Start build in a background thread and notify UI when done
    pub fn start_build_native_app_async(&self) -> Result<(), String> {
        let mgr = self
            .codebase_manager
            .as_ref()
            .ok_or_else(|| "Codebase manager not available".to_string())?
            .clone();
        let logger = self.logger.clone();
        let state = self.state.clone();
        let sender = self
            .ui_sender
            .clone()
            .ok_or_else(|| "UI sender unavailable".to_string())?;
        thread::spawn(move || {
            let res = mgr.build_native_app();
            if let Ok(mut s) = state.lock() {
                s.add_terminal_line(
                    "🛠️ Build completed".to_string(),
                    match &res {
                        Ok(_) => LineType::Info,
                        Err(_) => LineType::Error,
                    },
                );
                s.add_audit_log(
                    match &res {
                        Ok(_) => LogLevel::Success,
                        Err(_) => LogLevel::Error,
                    },
                    "Codebase".to_string(),
                    "Build finished".to_string(),
                );
            }
            let _ = logger.info("ℹ️ Build finished (async)");
            sender.send(UiMessage::BuildDone(res));
        });
        Ok(())
    }

    /// Handle Run Tests button click
    pub fn on_run_tests(&self) -> Result<String, String> {
        let manager = self
            .codebase_manager
            .as_ref()
            .ok_or_else(|| "Codebase manager not available".to_string())?;

        let result = manager.run_tests()?;
        let terminal_message = format!("🧪 Test run result: {}", result);

        {
            let mut state = self
                .state
                .lock()
                .map_err(|e| format!("State lock error: {}", e))?;

            state.terminal_output.push_back(TerminalLine {
                timestamp: Utc::now(),
                content: terminal_message.clone(),
                line_type: LineType::Info,
            });

            state.audit_logs.push_back(AuditLog {
                timestamp: Utc::now(),
                level: LogLevel::Success,
                component: "Codebase".to_string(),
                message: "Native app tests executed".to_string(),
            });
        }

        let _ = self
            .logger
            .info("✓ Native app tests executed via control panel");

        Ok(result)
    }

    /// Start tests in a background thread and notify UI when done
    pub fn start_run_tests_async(&self) -> Result<(), String> {
        let mgr = self
            .codebase_manager
            .as_ref()
            .ok_or_else(|| "Codebase manager not available".to_string())?
            .clone();
        let logger = self.logger.clone();
        let state = self.state.clone();
        let sender = self
            .ui_sender
            .clone()
            .ok_or_else(|| "UI sender unavailable".to_string())?;
        thread::spawn(move || {
            let res = mgr.run_tests();
            if let Ok(mut s) = state.lock() {
                s.add_terminal_line(
                    "🧪 Tests completed".to_string(),
                    match &res {
                        Ok(_) => LineType::Info,
                        Err(_) => LineType::Error,
                    },
                );
                s.add_audit_log(
                    match &res {
                        Ok(_) => LogLevel::Success,
                        Err(_) => LogLevel::Error,
                    },
                    "Codebase".to_string(),
                    "Tests finished".to_string(),
                );
            }
            let _ = logger.info("ℹ️ Tests finished (async)");
            sender.send(UiMessage::TestsDone(res));
        });
        Ok(())
    }

    /// Handle Git Status button click
    pub fn on_git_status(&self) -> Result<String, String> {
        let manager = self
            .codebase_manager
            .as_ref()
            .ok_or_else(|| "Codebase manager not available".to_string())?;

        let status = manager.get_git_status()?;
        let trimmed = status.trim();
        let status_message = if trimmed.is_empty() {
            "Working tree clean".to_string()
        } else {
            trimmed.to_string()
        };

        {
            let mut state = self
                .state
                .lock()
                .map_err(|e| format!("State lock error: {}", e))?;

            state.terminal_output.push_back(TerminalLine {
                timestamp: Utc::now(),
                content: format!("📁 Git status:\n{}", status_message),
                line_type: LineType::Info,
            });

            state.audit_logs.push_back(AuditLog {
                timestamp: Utc::now(),
                level: LogLevel::Info,
                component: "Codebase".to_string(),
                message: "Git status fetched".to_string(),
            });
        }

        let _ = self.logger.info("ℹ️ Git status fetched via control panel");

        Ok(status_message)
    }

    /// Start git status in a background thread and notify UI when done
    pub fn start_git_status_async(&self) -> Result<(), String> {
        let mgr = self
            .codebase_manager
            .as_ref()
            .ok_or_else(|| "Codebase manager not available".to_string())?
            .clone();
        let state = self.state.clone();
        let sender = self
            .ui_sender
            .clone()
            .ok_or_else(|| "UI sender unavailable".to_string())?;
        thread::spawn(move || {
            let res = mgr.get_git_status().map(|s| {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    "Working tree clean".to_string()
                } else {
                    trimmed.to_string()
                }
            });
            if let Ok(mut st) = state.lock() {
                match &res {
                    Ok(msg) => {
                        st.add_terminal_line(format!("📁 Git status:\n{}", msg), LineType::Info)
                    }
                    Err(err) => {
                        st.add_terminal_line(format!("Git status error: {}", err), LineType::Error)
                    }
                }
            }
            sender.send(UiMessage::GitStatusDone(res));
        });
        Ok(())
    }

    /// Handle Recent Commits button click
    pub fn on_recent_commits(&self, count: usize) -> Result<Vec<String>, String> {
        let manager = self
            .codebase_manager
            .as_ref()
            .ok_or_else(|| "Codebase manager not available".to_string())?;

        let commits = manager.get_recent_commits(count)?;
        let display = if commits.is_empty() {
            "No commits available".to_string()
        } else {
            commits
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        };

        {
            let mut state = self
                .state
                .lock()
                .map_err(|e| format!("State lock error: {}", e))?;

            state.terminal_output.push_back(TerminalLine {
                timestamp: Utc::now(),
                content: format!("📜 Recent commits:\n{}", display),
                line_type: LineType::Info,
            });

            state.audit_logs.push_back(AuditLog {
                timestamp: Utc::now(),
                level: LogLevel::Info,
                component: "Codebase".to_string(),
                message: format!(
                    "Fetched {} recent commit{}",
                    commits.len(),
                    if commits.len() == 1 { "" } else { "s" }
                ),
            });
        }

        let _ = self
            .logger
            .info("ℹ️ Recent commits fetched via control panel");

        Ok(commits)
    }

    /// Start recent commits fetch in a background thread and notify UI when done
    pub fn start_recent_commits_async(&self, count: usize) -> Result<(), String> {
        let mgr = self
            .codebase_manager
            .as_ref()
            .ok_or_else(|| "Codebase manager not available".to_string())?
            .clone();
        let state = self.state.clone();
        let sender = self
            .ui_sender
            .clone()
            .ok_or_else(|| "UI sender unavailable".to_string())?;
        thread::spawn(move || {
            let res = mgr.get_recent_commits(count);
            if let Ok(mut st) = state.lock() {
                match &res {
                    Ok(list) => {
                        let display = if list.is_empty() {
                            "No commits available".to_string()
                        } else {
                            list.join("\n")
                        };
                        st.add_terminal_line(
                            format!("📜 Recent commits:\n{}", display),
                            LineType::Info,
                        );
                    }
                    Err(err) => st.add_terminal_line(
                        format!("Commits fetch error: {}", err),
                        LineType::Error,
                    ),
                }
            }
            sender.send(UiMessage::CommitsDone(res));
        });
        Ok(())
    }
}
