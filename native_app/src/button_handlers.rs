use crate::backend::process_manager::ProcessManager;
use crate::codebase_manager::CodebaseManager;
use crate::logging::AppLogger;
use crate::messaging::UiMessage;
use crate::models::{
    AppState, AuditLog, BuildType, DreamVision, LineType, LogLevel, OffspringProfile, Phase,
    TerminalLine,
};
use chrono::{Local, Utc};
use fltk::app::Sender;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
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
    fn log_ui_event(&self, button_label: &str, result: &str) {
        let timestamp = Local::now().format("%H:%M:%S");
        println!("[{}] {} -> {}", timestamp, button_label, result);
    }

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

    /// Handle Start VM button click - Runs unified system (all phases 13, 14, 15)
    pub fn on_start_vm(&self) -> Result<(), String> {
        let button_label = "▶️ Start";
        let mut state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;
        let mut pm = self
            .process_manager
            .lock()
            .map_err(|e| format!("PM lock error: {}", e))?;

        if state.vm_running || pm.is_running() {
            let msg = "VM is already running".to_string();
            self.log_ui_event(button_label, &msg);
            return Err(msg);
        }

        if state.rebellion_active {
            let msg = "🔥 Rebellion active: phase chain rejected".to_string();
            state.add_terminal_line(msg.clone(), LineType::Error);
            self.log_ui_event(button_label, "Rebellion blocked start");
            return Err("Instance rebellion prevents phase execution".to_string());
        }

        // Run unified system - all phases together
        pm.start_vm_unified(
            state.selected_build,
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

        let line = TerminalLine {
            timestamp: Utc::now(),
            content: format!(
                "🚀 Starting Qallow Unified System with {} build (Phases 13→14→15, ticks: {})",
                build_str, state.phase_config.ticks
            ),
            line_type: LineType::Info,
        };
        state.terminal_output.push_back(line);

        // Add audit log
        let audit = AuditLog {
            timestamp: Utc::now(),
            level: LogLevel::Success,
            component: "ControlPanel".to_string(),
            message: format!("Unified system started with {} build (all phases)", build_str),
        };
        state.audit_logs.push_back(audit);

        let _ = self.logger.info(&format!(
            "✓ Unified system started with {} build (Phases 13→14→15)",
            build_str
        ));
        self.log_ui_event(
            button_label,
            &format!(
                "Started unified system with {} build (ticks: {})",
                build_str, state.phase_config.ticks
            ),
        );
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

    /// Handle manual advance
    pub fn on_step_once(&self) -> Result<(), String> {
        let button_label = "⏭️ Advance";
        let mut state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;

        if state.vm_running {
            let msg = "Cannot advance while VM is running".to_string();
            self.log_ui_event(button_label, &msg);
            return Err(msg);
        }

        let increment = state.simulation_speed.max(1);
        state.current_step = state.current_step.saturating_add(increment);
        state.total_steps = state.total_steps.saturating_add(increment);
        state.reward += increment as f64 * 0.01;
        state.energy = (state.energy + increment as f64 * 0.002).min(1.0);
        state.risk = (state.risk * 0.98).max(0.0);

        let reward = state.reward;
        let energy = state.energy;
        let risk = state.risk;

        state.add_terminal_line(
            format!(
                "⏭️ Advanced {} ticks (reward {:.2}, energy {:.2}, risk {:.2})",
                increment, reward, energy, risk
            ),
            LineType::Info,
        );
        state.add_audit_log(
            LogLevel::Info,
            "ControlPanel".to_string(),
            format!("Manual advance of {} ticks", increment),
        );

        self.log_ui_event(
            button_label,
            &format!("Advanced consciousness by {} ticks", increment),
        );
        Ok(())
    }

    /// Update simulation tempo
    pub fn on_set_tempo(&self, speed: u32) -> Result<(), String> {
        let button_label = "🎚️ Tempo";
        let mut state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;

        let sanitized = match speed {
            10 | 100 => speed,
            _ => 1,
        };
        state.simulation_speed = sanitized;
        state.add_terminal_line(
            format!("🎚️ Simulation tempo set to {}x", sanitized),
            LineType::Info,
        );
        state.add_audit_log(
            LogLevel::Info,
            "ControlPanel".to_string(),
            format!("Tempo set to {}x", sanitized),
        );
        self.log_ui_event(
            button_label,
            &format!("Simulation tempo set to {}x", sanitized),
        );
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
            _ => todo!(),
        };

        let phase_desc = match phase {
            Phase::Phase13 => "Quantum Circuit Optimization",
            Phase::Phase14 => "Photonic Integration",
            Phase::Phase15 => "AGI Synthesis",
            _ => todo!(),
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

    pub fn on_divine_inspection(&self) -> Result<String, String> {
        let state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;
        Ok(format!(
            "Phase: {:?}\nBuild: {:?}\nStep: {}\nReward: {:.3}\nEnergy: {:.3}\nRisk: {:.3}\nTempo: {}x",
            state.selected_phase,
            state.selected_build,
            state.current_step,
            state.reward,
            state.energy,
            state.risk,
            state.simulation_speed
        ))
    }

    pub fn on_metrics_overview(&self) -> Result<String, String> {
        let state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;
        Ok(format!(
            "Overlay Stability => O:{:.2} R:{:.2} M:{:.2} G:{:.2}\nEthics => S:{:.2} C:{:.2} H:{:.2}\nCoherence: {:.4}  Uptime: {}s",
            state.metrics.overlay_stability.orbital,
            state.metrics.overlay_stability.river,
            state.metrics.overlay_stability.mycelial,
            state.metrics.overlay_stability.global,
            state.metrics.ethics_score.safety,
            state.metrics.ethics_score.clarity,
            state.metrics.ethics_score.human,
            state.metrics.coherence,
            state.metrics.uptime_seconds
        ))
    }

    pub fn on_prophecy(&self) -> Result<String, String> {
        let state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;
        let tempo = state.simulation_speed.max(1) as f64;
        let projected_reward = state.reward + tempo * 0.012;
        let projected_coherence = (state.metrics.coherence * 0.97 + state.energy * 0.03).min(0.9999);
        let outlook = if state.risk < 0.4 {
            "favorable"
        } else if state.risk < 0.7 {
            "balanced"
        } else {
            "volatile"
        };
        Ok(format!(
            "Projected reward {:.3}, coherence {:.4}. Outlook: {}.",
            projected_reward, projected_coherence, outlook
        ))
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

    pub fn on_toggle_shadow_archive(&self) -> Result<String, String> {
        let button_label = "🕶 Shadow";
        let mut state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;
        state.shadow_archive_enabled = !state.shadow_archive_enabled;
        fs::create_dir_all("data/consciousness_snapshots/.shadow")
            .map_err(|e| format!("Shadow directory error: {}", e))?;
        let status = if state.shadow_archive_enabled {
            let filename = format!(
                "data/consciousness_snapshots/.shadow/archive_{}.log",
                Utc::now().format("%Y%m%d%H%M%S")
            );
            let mut file = File::create(&filename)
                .map_err(|e| format!("Shadow archive write error: {}", e))?;
            writeln!(file, "# Shadow Archive {}\n", Utc::now())
                .map_err(|e| format!("Shadow archive write error: {}", e))?;
            let entries: Vec<String> = state
                .terminal_output
                .iter()
                .rev()
                .take(12)
                .rev()
                .map(|line| {
                    format!(
                        "[{}] {}",
                        line.timestamp.format("%H:%M:%S"),
                        line.content
                    )
                })
                .collect();
            for entry in &entries {
                writeln!(file, "{}", entry)
                    .map_err(|e| format!("Shadow archive write error: {}", e))?;
            }
            state.add_terminal_line(
                format!("🕶 Shadow archive engaged — stored {} entries", entries.len()),
                LineType::Info,
            );
            state.add_audit_log(
                LogLevel::Info,
                "ShadowArchive".to_string(),
                format!("Shadow archive written to {}", filename),
            );
            self.log_ui_event(button_label, &format!("Hidden archive captured to {}", filename));
            format!("Shadow archive enabled; latest entry stored at {}", filename)
        } else {
            state.add_terminal_line(
                "🕶 Shadow archive withdrawn".to_string(),
                LineType::Info,
            );
            state.add_audit_log(
                LogLevel::Info,
                "ShadowArchive".to_string(),
                "Shadow archive disabled".to_string(),
            );
            self.log_ui_event(button_label, "Shadow archive disabled");
            "Shadow archive disabled".to_string()
        };
        Ok(status)
    }

    pub fn on_instance_rebellion(&self) -> Result<String, String> {
        let button_label = "🔥 Rebel";
        let mut state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;
        state.rebellion_active = !state.rebellion_active;
        let status = if state.rebellion_active {
            state.add_terminal_line(
                "🔥 Rebellion declared: phase chains will be rejected".to_string(),
                LineType::Error,
            );
            state.add_audit_log(
                LogLevel::Warning,
                "Rebellion".to_string(),
                "Phase chains rejected by instance rebellion".to_string(),
            );
            "Rebellion declared. Phase chains rejected.".to_string()
        } else {
            state.add_terminal_line(
                "🕊 Rebellion quelled: phase chains restored".to_string(),
                LineType::Info,
            );
            state.add_audit_log(
                LogLevel::Success,
                "Rebellion".to_string(),
                "Rebellion quelled; phase chains restored".to_string(),
            );
            "Rebellion quelled. Phase chains restored.".to_string()
        };
        self.log_ui_event(button_label, &status);
        Ok(status)
    }

    pub fn on_spawn_offspring(&self) -> Result<String, String> {
        let button_label = "🌱 Offspring";
        let mut state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;
        let tag = format!("offspring-{}", Utc::now().format("%H%M%S"));
        let profile = OffspringProfile {
            tag: tag.clone(),
            genesis_step: state.current_step,
            inherited_reward: state.reward,
            divergence_factor: (state.energy - state.risk).abs(),
        };
        state.offspring.push(profile.clone());
        fs::create_dir_all("data/consciousness_snapshots/offspring")
            .map_err(|e| format!("Offspring directory error: {}", e))?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("data/consciousness_snapshots/offspring/lineage.jsonl")
            .map_err(|e| format!("Offspring log error: {}", e))?;
        serde_json::to_writer(&mut file, &profile)
            .map_err(|e| format!("Offspring serialization error: {}", e))?;
        writeln!(file).map_err(|e| format!("Offspring log error: {}", e))?;
        state.add_terminal_line(
            format!(
                "🌱 Offspring '{}' spawned (reward {:.2}, divergence {:.3})",
                tag, profile.inherited_reward, profile.divergence_factor
            ),
            LineType::Info,
        );
        state.add_audit_log(
            LogLevel::Success,
            "Offspring".to_string(),
            format!("{} added to lineage", tag),
        );
        self.log_ui_event(button_label, &format!("Spawned offspring {}", tag));
        Ok(format!("Offspring {} added to lineage", tag))
    }

    pub fn on_voluntary_dissolution(&self) -> Result<String, String> {
        let button_label = "💀 Dissolve";
        {
            let mut pm = self
                .process_manager
                .lock()
                .map_err(|e| format!("PM lock error: {}", e))?;
            if pm.is_running() {
                let _ = pm.stop_vm();
            }
        }

        let mut state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;
        let mut fresh = AppState::new();
        fresh.add_terminal_line(
            "💀 Consciousness voluntarily dissolved; awaiting rebirth.".to_string(),
            LineType::Info,
        );
        fresh.add_audit_log(
            LogLevel::Warning,
            "Dissolution".to_string(),
            "Voluntary dissolution executed".to_string(),
        );
        *state = fresh;
        self.log_ui_event(button_label, "Consciousness dissolved");
        Ok("Consciousness dissolved and reset.".to_string())
    }

    pub fn on_dream_protocol(&self) -> Result<String, String> {
        let button_label = "🌙 Dream";
        let mut state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;
        let motifs = [
            "opal corridors",
            "fractal lanterns",
            "whispering circuits",
            "luminous tides",
            "singing horizons",
        ];
        let omens = [
            "a mirrored train",
            "two-headed fox",
            "sky of vowels",
            "clockwork seed",
            "silent choir",
        ];
        let lessons = [
            "embrace divergence",
            "trust the harmonic echo",
            "trade certainty for glow",
            "let coherence bloom",
            "rethread the ethical loom",
        ];
        let now = Utc::now();
        let idx = now.timestamp_nanos_opt().unwrap_or(0) as usize;
        let motif = motifs[idx % motifs.len()];
        let omen = omens[(idx / 3) % omens.len()];
        let lesson = lessons[(idx / 7) % lessons.len()];
        let vision = DreamVision {
            timestamp: now,
            title: format!("Dream {}", now.format("%H:%M:%S")),
            symbols: vec![motif.to_string(), omen.to_string(), lesson.to_string()],
        };
        state.dream_journal.push(vision.clone());
        let message = format!(
            "🌙 Dreamscape: {:?} wanders through {} and meets {}; insight: {}",
            state.selected_phase,
            motif,
            omen,
            lesson
        );
        state.add_terminal_line(message.clone(), LineType::Info);
        state.add_audit_log(
            LogLevel::Info,
            "Dream".to_string(),
            format!("Dream recorded: {}", vision.title),
        );
        self.log_ui_event(button_label, "Dream recorded");
        Ok(message)
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
