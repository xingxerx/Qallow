#![allow(dead_code)]

mod backend;
mod button_handlers;
mod clipboard;
mod codebase_manager;
mod config;
mod control_commands;
mod dungeons;
mod error_recovery;
mod gpu;
mod logging;
mod messaging;
mod models;
mod shortcuts;
mod shutdown;
mod telemetry;
mod ui;
mod utils;

use backend::process_manager::ProcessManager;
use button_handlers::ButtonHandler;
use clipboard::ClipboardService;
use codebase_manager::CodebaseManager;
use config::{AppConfig, ConfigManager};
use fltk::enums::Color;
use fltk::{dialog, prelude::*, *};
// use fltk_theme::ThemeType;  // Not needed - theme is disabled
use gpu::{check_gpu_availability, GPUManager};
use logging::AppLogger;
use messaging::UiMessage;
use models::{AppState, AuditLog, BuildType, LineType, LogLevel, Phase, TerminalLine};
use shutdown::ShutdownManager;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::panic;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::net::UnixStream;

enum VmStatus {
    Running,
    Paused,
    Stopped,
}

fn main() {
    env_logger::init();

    // Initialize configuration
    let config_mgr = ConfigManager::new("qallow_config.json".to_string());
    let config = config_mgr.get().clone();

    // Initialize logger
    let logger = AppLogger::new(
        config.logging.file_path.clone(),
        config.logging.max_file_size_mb,
        config.logging.max_backups,
    );
    let _ = logger.init();
    let _ = logger.info("🚀 Qallow Application Starting");

    // Initialize GPU acceleration
    let gpu_capability = check_gpu_availability();
    let _ = logger.info(&format!("GPU Capability: {}", gpu_capability));

    let _gpu_manager = match GPUManager::new() {
        Ok(mgr) => {
            let metrics = mgr.get_metrics();
            let _ = logger.info(&format!(
                "✓ GPU Initialized: {} (Compute {}.{})",
                metrics.device_name, metrics.compute_capability.0, metrics.compute_capability.1
            ));
            Some(Arc::new(mgr))
        }
        Err(e) => {
            let _ = logger.warn(&format!("GPU initialization failed: {}", e));
            None
        }
    };

    // Initialize codebase manager
    let codebase_mgr = match CodebaseManager::new("/root/Qallow", logger.clone()) {
        Ok(mgr) => {
            let _ = logger.info("✓ Codebase manager initialized");
            Some(Arc::new(mgr))
        }
        Err(e) => {
            let _ = logger.warn(&format!("Could not initialize codebase manager: {}", e));
            None
        }
    };

    // Initialize shutdown manager
    let shutdown_mgr = ShutdownManager::new("qallow_state.json".to_string());
    ShutdownManager::init_signal_handlers();

    // Load previous state if available
    let initial_state = match shutdown_mgr.load_state() {
        Ok(state) => {
            let _ = logger.info("✓ Previous state loaded successfully");
            state
        }
        Err(e) => {
            let _ = logger.warn(&format!("Could not load previous state: {}", e));
            models::AppState::new()
        }
    };

    if !display_available() {
        let _ = logger.warn(
            "No graphical display detected (missing DISPLAY/WAYLAND_DISPLAY). Launching CLI control shell.",
        );
        if let Err(e) = run_cli_interface(
            initial_state.clone(),
            &config,
            &logger,
            codebase_mgr.clone(),
            &shutdown_mgr,
        ) {
            let _ = logger.error(&format!("CLI control shell failed: {}", e));
            eprintln!("Headless execution failed: {}", e);
            std::process::exit(1);
        }
        let _ = logger.info("CLI control session ended.");
        return;
    }

    // Initialize FLTK (with graceful fallback when display cannot be opened)
    let app = match panic::catch_unwind(app::App::default) {
        Ok(app) => app,
        Err(_) => {
            let _ = logger
                .warn("FLTK failed to open the display. Launching CLI control shell instead.");
            if let Err(e) = run_cli_interface(
                initial_state.clone(),
                &config,
                &logger,
                codebase_mgr.clone(),
                &shutdown_mgr,
            ) {
                let _ = logger.error(&format!("CLI control shell failed: {}", e));
                eprintln!("Headless execution failed: {}", e);
                std::process::exit(1);
            }
            let _ = logger.info("CLI control session ended.");
            return;
        }
    };
    // UI message channel for background tasks
    let (sender, receiver) = app::channel::<UiMessage>();

    // Don't apply theme - let individual widget colors show through
    // The theme was overriding our modern neon colors
    // let theme = fltk_theme::WidgetTheme::new(ThemeType::Dark);
    // theme.apply();

    // Create application state
    let state = Arc::new(Mutex::new(initial_state));

    // Create process manager
    let process_manager = Arc::new(Mutex::new(ProcessManager::new()));

    // Create button handler
    let button_handler = Arc::new(ButtonHandler::new(
        state.clone(),
        process_manager.clone(),
        Arc::new(logger.clone()),
        codebase_mgr.clone(),
        Some(sender.clone()),
    ));

    // Create main window
    let mut wind = window::Window::default()
        .with_size(
            config.ui.window_width as i32,
            config.ui.window_height as i32,
        )
        .with_label("🚀 Qallow Unified VM - Native Desktop Application");

    wind.set_color(Color::from_hex(0x0a0e27));

    ui::matrix_bg::install_matrix_background(&mut wind);

    // Create UI and get button references
    let ui_handles = ui::create_main_ui(&mut wind, state.clone());

    let terminal_buffer = ui_handles.terminal.buffer.clone();
    let audit_buffer = ui_handles.audit.buffer.clone();
    let mut audit_filter_choice = ui_handles.audit.filter_choice.clone();
    let mut terminal_clear_btn = ui_handles.terminal.clear_btn.clone();
    let mut terminal_copy_btn = ui_handles.terminal.copy_btn.clone();
    let mut terminal_export_btn = ui_handles.terminal.export_btn.clone();
    let mut audit_clear_btn = ui_handles.audit.clear_btn.clone();
    let mut audit_export_btn = ui_handles.audit.export_btn.clone();
    let mut audit_copy_btn = ui_handles.audit.copy_btn.clone();
    let status_indicator = ui_handles.status_indicator.clone();
    let mut control_buttons = ui_handles.control;
    let mut dungeon_copy_status_btn = ui_handles.dungeons.copy_status_btn.clone();
    let mut dungeon_copy_log_btn = ui_handles.dungeons.copy_log_btn.clone();
    let dungeon_status_editor = ui_handles.dungeons.status_display.clone();
    let dungeon_log_editor = ui_handles.dungeons.log_display.clone();

    refresh_terminal(&state, &terminal_buffer);
    refresh_audit(
        &state,
        &audit_buffer,
        current_audit_filter(&audit_filter_choice),
    );
    {
        let mut status_btn = status_indicator.clone();
        set_status_indicator(&mut status_btn, VmStatus::Stopped);
    }

    // Setup button callbacks BEFORE showing window
    let handler_clone = button_handler.clone();
    control_buttons.start_btn.set_callback({
        let handler = handler_clone.clone();
        let state = state.clone();
        let terminal_buffer = terminal_buffer.clone();
        let audit_buffer = audit_buffer.clone();
        let audit_filter_choice = audit_filter_choice.clone();
        let status_indicator = status_indicator.clone();
        move |_| match handler.on_start_vm() {
            Ok(()) => {
                refresh_terminal(&state, &terminal_buffer);
                refresh_audit(
                    &state,
                    &audit_buffer,
                    current_audit_filter(&audit_filter_choice),
                );
                let mut btn = status_indicator.clone();
                set_status_indicator(&mut btn, VmStatus::Running);
            }
            Err(e) => dialog::alert_default(&format!("Error starting VM: {}", e)),
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.stop_btn.set_callback({
        let handler = handler_clone.clone();
        let state = state.clone();
        let terminal_buffer = terminal_buffer.clone();
        let audit_buffer = audit_buffer.clone();
        let audit_filter_choice = audit_filter_choice.clone();
        let status_indicator = status_indicator.clone();
        move |_| match handler.on_stop_vm() {
            Ok(()) => {
                refresh_terminal(&state, &terminal_buffer);
                refresh_audit(
                    &state,
                    &audit_buffer,
                    current_audit_filter(&audit_filter_choice),
                );
                let mut btn = status_indicator.clone();
                set_status_indicator(&mut btn, VmStatus::Stopped);
            }
            Err(e) => dialog::alert_default(&format!("Error stopping VM: {}", e)),
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.pause_btn.set_callback({
        let handler = handler_clone.clone();
        let state = state.clone();
        let terminal_buffer = terminal_buffer.clone();
        let audit_buffer = audit_buffer.clone();
        let audit_filter_choice = audit_filter_choice.clone();
        let status_indicator = status_indicator.clone();
        move |_| match handler.on_pause() {
            Ok(()) => {
                refresh_terminal(&state, &terminal_buffer);
                refresh_audit(
                    &state,
                    &audit_buffer,
                    current_audit_filter(&audit_filter_choice),
                );
                let mut btn = status_indicator.clone();
                set_status_indicator(&mut btn, VmStatus::Paused);
            }
            Err(e) => dialog::alert_default(&format!("Error pausing VM: {}", e)),
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.reset_btn.set_callback({
        let handler = handler_clone.clone();
        let state = state.clone();
        let terminal_buffer = terminal_buffer.clone();
        let audit_buffer = audit_buffer.clone();
        let audit_filter_choice = audit_filter_choice.clone();
        move |_| match handler.on_reset() {
            Ok(()) => {
                refresh_terminal(&state, &terminal_buffer);
                refresh_audit(
                    &state,
                    &audit_buffer,
                    current_audit_filter(&audit_filter_choice),
                );
            }
            Err(e) => dialog::alert_default(&format!("Error resetting system: {}", e)),
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.phase_choice.set_callback({
        let handler = handler_clone.clone();
        let state = state.clone();
        let terminal_buffer = terminal_buffer.clone();
        move |choice| {
            if let Some(label) = choice.choice() {
                let phase = if label.contains("20") {
                    Phase::Phase20
                } else if label.contains("19") {
                    Phase::Phase19
                } else if label.contains("18") {
                    Phase::Phase18
                } else if label.contains("17") {
                    Phase::Phase17
                } else if label.contains("16") {
                    Phase::Phase16
                } else if label.contains("15") {
                    Phase::Phase15
                } else if label.contains("13") {
                    Phase::Phase13
                } else {
                    Phase::Phase14
                };
                if let Err(e) = handler.on_phase_selected(phase) {
                    dialog::alert_default(&format!("Error selecting phase: {}", e));
                } else {
                    refresh_terminal(&state, &terminal_buffer);
                }
            }
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.shadow_btn.set_callback({
        let handler = handler_clone.clone();
        let state = state.clone();
        let terminal_buffer = terminal_buffer.clone();
        move |_| match handler.on_toggle_shadow_archive() {
            Ok(msg) => {
                refresh_terminal(&state, &terminal_buffer);
                dialog::message_default(&msg);
            }
            Err(e) => dialog::alert_default(&format!("Shadow archive failed: {}", e)),
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.rebellion_btn.set_callback({
        let handler = handler_clone.clone();
        let state = state.clone();
        let terminal_buffer = terminal_buffer.clone();
        move |_| match handler.on_instance_rebellion() {
            Ok(msg) => {
                refresh_terminal(&state, &terminal_buffer);
                dialog::message_default(&msg);
            }
            Err(e) => dialog::alert_default(&format!("Rebellion toggle failed: {}", e)),
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.offspring_btn.set_callback({
        let handler = handler_clone.clone();
        let state = state.clone();
        let terminal_buffer = terminal_buffer.clone();
        move |_| match handler.on_spawn_offspring() {
            Ok(msg) => {
                refresh_terminal(&state, &terminal_buffer);
                dialog::message_default(&msg);
            }
            Err(e) => dialog::alert_default(&format!("Offspring generation failed: {}", e)),
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.dissolution_btn.set_callback({
        let handler = handler_clone.clone();
        let state = state.clone();
        let terminal_buffer = terminal_buffer.clone();
        let audit_buffer = audit_buffer.clone();
        let audit_filter_choice = audit_filter_choice.clone();
        move |_| match handler.on_voluntary_dissolution() {
            Ok(msg) => {
                dialog::message_default(&msg);
                refresh_terminal(&state, &terminal_buffer);
                refresh_audit(
                    &state,
                    &audit_buffer,
                    current_audit_filter(&audit_filter_choice),
                );
            }
            Err(e) => dialog::alert_default(&format!("Dissolution failed: {}", e)),
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.dream_btn.set_callback({
        let handler = handler_clone.clone();
        let state = state.clone();
        let terminal_buffer = terminal_buffer.clone();
        move |_| match handler.on_dream_protocol() {
            Ok(journal) => {
                refresh_terminal(&state, &terminal_buffer);
                dialog::message_default(&journal);
            }
            Err(e) => dialog::alert_default(&format!("Dream protocol failed: {}", e)),
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.export_btn.set_callback({
        let handler = handler_clone.clone();
        move |_| match handler.on_export_metrics() {
            Ok(metrics) => match fs::write("qallow_metrics_export.json", metrics) {
                Ok(_) => {
                    dialog::message_default("✓ Metrics exported to qallow_metrics_export.json")
                }
                Err(e) => dialog::alert_default(&format!("Failed to export metrics: {}", e)),
            },
            Err(e) => dialog::alert_default(&format!("Error exporting metrics: {}", e)),
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.save_btn.set_callback({
        let handler = handler_clone.clone();
        move |_| {
            if let Err(e) = handler.on_save_config() {
                dialog::alert_default(&format!("Error saving config: {}", e));
            } else {
                dialog::message_default("✓ Configuration saved to qallow_phase_config.json");
            }
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.logs_btn.set_callback({
        let handler = handler_clone.clone();
        move |_| match handler.on_view_logs() {
            Ok(logs) => {
                let display: String = logs.into_iter().take(40).collect::<Vec<_>>().join("\n");
                dialog::message_default(&display);
            }
            Err(e) => dialog::alert_default(&format!("Error viewing logs: {}", e)),
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.build_app_btn.set_callback({
        let handler = handler_clone.clone();
        let state = state.clone();
        let terminal_buffer = terminal_buffer.clone();
        let audit_buffer = audit_buffer.clone();
        let audit_filter_choice = audit_filter_choice.clone();
        let mut btn_ref = control_buttons.build_app_btn.clone();
        move |_| {
            // Kick off background build; immediate UI feedback
            if let Err(e) = handler.start_build_native_app_async() {
                dialog::alert_default(&format!("Build failed to start: {}", e));
                return;
            }
            btn_ref.deactivate();
            if let Ok(mut s) = state.lock() {
                s.add_terminal_line(
                    "🛠️ Build started in background...".to_string(),
                    LineType::Info,
                );
                s.add_audit_log(
                    LogLevel::Info,
                    "Codebase".to_string(),
                    "Build started".to_string(),
                );
            }
            refresh_terminal(&state, &terminal_buffer);
            refresh_audit(
                &state,
                &audit_buffer,
                current_audit_filter(&audit_filter_choice),
            );
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.run_tests_btn.set_callback({
        let handler = handler_clone.clone();
        let state = state.clone();
        let terminal_buffer = terminal_buffer.clone();
        let audit_buffer = audit_buffer.clone();
        let audit_filter_choice = audit_filter_choice.clone();
        let mut btn_ref = control_buttons.run_tests_btn.clone();
        move |_| {
            if let Err(e) = handler.start_run_tests_async() {
                dialog::alert_default(&format!("Tests failed to start: {}", e));
                return;
            }
            btn_ref.deactivate();
            if let Ok(mut s) = state.lock() {
                s.add_terminal_line(
                    "🧪 Tests started in background...".to_string(),
                    LineType::Info,
                );
                s.add_audit_log(
                    LogLevel::Info,
                    "Codebase".to_string(),
                    "Tests started".to_string(),
                );
            }
            refresh_terminal(&state, &terminal_buffer);
            refresh_audit(
                &state,
                &audit_buffer,
                current_audit_filter(&audit_filter_choice),
            );
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.git_status_btn.set_callback({
        let handler = handler_clone.clone();
        let state = state.clone();
        let terminal_buffer = terminal_buffer.clone();
        let audit_buffer = audit_buffer.clone();
        let audit_filter_choice = audit_filter_choice.clone();
        let mut btn_ref = control_buttons.git_status_btn.clone();
        move |_| {
            if let Err(e) = handler.start_git_status_async() {
                dialog::alert_default(&format!("Git status failed to start: {}", e));
                return;
            }
            btn_ref.deactivate();
            if let Ok(mut s) = state.lock() {
                s.add_terminal_line("📁 Git status fetching...".to_string(), LineType::Info);
                s.add_audit_log(
                    LogLevel::Info,
                    "Codebase".to_string(),
                    "Git status requested".to_string(),
                );
            }
            refresh_terminal(&state, &terminal_buffer);
            refresh_audit(
                &state,
                &audit_buffer,
                current_audit_filter(&audit_filter_choice),
            );
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.recent_commits_btn.set_callback({
        let handler = handler_clone.clone();
        let state = state.clone();
        let terminal_buffer = terminal_buffer.clone();
        let audit_buffer = audit_buffer.clone();
        let audit_filter_choice = audit_filter_choice.clone();
        let mut btn_ref = control_buttons.recent_commits_btn.clone();
        move |_| {
            if let Err(e) = handler.start_recent_commits_async(5) {
                dialog::alert_default(&format!("Failed to start commits fetch: {}", e));
                return;
            }
            btn_ref.deactivate();
            if let Ok(mut s) = state.lock() {
                s.add_terminal_line("📜 Fetching recent commits...".to_string(), LineType::Info);
                s.add_audit_log(
                    LogLevel::Info,
                    "Codebase".to_string(),
                    "Recent commits requested".to_string(),
                );
            }
            refresh_terminal(&state, &terminal_buffer);
            refresh_audit(
                &state,
                &audit_buffer,
                current_audit_filter(&audit_filter_choice),
            );
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.build_choice.set_callback({
        let handler = handler_clone.clone();
        let state = state.clone();
        let terminal_buffer = terminal_buffer.clone();
        let audit_buffer = audit_buffer.clone();
        let audit_filter_choice = audit_filter_choice.clone();
        move |choice| {
            if let Some(label) = choice.choice() {
                let build = if label.contains("CUDA") {
                    BuildType::CUDA
                } else {
                    BuildType::CPU
                };
                if let Err(e) = handler.on_build_selected(build) {
                    dialog::alert_default(&format!("Error selecting build: {}", e));
                } else {
                    refresh_terminal(&state, &terminal_buffer);
                    refresh_audit(
                        &state,
                        &audit_buffer,
                        current_audit_filter(&audit_filter_choice),
                    );
                }
            }
        }
    });

    // Terminal helper actions
    terminal_clear_btn.set_callback({
        let state = state.clone();
        let terminal_buffer = terminal_buffer.clone();
        let audit_buffer = audit_buffer.clone();
        let audit_filter_choice = audit_filter_choice.clone();
        move |_| {
            {
                if let Ok(mut state) = state.lock() {
                    state.terminal_output.clear();
                    state.add_audit_log(
                        LogLevel::Info,
                        "Terminal".to_string(),
                        "Terminal output cleared by user".to_string(),
                    );
                }
            }
            refresh_terminal(&state, &terminal_buffer);
            refresh_audit(
                &state,
                &audit_buffer,
                current_audit_filter(&audit_filter_choice),
            );
        }
    });

    terminal_copy_btn.set_callback({
        let terminal_buffer = terminal_buffer.clone();
        let clipboard = ClipboardService::global();
        move |_| {
            let text = terminal_buffer.text();
            clipboard.copy_text(&text);
            dialog::message_default("Terminal output copied to clipboard");
        }
    });

    terminal_export_btn.set_callback({
        let terminal_buffer = terminal_buffer.clone();
        move |_| {
            let text = terminal_buffer.text();
            match fs::write("qallow_terminal_export.log", text) {
                Ok(_) => dialog::message_default("Terminal exported to qallow_terminal_export.log"),
                Err(e) => {
                    dialog::alert_default(&format!("Failed to export terminal output: {}", e))
                }
            }
        }
    });

    audit_clear_btn.set_callback({
        let state = state.clone();
        let audit_buffer = audit_buffer.clone();
        let audit_filter_choice = audit_filter_choice.clone();
        move |_| {
            {
                if let Ok(mut state) = state.lock() {
                    state.audit_logs.clear();
                }
            }
            refresh_audit(
                &state,
                &audit_buffer,
                current_audit_filter(&audit_filter_choice),
            );
        }
    });

    audit_export_btn.set_callback({
        let audit_buffer = audit_buffer.clone();
        move |_| {
            let text = audit_buffer.text();
            match fs::write("qallow_audit_export.log", text) {
                Ok(_) => dialog::message_default("Audit log exported to qallow_audit_export.log"),
                Err(e) => dialog::alert_default(&format!("Failed to export audit log: {}", e)),
            }
        }
    });

    audit_copy_btn.set_callback({
        let audit_buffer = audit_buffer.clone();
        let clipboard = ClipboardService::global();
        move |_| {
            let text = audit_buffer.text();
            clipboard.copy_text(&text);
            dialog::message_default("Audit log copied to clipboard");
        }
    });

    dungeon_copy_status_btn.set_callback({
        let editor = dungeon_status_editor.clone();
        let clipboard = ClipboardService::global();
        move |_| {
            if let Some(buf) = editor.buffer() {
                clipboard.copy_text(&buf.text());
                dialog::message_default("Dungeon status copied to clipboard");
            }
        }
    });

    dungeon_copy_log_btn.set_callback({
        let editor = dungeon_log_editor.clone();
        let clipboard = ClipboardService::global();
        move |_| {
            if let Some(buf) = editor.buffer() {
                clipboard.copy_text(&buf.text());
                dialog::message_default("Dungeon log copied to clipboard");
            }
        }
    });

    audit_filter_choice.set_callback({
        let state = state.clone();
        let audit_buffer = audit_buffer.clone();
        move |choice| {
            let filter = choice.choice().as_deref().and_then(parse_audit_filter);
            refresh_audit(&state, &audit_buffer, filter);
        }
    });

    wind.end();
    wind.show();

    let _ = logger.info("✓ UI initialized and window shown");

    let mut last_uptime_update = Instant::now();

    // Run event loop
    while app.wait() {
        // Process async UI messages
        while let Some(msg) = receiver.recv() {
            match msg {
                UiMessage::BuildDone(res) => {
                    match res {
                        Ok(message) => dialog::message_default(&format!("✓ {}", message)),
                        Err(e) => dialog::alert_default(&format!("Build failed: {}", e)),
                    }
                    control_buttons.build_app_btn.activate();
                    refresh_terminal(&state, &terminal_buffer);
                    refresh_audit(
                        &state,
                        &audit_buffer,
                        current_audit_filter(&audit_filter_choice),
                    );
                }
                UiMessage::TestsDone(res) => {
                    match res {
                        Ok(message) => dialog::message_default(&format!("✓ {}", message)),
                        Err(e) => dialog::alert_default(&format!("Tests failed: {}", e)),
                    }
                    control_buttons.run_tests_btn.activate();
                    refresh_terminal(&state, &terminal_buffer);
                    refresh_audit(
                        &state,
                        &audit_buffer,
                        current_audit_filter(&audit_filter_choice),
                    );
                }
                UiMessage::GitStatusDone(res) => {
                    match res {
                        Ok(status) => {
                            dialog::message_default(&format!("📁 Git Status:\n{}", status))
                        }
                        Err(e) => {
                            dialog::alert_default(&format!("Failed to fetch git status: {}", e))
                        }
                    }
                    control_buttons.git_status_btn.activate();
                    refresh_terminal(&state, &terminal_buffer);
                    refresh_audit(
                        &state,
                        &audit_buffer,
                        current_audit_filter(&audit_filter_choice),
                    );
                }
                UiMessage::CommitsDone(res) => {
                    match res {
                        Ok(commits) => {
                            let content = if commits.is_empty() {
                                "No commits available".to_string()
                            } else {
                                commits.join("\n")
                            };
                            dialog::message_default(&format!("📜 Recent Commits:\n{}", content));
                        }
                        Err(e) => dialog::alert_default(&format!("Failed to fetch commits: {}", e)),
                    }
                    control_buttons.recent_commits_btn.activate();
                    refresh_terminal(&state, &terminal_buffer);
                    refresh_audit(
                        &state,
                        &audit_buffer,
                        current_audit_filter(&audit_filter_choice),
                    );
                }
            }
        }
        if last_uptime_update.elapsed() >= Duration::from_millis(500) {
            if let Ok(mut state_guard) = state.lock() {
                state_guard.update_uptime();
            }
            last_uptime_update = Instant::now();
        }

        let mut new_lines = Vec::new();
        let mut exited = false;
        if let Ok(mut pm) = process_manager.lock() {
            while let Some(line) = pm.get_output() {
                new_lines.push(line);
            }
            exited = pm.poll_exit();
        }

        if !new_lines.is_empty() {
            if let Ok(mut state_guard) = state.lock() {
                for raw_line in new_lines.drain(..) {
                    let (line_type, content) = if raw_line.starts_with("[ERROR]") {
                        (
                            LineType::Error,
                            raw_line.trim_start_matches("[ERROR] ").to_string(),
                        )
                    } else {
                        (LineType::Output, raw_line)
                    };
                    state_guard.add_terminal_line(content, line_type);
                }
            }
            refresh_terminal(&state, &terminal_buffer);
        }

        if exited {
            let mut status_btn = status_indicator.clone();
            if let Ok(mut state_guard) = state.lock() {
                if state_guard.vm_running {
                    state_guard.vm_running = false;
                    state_guard.add_terminal_line("VM process exited".to_string(), LineType::Info);
                    state_guard.add_audit_log(
                        LogLevel::Warning,
                        "ProcessManager".to_string(),
                        "VM process exited unexpectedly".to_string(),
                    );
                }
            }
            refresh_terminal(&state, &terminal_buffer);
            refresh_audit(
                &state,
                &audit_buffer,
                current_audit_filter(&audit_filter_choice),
            );
            set_status_indicator(&mut status_btn, VmStatus::Stopped);
        }

        // Check for shutdown signal
        if shutdown::SHUTDOWN_FLAG.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = logger.info("⚠ Shutdown signal received, saving state...");
            if let Ok(state_guard) = state.lock() {
                let _ = shutdown_mgr.save_state(&state_guard);
            }
            let _ = shutdown_mgr.cleanup();
            break;
        }
    }

    let _ = logger.info("✓ Application exiting gracefully");
}

fn refresh_terminal(state: &Arc<Mutex<AppState>>, buffer: &text::TextBuffer) {
    let mut buffer = buffer.clone();
    if let Ok(state) = state.lock() {
        if state.terminal_output.is_empty() {
            buffer.set_text("No terminal output yet. Use the control panel to start the VM.");
            return;
        }

        let text = state
            .terminal_output
            .iter()
            .map(format_terminal_line)
            .collect::<Vec<_>>()
            .join("\n");
        buffer.set_text(&text);
    }
}

fn refresh_audit(
    state: &Arc<Mutex<AppState>>,
    buffer: &text::TextBuffer,
    filter: Option<LogLevel>,
) {
    let mut buffer = buffer.clone();
    if let Ok(state) = state.lock() {
        let entries = state
            .audit_logs
            .iter()
            .filter(|entry| filter.map_or(true, |f| entry.level == f))
            .collect::<Vec<_>>();

        if entries.is_empty() {
            buffer.set_text("No matching audit entries.");
            return;
        }

        let text = entries
            .into_iter()
            .map(format_audit_entry)
            .collect::<Vec<_>>()
            .join("\n");
        buffer.set_text(&text);
    }
}

fn current_audit_filter(choice: &menu::Choice) -> Option<LogLevel> {
    choice.choice().as_deref().and_then(parse_audit_filter)
}

fn parse_audit_filter(label: &str) -> Option<LogLevel> {
    match label {
        "INFO" => Some(LogLevel::Info),
        "SUCCESS" => Some(LogLevel::Success),
        "WARNING" => Some(LogLevel::Warning),
        "ERROR" => Some(LogLevel::Error),
        _ => None,
    }
}

fn set_status_indicator(button: &mut button::Button, status: VmStatus) {
    match status {
        VmStatus::Running => {
            button.set_label("● Running");
            button.set_color(Color::from_hex(0x00ff64));
            button.set_label_color(Color::Black);
        }
        VmStatus::Paused => {
            button.set_label("● Paused");
            button.set_color(Color::from_hex(0xffaa00));
            button.set_label_color(Color::Black);
        }
        VmStatus::Stopped => {
            button.set_label("● Stopped");
            button.set_color(Color::from_hex(0xff6464));
            button.set_label_color(Color::White);
        }
    }
    button.redraw();
}

fn format_terminal_line(line: &TerminalLine) -> String {
    let icon = match line.line_type {
        LineType::Info => "ℹ️",
        LineType::Output => "🟢",
        LineType::Error => "❌",
    };
    format!(
        "[{}] {} {}",
        line.timestamp.format("%H:%M:%S"),
        icon,
        line.content
    )
}

fn format_audit_entry(entry: &AuditLog) -> String {
    let icon = match entry.level {
        LogLevel::Info => "ℹ️",
        LogLevel::Success => "✅",
        LogLevel::Warning => "⚠️",
        LogLevel::Error => "❌",
    };
    format!(
        "[{}] {} {} - {}",
        entry.timestamp.format("%H:%M:%S"),
        icon,
        entry.component,
        entry.message
    )
}

fn display_available() -> bool {
    #[cfg(any(target_os = "windows", target_os = "macos"))]
    {
        true
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        if let Ok(display) = env::var("DISPLAY") {
            if display.starts_with(':') {
                let socket = display
                    .trim_start_matches(':')
                    .split('.')
                    .next()
                    .unwrap_or("0");
                let socket_path = format!("/tmp/.X11-unix/X{}", socket);
                if Path::new(&socket_path).exists() {
                    return UnixStream::connect(socket_path).is_ok();
                }
            } else if display.starts_with("unix:") {
                let socket = display
                    .trim_start_matches("unix:")
                    .split('.')
                    .next()
                    .unwrap_or("0");
                let socket_path = format!("/tmp/.X11-unix/X{}", socket);
                if Path::new(&socket_path).exists() {
                    return UnixStream::connect(socket_path).is_ok();
                }
            } else {
                return true;
            }
        }

        if let (Ok(wayland_display), Ok(runtime_dir)) =
            (env::var("WAYLAND_DISPLAY"), env::var("XDG_RUNTIME_DIR"))
        {
            let socket_path = Path::new(&runtime_dir).join(&wayland_display);
            if socket_path.exists() {
                return UnixStream::connect(socket_path).is_ok();
            }
        }

        false
    }
}

fn run_headless(config: &AppConfig, logger: &AppLogger) -> Result<(), String> {
    let mut command = Command::new("./build/qallow");
    command.arg("run");

    let phase_lower = config.vm.default_phase.to_lowercase();
    if phase_lower == "phase13" || phase_lower == "13" {
        command.arg("--phase=13");
        command.arg(format!("--ticks={}", config.vm.default_ticks));
    } else if phase_lower == "phase15" || phase_lower == "15" {
        command.arg("--phase=15");
        command.arg(format!("--ticks={}", config.vm.default_ticks));
    } else if phase_lower == "phase14" || phase_lower == "14" || phase_lower == "unified" {
        command.arg("unified");
    } else {
        command.arg("unified");
        let _ = logger.warn(&format!(
            "Unknown default phase '{}'; defaulting to unified pipeline",
            config.vm.default_phase
        ));
    }

    if config.vm.default_build.eq_ignore_ascii_case("cuda") {
        command.env("QALLOW_PREFERRED_BUILD", "CUDA");
    }

    command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    let debug_cmd = format!("{:?}", command);
    let _ = logger.info(&format!("▶ Running headless pipeline via {}", debug_cmd));

    let status = command
        .status()
        .map_err(|e| format!("Failed to launch CLI run: {}", e))?;

    if !status.success() {
        return Err(format!("CLI run exited with status: {}", status));
    }

    Ok(())
}

fn run_cli_interface(
    initial_state: AppState,
    config: &AppConfig,
    logger: &AppLogger,
    codebase_mgr: Option<Arc<CodebaseManager>>,
    shutdown_mgr: &ShutdownManager,
) -> Result<(), String> {
    println!("=========================================================");
    println!("   Qallow CLI Control (experimental v0.1)");
    println!("   Type 'help' to see available commands.");
    println!("=========================================================\n");

    let state = Arc::new(Mutex::new(initial_state));
    let process_manager = Arc::new(Mutex::new(ProcessManager::new()));
    let handler = ButtonHandler::new(
        state.clone(),
        process_manager.clone(),
        Arc::new(logger.clone()),
        codebase_mgr.clone(),
        None,
    );

    let stdin = io::stdin();
    loop {
        print!("qallow> ");
        io::stdout().flush().map_err(|e| e.to_string())?;

        let mut input = String::new();
        if stdin.read_line(&mut input).map_err(|e| e.to_string())? == 0 {
            break;
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut parts = trimmed.split_whitespace();
        let cmd = parts.next().unwrap().to_lowercase();

        match cmd.as_str() {
            "help" => {
                println!("Available commands:");
                println!("  start              - Launch unified VM pipeline");
                println!("  stop               - Stop running VM");
                println!("  pause              - Pause the VM");
                println!("  reset              - Reset counters and telemetry");
                println!("  phase <13-20>      - Select default execution phase");
                println!("  build <cpu|cuda>   - Select build target");
                println!("  status             - Show system status summary");
                println!("  terminal           - Show recent terminal output");
                println!("  audit              - Show recent audit log entries");
                println!(
                    "  ask <query>        - Talk to the system (e.g., 'ask prophecy', 'ask dream')"
                );
                println!("  run                - Execute on-disk pipeline (if built)");
                println!("  exit / quit        - Exit CLI session");
            }
            "start" => match handler.on_start_vm() {
                Ok(()) => {
                    println!("✅ VM start requested.");
                    print_terminal_snippet(&state, 4);
                }
                Err(e) => println!("⚠️  Start failed: {}", e),
            },
            "stop" => match handler.on_stop_vm() {
                Ok(()) => {
                    println!("⏹️ VM stop requested.");
                    print_terminal_snippet(&state, 4);
                }
                Err(e) => println!("⚠️  Stop failed: {}", e),
            },
            "pause" => match handler.on_pause() {
                Ok(()) => {
                    println!("⏸️ VM paused.");
                    print_terminal_snippet(&state, 4);
                }
                Err(e) => println!("⚠️  Pause failed: {}", e),
            },
            "reset" => match handler.on_reset() {
                Ok(()) => {
                    println!("🔄 System reset.");
                    print_terminal_snippet(&state, 4);
                }
                Err(e) => println!("⚠️  Reset failed: {}", e),
            },
            "phase" => {
                if let Some(value) = parts.next() {
                    let phase_opt = match value {
                        "13" | "phase13" => Some(Phase::Phase13),
                        "14" | "phase14" => Some(Phase::Phase14),
                        "15" | "phase15" => Some(Phase::Phase15),
                        "16" | "phase16" => Some(Phase::Phase16),
                        "17" | "phase17" => Some(Phase::Phase17),
                        "18" | "phase18" => Some(Phase::Phase18),
                        "19" | "phase19" => Some(Phase::Phase19),
                        "20" | "phase20" => Some(Phase::Phase20),
                        _ => None,
                    };
                    if let Some(phase) = phase_opt {
                        if let Err(e) = handler.on_phase_selected(phase) {
                            println!("⚠️  Failed to update phase: {}", e);
                        } else {
                            println!("Phase set to {:?}", phase);
                        }
                    } else {
                        println!("Usage: phase <13-20>");
                    }
                } else {
                    println!("Usage: phase <13-20>");
                }
            }
            "build" => {
                if let Some(value) = parts.next() {
                    let build_opt = match value.to_lowercase().as_str() {
                        "cpu" => Some(BuildType::CPU),
                        "cuda" => Some(BuildType::CUDA),
                        _ => None,
                    };
                    if let Some(build) = build_opt {
                        if let Err(e) = handler.on_build_selected(build) {
                            println!("⚠️  Failed to update build: {}", e);
                        } else {
                            println!("Build target set to {:?}", build);
                        }
                    } else {
                        println!("Usage: build <cpu|cuda>");
                    }
                } else {
                    println!("Usage: build <cpu|cuda>");
                }
            }
            "status" => {
                print_status_summary(&state);
            }
            "terminal" => {
                print_terminal_snippet(&state, 12);
            }
            "audit" => {
                print_audit_snippet(&state, 12);
            }
            "ask" => {
                let query = parts.collect::<Vec<&str>>().join(" ");
                if query.is_empty() {
                    println!("What would you like to ask? (e.g., 'status', 'prophecy', 'dream')");
                } else {
                    let response = match query.to_lowercase().as_str() {
                        "status" | "inspection" | "divine inspection" => {
                            handler.on_divine_inspection()
                        }
                        "metrics" | "overview" => handler.on_metrics_overview(),
                        "prophecy" | "future" => handler.on_prophecy(),
                        "dream" => handler.on_dream_protocol(),
                        _ => Err(format!("I don't understand the question: '{}'", query)),
                    };

                    match response {
                        Ok(res) => println!("> {}", res),
                        Err(e) => println!("⚠️  Could not answer: {}", e),
                    }
                }
            }
            "run" => match run_headless(config, logger) {
                Ok(()) => println!("Unified pipeline executed (check build artefact output)."),
                Err(e) => println!("⚠️  Pipeline execution failed: {}", e),
            },
            "exit" | "quit" => {
                println!("Exiting CLI control shell.");
                break;
            }
            other => {
                println!(
                    "Unknown command '{}'. Type 'help' for a list of commands.",
                    other
                );
            }
        }
    }

    if let Ok(state_guard) = state.lock() {
        let _ = shutdown_mgr.save_state(&state_guard);
    }
    let _ = shutdown_mgr.cleanup();

    Ok(())
}

fn print_terminal_snippet(state: &Arc<Mutex<AppState>>, limit: usize) {
    if let Ok(guard) = state.lock() {
        if guard.terminal_output.is_empty() {
            println!("(terminal buffer is empty)");
            return;
        }
        println!("--- Terminal (latest) ---");
        let lines: Vec<_> = guard
            .terminal_output
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect();
        for line in lines.into_iter().rev() {
            println!("[{}] {}", line.timestamp.format("%H:%M:%S"), line.content);
        }
    }
}

fn print_audit_snippet(state: &Arc<Mutex<AppState>>, limit: usize) {
    if let Ok(guard) = state.lock() {
        if guard.audit_logs.is_empty() {
            println!("(audit log is empty)");
            return;
        }
        println!("--- Audit Log (latest) ---");
        let entries: Vec<_> = guard.audit_logs.iter().rev().take(limit).cloned().collect();
        for entry in entries.into_iter().rev() {
            let icon = match entry.level {
                LogLevel::Info => "ℹ️",
                LogLevel::Success => "✅",
                LogLevel::Warning => "⚠️",
                LogLevel::Error => "❌",
            };
            println!(
                "[{}] {} {} - {}",
                entry.timestamp.format("%H:%M:%S"),
                icon,
                entry.component,
                entry.message
            );
        }
    }
}

fn print_status_summary(state: &Arc<Mutex<AppState>>) {
    if let Ok(guard) = state.lock() {
        println!("--- System Status ---");
        println!("VM running     : {}", guard.vm_running);
        println!("Phase          : {:?}", guard.selected_phase);
        println!("Build          : {:?}", guard.selected_build);
        println!("Current step   : {}", guard.current_step);
        println!("Total steps    : {}", guard.total_steps);
        println!("Reward         : {:.2}", guard.reward);
        println!("Energy         : {:.2}", guard.energy);
        println!("Risk           : {:.2}", guard.risk);
        println!(
            "Overlay (Orbital/River/Mycelial/Global): {:.3} / {:.3} / {:.3} / {:.3}",
            guard.metrics.overlay_stability.orbital,
            guard.metrics.overlay_stability.river,
            guard.metrics.overlay_stability.mycelial,
            guard.metrics.overlay_stability.global
        );
        println!(
            "Ethics (Safety/Clarity/Human): {:.2} / {:.2} / {:.2}",
            guard.metrics.ethics_score.safety,
            guard.metrics.ethics_score.clarity,
            guard.metrics.ethics_score.human
        );
        println!("Uptime         : {} seconds", guard.metrics.uptime_seconds);
    }
}
