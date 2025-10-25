mod backend;
mod button_handlers;
mod codebase_manager;
mod config;
mod error_recovery;
mod logging;
mod models;
mod shortcuts;
mod shutdown;
mod ui;
mod utils;

use backend::process_manager::ProcessManager;
use button_handlers::ButtonHandler;
use config::ConfigManager;
use fltk::enums::Color;
use fltk::{dialog, prelude::*, *};
use fltk_theme::ThemeType;
use logging::AppLogger;
use models::{AppState, AuditLog, BuildType, LineType, LogLevel, TerminalLine};
use shutdown::ShutdownManager;
use std::fs;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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

    // Initialize codebase manager
    let _codebase_mgr = match codebase_manager::CodebaseManager::new("/root/Qallow", logger.clone())
    {
        Ok(mgr) => {
            let _ = logger.info("✓ Codebase manager initialized");
            Some(mgr)
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

    // Initialize FLTK
    let app = app::App::default();
    let theme = fltk_theme::WidgetTheme::new(ThemeType::Dark);
    theme.apply();

    // Create application state
    let state = Arc::new(Mutex::new(initial_state));

    // Create process manager
    let process_manager = Arc::new(Mutex::new(ProcessManager::new()));

    // Create button handler
    let button_handler = Arc::new(ButtonHandler::new(
        state.clone(),
        process_manager.clone(),
        Arc::new(logger.clone()),
    ));

    // Create main window
    let mut wind = window::Window::default()
        .with_size(
            config.ui.window_width as i32,
            config.ui.window_height as i32,
        )
        .with_label("🚀 Qallow Unified VM - Native Desktop Application");

    wind.set_color(Color::from_hex(0x0a0e27));

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
    let status_indicator = ui_handles.status_indicator.clone();
    let mut control_buttons = ui_handles.control;

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
        move |_| {
            let text = terminal_buffer.text();
            fltk::app::copy(&text);
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
