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
use codebase_manager::CodebaseManager;
use config::{AppConfig, ConfigManager};
use fltk::enums::Color;
use fltk::{app, prelude::*, window::Window};
use models::{AppState, BuildType, LineType, LogLevel, Phase};
use native_app::{
    backend::api_client::ApiClient,
    logging::AppLogger,
    messaging::UiMessage,
};

use shutdown::ShutdownManager;
use std::env;
use std::io::{self, Write};
use std::panic;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use tokio::spawn;

#[cfg(unix)]
use std::os::unix::net::UnixStream;

enum VmStatus {
    Running,
    Paused,
    Stopped,
}

fn main() {
    let rt = Runtime::new().unwrap();
    let _enter = rt.enter();
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
    let gpu_capability = gpu::check_gpu_availability();
    let _ = logger.info(&format!("GPU Capability: {}", gpu_capability));

    let _gpu_manager = match gpu::GPUManager::new() {
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
        logger.clone(),
        codebase_mgr.clone(),
        Some(sender.clone()),
    ));

    // --- Main Window and UI Setup ---
    let mut main_win = ui::main_window::MainWindow::new(button_handler.clone());
    main_win.wind.show();

    // --- Chat Button Logic ---
    main_win.chat_panel.send_button.set_callback({
        let mut chat_input = main_win.chat_panel.input.clone();
        let mut chat_view = main_win.chat_panel.display.clone();
        let api_client = main_win.button_handler.api_client.clone();
        let logger = logger.clone();

        move |_| {
            let message = chat_input.value();
            if message.is_empty() {
                return;
            }
            chat_input.set_value("");
            chat_view.buffer().unwrap().append(&format!("You: {}\n", message));

            let api_client = api_client.clone();
            let logger = logger.clone();
            spawn(async move {
                match api_client.chat(&message).await {
                    Ok(response) => {
                        // Make sure to update UI in the main thread
                        fltk::app::awake_callback(move || {
                            chat_view
                                .buffer()
                                .unwrap()
                                .append(&format!("Agent: {}\n", response));
                        });
                    }
                    Err(e) => {
                        let _ = logger.error(&format!("API Error: {}", e));
                        fltk::app::awake_callback(move || {
                            chat_view
                                .buffer()
                                .unwrap()
                                .append("Agent: Sorry, I encountered an error.\n");
                        });
                    }
                }
            });
        }
    });

    // --- Main Application Loop ---
    let mut last_uptime_update = Instant::now();
    while main_win.wind.wait() {
        // Process async UI messages
        if let Ok(msg) = receiver.try_recv() {
            match msg {
                UiMessage::BuildDone(res) => {
                    match res {
                        Ok(message) => dialog::message_default(&format!("✓ {}", message)),
                        Err(e) => dialog::alert_default(&format!("Build failed: {}", e)),
                    }
                    main_win.control_panel.build_app_btn.activate();
                }
                UiMessage::TestsDone(res) => {
                    match res {
                        Ok(message) => dialog::message_default(&format!("✓ {}", message)),
                        Err(e) => dialog::alert_default(&format!("Tests failed: {}", e)),
                    }
                    main_win.control_panel.run_tests_btn.activate();
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
                    main_win.control_panel.git_status_btn.activate();
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
                    main_win.control_panel.recent_commits_btn.activate();
                }
                UiMessage::UpdateVmStatus(status) => {
                    update_vm_status_indicator(&mut main_win.control_panel.vm_status_button, status);
                }
                UiMessage::UpdateAll => {
                    // This logic is now handled by the respective views
                }
            }
        }

        if last_uptime_update.elapsed() >= Duration::from_millis(500) {
            if let Ok(mut state_guard) = state.lock() {
                state_guard.update_uptime();
            }
            last_uptime_update = Instant::now();
        }

        // Update uptime display in the dashboard
        if let Ok(state_guard) = state.lock() {
            let uptime = state_guard.metrics.uptime_seconds;
            let uptime_str = format!("{} seconds", uptime);
            main_win.dashboard_panel.uptime_value.set_label(&uptime_str);
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

// This function is kept as a placeholder for future CLI implementation.
fn run_cli_interface(
    _state: AppState,
    _config: &AppConfig,
    _logger: &AppLogger,
) -> Result<(), String> {
    println!("CLI mode is not yet fully implemented.");
    Ok(())
}

/// Updates the color and label of the VM status button.
fn update_vm_status_indicator(button: &mut button::Button, status: VmStatus) {
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
            }
        }
        if env::var("WAYLAND_DISPLAY").is_ok() {
            return true;
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
