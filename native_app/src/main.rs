mod ui;
mod backend;
mod models;
mod utils;
mod shutdown;
mod config;
mod logging;
mod error_recovery;
mod shortcuts;
mod button_handlers;
mod codebase_manager;

use fltk::{prelude::*, *};
use fltk::enums::Color;
use fltk_theme::ThemeType;
use std::sync::{Arc, Mutex};
use shutdown::ShutdownManager;
use config::ConfigManager;
use logging::AppLogger;
use backend::process_manager::ProcessManager;
use button_handlers::ButtonHandler;
use models::{BuildType, Phase};

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
    let codebase_mgr = match codebase_manager::CodebaseManager::new("/root/Qallow", logger.clone()) {
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
        .with_size(config.ui.window_width as i32, config.ui.window_height as i32)
        .with_label("🚀 Qallow Unified VM - Native Desktop Application");

    wind.set_color(Color::from_hex(0x0a0e27));

    // Create UI and get button references
    let mut control_buttons = ui::create_main_ui(&mut wind, state.clone());

    wind.end();
    wind.show();

    let _ = logger.info("✓ UI initialized and window shown");

    // Setup button callbacks
    let handler_clone = button_handler.clone();
    control_buttons.start_btn.set_callback({
        let handler = handler_clone.clone();
        move |_| {
            if let Err(e) = handler.on_start_vm() {
                eprintln!("Error starting VM: {}", e);
            }
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.stop_btn.set_callback({
        let handler = handler_clone.clone();
        move |_| {
            if let Err(e) = handler.on_stop_vm() {
                eprintln!("Error stopping VM: {}", e);
            }
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.pause_btn.set_callback({
        let handler = handler_clone.clone();
        move |_| {
            if let Err(e) = handler.on_pause() {
                eprintln!("Error pausing VM: {}", e);
            }
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.reset_btn.set_callback({
        let handler = handler_clone.clone();
        move |_| {
            if let Err(e) = handler.on_reset() {
                eprintln!("Error resetting system: {}", e);
            }
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.export_btn.set_callback({
        let handler = handler_clone.clone();
        move |_| {
            match handler.on_export_metrics() {
                Ok(metrics) => {
                    let _ = std::fs::write("qallow_metrics_export.json", metrics);
                    println!("✓ Metrics exported to qallow_metrics_export.json");
                }
                Err(e) => eprintln!("Error exporting metrics: {}", e),
            }
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.save_btn.set_callback({
        let handler = handler_clone.clone();
        move |_| {
            if let Err(e) = handler.on_save_config() {
                eprintln!("Error saving config: {}", e);
            }
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.logs_btn.set_callback({
        let handler = handler_clone.clone();
        move |_| {
            match handler.on_view_logs() {
                Ok(logs) => {
                    for log in logs {
                        println!("{}", log);
                    }
                }
                Err(e) => eprintln!("Error viewing logs: {}", e),
            }
        }
    });

    let handler_clone = button_handler.clone();
    control_buttons.build_choice.set_callback({
        let handler = handler_clone.clone();
        move |choice| {
            if let Some(label) = choice.choice() {
                let build = if label.contains("CUDA") {
                    BuildType::CUDA
                } else {
                    BuildType::CPU
                };
                if let Err(e) = handler.on_build_selected(build) {
                    eprintln!("Error selecting build: {}", e);
                }
            }
        }
    });

    // Run event loop
    while app.wait() {
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

