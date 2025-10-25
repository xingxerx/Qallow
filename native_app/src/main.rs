mod ui;
mod backend;
mod models;
mod utils;
mod shutdown;
mod config;
mod logging;
mod error_recovery;
mod shortcuts;

use fltk::{prelude::*, *};
use fltk::enums::Color;
use fltk_theme::ThemeType;
use std::sync::{Arc, Mutex};
use shutdown::ShutdownManager;
use config::ConfigManager;
use logging::AppLogger;

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

    // Create main window
    let mut wind = window::Window::default()
        .with_size(config.ui.window_width as i32, config.ui.window_height as i32)
        .with_label("🚀 Qallow Unified VM - Native Desktop Application");

    wind.set_color(Color::from_hex(0x0a0e27));

    // Create UI
    ui::create_main_ui(&mut wind, state.clone());

    wind.end();
    wind.show();

    let _ = logger.info("✓ UI initialized and window shown");

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

