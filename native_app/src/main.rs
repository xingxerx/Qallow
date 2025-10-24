mod ui;
mod backend;
mod models;
mod utils;

use fltk::{prelude::*, *};
use fltk_theme::{SchemeType, ThemeType};
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;

fn main() {
    env_logger::init();

    // Initialize FLTK
    let app = app::App::default();
    let theme = fltk_theme::WidgetTheme::new(ThemeType::Dark);
    theme.apply();

    // Create application state
    let state = Arc::new(Mutex::new(models::AppState::new()));

    // Create main window
    let mut wind = window::Window::default()
        .with_size(1600, 1000)
        .with_label("🚀 Qallow Unified VM - Native Desktop Application");

    wind.set_color(Color::from_hex(0x0a0e27));

    // Create UI
    ui::create_main_ui(&mut wind, state.clone());

    wind.end();
    wind.show();

    // Run event loop
    while app.wait() {
        // Handle any pending events
    }
}

