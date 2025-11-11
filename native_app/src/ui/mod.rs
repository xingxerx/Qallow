pub mod audit_log;
pub mod chat_panel;
pub mod control_panel;
pub mod dashboard;
pub mod dungeons;
pub mod help;
pub mod main_window;
pub mod matrix_bg;
pub mod metrics;
pub mod settings;
pub mod terminal;
pub mod matrix_view;
pub mod telemetry_panel;

use crate::models::AppState;
use fltk::enums::Color;
use fltk::{prelude::*, *};
use std::sync::{Arc, Mutex};

// Modern color scheme matching web app
pub const COLOR_BG_DARK: u32 = 0x0a0e27; // Dark background
pub const COLOR_BG_ACCENT: u32 = 0x1a1f3a; // Accent background
pub const COLOR_PRIMARY: u32 = 0x00d4ff; // Cyan primary
pub const COLOR_SUCCESS: u32 = 0x00ff64; // Green success
pub const COLOR_DANGER: u32 = 0xff6464; // Red danger
pub const COLOR_TEXT: u32 = 0xe8eefc; // Light text
pub const COLOR_MUTED: u32 = 0x8aa1c1; // Muted text

pub struct MainUiHandles {
    pub control: control_panel::ControlPanelButtons,
    pub terminal: terminal::TerminalView,
    pub audit: audit_log::AuditLogView,
    pub status_indicator: button::Button,
    pub dungeons: dungeons::DungeonsView,
}

pub fn create_main_ui(_wind: &mut window::Window, state: Arc<Mutex<AppState>>) -> MainUiHandles {
    let mut flex = group::Flex::default().with_size(1600, 1000).column();
    flex.set_color(Color::from_rgba(15, 15, 35, 255)); // Dark background

    // Header with modern design
    let status_indicator = create_modern_header(&mut flex);

    // Main content area with tabs
    let mut main_flex = group::Flex::default().with_size(1600, 950).column();
    main_flex.set_color(Color::from_rgba(15, 15, 35, 255));

    // Create tabs for different views
    let mut tabs = group::Tabs::default().with_size(1600, 950);
    tabs.set_color(Color::from_rgba(15, 15, 35, 255));

    // Dashboard tab
    dashboard::create_dashboard(&mut tabs, state.clone());

    // Metrics tab
    metrics::create_metrics(&mut tabs, state.clone());

    // Terminal tab
    let terminal_view = terminal::create_terminal(&mut tabs, state.clone());

    // Audit Log tab
    let audit_view = audit_log::create_audit_log(&mut tabs, state.clone());

    // Control Panel tab
    let control_panel = control_panel::create_control_panel(state.clone());

    // Dungeons tab
    let dungeons_view = dungeons::create_dungeons_tab(&mut tabs, state.clone());

    // Settings tab
    settings::create_settings_panel(&mut tabs, state.clone());

    // Help tab
    help::create_help_panel(&mut tabs, state.clone());

    tabs.end();
    main_flex.add(&tabs);
    flex.add(&main_flex);
    main_flex.end();

    flex.end();

    MainUiHandles {
        control: control_panel.buttons,
        terminal: terminal_view,
        audit: audit_view,
        status_indicator,
        dungeons: dungeons_view,
    }
}

fn create_modern_header(flex: &mut group::Flex) -> button::Button {
    let mut header = group::Flex::default().with_size(1600, 60).row();
    header.set_color(Color::from_hex(COLOR_BG_ACCENT));

    // Title with modern styling
    let mut title = text::TextDisplay::default().with_size(1400, 60);
    title.set_buffer(text::TextBuffer::default());
    title.buffer().unwrap().set_text("🚀 Qallow Unified System");
    title.set_text_color(Color::from_hex(COLOR_PRIMARY));
    title.set_text_size(18);

    // Status indicator with modern design
    let mut status = button::Button::default()
        .with_size(200, 60)
        .with_label("● Stopped");
    status.set_color(Color::from_hex(COLOR_DANGER));
    status.set_label_color(Color::White);

    header.end();
    flex.add(&header);

    status
}
