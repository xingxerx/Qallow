pub mod dashboard;
pub mod terminal;
pub mod metrics;
pub mod audit_log;
pub mod control_panel;
pub mod settings;
pub mod help;

use fltk::{prelude::*, *};
use fltk::enums::Color;
use std::sync::{Arc, Mutex};
use crate::models::AppState;

pub fn create_main_ui(_wind: &mut window::Window, state: Arc<Mutex<AppState>>) {
    let mut flex = group::Flex::default()
        .with_size(1600, 1000)
        .column();

    // Header
    create_header(&mut flex);

    // Main content area with sidebar
    let mut main_flex = group::Flex::default()
        .with_size(1600, 950)
        .row();

    // Sidebar navigation
    let mut sidebar = group::Flex::default()
        .with_size(150, 950)
        .column();
    sidebar.set_color(Color::from_hex(0x1a1f3a));

    create_sidebar(&mut sidebar, state.clone());

    sidebar.end();

    // Content area
    let mut content = group::Flex::default()
        .with_size(1450, 950)
        .column();
    content.set_color(Color::from_hex(0x0a0e27));

    // Create tabs for different views
    let mut tabs = group::Tabs::default()
        .with_size(1450, 950);

    // Dashboard tab
    dashboard::create_dashboard(&mut tabs, state.clone());

    // Metrics tab
    metrics::create_metrics(&mut tabs, state.clone());

    // Terminal tab
    terminal::create_terminal(&mut tabs, state.clone());

    // Audit Log tab
    audit_log::create_audit_log(&mut tabs, state.clone());

    // Control Panel tab
    control_panel::create_control_panel(&mut tabs, state.clone());

    // Settings tab
    settings::create_settings_panel(&mut tabs, state.clone());

    // Help tab
    help::create_help_panel(&mut tabs, state.clone());

    tabs.end();
    content.end();
    main_flex.end();

    flex.end();
}

fn create_header(flex: &mut group::Flex) {
    let mut header = group::Flex::default()
        .with_size(1600, 50)
        .row();
    header.set_color(Color::from_hex(0x1a1f3a));

    let mut title = text::TextDisplay::default()
        .with_size(1400, 50);
    title.set_buffer(text::TextBuffer::default());
    title.buffer().unwrap().set_text("🚀 Qallow Unified VM - Quantum-Photonic AGI System");
    title.set_text_color(Color::from_hex(0x00d4ff));

    let mut status = button::Button::default()
        .with_size(200, 50)
        .with_label("● Stopped");
    status.set_color(Color::from_hex(0xff6464));
    status.set_label_color(Color::White);

    header.end();
    flex.add(&header);
}

fn create_sidebar(flex: &mut group::Flex, _state: Arc<Mutex<AppState>>) {
    let mut dashboard_btn = button::Button::default()
        .with_size(150, 40)
        .with_label("📊 Dashboard");
    dashboard_btn.set_color(Color::from_hex(0x00d4ff));
    dashboard_btn.set_label_color(Color::Black);

    let mut metrics_btn = button::Button::default()
        .with_size(150, 40)
        .with_label("📈 Metrics");
    metrics_btn.set_color(Color::from_hex(0x1a1f3a));
    metrics_btn.set_label_color(Color::from_hex(0x00d4ff));

    let mut terminal_btn = button::Button::default()
        .with_size(150, 40)
        .with_label("💻 Terminal");
    terminal_btn.set_color(Color::from_hex(0x1a1f3a));
    terminal_btn.set_label_color(Color::from_hex(0x00d4ff));

    let mut audit_btn = button::Button::default()
        .with_size(150, 40)
        .with_label("🔍 Audit Log");
    audit_btn.set_color(Color::from_hex(0x1a1f3a));
    audit_btn.set_label_color(Color::from_hex(0x00d4ff));

    let mut control_btn = button::Button::default()
        .with_size(150, 40)
        .with_label("⚙️ Control");
    control_btn.set_color(Color::from_hex(0x1a1f3a));
    control_btn.set_label_color(Color::from_hex(0x00d4ff));

    flex.add(&dashboard_btn);
    flex.add(&metrics_btn);
    flex.add(&terminal_btn);
    flex.add(&audit_btn);
    flex.add(&control_btn);
}

