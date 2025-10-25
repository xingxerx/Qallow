use fltk::{prelude::*, *};
use fltk::enums::Color;
use std::sync::{Arc, Mutex};
use crate::models::AppState;

pub fn create_audit_log(tabs: &mut group::Tabs, _state: Arc<Mutex<AppState>>) {
    let mut group = group::Group::default()
        .with_label("🔍 Audit Log");
    group.set_color(Color::from_hex(0x0a0e27));

    let mut flex = group::Flex::default()
        .with_size(1450, 950)
        .column();
    flex.set_color(Color::from_hex(0x0a0e27));

    // Title
    let mut title = text::TextDisplay::default()
        .with_size(1450, 40);
    title.set_buffer(text::TextBuffer::default());
    title.buffer().unwrap().set_text("Event Audit Log");
    title.set_text_color(Color::from_hex(0x00d4ff));

    // Filter controls
    let mut filter_flex = group::Flex::default()
        .with_size(1450, 50)
        .row();
    filter_flex.set_color(Color::from_hex(0x0a0e27));

    let mut filter_label = text::TextDisplay::default()
        .with_size(100, 50);
    filter_label.set_buffer(text::TextBuffer::default());
    filter_label.buffer().unwrap().set_text("Filter:");
    filter_label.set_text_color(Color::from_hex(0x00d4ff));

    let mut filter_choice = menu::Choice::default()
        .with_size(200, 50);
    filter_choice.add_choice("ALL|INFO|SUCCESS|WARNING|ERROR");
    filter_choice.set_color(Color::from_hex(0x1a1f3a));
    filter_choice.set_text_color(Color::from_hex(0x00d4ff));

    filter_flex.end();

    // Audit log display
    let mut log_display = text::TextEditor::default()
        .with_size(1450, 800);
    log_display.set_buffer(text::TextBuffer::default());
    log_display.set_color(Color::from_hex(0x0a0e27));
    log_display.set_text_color(Color::from_hex(0x00ff64));

    let sample_logs = r#"[17:03:00] ℹ️  SYSTEM    - Qallow VM initialized
[17:03:01] ℹ️  BACKEND   - CPU backend loaded
[17:03:02] ✓ CUDA      - CUDA acceleration enabled (RTX 5080)
[17:03:03] ℹ️  PHASE14   - Coherence-Lattice Integration started
[17:03:04] ℹ️  CIRCUIT   - 256 nodes per overlay configured
[17:03:05] ✓ ETHICS    - Ethics monitoring active (E=2.39)
[17:03:06] ℹ️  QUANTUM   - Quantum state initialized
[17:03:07] ℹ️  EXECUTION - Tick loop started
[17:03:10] ✓ COHERENCE - Coherence level: 0.9993 (excellent)
[17:03:15] ✓ FIDELITY  - Target fidelity reached: 0.9810
[17:03:20] ✓ PHASE14   - Phase 14 completed successfully
[17:03:21] ℹ️  PHASE15   - Convergence & Lock-In started
[17:03:22] ℹ️  STABILITY - Monitoring stability constraints
[17:03:25] ✓ CONVERGE  - Convergence achieved at tick 201
[17:03:26] ✓ LOCKDOWN  - System locked in stable state
[17:03:27] ✓ COMPLETE  - All phases executed successfully
[17:03:28] ℹ️  TELEMETRY - Metrics exported to data/logs/
[17:03:29] ✓ SHUTDOWN  - Qallow VM shutdown complete"#;

    log_display.buffer().unwrap().set_text(sample_logs);

    // Control buttons
    let mut button_flex = group::Flex::default()
        .with_size(1450, 50)
        .row();
    button_flex.set_color(Color::from_hex(0x0a0e27));

    let mut clear_btn = button::Button::default()
        .with_size(100, 50)
        .with_label("Clear");
    clear_btn.set_color(Color::from_hex(0x1a1f3a));
    clear_btn.set_label_color(Color::from_hex(0x00d4ff));

    let mut export_btn = button::Button::default()
        .with_size(100, 50)
        .with_label("Export");
    export_btn.set_color(Color::from_hex(0x1a1f3a));
    export_btn.set_label_color(Color::from_hex(0x00d4ff));

    let mut search_label = text::TextDisplay::default()
        .with_size(100, 50);
    search_label.set_buffer(text::TextBuffer::default());
    search_label.buffer().unwrap().set_text("Search:");
    search_label.set_text_color(Color::from_hex(0x00d4ff));

    let mut search_input = text::TextEditor::default()
        .with_size(300, 50);
    search_input.set_buffer(text::TextBuffer::default());
    search_input.set_color(Color::from_hex(0x1a1f3a));
    search_input.set_text_color(Color::White);

    button_flex.end();

    flex.end();
    group.end();
    tabs.add(&group);
}

