use fltk::{prelude::*, *};
use fltk::enums::Color;
use std::sync::{Arc, Mutex};
use crate::models::AppState;

pub fn create_terminal(tabs: &mut group::Tabs, _state: Arc<Mutex<AppState>>) {
    let mut group = group::Group::default()
        .with_label("💻 Terminal");
    group.set_color(Color::from_hex(0x0a0e27));

    let mut flex = group::Flex::default()
        .with_size(1450, 950)
        .column();
    flex.set_color(Color::from_hex(0x0a0e27));

    // Title
    let mut title = text::TextDisplay::default()
        .with_size(1450, 40);
    title.set_buffer(text::TextBuffer::default());
    title.buffer().unwrap().set_text("Live Terminal Output");
    title.set_text_color(Color::from_hex(0x00d4ff));

    // Terminal output area
    let mut terminal_output = text::TextEditor::default()
        .with_size(1450, 850);
    terminal_output.set_buffer(text::TextBuffer::default());
    terminal_output.set_color(Color::from_hex(0x0a0e27));
    terminal_output.set_text_color(Color::from_hex(0x00ff64));
    terminal_output.set_cursor_color(Color::from_hex(0x00d4ff));

    // Add sample output
    let sample_output = r#"[2025-10-23 17:03:00] Starting Qallow Unified VM...
[2025-10-23 17:03:01] Initializing quantum circuits...
[2025-10-23 17:03:02] Loading Phase 14: Coherence-Lattice Integration
[2025-10-23 17:03:03] Configuring 256 nodes per overlay
[2025-10-23 17:03:04] Setting up 4 overlay types (Orbital, River, Mycelial, Global)
[2025-10-23 17:03:05] Initializing CUDA acceleration (NVIDIA RTX 5080)
[2025-10-23 17:03:06] Starting execution loop...
[2025-10-23 17:03:07] Tick 1: Coherence = 0.7500, Fidelity = 0.8200
[2025-10-23 17:03:08] Tick 2: Coherence = 0.8100, Fidelity = 0.8450
[2025-10-23 17:03:09] Tick 3: Coherence = 0.8650, Fidelity = 0.8680
[2025-10-23 17:03:10] Tick 4: Coherence = 0.9050, Fidelity = 0.8850
[2025-10-23 17:03:11] Tick 5: Coherence = 0.9350, Fidelity = 0.8980
[2025-10-23 17:03:12] Tick 6: Coherence = 0.9550, Fidelity = 0.9080
[2025-10-23 17:03:13] Tick 7: Coherence = 0.9700, Fidelity = 0.9150
[2025-10-23 17:03:14] Tick 8: Coherence = 0.9800, Fidelity = 0.9200
[2025-10-23 17:03:15] Tick 9: Coherence = 0.9850, Fidelity = 0.9250
[2025-10-23 17:03:16] Tick 10: Coherence = 0.9900, Fidelity = 0.9300
[2025-10-23 17:03:17] ✓ Phase 14 converged at tick 201
[2025-10-23 17:03:18] Final Fidelity: 0.981000 (TARGET REACHED)
[2025-10-23 17:03:19] Ethics Score: 2.39 (PASS ✓)
[2025-10-23 17:03:20] Execution complete. Total time: 20.5 seconds"#;

    terminal_output.buffer().unwrap().set_text(sample_output);

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

    let mut copy_btn = button::Button::default()
        .with_size(100, 50)
        .with_label("Copy");
    copy_btn.set_color(Color::from_hex(0x1a1f3a));
    copy_btn.set_label_color(Color::from_hex(0x00d4ff));

    let mut export_btn = button::Button::default()
        .with_size(100, 50)
        .with_label("Export");
    export_btn.set_color(Color::from_hex(0x1a1f3a));
    export_btn.set_label_color(Color::from_hex(0x00d4ff));

    button_flex.end();

    flex.end();
    group.end();
    tabs.add(&group);
}

