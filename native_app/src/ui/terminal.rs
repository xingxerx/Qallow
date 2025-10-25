use crate::models::AppState;
use fltk::enums::Color;
use fltk::{prelude::*, *};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct TerminalView {
    pub buffer: text::TextBuffer,
    pub clear_btn: button::Button,
    pub copy_btn: button::Button,
    pub export_btn: button::Button,
}

pub fn create_terminal(tabs: &mut group::Tabs, state: Arc<Mutex<AppState>>) -> TerminalView {
    let mut group = group::Group::default().with_label("💻 Terminal");
    group.set_color(Color::from_hex(0x0a0e27));

    let mut flex = group::Flex::default().with_size(1450, 950).column();
    flex.set_color(Color::from_hex(0x0a0e27));

    let mut title = text::TextDisplay::default().with_size(1450, 40);
    let mut title_buffer = text::TextBuffer::default();
    title.set_buffer(title_buffer.clone());
    title_buffer.set_text("Live Terminal Output");
    title.set_text_color(Color::from_hex(0x00d4ff));

    let mut terminal_buffer = text::TextBuffer::default();

    let mut terminal_output = text::TextEditor::default().with_size(1450, 850);
    terminal_output.set_buffer(terminal_buffer.clone());
    terminal_output.set_color(Color::from_hex(0x0a0e27));
    terminal_output.set_text_color(Color::from_hex(0x00ff64));
    terminal_output.set_cursor_color(Color::from_hex(0x00d4ff));

    if let Ok(state) = state.lock() {
        if state.terminal_output.is_empty() {
            terminal_buffer.set_text("No terminal output yet. Start a VM to see logs.");
        } else {
            let initial_text = state
                .terminal_output
                .iter()
                .map(|line| format!("[{}] {}", line.timestamp.format("%H:%M:%S"), line.content))
                .collect::<Vec<_>>()
                .join("\n");
            terminal_buffer.set_text(&initial_text);
        }
    }

    let mut button_flex = group::Flex::default().with_size(1450, 50).row();
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

    TerminalView {
        buffer: terminal_buffer,
        clear_btn,
        copy_btn,
        export_btn,
    }
}
