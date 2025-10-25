use crate::models::AppState;
use fltk::enums::Color;
use fltk::{prelude::*, *};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct AuditLogView {
    pub buffer: text::TextBuffer,
    pub filter_choice: menu::Choice,
    pub clear_btn: button::Button,
    pub export_btn: button::Button,
}

pub fn create_audit_log(tabs: &mut group::Tabs, state: Arc<Mutex<AppState>>) -> AuditLogView {
    let mut group = group::Group::default().with_label("🔍 Audit Log");
    group.set_color(Color::from_hex(0x0a0e27));

    let mut flex = group::Flex::default().with_size(1450, 950).column();
    flex.set_color(Color::from_hex(0x0a0e27));

    let mut title = text::TextDisplay::default().with_size(1450, 40);
    let mut title_buffer = text::TextBuffer::default();
    title.set_buffer(title_buffer.clone());
    title_buffer.set_text("Event Audit Log");
    title.set_text_color(Color::from_hex(0x00d4ff));

    let mut filter_flex = group::Flex::default().with_size(1450, 50).row();
    filter_flex.set_color(Color::from_hex(0x0a0e27));

    let mut filter_label = text::TextDisplay::default().with_size(100, 50);
    let mut filter_label_buffer = text::TextBuffer::default();
    filter_label.set_buffer(filter_label_buffer.clone());
    filter_label_buffer.set_text("Filter:");
    filter_label.set_text_color(Color::from_hex(0x00d4ff));

    let mut filter_choice = menu::Choice::default().with_size(200, 50);
    filter_choice.add_choice("ALL|INFO|SUCCESS|WARNING|ERROR");
    filter_choice.set_color(Color::from_hex(0x1a1f3a));
    filter_choice.set_text_color(Color::from_hex(0x00d4ff));
    filter_choice.set_value(0);

    filter_flex.end();

    let mut log_buffer = text::TextBuffer::default();
    let mut log_display = text::TextEditor::default().with_size(1450, 800);
    log_display.set_buffer(log_buffer.clone());
    log_display.set_color(Color::from_hex(0x0a0e27));
    log_display.set_text_color(Color::from_hex(0x00ff64));

    if let Ok(state) = state.lock() {
        if state.audit_logs.is_empty() {
            log_buffer.set_text(
                "No audit entries yet. Interact with the control panel to generate events.",
            );
        } else {
            let text = state
                .audit_logs
                .iter()
                .map(|log| {
                    format!(
                        "[{}] {} {} - {}",
                        log.timestamp.format("%H:%M:%S"),
                        match log.level {
                            crate::models::LogLevel::Info => "ℹ️",
                            crate::models::LogLevel::Success => "✓",
                            crate::models::LogLevel::Warning => "⚠️",
                            crate::models::LogLevel::Error => "❌",
                        },
                        log.component,
                        log.message
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            log_buffer.set_text(&text);
        }
    }

    let mut button_flex = group::Flex::default().with_size(1450, 50).row();
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

    let mut search_label = text::TextDisplay::default().with_size(100, 50);
    let mut search_label_buffer = text::TextBuffer::default();
    search_label.set_buffer(search_label_buffer.clone());
    search_label_buffer.set_text("Search:");
    search_label.set_text_color(Color::from_hex(0x00d4ff));

    let mut search_input = text::TextEditor::default().with_size(300, 50);
    let search_buffer = text::TextBuffer::default();
    search_input.set_buffer(search_buffer);
    search_input.set_color(Color::from_hex(0x1a1f3a));
    search_input.set_text_color(Color::White);

    button_flex.end();

    flex.end();
    group.end();
    tabs.add(&group);

    AuditLogView {
        buffer: log_buffer,
        filter_choice,
        clear_btn,
        export_btn,
    }
}
