use crate::models::AppState;
use fltk::enums::Color;
use fltk::{prelude::*, *};
use std::sync::{Arc, Mutex};

pub fn create_settings_panel(parent: &mut group::Tabs, _state: Arc<Mutex<AppState>>) {
    let settings_group = group::Group::default().with_label("⚙️ Settings");

    let mut flex = group::Flex::default().with_size(1450, 950).column();
    flex.set_color(Color::from_hex(0x0a0e27));

    // Title
    let mut title = text::TextDisplay::default().with_size(1450, 40);
    title.set_buffer(text::TextBuffer::default());
    title
        .buffer()
        .unwrap()
        .set_text("Application Settings & Preferences");
    title.set_text_color(Color::from_hex(0x00d4ff));

    // Settings sections
    create_app_settings(&mut flex);
    create_logging_settings(&mut flex);
    create_ui_settings(&mut flex);
    create_vm_settings(&mut flex);

    // Buttons
    let button_flex = group::Flex::default().with_size(1450, 50).row();

    let mut save_btn = button::Button::default()
        .with_size(200, 50)
        .with_label("💾 Save Settings");
    save_btn.set_color(Color::from_hex(0x00d4ff));
    save_btn.set_label_color(Color::Black);

    let mut reset_btn = button::Button::default()
        .with_size(200, 50)
        .with_label("🔄 Reset to Defaults");
    reset_btn.set_color(Color::from_hex(0xff6464));
    reset_btn.set_label_color(Color::White);

    let mut close_btn = button::Button::default()
        .with_size(200, 50)
        .with_label("✕ Close");
    close_btn.set_color(Color::from_hex(0x1a1f3a));
    close_btn.set_label_color(Color::from_hex(0x00d4ff));

    button_flex.end();
    flex.add(&button_flex);

    flex.end();
    settings_group.end();
    parent.add(&settings_group);
}

fn create_app_settings(flex: &mut group::Flex) {
    let mut section = group::Flex::default().with_size(1450, 150).column();
    section.set_color(Color::from_hex(0x1a1f3a));

    let mut title = text::TextDisplay::default().with_size(1450, 30);
    title.set_buffer(text::TextBuffer::default());
    title.buffer().unwrap().set_text("📱 Application Settings");
    title.set_text_color(Color::from_hex(0x00d4ff));

    // Auto-save interval
    let auto_save_flex = group::Flex::default().with_size(1450, 40).row();

    let mut auto_save_label = text::TextDisplay::default().with_size(400, 40);
    auto_save_label.set_buffer(text::TextBuffer::default());
    auto_save_label
        .buffer()
        .unwrap()
        .set_text("Auto-save interval (seconds):");
    auto_save_label.set_text_color(Color::White);

    let mut auto_save_input = text::TextEditor::default().with_size(100, 40);
    auto_save_input.set_buffer(text::TextBuffer::default());
    auto_save_input.buffer().unwrap().set_text("30");

    auto_save_flex.end();

    // Auto-recovery
    let auto_recovery_flex = group::Flex::default().with_size(1450, 40).row();

    let mut auto_recovery_label = text::TextDisplay::default().with_size(400, 40);
    auto_recovery_label.set_buffer(text::TextBuffer::default());
    auto_recovery_label
        .buffer()
        .unwrap()
        .set_text("Enable auto-recovery:");
    auto_recovery_label.set_text_color(Color::White);

    let mut auto_recovery_check = button::CheckButton::default()
        .with_size(50, 40)
        .with_label("");
    auto_recovery_check.set(true);

    auto_recovery_flex.end();

    section.end();
    flex.add(&section);
}

fn create_logging_settings(flex: &mut group::Flex) {
    let mut section = group::Flex::default().with_size(1450, 150).column();
    section.set_color(Color::from_hex(0x1a1f3a));

    let mut title = text::TextDisplay::default().with_size(1450, 30);
    title.set_buffer(text::TextBuffer::default());
    title.buffer().unwrap().set_text("📝 Logging Settings");
    title.set_text_color(Color::from_hex(0x00d4ff));

    // Log level
    let log_level_flex = group::Flex::default().with_size(1450, 40).row();

    let mut log_level_label = text::TextDisplay::default().with_size(400, 40);
    log_level_label.set_buffer(text::TextBuffer::default());
    log_level_label.buffer().unwrap().set_text("Log level:");
    log_level_label.set_text_color(Color::White);

    let mut log_level_choice = menu::Choice::default().with_size(150, 40);
    log_level_choice.add_choice("DEBUG|INFO|WARN|ERROR");

    log_level_flex.end();

    // Log file path
    let log_path_flex = group::Flex::default().with_size(1450, 40).row();

    let mut log_path_label = text::TextDisplay::default().with_size(400, 40);
    log_path_label.set_buffer(text::TextBuffer::default());
    log_path_label.buffer().unwrap().set_text("Log file path:");
    log_path_label.set_text_color(Color::White);

    let mut log_path_input = text::TextEditor::default().with_size(400, 40);
    log_path_input.set_buffer(text::TextBuffer::default());
    log_path_input.buffer().unwrap().set_text("qallow.log");

    log_path_flex.end();

    section.end();
    flex.add(&section);
}

fn create_ui_settings(flex: &mut group::Flex) {
    let mut section = group::Flex::default().with_size(1450, 150).column();
    section.set_color(Color::from_hex(0x1a1f3a));

    let mut title = text::TextDisplay::default().with_size(1450, 30);
    title.set_buffer(text::TextBuffer::default());
    title.buffer().unwrap().set_text("🎨 UI Settings");
    title.set_text_color(Color::from_hex(0x00d4ff));

    // Theme
    let theme_flex = group::Flex::default().with_size(1450, 40).row();

    let mut theme_label = text::TextDisplay::default().with_size(400, 40);
    theme_label.set_buffer(text::TextBuffer::default());
    theme_label.buffer().unwrap().set_text("Theme:");
    theme_label.set_text_color(Color::White);

    let mut theme_choice = menu::Choice::default().with_size(150, 40);
    theme_choice.add_choice("Dark|Light|Auto");

    theme_flex.end();

    // Auto-scroll terminal
    let auto_scroll_flex = group::Flex::default().with_size(1450, 40).row();

    let mut auto_scroll_label = text::TextDisplay::default().with_size(400, 40);
    auto_scroll_label.set_buffer(text::TextBuffer::default());
    auto_scroll_label
        .buffer()
        .unwrap()
        .set_text("Auto-scroll terminal:");
    auto_scroll_label.set_text_color(Color::White);

    let mut auto_scroll_check = button::CheckButton::default()
        .with_size(50, 40)
        .with_label("");
    auto_scroll_check.set(true);

    auto_scroll_flex.end();

    section.end();
    flex.add(&section);
}

fn create_vm_settings(flex: &mut group::Flex) {
    let mut section = group::Flex::default().with_size(1450, 150).column();
    section.set_color(Color::from_hex(0x1a1f3a));

    let mut title = text::TextDisplay::default().with_size(1450, 30);
    title.set_buffer(text::TextBuffer::default());
    title.buffer().unwrap().set_text("⚡ VM Settings");
    title.set_text_color(Color::from_hex(0x00d4ff));

    // Process timeout
    let timeout_flex = group::Flex::default().with_size(1450, 40).row();

    let mut timeout_label = text::TextDisplay::default().with_size(400, 40);
    timeout_label.set_buffer(text::TextBuffer::default());
    timeout_label
        .buffer()
        .unwrap()
        .set_text("Process timeout (seconds):");
    timeout_label.set_text_color(Color::White);

    let mut timeout_input = text::TextEditor::default().with_size(100, 40);
    timeout_input.set_buffer(text::TextBuffer::default());
    timeout_input.buffer().unwrap().set_text("300");

    timeout_flex.end();

    // Metrics collection
    let metrics_flex = group::Flex::default().with_size(1450, 40).row();

    let mut metrics_label = text::TextDisplay::default().with_size(400, 40);
    metrics_label.set_buffer(text::TextBuffer::default());
    metrics_label
        .buffer()
        .unwrap()
        .set_text("Enable metrics collection:");
    metrics_label.set_text_color(Color::White);

    let mut metrics_check = button::CheckButton::default()
        .with_size(50, 40)
        .with_label("");
    metrics_check.set(true);

    metrics_flex.end();

    section.end();
    flex.add(&section);
}
