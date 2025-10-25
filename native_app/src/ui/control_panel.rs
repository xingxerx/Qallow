use crate::models::AppState;
use fltk::enums::Color;
use fltk::{prelude::*, *};
use std::sync::{Arc, Mutex};

pub struct ControlPanelButtons {
    pub start_btn: button::Button,
    pub stop_btn: button::Button,
    pub pause_btn: button::Button,
    pub reset_btn: button::Button,
    pub export_btn: button::Button,
    pub save_btn: button::Button,
    pub logs_btn: button::Button,
    pub build_choice: menu::Choice,
    pub build_app_btn: button::Button,
    pub run_tests_btn: button::Button,
    pub git_status_btn: button::Button,
    pub recent_commits_btn: button::Button,
}

pub fn create_control_panel(
    tabs: &mut group::Tabs,
    _state: Arc<Mutex<AppState>>,
) -> ControlPanelButtons {
    let mut group = group::Group::default().with_label("⚙️ Control");
    group.set_color(Color::from_hex(0x0a0e27));

    let mut flex = group::Flex::default().with_size(1450, 950).column();
    flex.set_color(Color::from_hex(0x0a0e27));

    // Title
    let mut title = text::TextDisplay::default().with_size(1450, 40);
    title.set_buffer(text::TextBuffer::default());
    title.buffer().unwrap().set_text("System Control Panel");
    title.set_text_color(Color::from_hex(0x00d4ff));

    // Control section
    let mut control_flex = group::Flex::default().with_size(1450, 150).row();
    control_flex.set_color(Color::from_hex(0x0a0e27));

    let mut start_btn = button::Button::default()
        .with_size(150, 150)
        .with_label("▶️ Start VM");
    start_btn.set_color(Color::from_hex(0x00ff64));
    start_btn.set_label_color(Color::Black);
    control_flex.add(&start_btn);

    let mut stop_btn = button::Button::default()
        .with_size(150, 150)
        .with_label("⏹️ Stop VM");
    stop_btn.set_color(Color::from_hex(0xff6464));
    stop_btn.set_label_color(Color::White);
    control_flex.add(&stop_btn);

    let mut pause_btn = button::Button::default()
        .with_size(150, 150)
        .with_label("⏸️ Pause");
    pause_btn.set_color(Color::from_hex(0xffaa00));
    pause_btn.set_label_color(Color::Black);
    control_flex.add(&pause_btn);

    let mut reset_btn = button::Button::default()
        .with_size(150, 150)
        .with_label("🔄 Reset");
    reset_btn.set_color(Color::from_hex(0x1a1f3a));
    reset_btn.set_label_color(Color::from_hex(0x00d4ff));
    control_flex.add(&reset_btn);

    control_flex.end();

    // Build selection
    let mut build_flex = group::Flex::default().with_size(1450, 100).row();
    build_flex.set_color(Color::from_hex(0x0a0e27));

    let mut build_label = text::TextDisplay::default().with_size(200, 100);
    build_label.set_buffer(text::TextBuffer::default());
    build_label.buffer().unwrap().set_text("Select Build:");
    build_label.set_text_color(Color::from_hex(0x00d4ff));
    build_flex.add(&build_label);

    let mut build_choice = menu::Choice::default().with_size(300, 100);
    build_choice.add_choice("CPU|CUDA");
    build_choice.set_color(Color::from_hex(0x1a1f3a));
    build_choice.set_text_color(Color::from_hex(0x00d4ff));
    build_flex.add(&build_choice);

    build_flex.end();

    // Phase configuration
    let mut phase_flex = group::Flex::default().with_size(1450, 150).column();
    phase_flex.set_color(Color::from_hex(0x0a0e27));

    let mut phase_title = text::TextDisplay::default().with_size(1450, 40);
    phase_title.set_buffer(text::TextBuffer::default());
    phase_title
        .buffer()
        .unwrap()
        .set_text("Phase Configuration");
    phase_title.set_text_color(Color::from_hex(0x00d4ff));

    let mut phase_config_flex = group::Flex::default().with_size(1450, 110).row();
    phase_config_flex.set_color(Color::from_hex(0x0a0e27));

    create_config_input(&mut phase_config_flex, "Phase:", "Phase 14");
    create_config_input(&mut phase_config_flex, "Ticks:", "1000");
    create_config_input(&mut phase_config_flex, "Fidelity:", "0.981");
    create_config_input(&mut phase_config_flex, "Epsilon:", "5e-6");

    phase_config_flex.end();
    phase_flex.end();

    // Quick actions
    let mut actions_flex = group::Flex::default().with_size(1450, 100).row();
    actions_flex.set_color(Color::from_hex(0x0a0e27));

    let mut export_btn = button::Button::default()
        .with_size(200, 100)
        .with_label("📈 Export Metrics");
    export_btn.set_color(Color::from_hex(0x1a1f3a));
    export_btn.set_label_color(Color::from_hex(0x00d4ff));
    actions_flex.add(&export_btn);

    let mut save_btn = button::Button::default()
        .with_size(200, 100)
        .with_label("💾 Save Config");
    save_btn.set_color(Color::from_hex(0x1a1f3a));
    save_btn.set_label_color(Color::from_hex(0x00d4ff));
    actions_flex.add(&save_btn);

    let mut logs_btn = button::Button::default()
        .with_size(200, 100)
        .with_label("📋 View Logs");
    logs_btn.set_color(Color::from_hex(0x1a1f3a));
    logs_btn.set_label_color(Color::from_hex(0x00d4ff));
    actions_flex.add(&logs_btn);

    actions_flex.end();

    // Codebase operations
    let mut codebase_flex = group::Flex::default().with_size(1450, 100).row();
    codebase_flex.set_color(Color::from_hex(0x0a0e27));

    let mut build_app_btn = button::Button::default()
        .with_size(220, 100)
        .with_label("🛠️ Build Native App");
    build_app_btn.set_color(Color::from_hex(0x1a1f3a));
    build_app_btn.set_label_color(Color::from_hex(0x00d4ff));
    codebase_flex.add(&build_app_btn);

    let mut run_tests_btn = button::Button::default()
        .with_size(220, 100)
        .with_label("🧪 Run Tests");
    run_tests_btn.set_color(Color::from_hex(0x1a1f3a));
    run_tests_btn.set_label_color(Color::from_hex(0x00d4ff));
    codebase_flex.add(&run_tests_btn);

    let mut git_status_btn = button::Button::default()
        .with_size(220, 100)
        .with_label("📁 Git Status");
    git_status_btn.set_color(Color::from_hex(0x1a1f3a));
    git_status_btn.set_label_color(Color::from_hex(0x00d4ff));
    codebase_flex.add(&git_status_btn);

    let mut recent_commits_btn = button::Button::default()
        .with_size(220, 100)
        .with_label("📜 Recent Commits");
    recent_commits_btn.set_color(Color::from_hex(0x1a1f3a));
    recent_commits_btn.set_label_color(Color::from_hex(0x00d4ff));
    codebase_flex.add(&recent_commits_btn);

    codebase_flex.end();

    // System information
    let mut info_flex = group::Flex::default().with_size(1450, 250).column();
    info_flex.set_color(Color::from_hex(0x0a0e27));

    let mut info_title = text::TextDisplay::default().with_size(1450, 40);
    info_title.set_buffer(text::TextBuffer::default());
    info_title.buffer().unwrap().set_text("System Information");
    info_title.set_text_color(Color::from_hex(0x00d4ff));

    let mut info_display = text::TextDisplay::default().with_size(1450, 210);
    info_display.set_buffer(text::TextBuffer::default());
    let info_text = r#"Build Version:        1.0.0
GPU:                  NVIDIA RTX 5080
CUDA Version:         12.0
Memory:               15.9 GB
Uptime:               2h 34m
Last Update:          5:03:00 PM
Status:               🟢 Running
Coherence:            0.9993 (Excellent)
Ethics Score:         2.39 (PASS ✓)"#;
    info_display.buffer().unwrap().set_text(info_text);
    info_display.set_text_color(Color::from_hex(0x00ff64));

    info_flex.end();

    flex.end();
    group.end();
    tabs.add(&group);

    ControlPanelButtons {
        start_btn,
        stop_btn,
        pause_btn,
        reset_btn,
        export_btn,
        save_btn,
        logs_btn,
        build_choice,
        build_app_btn,
        run_tests_btn,
        git_status_btn,
        recent_commits_btn,
    }
}

fn create_config_input(flex: &mut group::Flex, label: &str, value: &str) {
    let mut input_flex = group::Flex::default().with_size(350, 110).column();
    input_flex.set_color(Color::from_hex(0x0a0e27));

    let mut label_text = text::TextDisplay::default().with_size(350, 40);
    label_text.set_buffer(text::TextBuffer::default());
    label_text.buffer().unwrap().set_text(label);
    label_text.set_text_color(Color::from_hex(0x00d4ff));

    let mut input = text::TextEditor::default().with_size(350, 70);
    input.set_buffer(text::TextBuffer::default());
    input.buffer().unwrap().set_text(value);
    input.set_color(Color::from_hex(0x1a1f3a));
    input.set_text_color(Color::White);

    input_flex.end();
    flex.add(&input_flex);
}
