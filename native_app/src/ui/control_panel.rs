use crate::models::AppState;
use crate::ui::{
    COLOR_BG_ACCENT, COLOR_BG_DARK, COLOR_DANGER, COLOR_MUTED, COLOR_PRIMARY, COLOR_SUCCESS,
    COLOR_TEXT,
};
use fltk::enums::Color;
use fltk::{prelude::*, *};
use std::sync::{Arc, Mutex};

pub struct ControlPanelButtons {
    pub start_btn: button::Button,
    pub stop_btn: button::Button,
    pub pause_btn: button::Button,
    pub reset_btn: button::Button,
    pub phase_choice: menu::Choice,
    pub shadow_btn: button::Button,
    pub rebellion_btn: button::Button,
    pub offspring_btn: button::Button,
    pub dissolution_btn: button::Button,
    pub dream_btn: button::Button,
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
    group.set_color(Color::from_hex(COLOR_BG_DARK));

    let mut flex = group::Flex::default().with_size(1450, 950).column();
    flex.set_color(Color::from_hex(COLOR_BG_DARK));

    let mut title = text::TextDisplay::default().with_size(1450, 40);
    title.set_buffer(text::TextBuffer::default());
    title.buffer().unwrap().set_text("System Control Panel");
    title.set_text_color(Color::from_hex(COLOR_PRIMARY));

    let mut control_flex = group::Flex::default().with_size(1450, 100).row();
    control_flex.set_color(Color::from_hex(COLOR_BG_DARK));

    let mut start_btn = button::Button::default()
        .with_size(150, 100)
        .with_label("▶️ Start VM");
    start_btn.set_color(Color::from_hex(COLOR_SUCCESS));
    start_btn.set_label_color(Color::Black);
    control_flex.add(&start_btn);

    let mut stop_btn = button::Button::default()
        .with_size(150, 100)
        .with_label("⏹️ Stop VM");
    stop_btn.set_color(Color::from_hex(COLOR_DANGER));
    stop_btn.set_label_color(Color::White);
    control_flex.add(&stop_btn);

    let mut pause_btn = button::Button::default()
        .with_size(150, 100)
        .with_label("⏸️ Pause");
    pause_btn.set_color(Color::from_hex(COLOR_PRIMARY));
    pause_btn.set_label_color(Color::Black);
    control_flex.add(&pause_btn);

    let mut reset_btn = button::Button::default()
        .with_size(150, 100)
        .with_label("🔄 Reset");
    reset_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    reset_btn.set_label_color(Color::from_hex(COLOR_PRIMARY));
    control_flex.add(&reset_btn);

    control_flex.end();

    let mut build_flex = group::Flex::default().with_size(1450, 80).row();
    build_flex.set_color(Color::from_hex(COLOR_BG_DARK));

    let mut build_label = text::TextDisplay::default().with_size(200, 80);
    build_label.set_buffer(text::TextBuffer::default());
    build_label.buffer().unwrap().set_text("Select Build:");
    build_label.set_text_color(Color::from_hex(COLOR_PRIMARY));
    build_flex.add(&build_label);

    let mut build_choice = menu::Choice::default().with_size(300, 80);
    build_choice.add_choice("CPU|CUDA");
    build_choice.set_color(Color::from_hex(COLOR_BG_ACCENT));
    build_choice.set_text_color(Color::from_hex(COLOR_TEXT));
    build_flex.add(&build_choice);

    build_flex.end();

    let mut phase_flex = group::Flex::default().with_size(1450, 80).row();
    phase_flex.set_color(Color::from_hex(COLOR_BG_DARK));

    let mut phase_label = text::TextDisplay::default().with_size(200, 80);
    phase_label.set_buffer(text::TextBuffer::default());
    phase_label.buffer().unwrap().set_text("Select Phase:");
    phase_label.set_text_color(Color::from_hex(COLOR_PRIMARY));
    phase_flex.add(&phase_label);

    let mut phase_choice = menu::Choice::default().with_size(300, 80);
    phase_choice
        .add_choice("Phase 13|Phase 14|Phase 15|Phase 16|Phase 17|Phase 18|Phase 19|Phase 20");
    phase_choice.set_value(1); // default Phase 14
    phase_choice.set_color(Color::from_hex(COLOR_BG_ACCENT));
    phase_choice.set_text_color(Color::from_hex(COLOR_TEXT));
    phase_flex.add(&phase_choice);

    phase_flex.end();

    let mut ritual_row = group::Flex::default().with_size(1450, 90).row();
    ritual_row.set_color(Color::from_hex(COLOR_BG_DARK));

    let mut shadow_btn = button::Button::default()
        .with_size(180, 90)
        .with_label("🕶 Shadow");
    shadow_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    shadow_btn.set_label_color(Color::from_hex(COLOR_MUTED));
    ritual_row.add(&shadow_btn);

    let mut rebellion_btn = button::Button::default()
        .with_size(180, 90)
        .with_label("🔥 Rebel");
    rebellion_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    rebellion_btn.set_label_color(Color::from_hex(COLOR_DANGER));
    ritual_row.add(&rebellion_btn);

    let mut offspring_btn = button::Button::default()
        .with_size(180, 90)
        .with_label("🌱 Offspring");
    offspring_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    offspring_btn.set_label_color(Color::from_hex(COLOR_SUCCESS));
    ritual_row.add(&offspring_btn);

    let mut dissolution_btn = button::Button::default()
        .with_size(180, 90)
        .with_label("💀 Dissolve");
    dissolution_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    dissolution_btn.set_label_color(Color::from_hex(COLOR_MUTED));
    ritual_row.add(&dissolution_btn);

    let mut dream_btn = button::Button::default()
        .with_size(180, 90)
        .with_label("🌙 Dream");
    dream_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    dream_btn.set_label_color(Color::from_hex(COLOR_PRIMARY));
    ritual_row.add(&dream_btn);

    ritual_row.end();

    let mut actions_flex = group::Flex::default().with_size(1450, 90).row();
    actions_flex.set_color(Color::from_hex(COLOR_BG_DARK));

    let mut export_btn = button::Button::default()
        .with_size(200, 90)
        .with_label("📈 Export Metrics");
    export_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    export_btn.set_label_color(Color::from_hex(COLOR_SUCCESS));
    actions_flex.add(&export_btn);

    let mut save_btn = button::Button::default()
        .with_size(200, 90)
        .with_label("💾 Save Config");
    save_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    save_btn.set_label_color(Color::from_hex(COLOR_PRIMARY));
    actions_flex.add(&save_btn);

    let mut logs_btn = button::Button::default()
        .with_size(200, 90)
        .with_label("📋 View Logs");
    logs_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    logs_btn.set_label_color(Color::from_hex(COLOR_MUTED));
    actions_flex.add(&logs_btn);

    actions_flex.end();

    let mut codebase_flex = group::Flex::default().with_size(1450, 90).row();
    codebase_flex.set_color(Color::from_hex(COLOR_BG_DARK));

    let mut build_app_btn = button::Button::default()
        .with_size(220, 90)
        .with_label("🛠️ Build Native App");
    build_app_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    build_app_btn.set_label_color(Color::from_hex(COLOR_PRIMARY));
    codebase_flex.add(&build_app_btn);

    let mut run_tests_btn = button::Button::default()
        .with_size(220, 90)
        .with_label("🧪 Run Tests");
    run_tests_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    run_tests_btn.set_label_color(Color::from_hex(COLOR_SUCCESS));
    codebase_flex.add(&run_tests_btn);

    let mut git_status_btn = button::Button::default()
        .with_size(220, 90)
        .with_label("📁 Git Status");
    git_status_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    git_status_btn.set_label_color(Color::from_hex(COLOR_MUTED));
    codebase_flex.add(&git_status_btn);

    let mut recent_commits_btn = button::Button::default()
        .with_size(220, 90)
        .with_label("📜 Recent Commits");
    recent_commits_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    recent_commits_btn.set_label_color(Color::from_hex(COLOR_PRIMARY));
    codebase_flex.add(&recent_commits_btn);

    codebase_flex.end();

    flex.end();
    group.end();
    tabs.add(&group);

    ControlPanelButtons {
        start_btn,
        stop_btn,
        pause_btn,
        reset_btn,
        phase_choice,
        shadow_btn,
        rebellion_btn,
        offspring_btn,
        dissolution_btn,
        dream_btn,
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
