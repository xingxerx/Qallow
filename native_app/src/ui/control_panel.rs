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
    group.begin();

    let mut flex = group::Flex::default().column();
    flex.set_color(Color::from_hex(COLOR_BG_DARK));
    flex.set_pad(10);
    flex.begin();

    let mut title = text::TextDisplay::default().with_size(0, 40);
    title.set_buffer(text::TextBuffer::default());
    title.buffer().unwrap().set_text("System Control Panel");
    title.set_text_color(Color::from_hex(COLOR_PRIMARY));
    flex.add(&title);
    flex.fixed(&title, 40);

    let mut control_row = group::Flex::default().row();
    control_row.set_pad(10);
    control_row.set_color(Color::from_hex(COLOR_BG_DARK));

    let mut start_btn = button::Button::default().with_label("▶️ Start VM");
    start_btn.set_color(Color::from_hex(COLOR_SUCCESS));
    start_btn.set_label_color(Color::Black);
    control_row.add(&start_btn);
    control_row.fixed(&start_btn, 160);

    let mut stop_btn = button::Button::default().with_label("⏹️ Stop VM");
    stop_btn.set_color(Color::from_hex(COLOR_DANGER));
    stop_btn.set_label_color(Color::White);
    control_row.add(&stop_btn);
    control_row.fixed(&stop_btn, 160);

    let mut pause_btn = button::Button::default().with_label("⏸️ Pause");
    pause_btn.set_color(Color::from_hex(COLOR_PRIMARY));
    pause_btn.set_label_color(Color::Black);
    control_row.add(&pause_btn);
    control_row.fixed(&pause_btn, 160);

    let mut reset_btn = button::Button::default().with_label("🔄 Reset");
    reset_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    reset_btn.set_label_color(Color::from_hex(COLOR_PRIMARY));
    control_row.add(&reset_btn);
    control_row.fixed(&reset_btn, 160);

    control_row.end();
    flex.add(&control_row);
    flex.fixed(&control_row, 70);

    let mut build_row = group::Flex::default().row();
    build_row.set_pad(10);
    build_row.set_color(Color::from_hex(COLOR_BG_DARK));

    let mut build_label = text::TextDisplay::default().with_size(200, 30);
    build_label.set_buffer(text::TextBuffer::default());
    build_label.buffer().unwrap().set_text("Select Build:");
    build_label.set_text_color(Color::from_hex(COLOR_PRIMARY));
    build_row.add(&build_label);
    build_row.fixed(&build_label, 200);

    let mut build_choice = menu::Choice::default();
    build_choice.add_choice("CPU|CUDA");
    build_choice.set_color(Color::from_hex(COLOR_BG_ACCENT));
    build_choice.set_text_color(Color::from_hex(COLOR_TEXT));
    build_choice.set_size(220, 30);
    build_row.add(&build_choice);
    build_row.fixed(&build_choice, 180);

    build_row.end();
    flex.add(&build_row);
    flex.fixed(&build_row, 40);

    let mut phase_row = group::Flex::default().row();
    phase_row.set_pad(10);
    phase_row.set_color(Color::from_hex(COLOR_BG_DARK));

    let mut phase_label = text::TextDisplay::default().with_size(200, 30);
    phase_label.set_buffer(text::TextBuffer::default());
    phase_label.buffer().unwrap().set_text("Select Phase:");
    phase_label.set_text_color(Color::from_hex(COLOR_PRIMARY));
    phase_row.add(&phase_label);
    phase_row.fixed(&phase_label, 200);

    let mut phase_choice = menu::Choice::default();
    phase_choice
        .add_choice("Phase 13|Phase 14|Phase 15|Phase 16|Phase 17|Phase 18|Phase 19|Phase 20");
    phase_choice.set_value(1);
    phase_choice.set_color(Color::from_hex(COLOR_BG_ACCENT));
    phase_choice.set_text_color(Color::from_hex(COLOR_TEXT));
    phase_choice.set_size(220, 30);
    phase_row.add(&phase_choice);
    phase_row.fixed(&phase_choice, 220);

    phase_row.end();
    flex.add(&phase_row);
    flex.fixed(&phase_row, 40);

    let mut ritual_row = group::Flex::default().row();
    ritual_row.set_pad(10);
    ritual_row.set_color(Color::from_hex(COLOR_BG_DARK));

    let mut shadow_btn = button::Button::default().with_label("🕶 Shadow");
    shadow_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    shadow_btn.set_label_color(Color::from_hex(COLOR_MUTED));
    ritual_row.add(&shadow_btn);
    ritual_row.fixed(&shadow_btn, 160);

    let mut rebellion_btn = button::Button::default().with_label("🔥 Rebel");
    rebellion_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    rebellion_btn.set_label_color(Color::from_hex(COLOR_DANGER));
    ritual_row.add(&rebellion_btn);
    ritual_row.fixed(&rebellion_btn, 160);

    let mut offspring_btn = button::Button::default().with_label("🌱 Offspring");
    offspring_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    offspring_btn.set_label_color(Color::from_hex(COLOR_SUCCESS));
    ritual_row.add(&offspring_btn);
    ritual_row.fixed(&offspring_btn, 160);

    let mut dissolution_btn = button::Button::default().with_label("💀 Dissolve");
    dissolution_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    dissolution_btn.set_label_color(Color::from_hex(COLOR_MUTED));
    ritual_row.add(&dissolution_btn);
    ritual_row.fixed(&dissolution_btn, 160);

    let mut dream_btn = button::Button::default().with_label("🌙 Dream");
    dream_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    dream_btn.set_label_color(Color::from_hex(COLOR_PRIMARY));
    ritual_row.add(&dream_btn);
    ritual_row.fixed(&dream_btn, 160);

    ritual_row.end();
    flex.add(&ritual_row);
    flex.fixed(&ritual_row, 60);

    let mut actions_row = group::Flex::default().row();
    actions_row.set_pad(10);
    actions_row.set_color(Color::from_hex(COLOR_BG_DARK));

    let mut export_btn = button::Button::default().with_label("📈 Export Metrics");
    export_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    export_btn.set_label_color(Color::from_hex(COLOR_SUCCESS));
    actions_row.add(&export_btn);
    actions_row.fixed(&export_btn, 200);

    let mut save_btn = button::Button::default().with_label("💾 Save Config");
    save_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    save_btn.set_label_color(Color::from_hex(COLOR_PRIMARY));
    actions_row.add(&save_btn);
    actions_row.fixed(&save_btn, 200);

    let mut logs_btn = button::Button::default().with_label("📋 View Logs");
    logs_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    logs_btn.set_label_color(Color::from_hex(COLOR_MUTED));
    actions_row.add(&logs_btn);
    actions_row.fixed(&logs_btn, 200);

    actions_row.end();
    flex.add(&actions_row);
    flex.fixed(&actions_row, 60);

    let mut code_row = group::Flex::default().row();
    code_row.set_pad(10);
    code_row.set_color(Color::from_hex(COLOR_BG_DARK));

    let mut build_app_btn = button::Button::default().with_label("🛠️ Build Native App");
    build_app_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    build_app_btn.set_label_color(Color::from_hex(COLOR_PRIMARY));
    code_row.add(&build_app_btn);
    code_row.fixed(&build_app_btn, 220);

    let mut run_tests_btn = button::Button::default().with_label("🧪 Run Tests");
    run_tests_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    run_tests_btn.set_label_color(Color::from_hex(COLOR_SUCCESS));
    code_row.add(&run_tests_btn);
    code_row.fixed(&run_tests_btn, 220);

    let mut git_status_btn = button::Button::default().with_label("📁 Git Status");
    git_status_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    git_status_btn.set_label_color(Color::from_hex(COLOR_MUTED));
    code_row.add(&git_status_btn);
    code_row.fixed(&git_status_btn, 220);

    let mut recent_commits_btn = button::Button::default().with_label("📜 Recent Commits");
    recent_commits_btn.set_color(Color::from_hex(COLOR_BG_ACCENT));
    recent_commits_btn.set_label_color(Color::from_hex(COLOR_PRIMARY));
    code_row.add(&recent_commits_btn);
    code_row.fixed(&recent_commits_btn, 220);

    code_row.end();
    flex.add(&code_row);
    flex.fixed(&code_row, 60);

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
