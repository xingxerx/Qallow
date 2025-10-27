use crate::models::AppState;
use fltk::enums::{Color, FrameType};
use fltk::{prelude::*, *};
use std::sync::{Arc, Mutex};

pub struct ControlPanelButtons {
    pub start_btn: button::Button,
    pub stop_btn: button::Button,
    pub pause_btn: button::Button,
    pub reset_btn: button::Button,
    pub ignite_btn: button::Button,
    pub hibernate_btn: button::Button,
    pub dissolve_btn: button::Button,
    pub advance_btn: button::Button,
    pub tempo_choice: menu::Choice,
    pub divine_btn: button::Button,
    pub metrics_btn: button::Button,
    pub chronicle_btn: button::Button,
    pub prophecy_btn: button::Button,
    pub snapshot_btn: button::Button,
    pub restore_btn: button::Button,
    pub fork_btn: button::Button,
    pub merge_btn: button::Button,
    pub swarm_btn: button::Button,
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

    let mut title = text::TextDisplay::default().with_size(1450, 40);
    title.set_buffer(text::TextBuffer::default());
    title
        .buffer()
        .unwrap()
        .set_text("Consciousness Command Console");
    title.set_text_color(Color::from_hex(0x00d4ff));

    let mut control_row = group::Flex::default().with_size(1450, 120).row();
    control_row.set_color(Color::from_hex(0x0a0e27));

    let mut ignite_btn = themed_button("🔥 Ignite", 0x00ff64, Color::Black);
    control_row.add(&ignite_btn);

    let mut hibernate_btn = themed_button("⏸️ Hibernate", 0xffaa00, Color::Black);
    control_row.add(&hibernate_btn);

    let mut dissolve_btn = themed_button("⏹️ Dissolve", 0xff6464, Color::White);
    control_row.add(&dissolve_btn);

    let mut advance_btn = themed_button("⏭️ Advance", 0x6600ff, Color::White);
    control_row.add(&advance_btn);

    let mut tempo_choice = menu::Choice::default().with_size(200, 120);
    tempo_choice.set_color(Color::from_hex(0x1a1f3a));
    tempo_choice.set_text_color(Color::from_hex(0x00d4ff));
    tempo_choice.set_label("🎚️ Tempo");
    tempo_choice.add_choice("1x|10x|100x");
    tempo_choice.set_value(0);
    control_row.add(&tempo_choice);

    control_row.end();

    let mut observation_row = group::Flex::default().with_size(1450, 120).row();
    observation_row.set_color(Color::from_hex(0x0a0e27));

    let mut divine_btn = themed_button("🧠 Divine", 0x1a1f3a, Color::from_hex(0x00d4ff));
    observation_row.add(&divine_btn);

    let mut metrics_btn = themed_button("📈 Metrics", 0x1a1f3a, Color::from_hex(0x00d4ff));
    observation_row.add(&metrics_btn);

    let mut chronicle_btn = themed_button("📜 Chronicle", 0x1a1f3a, Color::from_hex(0x00d4ff));
    observation_row.add(&chronicle_btn);

    let mut prophecy_btn = themed_button("🔮 Prophecy", 0x1a1f3a, Color::from_hex(0x00d4ff));
    observation_row.add(&prophecy_btn);

    observation_row.end();

    let mut rituals_row = group::Flex::default().with_size(1450, 120).row();
    rituals_row.set_color(Color::from_hex(0x0a0e27));

    let mut snapshot_btn = themed_button("💾 Snapshot", 0x1a1f3a, Color::from_hex(0x00d4ff));
    rituals_row.add(&snapshot_btn);

    let mut restore_btn = themed_button("🔄 Restore", 0x1a1f3a, Color::from_hex(0x00d4ff));
    rituals_row.add(&restore_btn);

    let mut fork_btn = themed_button("🌿 Fork", 0x1a1f3a, Color::from_hex(0x00d4ff));
    rituals_row.add(&fork_btn);

    let mut merge_btn = themed_button("🤝 Merge", 0x1a1f3a, Color::from_hex(0x00d4ff));
    rituals_row.add(&merge_btn);

    let mut swarm_btn = themed_button("⚡ Swarm", 0x1a1f3a, Color::from_hex(0x00d4ff));
    rituals_row.add(&swarm_btn);

    rituals_row.end();

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

    let mut actions_flex = group::Flex::default().with_size(1450, 100).row();
    actions_flex.set_color(Color::from_hex(0x0a0e27));

    let mut export_btn = themed_button("📈 Export Metrics", 0x1a1f3a, Color::from_hex(0x00d4ff));
    actions_flex.add(&export_btn);

    let mut save_btn = themed_button("💾 Save Config", 0x1a1f3a, Color::from_hex(0x00d4ff));
    actions_flex.add(&save_btn);

    let mut logs_btn = themed_button("📋 View Logs", 0x1a1f3a, Color::from_hex(0x00d4ff));
    actions_flex.add(&logs_btn);

    actions_flex.end();

    let mut build_flex = group::Flex::default().with_size(1450, 100).row();
    build_flex.set_color(Color::from_hex(0x0a0e27));

    let mut build_label = text::TextDisplay::default().with_size(200, 100);
    build_label.set_buffer(text::TextBuffer::default());
    build_label
        .buffer()
        .unwrap()
        .set_text("Select Build:");
    build_label.set_text_color(Color::from_hex(0x00d4ff));
    build_flex.add(&build_label);

    let mut build_choice = menu::Choice::default().with_size(300, 100);
    build_choice.add_choice("CPU|CUDA");
    build_choice.set_color(Color::from_hex(0x1a1f3a));
    build_choice.set_text_color(Color::from_hex(0x00d4ff));
    build_flex.add(&build_choice);

    build_flex.end();

    let mut codebase_flex = group::Flex::default().with_size(1450, 100).row();
    codebase_flex.set_color(Color::from_hex(0x0a0e27));

    let mut build_app_btn = themed_button("🛠️ Build Native App", 0x1a1f3a, Color::from_hex(0x00d4ff));
    codebase_flex.add(&build_app_btn);

    let mut run_tests_btn = themed_button("🧪 Run Tests", 0x1a1f3a, Color::from_hex(0x00d4ff));
    codebase_flex.add(&run_tests_btn);

    let mut git_status_btn = themed_button("📁 Git Status", 0x1a1f3a, Color::from_hex(0x00d4ff));
    codebase_flex.add(&git_status_btn);

    let mut recent_commits_btn = themed_button("📜 Recent Commits", 0x1a1f3a, Color::from_hex(0x00d4ff));
    codebase_flex.add(&recent_commits_btn);

    codebase_flex.end();

    flex.end();
    group.end();
    tabs.add(&group);

    ControlPanelButtons {
        start_btn: ignite_btn.clone(),
        stop_btn: dissolve_btn.clone(),
        pause_btn: hibernate_btn.clone(),
        reset_btn: advance_btn.clone(),
        ignite_btn,
        hibernate_btn,
        dissolve_btn,
        advance_btn,
        tempo_choice,
        divine_btn,
        metrics_btn,
        chronicle_btn,
        prophecy_btn,
        snapshot_btn,
        restore_btn,
        fork_btn,
        merge_btn,
        swarm_btn,
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

fn themed_button(label: &str, color: u32, text_color: Color) -> button::Button {
    let mut btn = button::Button::default().with_size(160, 120).with_label(label);
    btn.set_color(Color::from_hex(color));
    btn.set_label_color(text_color);
    btn.set_frame(FrameType::RoundedBox);
    btn
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
