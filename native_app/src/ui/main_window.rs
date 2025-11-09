use crate::button_handlers::ButtonHandler;
use crate::ui::{
    audit_log, control_panel, dashboard, dungeons, help, metrics, settings, terminal,
};
use fltk::{prelude::*, *};
use std::sync::Arc;

pub struct MainWindow {
    pub wind: window::Window,
    pub dashboard_panel: dashboard::Dashboard,
    pub control_panel: control_panel::ControlPanel,
    pub terminal_panel: terminal::TerminalView,
    pub audit_panel: audit_log::AuditLogView,
    pub chat_panel: super::chat_panel::ChatPanel,
    pub button_handler: Arc<ButtonHandler>,
}

impl MainWindow {
    pub fn new(button_handler: Arc<ButtonHandler>) -> Self {
        let mut wind = window::Window::default()
            .with_size(1600, 1000)
            .with_label("Qallow");
        wind.set_color(enums::Color::from_hex(0x0a0e27));
        wind.begin();

        let mut root_flex = group::Flex::default_fill().column();
        root_flex.set_margin(5);

        let mut main_flex = group::Flex::default().row();
        main_flex.set_margin(5);

        let mut tabs = group::Tabs::default();
        tabs.set_tab_align(enums::Align::Left);
        tabs.handle(move |_, ev| {
            if ev == enums::Event::Push {
                if app::event_x() > tabs.x() + tabs.width() - 30 {
                    return true;
                }
            }
            false
        });

        let dashboard_panel = dashboard::create_dashboard(&mut tabs, button_handler.state.clone());
        let terminal_panel = terminal::create_terminal(&mut tabs, button_handler.state.clone());
        let audit_panel = audit_log::create_audit_log(&mut tabs, button_handler.state.clone());
        let _metrics_panel = metrics::create_metrics(&mut tabs, button_handler.state.clone());
        let _dungeons_panel = dungeons::create_dungeons_tab(&mut tabs, button_handler.state.clone());
        let _settings_panel = settings::create_settings_panel(&mut tabs, button_handler.state.clone());
        let _help_panel = help::create_help_panel(&mut tabs, button_handler.state.clone());

        tabs.end();
        main_flex.add(&tabs);

        let control_panel = control_panel::create_control_panel(&mut tabs, button_handler.clone());
        main_flex.add(&control_panel.flex);
        main_flex.fixed(&control_panel.flex, 200);

        main_flex.end();
        root_flex.add(&main_flex);

        let chat_panel = super::chat_panel::ChatPanel::new();
        root_flex.add(&chat_panel.pack);
        root_flex.fixed(&chat_panel.pack, 150);

        root_flex.end();
        wind.end();
        wind.make_resizable(true);

        Self {
            wind,
            dashboard_panel,
            control_panel,
            terminal_panel,
            audit_panel,
            chat_panel,
            button_handler,
        }
    }
}
