use crate::button_handlers::ButtonHandler;
use crate::ui::{control_panel, matrix_view, telemetry_panel};
use fltk::{prelude::*, *};
use std::sync::Arc;

#[derive(Clone)]
pub struct MainWindow {
    pub wind: window::Window,
    pub matrix_view: matrix_view::MatrixView,
    pub telemetry_panel: telemetry_panel::TelemetryPanel,
    pub control_panel: control_panel::ControlPanel,
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
        root_flex.set_margin(8);
        root_flex.set_spacing(5);

        let mut main_flex = group::Flex::default().row();
        main_flex.set_margin(5);
        main_flex.set_spacing(5);

        let matrix_view = matrix_view::MatrixView::new();
        main_flex.add(&matrix_view.table);

        let mut right_flex = group::Flex::default().column();
                let control_panel = control_panel::create_control_panel(button_handler.state.clone());
        let telemetry_panel = telemetry_panel::TelemetryPanel::new();
        right_flex.add(&control_panel.flex);
        right_flex.add(&telemetry_panel.display);
        right_flex.end();
        
        main_flex.add(&right_flex);
        main_flex.fixed(&right_flex, 200);

        main_flex.end();
        root_flex.add(&main_flex);

        root_flex.end();
        wind.end();
        wind.make_resizable(true);
        wind.set_callback(|_| {
            if app::event() == enums::Event::Close {
                app::quit();
            }
        });

        Self {
            wind,
            matrix_view,
            telemetry_panel,
            control_panel,
            button_handler,
        }
    }
}
