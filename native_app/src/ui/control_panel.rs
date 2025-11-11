use crate::models::AppState;
use crate::ui::{COLOR_PRIMARY, COLOR_SUCCESS};
use fltk::enums::Color;
use fltk::{prelude::*, *};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct ControlPanel {
    pub flex: group::Flex,
    pub buttons: ControlPanelButtons,
}

#[derive(Clone)]
pub struct ControlPanelButtons {
    pub phase_buttons: Vec<button::Button>,
    pub unified_button: button::Button,
}

pub fn create_control_panel(
    _state: Arc<Mutex<AppState>>,
) -> ControlPanel {
    let mut flex = group::Flex::default().with_size(200, 0).column();
    flex.set_pad(10);

    let mut phase_buttons = Vec::new();
    for i in 1..=15 {
        let mut btn = button::Button::default().with_label(&format!("Phase {}", i));
        btn.set_color(Color::from_hex(COLOR_PRIMARY));
        btn.set_label_color(Color::Black);
        flex.add(&btn);
        phase_buttons.push(btn);
    }
    
    let mut unified_button = button::Button::default().with_label("Run Unified");
    unified_button.set_color(Color::from_hex(COLOR_SUCCESS));
    unified_button.set_label_color(Color::Black);
    flex.add(&unified_button);

    flex.end();

    let buttons = ControlPanelButtons {
        phase_buttons,
        unified_button,
    };

    ControlPanel { flex, buttons }
}
