//! The main application window: phase/unified control buttons plus a
//! terminal-style output area.

use crate::button_handlers::ButtonHandler;
use fltk::{
    button::Button,
    group::{Flex, FlexType},
    output::MultilineOutput,
    prelude::*,
    text::{TextBuffer, TextDisplay},
    window::Window,
};
use std::sync::Arc;

pub struct ButtonGroup {
    pub phase_buttons: Vec<Button>,
    pub unified_button: Button,
}

pub struct ControlPanel {
    pub buttons: ButtonGroup,
}

pub struct MainWindow {
    pub wind: Window,
    pub control_panel: ControlPanel,
    pub button_handler: Arc<ButtonHandler>,
    pub terminal_display: TextDisplay,
    pub status_output: MultilineOutput,
}

impl MainWindow {
    pub fn new(button_handler: Arc<ButtonHandler>) -> Self {
        let mut wind = Window::new(100, 100, 800, 600, "Qallow");
        wind.make_resizable(true);

        let mut root = Flex::new(10, 10, 780, 580, None);
        root.set_type(FlexType::Column);

        let button_row = Flex::default().row();
        let mut phase_buttons = Vec::new();
        for i in 1..=4 {
            let btn = Button::default().with_label(&format!("Phase {}", i));
            phase_buttons.push(btn);
        }
        let unified_button = Button::default().with_label("Unified");
        button_row.end();
        root.fixed(&button_row, 40);

        let mut status_output = MultilineOutput::default();
        status_output.set_value("Ready.");
        root.fixed(&status_output, 30);

        let buffer = TextBuffer::default();
        let mut terminal_display = TextDisplay::default();
        terminal_display.set_buffer(buffer);

        root.end();
        wind.end();

        Self {
            wind,
            control_panel: ControlPanel {
                buttons: ButtonGroup {
                    phase_buttons,
                    unified_button,
                },
            },
            button_handler,
            terminal_display,
            status_output,
        }
    }
}
