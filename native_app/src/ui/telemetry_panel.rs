
use fltk::{
    prelude::*,
    text::{TextBuffer, TextDisplay},
    enums::{Color, FrameType},
};

#[derive(Clone)]
pub struct TelemetryPanel {
    pub display: TextDisplay,
    buffer: TextBuffer,
}

impl TelemetryPanel {
    pub fn new() -> Self {
        let mut display = TextDisplay::new(0, 0, 0, 0, "Telemetry");
        let buffer = TextBuffer::default();
        display.set_buffer(buffer.clone());
        display.set_color(Color::from_hex(0x0a0e27));
        display.set_text_color(Color::from_hex(0x00ff64));
        display.set_frame(FrameType::FlatBox);

        Self { display, buffer }
    }

    pub fn append(&mut self, text: &str) {
        self.buffer.append(text);
        self.display.scroll(
            self.display.count_lines(0, self.buffer.length(), true),
            0,
        );
    }
}
