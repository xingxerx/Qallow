
use fltk::{
    app,
    button::Button,
    enums::{Align, Color, FrameType},
    frame::Frame,
    group::{Flex, Pack},
    input::Input,
    prelude::*,
    text::{TextBuffer, TextDisplay},
};

pub struct ChatPanel {
    pub pack: Pack,
    pub conversation_display: TextDisplay,
    pub input: Input,
    pub send_button: Button,
}

impl ChatPanel {
    pub fn new() -> Self {
        let mut pack = Pack::new(0, 0, 600, 800, "Agent Chat");
        pack.set_spacing(10);
        pack.set_type(fltk::group::PackType::Vertical);

        let mut conversation_display = TextDisplay::default().with_size(580, 500);
        conversation_display.set_buffer(TextBuffer::default());
        conversation_display.set_color(Color::from_rgb(30, 30, 30));
        conversation_display.set_text_color(Color::White);
        conversation_display.set_frame(FrameType::DownBox);

        let input_flex = Flex::default().with_size(580, 40).row();
        let mut input = Input::default().with_size(480, 40);
        input.set_text_color(Color::White);
        input.set_color(Color::from_rgb(50, 50, 50));

        let mut send_button = Button::new(0, 0, 90, 40, "Send");
        send_button.set_color(Color::from_rgb(80, 80, 120));
        send_button.set_label_color(Color::White);
        input_flex.end();

        pack.end();

        Self {
            pack,
            conversation_display,
            input,
            send_button,
        }
    }

    // ... existing code ...
    pub fn add_message(&mut self, author: &str, text: &str) {
        if let Some(mut buffer) = self.conversation_display.buffer() {
            let formatted_message = format!("{}: {}\n", author, text);
            buffer.append(&formatted_message);
            self.conversation_display.scroll(
                buffer.count_lines(0, buffer.length()),
                0,
            );
        }
    }

}
