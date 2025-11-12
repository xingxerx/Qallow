use fltk::{
    button::Button,
    enums::{Color, FrameType},
    group::{Flex, Pack},
    input::Input,
    prelude::*,
    text::{TextBuffer, TextDisplay},
};

#[derive(Clone)]
pub struct ChatPanel {
    pub pack: Pack,
    pub conversation_display: TextDisplay,
    pub input: Input,
    pub send_button: Button,
}

impl ChatPanel {
    pub fn new() -> Self {
        let mut pack = Pack::new(0, 0, 600, 200, "");
        pack.set_spacing(8);
        pack.set_type(fltk::group::PackType::Vertical);
        pack.set_color(Color::from_rgb(15, 15, 25));
        pack.begin();

        // Title label
        let mut title = fltk::frame::Frame::new(0, 0, 600, 25, "🤖 AI Agent Chat");
        title.set_label_size(14);
        title.set_label_color(Color::from_rgb(100, 200, 255));
        pack.add(&title);

        // Conversation display with enhanced styling
        let mut conversation_display = TextDisplay::new(0, 25, 600, 130, "");
        let mut buffer = TextBuffer::default();
        buffer.append("Welcome to Qallow AI Agent Chat\n");
        buffer.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");
        conversation_display.set_buffer(buffer);
        conversation_display.set_color(Color::from_rgb(20, 20, 35));
        conversation_display.set_text_color(Color::from_rgb(200, 220, 255));
        conversation_display.set_frame(FrameType::DownBox);
        conversation_display.set_text_font(fltk::enums::Font::Courier);
        pack.add(&conversation_display);

        // Input section with enhanced styling
        let mut input_flex = Flex::new(0, 155, 600, 45, "").row();
        input_flex.set_color(Color::from_rgb(15, 15, 25));
        input_flex.set_margin(5);
        input_flex.begin();

        let mut input = Input::default();
        input.set_text_color(Color::from_rgb(200, 220, 255));
        input.set_color(Color::from_rgb(40, 40, 60));
        input.set_frame(FrameType::DownBox);
        input.set_text_font(fltk::enums::Font::Courier);

        let mut send_button = Button::new(0, 0, 100, 0, "Send");
        send_button.set_color(Color::from_rgb(0, 150, 200));
        send_button.set_label_color(Color::White);
        send_button.set_frame(FrameType::UpBox);
        send_button.set_label_size(12);

        input_flex.fixed(&send_button, 100);
        input_flex.end();
        pack.add(&input_flex);

        pack.end();

        Self {
            pack,
            conversation_display,
            input,
            send_button,
        }
    }

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
