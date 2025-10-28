/// Matrix background effect for the native app
/// Provides a visual effect similar to the web app's matrix rain animation
/// This module creates a decorative background that enhances the modern UI

use fltk::{prelude::*, *};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct MatrixBackground {
    pub enabled: bool,
    pub opacity: f32,
    pub speed: f32,
    pub last_update: Instant,
}

impl MatrixBackground {
    pub fn new() -> Self {
        MatrixBackground {
            enabled: true,
            opacity: 0.15,
            speed: 1.0,
            last_update: Instant::now(),
        }
    }

    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
    }

    pub fn set_opacity(&mut self, opacity: f32) {
        self.opacity = opacity.clamp(0.0, 1.0);
    }

    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed.clamp(0.1, 5.0);
    }

    pub fn should_update(&mut self) -> bool {
        let elapsed = self.last_update.elapsed().as_millis() as f32;
        let update_interval = (1000.0 / (60.0 * self.speed)) as u128;
        
        if elapsed as u128 >= update_interval {
            self.last_update = Instant::now();
            true
        } else {
            false
        }
    }
}

/// Create a visual background effect using FLTK widgets
/// This simulates the matrix rain effect from the web app
pub fn create_matrix_background_widget(parent: &mut group::Group) -> Arc<Mutex<MatrixBackground>> {
    let matrix = Arc::new(Mutex::new(MatrixBackground::new()));

    // Create a decorative box with gradient-like appearance
    let mut bg_box = fltk::widget::Widget::default()
        .with_size(parent.width(), parent.height());

    // Set dark background color matching the theme
    bg_box.set_color(fltk::enums::Color::from_hex(0x0a0e27));

    parent.add(&bg_box);

    matrix
}

/// Create a modern panel with neon border effect
pub fn create_neon_panel(
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    title: &str,
) -> group::Group {
    let mut panel = group::Group::default()
        .with_pos(x, y)
        .with_size(w, h);
    
    panel.set_color(fltk::enums::Color::from_hex(0x1a1f3a));
    
    // Add title
    let mut title_box = text::TextDisplay::default()
        .with_pos(x + 10, y + 5)
        .with_size(w - 20, 25);
    
    title_box.set_buffer(text::TextBuffer::default());
    title_box.buffer().unwrap().set_text(title);
    title_box.set_text_color(fltk::enums::Color::from_hex(0x00d4ff));
    title_box.set_text_size(14);
    
    panel.add(&title_box);
    panel.end();
    
    panel
}

/// Create a modern button with neon styling
pub fn create_neon_button(
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    label: &str,
    is_active: bool,
) -> button::Button {
    let mut btn = button::Button::default()
        .with_pos(x, y)
        .with_size(w, h)
        .with_label(label);
    
    if is_active {
        btn.set_color(fltk::enums::Color::from_hex(0x00d4ff));
        btn.set_label_color(fltk::enums::Color::from_hex(0x0a0e27));
    } else {
        btn.set_color(fltk::enums::Color::from_hex(0x1a1f3a));
        btn.set_label_color(fltk::enums::Color::from_hex(0x00d4ff));
    }
    
    btn
}

/// Create a status indicator with color coding
pub fn create_status_indicator(
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    status: &str,
) -> button::Button {
    let mut indicator = button::Button::default()
        .with_pos(x, y)
        .with_size(w, h)
        .with_label(status);
    
    let color = match status {
        "Running" => 0x00ff64,    // Green
        "Stopped" => 0xff6464,    // Red
        "Paused" => 0xffd166,     // Yellow
        _ => 0x8aa1c1,            // Muted
    };
    
    indicator.set_color(fltk::enums::Color::from_hex(color));
    indicator.set_label_color(fltk::enums::Color::White);
    
    indicator
}

/// Create a modern input field with neon border styling
pub fn create_neon_input(
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    _label: &str,
) -> text::TextEditor {
    let mut input = text::TextEditor::default()
        .with_pos(x, y)
        .with_size(w, h);

    input.set_color(fltk::enums::Color::from_hex(0x0a0e27));
    input.set_text_color(fltk::enums::Color::from_hex(0x00d4ff));
    input.set_text_size(12);

    input
}

/// Create a metrics display card with modern styling
pub fn create_metrics_card(
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    title: &str,
    value: &str,
) -> group::Group {
    let mut card = group::Group::default()
        .with_pos(x, y)
        .with_size(w, h);
    
    card.set_color(fltk::enums::Color::from_hex(0x1a1f3a));
    
    // Title
    let mut title_box = text::TextDisplay::default()
        .with_pos(x + 10, y + 5)
        .with_size(w - 20, 20);
    title_box.set_buffer(text::TextBuffer::default());
    title_box.buffer().unwrap().set_text(title);
    title_box.set_text_color(fltk::enums::Color::from_hex(0x8aa1c1));
    title_box.set_text_size(11);
    
    // Value
    let mut value_box = text::TextDisplay::default()
        .with_pos(x + 10, y + 30)
        .with_size(w - 20, 30);
    value_box.set_buffer(text::TextBuffer::default());
    value_box.buffer().unwrap().set_text(value);
    value_box.set_text_color(fltk::enums::Color::from_hex(0x00ff64));
    value_box.set_text_size(18);
    
    card.add(&title_box);
    card.add(&value_box);
    card.end();
    
    card
}

