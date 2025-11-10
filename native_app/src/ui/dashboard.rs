use crate::models::AppState;
use fltk::enums::Color;
use fltk::{prelude::*, *};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Dashboard {
    pub uptime_value: text::TextDisplay,
}

pub fn create_dashboard(tabs: &mut group::Tabs, _state: Arc<Mutex<AppState>>) -> Dashboard {
    let mut group = group::Group::default().with_label("📊 Dashboard");
    group.set_color(Color::from_hex(0x0a0e27));
    group.begin();

    let mut flex = group::Flex::default().with_size(1450, 950).column();
    flex.set_color(Color::from_hex(0x0a0e27));
    flex.begin();

    // Title
    let mut title = text::TextDisplay::default().with_size(1450, 40);
    title.set_buffer(text::TextBuffer::default());
    title.buffer().unwrap().set_text("Real-Time System Metrics");
    title.set_text_color(Color::from_hex(0x00d4ff));
    flex.add(&title);
    flex.fixed(&title, 40);

    // Metrics grid
    let mut metrics_flex = group::Flex::default().with_size(1450, 400).row();
    metrics_flex.set_color(Color::from_hex(0x0a0e27));

    // Overlay Stability
    create_metric_card(
        &mut metrics_flex,
        "Overlay Stability",
        "Orbital: 0.9575\nRiver: 0.9684\nMycelial: 0.9984\nGlobal: 0.9575",
    );

    // Ethics Monitoring
    create_metric_card(
        &mut metrics_flex,
        "Ethics Monitoring",
        "Safety: 0.85\nClarity: 0.88\nHuman: 0.82\nScore: 2.39",
    );

    // Coherence Tracking
    create_metric_card(
        &mut metrics_flex,
        "Coherence Tracking",
        "Coherence: 0.9993\nDecoherence: 0.0007\nStability: Excellent",
    );

    // System Status
    create_metric_card(
        &mut metrics_flex,
        "System Status",
        "GPU: NVIDIA RTX 5080
CUDA: 12.0
Memory: 15.9 GB
Uptime: 2h 34m",
    );
    let uptime_value = text::TextDisplay::default();

    metrics_flex.end();
    flex.add(&metrics_flex);
    flex.fixed(&metrics_flex, 400);

    // Status indicators
    let mut status_flex = group::Flex::default().with_size(1450, 200).row();
    status_flex.set_color(Color::from_hex(0x0a0e27));

    create_status_indicator(
        &mut status_flex,
        "Phase 13",
        "✓ Complete",
        Color::from_hex(0x00ff64),
    );
    create_status_indicator(
        &mut status_flex,
        "Phase 14",
        "✓ Running",
        Color::from_hex(0x00ff64),
    );
    create_status_indicator(
        &mut status_flex,
        "Phase 15",
        "○ Idle",
        Color::from_hex(0xffaa00),
    );
    create_status_indicator(
        &mut status_flex,
        "GPU",
        "✓ Enabled",
        Color::from_hex(0x00ff64),
    );

    status_flex.end();
    flex.add(&status_flex);
    flex.fixed(&status_flex, 200);

    // Progress bars
    let mut progress_flex = group::Flex::default().with_size(1450, 200).column();
    progress_flex.set_color(Color::from_hex(0x0a0e27));

    create_progress_bar(&mut progress_flex, "Execution Progress", 0.75);
    create_progress_bar(&mut progress_flex, "Coherence Level", 0.9993);
    create_progress_bar(&mut progress_flex, "Ethics Compliance", 0.95);

    progress_flex.end();
    flex.add(&progress_flex);

    flex.end();
    group.end();
    tabs.add(&group);
    Dashboard { uptime_value }
}

fn create_metric_card(flex: &mut group::Flex, title: &str, content: &str) {
    let mut card = group::Flex::default().with_size(350, 400).column();
    card.set_color(Color::from_hex(0x1a1f3a));

    let mut card_title = text::TextDisplay::default().with_size(350, 40);
    card_title.set_buffer(text::TextBuffer::default());
    card_title.buffer().unwrap().set_text(title);
    card_title.set_text_color(Color::from_hex(0x00d4ff));

    let mut card_content = text::TextDisplay::default().with_size(350, 360);
    card_content.set_buffer(text::TextBuffer::default());
    card_content.buffer().unwrap().set_text(content);
    card_content.set_text_color(Color::White);

    card.end();
    flex.add(&card);
}

fn create_status_indicator(flex: &mut group::Flex, label: &str, status: &str, color: Color) {
    let mut indicator = group::Flex::default().with_size(350, 200).column();
    indicator.set_color(Color::from_hex(0x1a1f3a));

    let mut label_text = text::TextDisplay::default().with_size(350, 50);
    label_text.set_buffer(text::TextBuffer::default());
    label_text.buffer().unwrap().set_text(label);
    label_text.set_text_color(Color::from_hex(0x00d4ff));

    let mut status_text = text::TextDisplay::default().with_size(350, 150);
    status_text.set_buffer(text::TextBuffer::default());
    status_text.buffer().unwrap().set_text(status);
    status_text.set_text_color(color);

    indicator.end();
    flex.add(&indicator);
}

fn create_progress_bar(flex: &mut group::Flex, label: &str, value: f64) {
    let mut bar_flex = group::Flex::default().with_size(1450, 50).row();
    bar_flex.set_color(Color::from_hex(0x0a0e27));

    let mut label_text = text::TextDisplay::default().with_size(200, 50);
    label_text.set_buffer(text::TextBuffer::default());
    label_text.buffer().unwrap().set_text(label);
    label_text.set_text_color(Color::from_hex(0x00d4ff));

    let mut progress = valuator::Slider::default().with_size(1200, 50);
    progress.set_value(value);
    progress.set_color(Color::from_hex(0x1a1f3a));
    progress.set_selection_color(Color::from_hex(0x00d4ff));

    let mut percent_text = text::TextDisplay::default().with_size(50, 50);
    percent_text.set_buffer(text::TextBuffer::default());
    percent_text
        .buffer()
        .unwrap()
        .set_text(&format!("{:.1}%", value * 100.0));
    percent_text.set_text_color(Color::from_hex(0x00ff64));

    bar_flex.end();
    flex.add(&bar_flex);
}
