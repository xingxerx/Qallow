use fltk::{prelude::*, *};
use fltk::enums::Color;
use std::sync::{Arc, Mutex};
use crate::models::AppState;

pub fn create_metrics(tabs: &mut group::Tabs, _state: Arc<Mutex<AppState>>) {
    let mut group = group::Group::default()
        .with_label("📈 Metrics");
    group.set_color(Color::from_hex(0x0a0e27));

    let mut flex = group::Flex::default()
        .with_size(1450, 950)
        .column();
    flex.set_color(Color::from_hex(0x0a0e27));

    // Title
    let mut title = text::TextDisplay::default()
        .with_size(1450, 40);
    title.set_buffer(text::TextBuffer::default());
    title.buffer().unwrap().set_text("Performance Metrics");
    title.set_text_color(Color::from_hex(0x00d4ff));

    // Metrics grid
    let mut metrics_grid = group::Flex::default()
        .with_size(1450, 400)
        .row();
    metrics_grid.set_color(Color::from_hex(0x0a0e27));

    create_metric_box(&mut metrics_grid, "Phase Status", "Phase 14\nRunning\n201/600 ticks");
    create_metric_box(&mut metrics_grid, "Memory Usage", "GPU: 8.5 GB\nCPU: 4.2 GB\nTotal: 12.7 GB");
    create_metric_box(&mut metrics_grid, "Network Stats", "Packets: 1.2M\nLatency: 0.5ms\nBandwidth: 850 Mbps");
    create_metric_box(&mut metrics_grid, "Performance", "Throughput: 2.1M ops/s\nEfficiency: 94.2%\nTemp: 65°C");

    metrics_grid.end();

    // Detailed metrics table
    let mut table_flex = group::Flex::default()
        .with_size(1450, 450)
        .column();
    table_flex.set_color(Color::from_hex(0x0a0e27));

    let mut table_title = text::TextDisplay::default()
        .with_size(1450, 40);
    table_title.set_buffer(text::TextBuffer::default());
    table_title.buffer().unwrap().set_text("Detailed Metrics");
    table_title.set_text_color(Color::from_hex(0x00d4ff));

    let mut table = text::TextDisplay::default()
        .with_size(1450, 410);
    table.set_buffer(text::TextBuffer::default());
    let metrics_text = r#"Metric                          Value           Status
─────────────────────────────────────────────────────
Coherence Level                 0.9993          ✓ Excellent
Fidelity                        0.9810          ✓ Target Reached
Ethics Score                    2.39            ✓ Pass
Overlay Stability (Orbital)     0.9575          ✓ Good
Overlay Stability (River)       0.9684          ✓ Good
Overlay Stability (Mycelial)    0.9984          ✓ Excellent
Overlay Stability (Global)      0.9575          ✓ Good
GPU Utilization                 87.3%           ✓ High
CPU Utilization                 42.1%           ○ Normal
Memory Bandwidth                850 Mbps        ✓ Good
Quantum Gate Fidelity           0.9950          ✓ Excellent
Circuit Depth                   256 gates       ○ Normal
Execution Time (avg)            2.1 ms          ✓ Fast
Decoherence Rate                0.0007          ✓ Low"#;
    table.buffer().unwrap().set_text(metrics_text);
    table.set_text_color(Color::from_hex(0x00ff64));

    table_flex.end();

    // Control buttons
    let mut button_flex = group::Flex::default()
        .with_size(1450, 50)
        .row();
    button_flex.set_color(Color::from_hex(0x0a0e27));

    let mut refresh_btn = button::Button::default()
        .with_size(100, 50)
        .with_label("🔄 Refresh");
    refresh_btn.set_color(Color::from_hex(0x1a1f3a));
    refresh_btn.set_label_color(Color::from_hex(0x00d4ff));

    let mut export_btn = button::Button::default()
        .with_size(100, 50)
        .with_label("📥 Export");
    export_btn.set_color(Color::from_hex(0x1a1f3a));
    export_btn.set_label_color(Color::from_hex(0x00d4ff));

    button_flex.end();

    flex.end();
    group.end();
    tabs.add(&group);
}

fn create_metric_box(flex: &mut group::Flex, title: &str, content: &str) {
    let mut box_flex = group::Flex::default()
        .with_size(350, 400)
        .column();
    box_flex.set_color(Color::from_hex(0x1a1f3a));

    let mut box_title = text::TextDisplay::default()
        .with_size(350, 50);
    box_title.set_buffer(text::TextBuffer::default());
    box_title.buffer().unwrap().set_text(title);
    box_title.set_text_color(Color::from_hex(0x00d4ff));

    let mut box_content = text::TextDisplay::default()
        .with_size(350, 350);
    box_content.set_buffer(text::TextBuffer::default());
    box_content.buffer().unwrap().set_text(content);
    box_content.set_text_color(Color::White);

    box_flex.end();
    flex.add(&box_flex);
}

