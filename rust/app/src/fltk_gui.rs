//! Native GUI for the Qallow unified dashboard using FLTK.

use fltk::{prelude::*, *};
use fltk::enums::Color;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process::{Command, Child, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct MetricsData {
    pub tick: u32,
    pub orbital: f32,
    pub river: f32,
    pub mycelial: f32,
    pub global: f32,
    pub decoherence: f32,
    pub mode: String,
}

impl Default for MetricsData {
    fn default() -> Self {
        Self {
            tick: 0,
            orbital: 0.9782,
            river: 0.9781,
            mycelial: 0.9782,
            global: 0.9782,
            decoherence: 0.0,
            mode: "CPU".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Message {
    RunVM,
    StopVM,
    RestartVM,
    RefreshMetrics,
    UpdateMetrics,
}

/// Read the latest telemetry row from CSV file
fn read_latest_telemetry(path: &str) -> Option<MetricsData> {
    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Skip header
    lines.next()?;

    // Get last line
    let mut last_line = None;
    for line in lines {
        if let Ok(l) = line {
            last_line = Some(l);
        }
    }

    let line = last_line?;
    let parts: Vec<&str> = line.split(',').collect();

    if parts.len() < 7 {
        return None;
    }

    Some(MetricsData {
        tick: parts[0].parse().unwrap_or(0),
        orbital: parts[1].parse().unwrap_or(0.0),
        river: parts[2].parse().unwrap_or(0.0),
        mycelial: parts[3].parse().unwrap_or(0.0),
        global: parts[4].parse().unwrap_or(0.0),
        decoherence: parts[5].parse().unwrap_or(0.0),
        mode: parts[6].trim().to_string(),
    })
}

pub fn run_gui() -> anyhow::Result<()> {
    let app = app::App::default();
    let (tx, rx) = app::channel::<Message>();

    let mut wind = window::Window::default()
        .with_size(1600, 1000)
        .with_label("🚀 Qallow Unified Dashboard");

    wind.set_color(Color::from_hex(0x1e1e1e));

    // Main vertical flex layout
    let main_flex = group::Flex::default()
        .with_size(1600, 1000)
        .column();

    // ===== HEADER =====
    let header = frame::Frame::default()
        .with_size(1600, 70);
    {
        let mut h = header;
        h.set_label("🚀 QALLOW UNIFIED DASHBOARD");
        h.set_label_size(24);
        h.set_color(Color::from_hex(0x2d2d2d));
        h.set_label_color(Color::from_hex(0x00ff88));
    }

    // ===== TOP SECTION: METRICS + BUTTONS =====
    let top_flex = group::Flex::default()
        .with_size(1600, 300)
        .row();

    // Metrics grid (left side)
    let metrics_flex = group::Flex::default()
        .with_size(1200, 300)
        .row();

    let mut tick_value = frame::Frame::default();
    let mut global_value = frame::Frame::default();
    let mut orbital_value = frame::Frame::default();
    let mut decoherence_value = frame::Frame::default();

    create_metric_card(&mut tick_value, "📍 Tick", "0");
    create_metric_card(&mut global_value, "🌍 Global", "0.9782");
    create_metric_card(&mut orbital_value, "🛰️ Orbital", "0.9782");
    create_metric_card(&mut decoherence_value, "⚡ Decoherence", "0.0000");

    metrics_flex.end();

    // Buttons (right side)
    let button_col = group::Flex::default()
        .with_size(400, 300)
        .column();

    let mut run_btn = button::Button::default()
        .with_size(380, 60)
        .with_label("▶️  RUN VM");
    style_button(&mut run_btn, 0x00ff88);

    let mut stop_btn = button::Button::default()
        .with_size(380, 60)
        .with_label("⏹️  STOP VM");
    style_button(&mut stop_btn, 0xff6464);

    let mut restart_btn = button::Button::default()
        .with_size(380, 60)
        .with_label("🔄  RESTART");
    style_button(&mut restart_btn, 0xffaa00);

    let mut refresh_btn = button::Button::default()
        .with_size(380, 60)
        .with_label("🔃  REFRESH");
    style_button(&mut refresh_btn, 0x00aaff);

    button_col.end();
    top_flex.end();

    // ===== STATUS BAR =====
    let mut status = frame::Frame::default()
        .with_size(1600, 40);
    status.set_label("✅ Status: Ready");
    status.set_label_size(14);
    status.set_color(Color::from_hex(0x2d2d2d));
    status.set_label_color(Color::from_hex(0x00ff88));

    // ===== TERMINAL OUTPUT =====
    let mut terminal_label = frame::Frame::default()
        .with_size(1600, 30)
        .with_label("📋 Terminal Output (Selectable)");
    terminal_label.set_label_size(12);
    terminal_label.set_color(Color::from_hex(0x1a1a1a));
    terminal_label.set_label_color(Color::from_hex(0x888888));

    let mut terminal = text::TextEditor::default()
        .with_size(1600, 500);
    terminal.set_buffer(text::TextBuffer::default());
    terminal.set_color(Color::from_hex(0x0d0d0d));
    terminal.set_text_color(Color::from_hex(0x00ff88));
    terminal.set_cursor_color(Color::from_hex(0x00ff88));
    terminal.set_linenumber_width(40);
    terminal.set_linenumber_bgcolor(Color::from_hex(0x1a1a1a));
    terminal.set_linenumber_fgcolor(Color::from_hex(0x666666));

    main_flex.end();
    wind.end();
    wind.show();

    // Shared state
    let metrics = Arc::new(Mutex::new(MetricsData::default()));
    let vm_process: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));
    let mut terminal_buf = terminal.buffer().unwrap();

    // Button event handlers
    {
        let tx = tx.clone();
        run_btn.emit(tx, Message::RunVM);
    }
    {
        let tx = tx.clone();
        stop_btn.emit(tx, Message::StopVM);
    }
    {
        let tx = tx.clone();
        restart_btn.emit(tx, Message::RestartVM);
    }
    {
        let tx = tx.clone();
        refresh_btn.emit(tx, Message::RefreshMetrics);
    }

    // Background thread to update metrics every 500ms
    let metrics_clone = metrics.clone();
    let tx_clone = tx.clone();
    thread::spawn(move || loop {
        thread::sleep(Duration::from_millis(500));
        if let Some(m) = read_latest_telemetry("data/logs/telemetry_stream.csv") {
            *metrics_clone.lock().unwrap() = m;
            tx_clone.send(Message::UpdateMetrics);
        }
    });

    // Main event loop
    while app.wait() {
        if let Some(msg) = rx.recv() {
            match msg {
                Message::RunVM => {
                    let msg_text = "▶️ VM Started\n";
                    terminal_buf.append(msg_text);
                    status.set_label("✅ Status: VM Running");

                    let mut proc = vm_process.lock().unwrap();
                    if proc.is_none() {
                        if let Ok(child) = Command::new("./build/qallow")
                            .arg("phase")
                            .arg("14")
                            .stdout(Stdio::piped())
                            .stderr(Stdio::piped())
                            .spawn()
                        {
                            *proc = Some(child);
                        }
                    }
                }
                Message::StopVM => {
                    let msg_text = "⏹️ VM Stopped\n";
                    terminal_buf.append(msg_text);
                    status.set_label("⏹️ Status: VM Stopped");

                    let mut proc = vm_process.lock().unwrap();
                    if let Some(mut child) = proc.take() {
                        let _ = child.kill();
                    }
                }
                Message::RestartVM => {
                    let msg_text = "🔄 VM Restarted\n";
                    terminal_buf.append(msg_text);
                    status.set_label("🔄 Status: VM Restarting...");

                    let mut proc = vm_process.lock().unwrap();
                    if let Some(mut child) = proc.take() {
                        let _ = child.kill();
                    }
                    if let Ok(child) = Command::new("./build/qallow")
                        .arg("phase")
                        .arg("14")
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .spawn()
                    {
                        *proc = Some(child);
                    }
                }
                Message::RefreshMetrics => {
                    let msg_text = "🔃 Metrics Refreshed\n";
                    terminal_buf.append(msg_text);
                    status.set_label("✅ Status: Metrics Updated");
                }
                Message::UpdateMetrics => {
                    let m = metrics.lock().unwrap();
                    tick_value.set_label(&format!("{}", m.tick));
                    global_value.set_label(&format!("{:.4}", m.global));
                    orbital_value.set_label(&format!("{:.4}", m.orbital));
                    decoherence_value.set_label(&format!("{:.4}", m.decoherence));
                }
            }
        }
    }

    Ok(())
}

fn create_metric_card(value: &mut frame::Frame, label: &str, default: &str) {
    let card = group::Group::default().with_size(240, 300);
    let mut label_frame = frame::Frame::default()
        .with_size(240, 40)
        .with_label(label);
    label_frame.set_label_size(14);
    label_frame.set_color(Color::from_hex(0x2d2d2d));
    label_frame.set_label_color(Color::from_hex(0x00ff88));

    *value = frame::Frame::default()
        .with_size(240, 100)
        .with_label(default);
    value.set_label_size(42);
    value.set_color(Color::from_hex(0x1a1a1a));
    value.set_label_color(Color::from_hex(0x00ff88));
    card.end();
}

fn style_button(btn: &mut button::Button, color_hex: u32) {
    btn.set_label_size(14);
    btn.set_color(Color::from_hex(color_hex));
    btn.set_label_color(Color::from_hex(0x000000));
    btn.set_selection_color(Color::from_hex(0xffffff));
}

