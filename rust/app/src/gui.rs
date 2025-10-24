//! GUI module for the Qallow unified dashboard.

use egui::{RichText, Ui};
use std::path::PathBuf;

/// Main application state for the Qallow GUI.
pub struct QallowApp {
    /// Current active tab
    active_tab: Tab,
    /// Terminal output buffer
    terminal_output: String,
    /// Metrics data
    metrics: MetricsData,
    /// Build selection
    selected_build: String,
    /// Telemetry CSV path
    telemetry_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Dashboard,
    Metrics,
    Terminal,
    AuditLog,
    ControlPanel,
}

#[derive(Debug, Clone, Default)]
struct MetricsData {
    cpu_usage: f32,
    memory_usage: f32,
    phase: u32,
    fidelity: f32,
    entanglement: f32,
}

impl Default for QallowApp {
    fn default() -> Self {
        Self {
            active_tab: Tab::Dashboard,
            terminal_output: String::new(),
            metrics: MetricsData::default(),
            selected_build: "CPU".to_string(),
            telemetry_path: PathBuf::from("data/logs/telemetry_stream.csv"),
        }
    }
}

impl QallowApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }

    fn render_tab_buttons(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            if ui
                .selectable_label(self.active_tab == Tab::Dashboard, "📊 Dashboard")
                .clicked()
            {
                self.active_tab = Tab::Dashboard;
            }
            if ui
                .selectable_label(self.active_tab == Tab::Metrics, "📈 Metrics")
                .clicked()
            {
                self.active_tab = Tab::Metrics;
            }
            if ui
                .selectable_label(self.active_tab == Tab::Terminal, "⌨️ Terminal")
                .clicked()
            {
                self.active_tab = Tab::Terminal;
            }
            if ui
                .selectable_label(self.active_tab == Tab::AuditLog, "📋 Audit Log")
                .clicked()
            {
                self.active_tab = Tab::AuditLog;
            }
            if ui
                .selectable_label(self.active_tab == Tab::ControlPanel, "⚙️ Control")
                .clicked()
            {
                self.active_tab = Tab::ControlPanel;
            }
        });
    }

    fn render_dashboard(&mut self, ui: &mut Ui) {
        ui.heading("🎯 Qallow Unified Dashboard");

        ui.separator();

        // Status cards
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new("Phase").strong());
                ui.label(format!("{}", self.metrics.phase));
            });

            ui.vertical(|ui| {
                ui.label(RichText::new("Fidelity").strong());
                ui.label(format!("{:.4}", self.metrics.fidelity));
            });

            ui.vertical(|ui| {
                ui.label(RichText::new("Entanglement").strong());
                ui.label(format!("{:.4}", self.metrics.entanglement));
            });
        });

        ui.separator();

        // Progress bars
        ui.label("CPU Usage:");
        ui.add(egui::ProgressBar::new(self.metrics.cpu_usage).show_percentage());

        ui.label("Memory Usage:");
        ui.add(egui::ProgressBar::new(self.metrics.memory_usage).show_percentage());
    }

    fn render_metrics(&mut self, ui: &mut Ui) {
        ui.heading("📈 Metrics");

        ui.label("Real-time metrics from telemetry stream:");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Phase:");
            ui.add(egui::Slider::new(&mut self.metrics.phase, 1..=16).text(""));
        });

        ui.horizontal(|ui| {
            ui.label("Fidelity:");
            ui.add(egui::Slider::new(&mut self.metrics.fidelity, 0.0..=1.0).text(""));
        });

        ui.horizontal(|ui| {
            ui.label("Entanglement:");
            ui.add(egui::Slider::new(&mut self.metrics.entanglement, 0.0..=1.0).text(""));
        });

        ui.horizontal(|ui| {
            ui.label("CPU Usage:");
            ui.add(egui::Slider::new(&mut self.metrics.cpu_usage, 0.0..=1.0).text(""));
        });

        ui.horizontal(|ui| {
            ui.label("Memory Usage:");
            ui.add(egui::Slider::new(&mut self.metrics.memory_usage, 0.0..=1.0).text(""));
        });
    }

    fn render_terminal(&mut self, ui: &mut Ui) {
        ui.heading("⌨️ Terminal");

        ui.label("Terminal output:");
        ui.text_edit_multiline(&mut self.terminal_output);

        ui.separator();

        if ui.button("Clear Terminal").clicked() {
            self.terminal_output.clear();
        }
    }

    fn render_audit_log(&mut self, ui: &mut Ui) {
        ui.heading("📋 Audit Log");

        ui.label("Audit trail of operations:");
        ui.separator();

        ui.text_edit_multiline(&mut self.terminal_output);
    }

    fn render_control_panel(&mut self, ui: &mut Ui) {
        ui.heading("⚙️ Control Panel");

        ui.separator();

        ui.label("Build Selection:");
        ui.horizontal(|ui| {
            if ui.button("CPU").clicked() {
                self.selected_build = "CPU".to_string();
            }
            if ui.button("CUDA").clicked() {
                self.selected_build = "CUDA".to_string();
            }
        });
        ui.label(format!("Selected: {}", self.selected_build));

        ui.separator();

        ui.label("Actions:");
        if ui.button("🚀 Run Qallow VM").clicked() {
            self.terminal_output
                .push_str("Starting Qallow VM...\n");
        }

        if ui.button("⏹️ Stop").clicked() {
            self.terminal_output.push_str("Stopping Qallow VM...\n");
        }

        if ui.button("🔄 Restart").clicked() {
            self.terminal_output
                .push_str("Restarting Qallow VM...\n");
        }

        ui.separator();

        ui.label("Telemetry Path:");
        ui.text_edit_singleline(
            &mut self
                .telemetry_path
                .to_string_lossy()
                .to_string(),
        );
    }
}

impl eframe::App for QallowApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_tab_buttons(ui);
            ui.separator();

            match self.active_tab {
                Tab::Dashboard => self.render_dashboard(ui),
                Tab::Metrics => self.render_metrics(ui),
                Tab::Terminal => self.render_terminal(ui),
                Tab::AuditLog => self.render_audit_log(ui),
                Tab::ControlPanel => self.render_control_panel(ui),
            }
        });
    }
}

