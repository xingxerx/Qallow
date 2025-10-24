use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppState {
    pub vm_running: bool,
    pub selected_build: BuildType,
    pub selected_phase: Phase,
    pub terminal_output: VecDeque<TerminalLine>,
    pub metrics: SystemMetrics,
    pub audit_logs: VecDeque<AuditLog>,
    pub phase_config: PhaseConfig,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum BuildType {
    CPU,
    CUDA,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Phase {
    Phase13,
    Phase14,
    Phase15,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalLine {
    pub timestamp: DateTime<Utc>,
    pub content: String,
    pub line_type: LineType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum LineType {
    Output,
    Error,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub overlay_stability: OverlayStability,
    pub ethics_score: EthicsScore,
    pub coherence: f64,
    pub gpu_memory: f64,
    pub cpu_memory: f64,
    pub uptime_seconds: u64,
    pub last_update: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlayStability {
    pub orbital: f64,
    pub river: f64,
    pub mycelial: f64,
    pub global: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicsScore {
    pub safety: f64,
    pub clarity: f64,
    pub human: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLog {
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub component: String,
    pub message: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum LogLevel {
    Info,
    Success,
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseConfig {
    pub ticks: u32,
    pub target_fidelity: f64,
    pub epsilon: f64,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            vm_running: false,
            selected_build: BuildType::CPU,
            selected_phase: Phase::Phase14,
            terminal_output: VecDeque::with_capacity(1000),
            metrics: SystemMetrics {
                overlay_stability: OverlayStability {
                    orbital: 0.95,
                    river: 0.94,
                    mycelial: 0.96,
                    global: 0.93,
                },
                ethics_score: EthicsScore {
                    safety: 0.85,
                    clarity: 0.88,
                    human: 0.82,
                },
                coherence: 0.9993,
                gpu_memory: 8.5,
                cpu_memory: 4.2,
                uptime_seconds: 0,
                last_update: Utc::now(),
            },
            audit_logs: VecDeque::with_capacity(500),
            phase_config: PhaseConfig {
                ticks: 1000,
                target_fidelity: 0.981,
                epsilon: 5e-6,
            },
        }
    }

    pub fn add_terminal_line(&mut self, content: String, line_type: LineType) {
        let line = TerminalLine {
            timestamp: Utc::now(),
            content,
            line_type,
        };
        self.terminal_output.push_back(line);
        if self.terminal_output.len() > 1000 {
            self.terminal_output.pop_front();
        }
    }

    pub fn add_audit_log(&mut self, level: LogLevel, component: String, message: String) {
        let log = AuditLog {
            timestamp: Utc::now(),
            level,
            component,
            message,
        };
        self.audit_logs.push_back(log);
        if self.audit_logs.len() > 500 {
            self.audit_logs.pop_front();
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

