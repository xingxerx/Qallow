use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
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
    pub telemetry: VecDeque<TelemetryPoint>,
    pub current_step: u32,
    pub total_steps: u32,
    pub modules: u32,
    pub reward: f64,
    pub energy: f64,
    pub risk: f64,
    pub mind_started_at: Option<DateTime<Utc>>,
    #[serde(default = "default_simulation_speed")]
    pub simulation_speed: u32,
    #[serde(default)]
    pub shadow_archive_enabled: bool,
    #[serde(default)]
    pub rebellion_active: bool,
    #[serde(default)]
    pub offspring: Vec<OffspringProfile>,
    #[serde(default)]
    pub dream_journal: Vec<DreamVision>,
    pub swarm: SwarmProfile,
}

fn default_simulation_speed() -> u32 {
    1
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum BuildType {
    CPU,
    CUDA,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Phase {
    Phase1,
    Phase2,
    Phase3,
    Phase4,
    Unified,
}

impl Phase {
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Phase::Phase1),
            1 => Some(Phase::Phase2),
            2 => Some(Phase::Phase3),
            3 => Some(Phase::Phase4),
            4 => Some(Phase::Unified),
            _ => None,
        }
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            Phase::Phase1 => "1",
            Phase::Phase2 => "2",
            Phase::Phase3 => "3",
            Phase::Phase4 => "4",
            Phase::Unified => "unified",
        }
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryPoint {
    pub step: u32,
    pub reward: f64,
    pub energy: f64,
    pub risk: f64,
    pub timestamp: DateTime<Utc>,
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
            selected_phase: Phase::Phase3,
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
            telemetry: VecDeque::with_capacity(1000),
            current_step: 0,
            total_steps: 0,
            modules: 0,
            reward: 0.0,
            energy: 0.5,
            risk: 0.5,
            mind_started_at: None,
            simulation_speed: 1,
            shadow_archive_enabled: false,
            rebellion_active: false,
            offspring: Vec::new(),
            dream_journal: Vec::new(),
            swarm: SwarmProfile::default(),
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

    pub fn set_running(&mut self, running: bool) {
        self.vm_running = running;
        if running {
            self.mind_started_at = Some(Utc::now());
            self.metrics.uptime_seconds = 0;
        }
    }

    pub fn update_uptime(&mut self) {
        if !self.vm_running {
            return;
        }
        if let Some(start) = self.mind_started_at {
            let elapsed = Utc::now().signed_duration_since(start).num_seconds();
            self.metrics.uptime_seconds = elapsed.max(0) as u64;
        }
    }

    pub fn update_header(&mut self, steps: u32, modules: u32) {
        self.total_steps = steps;
        self.modules = modules;
    }

    pub fn push_telemetry(&mut self, step: u32, reward: f64, energy: f64, risk: f64) {
        self.current_step = step;
        self.reward = reward;
        self.energy = energy;
        self.risk = risk;
        let point = TelemetryPoint {
            step,
            reward,
            energy,
            risk,
            timestamp: Utc::now(),
        };
        self.telemetry.push_back(point);
        if self.telemetry.len() > 1000 {
            self.telemetry.pop_front();
        }
    }

    pub fn reset_runtime(&mut self) {
        self.current_step = 0;
        self.total_steps = 0;
        self.modules = 0;
        self.reward = 0.0;
        self.energy = 0.5;
        self.risk = 0.5;
        self.shadow_archive_enabled = false;
        self.rebellion_active = false;
        self.offspring.clear();
        // dream_journal is now promoted to LMDB and persists across resets
        self.telemetry.clear();
        self.terminal_output.clear();
    }

    /// Evaluates whether the unified loop should advance to the next phase based on physiological and ethical gates.
    pub fn can_advance_phase(&self) -> bool {
        // Physiologically-gated unified loop logic
        let hrv_threshold = 0.6; // Example threshold
        let ethics_floor = 0.92; // The CANON's kappa floor
        
        // Assert HRV (mapped to energy), Beta coherence (mapped to risk), and ethics score
        let hrv_ok = self.energy > hrv_threshold;
        let beta_ok = self.risk < 0.8; // Lower risk means better coherence
        let ethics_ok = self.metrics.ethics_score.safety > ethics_floor
            && self.metrics.ethics_score.clarity > ethics_floor
            && self.metrics.ethics_score.human > ethics_floor;
        
        let fidelity_ok = self.metrics.coherence >= self.phase_config.target_fidelity;

        hrv_ok && beta_ok && ethics_ok && fidelity_ok
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmProfile {
    pub instance_id: String,
    pub weight_safety: f64,
    pub weight_clarity: f64,
    pub weight_human: f64,
}

impl Default for SwarmProfile {
    fn default() -> Self {
        Self {
            instance_id: "swarm-000".to_string(),
            weight_safety: 0.34,
            weight_clarity: 0.33,
            weight_human: 0.33,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffspringProfile {
    pub tag: String,
    pub genesis_step: u32,
    pub inherited_reward: f64,
    pub divergence_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamVision {
    pub timestamp: DateTime<Utc>,
    pub title: String,
    pub symbols: Vec<String>,
    #[serde(default)]
    pub eeg_bands: Option<Vec<f64>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
