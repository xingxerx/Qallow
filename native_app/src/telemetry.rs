//! Telemetry point helpers. Currently the UI reads telemetry directly from
//! `AppState`; this module is a placeholder for future background polling of
//! the backend's `/metrics` endpoint into `AppState::telemetry`.

use crate::models::TelemetryPoint;

/// Builds a telemetry point from raw values, timestamping it as "now".
pub fn make_point(step: u32, reward: f64, energy: f64, risk: f64) -> TelemetryPoint {
    TelemetryPoint {
        step,
        reward,
        energy,
        risk,
        timestamp: chrono::Utc::now(),
    }
}
