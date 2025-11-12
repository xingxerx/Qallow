use serde::{Deserialize, Serialize};
use serde_json::json;

// ...existing code...
#[derive(Clone)]
pub struct ApiClient;

impl ApiClient {
    pub fn new(_base_url: &str) -> Self {
        ApiClient
    }
// ...existing code...

    pub async fn get_metrics(&self) -> Result<serde_json::Value, String> {
        // Placeholder for API call to web dashboard
        Ok(json!({
            "overlay_stability": {
                "orbital": 0.9575,
                "river": 0.9684,
                "mycelial": 0.9984,
                "global": 0.9575
            },
            "ethics_score": {
                "safety": 0.85,
                "clarity": 0.88,
                "human": 0.82
            },
            "coherence": 0.9993,
            "gpu_memory": 8.5,
            "cpu_memory": 4.2
        }))
    }

    pub async fn get_audit_logs(&self) -> Result<Vec<serde_json::Value>, String> {
        // Placeholder for API call to get audit logs
        Ok(vec![
            json!({
                "timestamp": "2025-10-23T17:03:00Z",
                "level": "INFO",
                "component": "SYSTEM",
                "message": "Qallow VM initialized"
            }),
            json!({
                "timestamp": "2025-10-23T17:03:01Z",
                "level": "SUCCESS",
                "component": "CUDA",
                "message": "CUDA acceleration enabled"
            }),
        ])
    }

    pub async fn start_phase(&self, _phase: u32, _ticks: u32) -> Result<(), String> {
        // Placeholder for API call to start a phase
        Ok(())
    }

    pub async fn stop_phase(&self) -> Result<(), String> {
        // Placeholder for API call to stop current phase
        Ok(())
    }

    pub async fn export_metrics(&self, format: &str) -> Result<String, String> {
        // Placeholder for API call to export metrics
        Ok(format!("Metrics exported as {}", format))
    }

    pub async fn chat(&self, message: &str) -> Result<String, reqwest::Error> {
        let client = reqwest::Client::new();
        let request = ChatRequest {
            message: message.to_string(),
        };

        let res = client
            .post("http://127.0.0.1:8008/chat")
            .json(&request)
            .send()
            .await?;

        let chat_response: ChatResponse = res.json().await?;
        Ok(chat_response.reply)
    }
}

impl Default for ApiClient {
    fn default() -> Self {
        Self::new("http://localhost:5000")
    }
}

#[derive(Serialize, Deserialize)]
struct ChatRequest {
    message: String,
}

#[derive(Serialize, Deserialize)]
struct ChatResponse {
    reply: String,
}
