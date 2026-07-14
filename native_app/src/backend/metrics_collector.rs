//! Thin helper that polls the backend REST API for metrics on demand.

use crate::backend::api_client::ApiClient;

pub struct MetricsCollector {
    client: ApiClient,
}

impl MetricsCollector {
    pub fn new(client: ApiClient) -> Self {
        Self { client }
    }

    pub async fn collect(&self) -> Result<serde_json::Value, String> {
        self.client.get_metrics().await
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new(ApiClient::default())
    }
}
