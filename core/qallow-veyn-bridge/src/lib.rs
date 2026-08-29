use futures_util::StreamExt;
use lmdb::{DatabaseFlags, Environment, EnvironmentFlags, Transaction, WriteFlags};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokio_tungstenite::connect_async;
use url::Url;

use async_trait::async_trait;

#[derive(Debug, Deserialize, Serialize)]
pub struct VeynEvent {
    pub metric: String,
    pub value: f64,
    pub timestamp: u64,
}

#[async_trait]
pub trait SnapshotNotifier: Send + Sync {
    async fn trigger_snapshot(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

pub struct HttpSnapshotNotifier {
    base_url: String,
}

impl HttpSnapshotNotifier {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
        }
    }
}

#[async_trait]
impl SnapshotNotifier for HttpSnapshotNotifier {
    async fn trigger_snapshot(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Minimal HTTP GET using tokio TcpStream to avoid heavy deps
        let base = Url::parse(&self.base_url)?;
        let host = base.host_str().ok_or("missing host")?;
        let port = base.port().unwrap_or_else(|| if base.scheme() == "https" { 443 } else { 80 });
        let addr = format!("{}:{}", host, port);

        let mut stream = tokio::net::TcpStream::connect(addr).await?;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let req = format!(
            "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
            "/export", host
        );
        stream.write_all(req.as_bytes()).await?;
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).await?;
        Ok(())
    }
}

pub struct VeynBridge {
    env: Arc<Environment>,
    notifier: Arc<dyn SnapshotNotifier>,
}

impl VeynBridge {
    pub fn new(db_path: &Path) -> Result<Self, lmdb::Error> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let env = Environment::new()
            .set_flags(EnvironmentFlags::NO_SUB_DIR | EnvironmentFlags::NO_TLS)
            .set_max_dbs(1)
            .open(db_path)?;

        Ok(Self {
            env: Arc::new(env),
            notifier: Arc::new(HttpSnapshotNotifier::new("http://localhost:5000")),
        })
    }

    pub fn new_with_notifier(
        db_path: &Path,
        notifier: Arc<dyn SnapshotNotifier>,
    ) -> Result<Self, lmdb::Error> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let env = Environment::new()
            .set_flags(EnvironmentFlags::NO_SUB_DIR | EnvironmentFlags::NO_TLS)
            .set_max_dbs(1)
            .open(db_path)?;

        Ok(Self {
            env: Arc::new(env),
            notifier,
        })
    }

    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let url = Url::parse("ws://localhost:7700/stream")?;
        let (ws_stream, _) = connect_async(url).await?;
        println!("Connected to VEYN stream");

        let (_, mut read) = ws_stream.split();

        while let Some(msg) = read.next().await {
            let msg = msg?;
            if msg.is_text() {
                if let Ok(event) = serde_json::from_str::<VeynEvent>(msg.to_text()?) {
                    println!("Received VEYN event: {:?}", event);
                    self.process_event(event).await?;
                }
            }
        }

        Ok(())
    }

    pub async fn process_event(
        &self,
        event: VeynEvent,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Map metrics to ql_state fields
        if let Some((key, val)) = map_metric_to_state_key(&event.metric, event.value) {
            // Write to LMDB
            let db = self
                .env
                .create_db(Some("veyn_metrics"), DatabaseFlags::empty())?;
            let mut txn = self.env.begin_rw_txn()?;
            let val_bytes = val.to_le_bytes();
            txn.put(db, &key, &val_bytes, WriteFlags::empty())?;
            txn.commit()?;
        }

        // Snapshot triggers
        if should_trigger_snapshot(&event) {
            println!("REM detected. Triggering Qallow snapshot via /export...");
            if let Err(e) = self.notifier.trigger_snapshot().await {
                eprintln!("Snapshot trigger failed: {}", e);
            }
        }

        Ok(())
    }
}

fn map_metric_to_state_key(metric: &str, value: f64) -> Option<(&'static str, f64)> {
    match metric {
        "hrv" => Some(("energy", value)),
        "eeg_beta" => Some(("risk", value)),
        "spo2" => Some(("reward_mod", value)),
        "presence" => Some(("autonomy", value)),
        "sleep_stage" => Some(("sleep_stage", value)),
        _ => None,
    }
}

fn should_trigger_snapshot(event: &VeynEvent) -> bool {
    // Legacy path: sleep_stage == 3.0
    if event.metric == "sleep_stage" && event.value == 3.0 {
        return true;
    }
    // New discrete cues
    if (event.metric == "rem_detected" || event.metric == "veyn.rem_event")
        && event.value == 1.0
    {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct MockNotifier {
        calls: AtomicUsize,
    }

    impl MockNotifier {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
        fn count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl SnapshotNotifier for MockNotifier {
        async fn trigger_snapshot(
            &self,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn test_db_path(name: &str) -> std::path::PathBuf {
        let mut p = std::path::PathBuf::from("target/test-data");
        std::fs::create_dir_all(&p).ok();
        p.push(format!("{}.lmdb", name));
        p
    }

    #[tokio::test]
    async fn triggers_on_rem_detected_value_one() {
        let notifier = Arc::new(MockNotifier::new());
        let db_path = test_db_path("rem_detected");
        let bridge = VeynBridge::new_with_notifier(&db_path, notifier.clone()).unwrap();

        bridge
            .process_event(VeynEvent {
                metric: "rem_detected".to_string(),
                value: 1.0,
                timestamp: 0,
            })
            .await
            .unwrap();

        assert_eq!(notifier.count(), 1);
    }

    #[tokio::test]
    async fn triggers_on_veyn_rem_event_value_one() {
        let notifier = Arc::new(MockNotifier::new());
        let db_path = test_db_path("veyn_rem_event");
        let bridge = VeynBridge::new_with_notifier(&db_path, notifier.clone()).unwrap();

        bridge
            .process_event(VeynEvent {
                metric: "veyn.rem_event".to_string(),
                value: 1.0,
                timestamp: 0,
            })
            .await
            .unwrap();

        assert_eq!(notifier.count(), 1);
    }

    #[tokio::test]
    async fn triggers_on_sleep_stage_legacy() {
        let notifier = Arc::new(MockNotifier::new());
        let db_path = test_db_path("sleep_stage_legacy");
        let bridge = VeynBridge::new_with_notifier(&db_path, notifier.clone()).unwrap();

        bridge
            .process_event(VeynEvent {
                metric: "sleep_stage".to_string(),
                value: 3.0,
                timestamp: 0,
            })
            .await
            .unwrap();

        assert_eq!(notifier.count(), 1);
    }

    #[tokio::test]
    async fn does_not_trigger_on_rem_detected_zero() {
        let notifier = Arc::new(MockNotifier::new());
        let db_path = test_db_path("rem_detected_zero");
        let bridge = VeynBridge::new_with_notifier(&db_path, notifier.clone()).unwrap();

        bridge
            .process_event(VeynEvent {
                metric: "rem_detected".to_string(),
                value: 0.0,
                timestamp: 0,
            })
            .await
            .unwrap();

        assert_eq!(notifier.count(), 0);
    }
}
