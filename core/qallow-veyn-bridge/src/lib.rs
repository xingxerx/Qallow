use futures_util::StreamExt;
use lmdb::{DatabaseFlags, Environment, EnvironmentFlags, Transaction, WriteFlags};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokio_tungstenite::connect_async;
use url::Url;

#[derive(Debug, Deserialize, Serialize)]
pub struct VeynEvent {
    pub metric: String,
    pub value: f64,
    pub timestamp: u64,
}

pub struct VeynBridge {
    env: Arc<Environment>,
}

impl VeynBridge {
    pub fn new(db_path: &Path) -> Result<Self, lmdb::Error> {
        std::fs::create_dir_all(db_path).ok();
        let env = Environment::new()
            .set_flags(EnvironmentFlags::NO_SUB_DIR | EnvironmentFlags::NO_TLS)
            .set_max_dbs(1)
            .open(db_path)?;

        Ok(Self { env: Arc::new(env) })
    }

    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let url = Url::parse("ws://localhost:7700/stream")?;
        let (ws_stream, _) = connect_async(url).await?;
        println!("Connected to VEYN stream");

        let (_, mut read) = ws_stream.split();

        let db = self.env.create_db(Some("veyn_metrics"), DatabaseFlags::empty())?;

        while let Some(msg) = read.next().await {
            let msg = msg?;
            if msg.is_text() {
                if let Ok(event) = serde_json::from_str::<VeynEvent>(msg.to_text()?) {
                    println!("Received VEYN event: {:?}", event);
                    
                    // Map metrics to ql_state fields
                    let (key, val) = match event.metric.as_str() {
                        "hrv" => ("energy", event.value),
                        "eeg_beta" => ("risk", event.value),
                        "spo2" => ("reward_mod", event.value),
                        "presence" => ("autonomy", event.value),
                        "sleep_stage" => ("sleep_stage", event.value),
                        _ => continue,
                    };

                    // Write to LMDB
                    let mut txn = self.env.begin_rw_txn()?;
                    let val_bytes = val.to_le_bytes();
                    txn.put(db, &key, &val_bytes, WriteFlags::empty())?;
                    txn.commit()?;

                    // Trigger phase snapshot for REM sleep
                    if event.metric == "sleep_stage" && event.value == 3.0 {
                        println!("REM sleep detected. Triggering Qallow phase snapshot...");
                        // Call HTTP endpoint or send IPC to daemon to freeze cognitive state
                        // reqwest::Client::new().post("http://localhost:5000/snapshot").send().await?;
                    }
                }
            }
        }

        Ok(())
    }
}
