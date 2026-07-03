use qallow_veyn_bridge::VeynBridge;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = Path::new("veyn_state.lmdb");
    let bridge = VeynBridge::new(db_path)?;
    bridge.run().await
}
