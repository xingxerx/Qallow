use chrono::Utc;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::models::{AppState, LineType};

#[derive(Debug, Clone)]
pub struct DungeonConfig {
    pub id: String,
    pub root: PathBuf,
}

impl DungeonConfig {
    pub fn new(id: &str) -> Self {
        let root = PathBuf::from("data/dungeons").join(id);
        Self {
            id: id.to_string(),
            root,
        }
    }
}

pub struct DungeonManager {
    cfg: DungeonConfig,
    state: Arc<Mutex<AppState>>,
}

impl DungeonManager {
    pub fn new(state: Arc<Mutex<AppState>>, id: &str) -> Self {
        let cfg = DungeonConfig::new(id);
        let _ = fs::create_dir_all(&cfg.root);
        Self { cfg, state }
    }

    fn log_path(&self) -> PathBuf {
        self.cfg.root.join("ethical_deliberations.log")
    }

    fn sentinel_path(&self) -> PathBuf {
        self.cfg.root.join("RUNNING")
    }

    fn ensure_files(&self) {
        let _ = fs::create_dir_all(&self.cfg.root);
        if !self.log_path().exists() {
            let _ = File::create(self.log_path());
        }
    }

    pub fn start_simulated_run(&self) {
        self.ensure_files();
        let running = self.sentinel_path();
        if File::create(&running).is_err() {
            return;
        }

        let cfg = self.cfg.clone();
        let state = self.state.clone();

        thread::spawn(move || {
            let mut consensus: u32 = 40;
            let mut tick: u32 = 0;

            append_log(&cfg, "Dungeon run started");
            if let Ok(mut guard) = state.lock() {
                guard.add_terminal_line(
                    format!("🗺️ Dungeon '{}' started", cfg.id),
                    LineType::Info,
                );
            }

            while running.exists() {
                tick += 1;
                consensus = (consensus + 1).min(100);
                append_log(
                    &cfg,
                    &format!(
                        "[{}] Deliberation tick {} consensus {}%",
                        Utc::now().to_rfc3339(),
                        tick,
                        consensus
                    ),
                );

                if consensus % 10 == 0 {
                    if let Ok(mut guard) = state.lock() {
                        guard.add_terminal_line(
                            format!("Consensus reached {}%", consensus),
                            LineType::Info,
                        );
                    }
                }

                if consensus >= 95 {
                    append_log(&cfg, "Dungeon victory achieved");
                    let _ = fs::remove_file(&running);
                    if let Ok(mut guard) = state.lock() {
                        guard.add_terminal_line(
                            "🏆 Dungeon cleared".to_string(),
                            LineType::Info,
                        );
                    }
                    break;
                }

                thread::sleep(Duration::from_millis(500));
            }

            append_log(&cfg, "Dungeon run finished");
        });
    }

    pub fn stop(&self) {
        let _ = fs::remove_file(self.sentinel_path());
    }
}

fn append_log(cfg: &DungeonConfig, line: &str) {
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(cfg.root.join("ethical_deliberations.log"))
    {
        let _ = writeln!(file, "{}", line);
    }
}

pub fn read_recent_deliberations(cfg: &DungeonConfig, max_lines: usize) -> String {
    let path = cfg.root.join("ethical_deliberations.log");
    if let Ok(file) = File::open(&path) {
        let reader = BufReader::new(file);
        let lines: Vec<_> = reader.lines().filter_map(|l| l.ok()).collect();
        let start = lines.len().saturating_sub(max_lines);
        return lines[start..].join("\n");
    }
    "No deliberations yet...".to_string()
}
