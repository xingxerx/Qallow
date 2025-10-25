use crate::models::{BuildType, Phase};
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
#[allow(unused_imports)]
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct ProcessMetadata {
    pub start_time: u64,
    pub phase: Phase,
    pub build: BuildType,
    pub ticks: u32,
}

pub struct ProcessManager {
    process: Option<Child>,
    output_tx: Sender<String>,
    output_rx: Receiver<String>,
    metadata: Option<ProcessMetadata>,
    retry_count: u32,
    max_retries: u32,
}

impl ProcessManager {
    pub fn new() -> Self {
        let (tx, rx) = unbounded();
        Self {
            process: None,
            output_tx: tx,
            output_rx: rx,
            metadata: None,
            retry_count: 0,
            max_retries: 3,
        }
    }

    pub fn set_max_retries(&mut self, max_retries: u32) {
        self.max_retries = max_retries;
    }

    pub fn get_retry_count(&self) -> u32 {
        self.retry_count
    }

    pub fn reset_retry_count(&mut self) {
        self.retry_count = 0;
    }

    pub fn get_metadata(&self) -> Option<&ProcessMetadata> {
        self.metadata.as_ref()
    }

    pub fn get_uptime_secs(&self) -> Option<u64> {
        self.metadata.as_ref().map(|m| {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            now.saturating_sub(m.start_time)
        })
    }

    pub fn start_vm(&mut self, build: BuildType, phase: Phase, ticks: u32) -> Result<(), String> {
        if self.process.is_some() {
            return Err("VM already running".to_string());
        }

        let binary_path = match build {
            BuildType::CPU => "/root/Qallow/build/qallow",
            BuildType::CUDA => "/root/Qallow/build/qallow_unified",
        };

        let phase_arg = match phase {
            Phase::Phase13 => "13",
            Phase::Phase14 => "14",
            Phase::Phase15 => "15",
        };

        let ticks_arg = format!("--ticks={}", ticks);

        let mut cmd = Command::new(binary_path);
        cmd.arg("phase")
            .arg(phase_arg)
            .arg(&ticks_arg)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        match cmd.spawn() {
            Ok(mut child) => {
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();
                let tx = self.output_tx.clone();

                // Store metadata
                let start_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                self.metadata = Some(ProcessMetadata {
                    start_time,
                    phase,
                    build,
                    ticks,
                });
                self.retry_count = 0;

                // Spawn thread to read stdout
                if let Some(stdout) = stdout {
                    let tx_clone = tx.clone();
                    std::thread::spawn(move || {
                        let reader = BufReader::new(stdout);
                        for line in reader.lines() {
                            if let Ok(line) = line {
                                let _ = tx_clone.send(line);
                            }
                        }
                    });
                }

                // Spawn thread to read stderr
                if let Some(stderr) = stderr {
                    let tx_clone = tx.clone();
                    std::thread::spawn(move || {
                        let reader = BufReader::new(stderr);
                        for line in reader.lines() {
                            if let Ok(line) = line {
                                let _ = tx_clone.send(format!("[ERROR] {}", line));
                            }
                        }
                    });
                }

                self.process = Some(child);
                Ok(())
            }
            Err(e) => {
                self.retry_count += 1;
                Err(format!(
                    "Failed to start process (attempt {}): {}",
                    self.retry_count, e
                ))
            }
        }
    }

    pub fn stop_vm(&mut self) -> Result<(), String> {
        if let Some(mut child) = self.process.take() {
            // Try graceful termination first
            #[cfg(unix)]
            {
                let pid = child.id();
                // Send SIGTERM using kill command
                let _ = std::process::Command::new("kill")
                    .arg("-TERM")
                    .arg(pid.to_string())
                    .output();
                std::thread::sleep(std::time::Duration::from_secs(2));
            }

            // Force kill if still running
            match child.kill() {
                Ok(_) => {
                    let _ = child.wait();
                    self.metadata = None;
                    Ok(())
                }
                Err(e) => {
                    self.metadata = None;
                    Err(format!("Failed to stop process: {}", e))
                }
            }
        } else {
            Err("No VM running".to_string())
        }
    }

    pub fn try_graceful_stop(&mut self, timeout_secs: u64) -> Result<(), String> {
        if let Some(mut child) = self.process.take() {
            #[cfg(unix)]
            {
                let pid = child.id();
                // Send SIGTERM using kill command
                let _ = std::process::Command::new("kill")
                    .arg("-TERM")
                    .arg(pid.to_string())
                    .output();

                // Wait for graceful shutdown
                let start = std::time::Instant::now();
                loop {
                    match child.try_wait() {
                        Ok(Some(_)) => {
                            self.metadata = None;
                            return Ok(());
                        }
                        Ok(None) => {
                            if start.elapsed().as_secs() > timeout_secs {
                                // Force kill
                                let _ = child.kill();
                                let _ = child.wait();
                                self.metadata = None;
                                return Err("Process killed after timeout".to_string());
                            }
                            std::thread::sleep(std::time::Duration::from_millis(100));
                        }
                        Err(e) => {
                            self.metadata = None;
                            return Err(format!("Error waiting for process: {}", e));
                        }
                    }
                }
            }

            #[cfg(not(unix))]
            {
                // Fallback: force kill for non-Unix systems
                match child.kill() {
                    Ok(_) => {
                        let _ = child.wait();
                        self.metadata = None;
                        Ok(())
                    }
                    Err(e) => {
                        self.metadata = None;
                        Err(format!("Failed to kill process: {}", e))
                    }
                }
            }
        } else {
            Err("No VM running".to_string())
        }
    }

    pub fn is_running(&self) -> bool {
        self.process.is_some()
    }

    pub fn get_output(&self) -> Option<String> {
        self.output_rx.try_recv().ok()
    }

    pub fn get_all_output(&self) -> Vec<String> {
        let mut output = Vec::new();
        while let Ok(line) = self.output_rx.try_recv() {
            output.push(line);
        }
        output
    }

    pub fn poll_exit(&mut self) -> bool {
        if let Some(child) = self.process.as_mut() {
            match child.try_wait() {
                Ok(Some(_status)) => {
                    self.process = None;
                    return true;
                }
                Ok(None) => {}
                Err(_) => {
                    self.process = None;
                    return true;
                }
            }
        }
        false
    }
}

impl Default for ProcessManager {
    fn default() -> Self {
        Self::new()
    }
}
