use std::process::{Command, Child, Stdio};
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex};
use crossbeam_channel::{Sender, Receiver, unbounded};
use crate::models::{BuildType, Phase};

pub struct ProcessManager {
    process: Option<Child>,
    output_tx: Sender<String>,
    output_rx: Receiver<String>,
}

impl ProcessManager {
    pub fn new() -> Self {
        let (tx, rx) = unbounded();
        Self {
            process: None,
            output_tx: tx,
            output_rx: rx,
        }
    }

    pub fn start_vm(&mut self, build: BuildType, phase: Phase) -> Result<(), String> {
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

        let mut cmd = Command::new(binary_path);
        cmd.arg("phase")
            .arg(phase_arg)
            .arg("--ticks=1000")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        match cmd.spawn() {
            Ok(mut child) => {
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();
                let tx = self.output_tx.clone();

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
            Err(e) => Err(format!("Failed to start process: {}", e)),
        }
    }

    pub fn stop_vm(&mut self) -> Result<(), String> {
        if let Some(mut child) = self.process.take() {
            match child.kill() {
                Ok(_) => {
                    let _ = child.wait();
                    Ok(())
                }
                Err(e) => Err(format!("Failed to kill process: {}", e)),
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
}

impl Default for ProcessManager {
    fn default() -> Self {
        Self::new()
    }
}

