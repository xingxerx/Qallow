use std::path::{Path, PathBuf};
use std::fs;
use std::process::Command;
use crate::logging::AppLogger;

/// Manages the Qallow codebase and integrates it with the native app
pub struct CodebaseManager {
    root_path: PathBuf,
    logger: AppLogger,
}

impl CodebaseManager {
    pub fn new(root_path: &str, logger: AppLogger) -> Result<Self, String> {
        let path = PathBuf::from(root_path);
        
        if !path.exists() {
            return Err(format!("Codebase path does not exist: {}", root_path));
        }

        Ok(CodebaseManager {
            root_path: path,
            logger,
        })
    }

    /// Get the root path of the codebase
    pub fn get_root_path(&self) -> &Path {
        &self.root_path
    }

    /// List all phases available in the codebase
    pub fn list_phases(&self) -> Result<Vec<String>, String> {
        let phases_dir = self.root_path.join("phases");
        
        if !phases_dir.exists() {
            return Ok(vec!["Phase13".to_string(), "Phase14".to_string(), "Phase15".to_string()]);
        }

        let mut phases = Vec::new();
        for entry in fs::read_dir(&phases_dir)
            .map_err(|e| format!("Failed to read phases directory: {}", e))?
        {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name() {
                    if let Some(name_str) = name.to_str() {
                        phases.push(name_str.to_string());
                    }
                }
            }
        }

        Ok(phases)
    }

    /// Get available build types
    pub fn list_builds(&self) -> Vec<String> {
        vec!["CPU".to_string(), "CUDA".to_string()]
    }

    /// Get codebase statistics
    pub fn get_statistics(&self) -> Result<CodebaseStats, String> {
        let mut stats = CodebaseStats::default();

        // Count Rust files
        self.count_files(&self.root_path, "rs", &mut stats.rust_files)?;
        
        // Count Python files
        self.count_files(&self.root_path, "py", &mut stats.python_files)?;
        
        // Count TOML files
        self.count_files(&self.root_path, "toml", &mut stats.config_files)?;

        // Get git status
        if let Ok(output) = Command::new("git")
            .arg("rev-parse")
            .arg("HEAD")
            .current_dir(&self.root_path)
            .output()
        {
            if output.status.success() {
                stats.git_commit = String::from_utf8_lossy(&output.stdout).trim().to_string();
            }
        }

        // Get git branch
        if let Ok(output) = Command::new("git")
            .arg("rev-parse")
            .arg("--abbrev-ref")
            .arg("HEAD")
            .current_dir(&self.root_path)
            .output()
        {
            if output.status.success() {
                stats.git_branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
            }
        }

        Ok(stats)
    }

    /// Count files with a specific extension
    fn count_files(&self, dir: &Path, extension: &str, count: &mut usize) -> Result<(), String> {
        for entry in fs::read_dir(dir)
            .map_err(|e| format!("Failed to read directory: {}", e))?
        {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();

            if path.is_dir() {
                // Skip hidden directories and common non-code directories
                if let Some(name) = path.file_name() {
                    if let Some(name_str) = name.to_str() {
                        if name_str.starts_with('.') || name_str == "target" || name_str == "__pycache__" {
                            continue;
                        }
                    }
                }
                let _ = self.count_files(&path, extension, count);
            } else if path.extension().and_then(|s| s.to_str()) == Some(extension) {
                *count += 1;
            }
        }

        Ok(())
    }

    /// Build the native app
    pub fn build_native_app(&self) -> Result<String, String> {
        let native_app_path = self.root_path.join("native_app");
        
        if !native_app_path.exists() {
            return Err("Native app directory not found".to_string());
        }

        let output = Command::new("cargo")
            .arg("build")
            .arg("--release")
            .current_dir(&native_app_path)
            .output()
            .map_err(|e| format!("Failed to build: {}", e))?;

        if output.status.success() {
            let _ = self.logger.info("✓ Native app built successfully");
            Ok("Build successful".to_string())
        } else {
            let error = String::from_utf8_lossy(&output.stderr);
            Err(format!("Build failed: {}", error))
        }
    }

    /// Run tests for the codebase
    pub fn run_tests(&self) -> Result<String, String> {
        let native_app_path = self.root_path.join("native_app");
        
        let output = Command::new("cargo")
            .arg("test")
            .current_dir(&native_app_path)
            .output()
            .map_err(|e| format!("Failed to run tests: {}", e))?;

        if output.status.success() {
            let _ = self.logger.info("✓ Tests passed");
            Ok("All tests passed".to_string())
        } else {
            let error = String::from_utf8_lossy(&output.stderr);
            Err(format!("Tests failed: {}", error))
        }
    }

    /// Get the current git status
    pub fn get_git_status(&self) -> Result<String, String> {
        let output = Command::new("git")
            .arg("status")
            .arg("--short")
            .current_dir(&self.root_path)
            .output()
            .map_err(|e| format!("Failed to get git status: {}", e))?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Get recent commits
    pub fn get_recent_commits(&self, count: usize) -> Result<Vec<String>, String> {
        let output = Command::new("git")
            .arg("log")
            .arg(format!("-{}", count))
            .arg("--oneline")
            .current_dir(&self.root_path)
            .output()
            .map_err(|e| format!("Failed to get commits: {}", e))?;

        let commits = String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(|s| s.to_string())
            .collect();

        Ok(commits)
    }
}

#[derive(Debug, Default)]
pub struct CodebaseStats {
    pub rust_files: usize,
    pub python_files: usize,
    pub config_files: usize,
    pub git_commit: String,
    pub git_branch: String,
}

impl std::fmt::Display for CodebaseStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Codebase Statistics:\n  Rust Files: {}\n  Python Files: {}\n  Config Files: {}\n  Git Branch: {}\n  Git Commit: {}",
            self.rust_files, self.python_files, self.config_files, self.git_branch, self.git_commit
        )
    }
}

