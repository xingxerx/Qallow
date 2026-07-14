//! Tracks child processes spawned by the UI (e.g. phase runs) so they can be
//! queried or terminated later.

use std::collections::HashMap;
use std::process::Child;

pub struct ProcessManager {
    processes: HashMap<String, Child>,
}

impl ProcessManager {
    pub fn new() -> Self {
        Self {
            processes: HashMap::new(),
        }
    }

    /// Registers a spawned child process under `name`, replacing any previous
    /// process registered under the same name.
    pub fn insert(&mut self, name: String, child: Child) {
        self.processes.insert(name, child);
    }

    /// Returns true if the named process is still running.
    pub fn is_running(&mut self, name: &str) -> bool {
        match self.processes.get_mut(name) {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => false,
        }
    }

    /// Kills and forgets the named process, if any.
    pub fn kill(&mut self, name: &str) -> std::io::Result<()> {
        if let Some(mut child) = self.processes.remove(name) {
            child.kill()?;
        }
        Ok(())
    }
}

impl Default for ProcessManager {
    fn default() -> Self {
        Self::new()
    }
}
