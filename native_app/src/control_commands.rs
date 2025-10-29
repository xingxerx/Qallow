//! Qallow Control Commands
//!
//! Sends control commands to C-side simulation via POSIX message queues.
//! Supports: START, PAUSE, INJECT_CONSTRAINT, EXPORT_SPEC

use libc::{mq_open, mq_send, O_CREAT, O_WRONLY, O_NONBLOCK};
use std::ffi::CString;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlCommand {
    Start = 0,
    Pause = 1,
    InjectConstraint = 2,
    ExportSpec = 3,
}

pub struct ControlCommandSender {
    mq_name: String,
}

impl ControlCommandSender {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            mq_name: "/qallow_control".to_string(),
        })
    }

    /// Send a control command to the C core
    pub fn send_command(&self, cmd: ControlCommand, payload: Option<&str>) -> Result<(), String> {
        let mq_name = CString::new(self.mq_name.as_str())
            .map_err(|e| format!("Invalid mq name: {}", e))?;

        let flags = O_CREAT | O_WRONLY | O_NONBLOCK;
        let mode = 0o666;

        let mq = unsafe {
            mq_open(mq_name.as_ptr(), flags, mode, std::ptr::null::<libc::mq_attr>())
        };

        if mq < 0 {
            return Err("Failed to open control mq".to_string());
        }

        let mut msg = vec![0u8; 256];
        let cmd_byte = cmd as u8;
        msg[0] = cmd_byte;

        if let Some(payload_str) = payload {
            let payload_bytes = payload_str.as_bytes();
            let copy_len = std::cmp::min(payload_bytes.len(), 255);
            msg[1..1 + copy_len].copy_from_slice(&payload_bytes[..copy_len]);
        }

        let result = unsafe {
            mq_send(mq, msg.as_ptr() as *const i8, msg.len(), cmd as u32)
        };

        if result < 0 {
            return Err("Failed to send control command".to_string());
        }

        Ok(())
    }

    /// Send START command
    pub fn send_start(&self) -> Result<(), String> {
        self.send_command(ControlCommand::Start, None)
    }

    /// Send PAUSE command
    pub fn send_pause(&self) -> Result<(), String> {
        self.send_command(ControlCommand::Pause, None)
    }

    /// Send INJECT_CONSTRAINT command with constraint name
    pub fn send_inject_constraint(&self, constraint: &str) -> Result<(), String> {
        self.send_command(ControlCommand::InjectConstraint, Some(constraint))
    }

    /// Send EXPORT_SPEC command
    pub fn send_export_spec(&self) -> Result<(), String> {
        self.send_command(ControlCommand::ExportSpec, None)
    }
}

impl Default for ControlCommandSender {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            mq_name: "/qallow_control".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_command_values() {
        assert_eq!(ControlCommand::Start as u8, 0);
        assert_eq!(ControlCommand::Pause as u8, 1);
        assert_eq!(ControlCommand::InjectConstraint as u8, 2);
        assert_eq!(ControlCommand::ExportSpec as u8, 3);
    }

    #[test]
    fn test_sender_creation() {
        let sender = ControlCommandSender::new();
        assert!(sender.is_ok());
    }
}

