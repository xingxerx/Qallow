//! Well-known CLI subcommands/flags used when invoking the `qallow` binary,
//! kept in one place so callers don't hand-roll argument strings.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlCommand {
    Run,
    Stop,
    Status,
}

impl ControlCommand {
    pub fn as_str(&self) -> &'static str {
        match self {
            ControlCommand::Run => "run",
            ControlCommand::Stop => "stop",
            ControlCommand::Status => "status",
        }
    }
}
