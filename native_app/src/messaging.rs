//! Message type sent from background threads/tasks to the FLTK UI thread
//! over `fltk::app::channel`.

#[derive(Debug, Clone, Copy)]
pub enum UiMessage {
    /// Ask the UI to refresh whatever state-derived widgets it shows.
    Refresh,
    /// A new log line is available.
    Log,
}
