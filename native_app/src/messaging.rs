// UI messaging between background workers and the FLTK UI thread

#[derive(Debug, Clone)]
pub enum UiMessage {
    BuildDone(Result<String, String>),
    TestsDone(Result<String, String>),
    GitStatusDone(Result<String, String>),
    CommitsDone(Result<Vec<String>, String>),
}
