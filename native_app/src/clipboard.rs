use fltk::app;
use std::sync::{Arc, Mutex, OnceLock};

pub struct ClipboardService {
    last: Arc<Mutex<String>>,
}

impl ClipboardService {
    pub fn global() -> &'static ClipboardService {
        static SERVICE: OnceLock<ClipboardService> = OnceLock::new();
        SERVICE.get_or_init(|| ClipboardService {
            last: Arc::new(Mutex::new(String::new())),
        })
    }

    pub fn copy_text(&self, text: &str) {
        if let Ok(mut last) = self.last.lock() {
            last.clear();
            last.push_str(text);
        }
        app::copy(text);
    }

    pub fn last_text(&self) -> String {
        self.last.lock().map(|s| s.clone()).unwrap_or_default()
    }

    pub fn paste_text(&self) -> Option<String> {
        // FLTK doesn't provide a direct clipboard read function
        // Return the last copied text instead
        let cached = self.last_text();
        if cached.is_empty() {
            None
        } else {
            Some(cached)
        }
    }
}
