//! Clipboard helpers backed by FLTK's built-in clipboard support (no extra
//! dependency needed).

use fltk::app;

pub fn copy_to_clipboard(text: &str) {
    app::copy(text);
}
