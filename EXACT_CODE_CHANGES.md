# 📝 Exact Code Changes - CI Build Fix

**File**: `native_app/src/main.rs`  
**Status**: ✅ Fixed and verified  
**Build**: ✅ Compiles successfully

---

## Change 1: Remove Unused Imports (Lines 21-34)

### Before
```rust
use backend::process_manager::ProcessManager;
use button_handlers::ButtonHandler;
use codebase_manager::CodebaseManager;
use config::{AppConfig, ConfigManager};
use fltk::enums::Color;
use fltk::{app, button, dialog, prelude::*};
use crate::ui::main_window::MainWindow;  // ❌ UNUSED
use models::AppState;
use std::sync::{Arc, Mutex};
use std::thread;  // ❌ UNUSED
use tokio::runtime::Runtime;
use crate::{
    control_commands::ControlCommand,  // ❌ UNUSED
    logging::AppLogger,
    messaging::UiMessage,
    shutdown::ShutdownManager,
};
```

### After
```rust
use backend::process_manager::ProcessManager;
use button_handlers::ButtonHandler;
use codebase_manager::CodebaseManager;
use config::{AppConfig, ConfigManager};
use fltk::enums::Color;
use fltk::{app, button, dialog, prelude::*};
use models::AppState;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;
use crate::{
    logging::AppLogger,
    messaging::UiMessage,
    shutdown::ShutdownManager,
};
```

**Changes**:
- ✅ Removed `use crate::ui::main_window::MainWindow;`
- ✅ Removed `use std::thread;`
- ✅ Removed `use crate::control_commands::ControlCommand;`

---

## Change 2: Fix Chat View Handling (Lines 186-220)

### Before
```rust
// --- Chat Button Logic ---
main_win.chat_panel.send_button.set_callback({
    let mut chat_input = main_win.chat_panel.input.clone();
    let mut chat_view = main_win.chat_panel.conversation_display.clone();  // ❌ mut
    let api_client = main_win.button_handler.api_client.clone();
    let logger = logger.clone();

    move |_| {
        let message = chat_input.value();
        if message.is_empty() {
            return;
        }
        chat_input.set_value("");
        chat_view.buffer().unwrap().append(&format!("You: {}\n", message));

        let api_client = api_client.clone();
        let logger = logger.clone();
        spawn(async move {  // ❌ ERROR: chat_view moved here
            match api_client.chat(&message).await {
                Ok(response) => {
                    // Make sure to update UI in the main thread
                    fltk::app::awake_callback(move || {
                        chat_view  // ❌ ERROR: used here too
                            .buffer()
                            .unwrap()
                            .append(&format!("Agent: {}\n", response));
                    });
                }
                Err(e) => {
                    let _ = logger.error(&format!("API Error: {}", e));
                    fltk::app::awake_callback(move || {
                        chat_view  // ❌ ERROR: used here too
                            .buffer()
                            .unwrap()
                            .append("Agent: Sorry, I encountered an error.\n");
                    });
                }
            }
        });
    }
});
```

### After
```rust
// --- Chat Button Logic ---
main_win.chat_panel.send_button.set_callback({
    let mut chat_input = main_win.chat_panel.input.clone();
    let chat_view = main_win.chat_panel.conversation_display.clone();  // ✅ no mut
    let api_client = main_win.button_handler.api_client.clone();
    let logger = logger.clone();

    move |_| {
        let message = chat_input.value();
        if message.is_empty() {
            return;
        }
        chat_input.set_value("");
        chat_view.buffer().unwrap().append(&format!("You: {}\n", message));

        let api_client = api_client.clone();
        let logger = logger.clone();
        let chat_view_clone = chat_view.clone();  // ✅ Clone for async
        spawn(async move {  // ✅ Now uses chat_view_clone
            match api_client.chat(&message).await {
                Ok(response) => {
                    // Make sure to update UI in the main thread
                    let chat_view_for_callback = chat_view_clone.clone();  // ✅ Clone for callback
                    fltk::app::awake_callback(move || {
                        chat_view_for_callback  // ✅ Uses its own clone
                            .buffer()
                            .unwrap()
                            .append(&format!("Agent: {}\n", response));
                    });
                }
                Err(e) => {
                    let _ = logger.error(&format!("API Error: {}", e));
                    let chat_view_for_callback = chat_view_clone.clone();  // ✅ Clone for callback
                    fltk::app::awake_callback(move || {
                        chat_view_for_callback  // ✅ Uses its own clone
                            .buffer()
                            .unwrap()
                            .append("Agent: Sorry, I encountered an error.\n");
                    });
                }
            }
        });
    }
});
```

**Changes**:
- ✅ Removed `mut` from `chat_view` (line 186)
- ✅ Added `let chat_view_clone = chat_view.clone();` (line 199)
- ✅ Added `let chat_view_for_callback = chat_view_clone.clone();` (line 203)
- ✅ Changed `chat_view` to `chat_view_for_callback` in Ok branch (line 205)
- ✅ Added `let chat_view_for_callback = chat_view_clone.clone();` (line 212)
- ✅ Changed `chat_view` to `chat_view_for_callback` in Err branch (line 214)

---

## Summary of Changes

### Files Modified
- `native_app/src/main.rs` - 1 file

### Lines Changed
- Additions: ~10 lines
- Deletions: ~3 lines
- Net change: ~7 lines

### Errors Fixed
- ✅ E0507: Cannot move out of captured variable

### Warnings Fixed
- ✅ Unused import: `MainWindow`
- ✅ Unused import: `std::thread`
- ✅ Unused import: `ControlCommand`
- ✅ Mutable binding: `chat_view`

### Build Status
- ✅ `cargo check` - PASS
- ✅ `cargo build` - PASS
- ✅ No errors
- ✅ No blocking warnings

---

## Ownership Pattern Explanation

### The Problem
```
FnMut closure captures chat_view
    ↓
    Can be called multiple times
    ↓
    But async move consumes chat_view once
    ↓
    ERROR: Can't consume multiple times!
```

### The Solution
```
Original chat_view
    ↓
    Used in button callback (can be called multiple times)
    ↓
    Clone 1: chat_view_clone (moved into async block)
    ↓
    Clone 2: chat_view_for_callback (moved into each callback)
    ↓
    Each scope has its own owned copy - NO CONFLICTS!
```

---

## Verification Commands

```bash
# Check for errors
cargo check

# Build debug version
cargo build

# Build release version
cargo build --release

# View the diff
git diff native_app/src/main.rs
```

---

## Commit Message

```
fix: resolve borrow checker violation in chat callback

The chat_view variable was being moved into an async block while also
being captured by an FnMut closure. This violated Rust's ownership rules.

Solution: Clone the variable before moving it into the async block, and
clone again for each nested callback closure.

Changes:
- Remove unused imports (MainWindow, std::thread, ControlCommand)
- Fix mutable binding warning on chat_view
- Resolve E0507 compilation error

Fixes: CI build failure in native_app
```

---

**Status**: ✅ **READY FOR MERGE**

