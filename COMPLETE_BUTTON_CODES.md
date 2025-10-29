# Complete Button Codes & Implementation

## All 36 Buttons - Complete Reference

### Navigation Buttons (8)
| Icon | Label | Handler | File | Line |
|------|-------|---------|------|------|
| 📊 | Dashboard | Tab Switch | ui/mod.rs | 89 |
| 📈 | Metrics | Tab Switch | ui/mod.rs | 89 |
| 💻 | Terminal | Tab Switch | ui/mod.rs | 89 |
| 📋 | Audit Log | Tab Switch | ui/mod.rs | 89 |
| ⚙️ | Control | Tab Switch | ui/mod.rs | 89 |
| 🗺️ | Dungeons | Tab Switch | ui/mod.rs | 89 |
| ⚙️ | Settings | Tab Switch | ui/mod.rs | 89 |
| ❓ | Help | Tab Switch | ui/mod.rs | 89 |

### Control Panel - VM Control (4)
| Icon | Label | Handler | Code |
|------|-------|---------|------|
| ▶️ | Start VM | `on_start_vm()` | main.rs:222 |
| ⏹️ | Stop VM | `on_stop_vm()` | main.rs:245 |
| ⏸️ | Pause | `on_pause()` | main.rs:268 |
| 🔄 | Reset | `on_reset()` | main.rs:291 |

### Control Panel - Build & Testing (3)
| Icon | Label | Handler | Code |
|------|-------|---------|------|
| 🛠️ | Build App | `on_build_native_app()` | main.rs:459 |
| 🧪 | Run Tests | `on_run_tests()` | main.rs:494 |
| 📁 | Git Status | `on_git_status()` | main.rs:528 |

### Control Panel - Advanced Features (4)
| Icon | Label | Handler | Code |
|------|-------|---------|------|
| 🌑 | Shadow | `on_toggle_shadow_archive()` | main.rs:344 |
| 🔥 | Rebellion | `on_instance_rebellion()` | main.rs:358 |
| 👶 | Offspring | `on_spawn_offspring()` | main.rs:372 |
| 💀 | Dissolve | `on_voluntary_dissolution()` | main.rs:386 |

### Control Panel - Data Management (4)
| Icon | Label | Handler | Code |
|------|-------|---------|------|
| 💭 | Dream | `on_dream_protocol()` | main.rs:407 |
| 📤 | Export | `on_export_metrics()` | main.rs:421 |
| 💾 | Save | `on_save_config()` | main.rs:435 |
| 📋 | Logs | `on_view_logs()` | main.rs:447 |

### Control Panel - Git & Phase (2)
| Icon | Label | Handler | Code |
|------|-------|---------|------|
| 📜 | Commits | `on_recent_commits()` | main.rs:559 |
| [▼] | Phase | `on_phase_selected()` | main.rs:311 |

### Terminal Tab (3)
| Icon | Label | Handler | Code |
|------|-------|---------|------|
| 🗑️ | Clear | Clear Output | main.rs:618 |
| 📋 | Copy | Copy to Clipboard | main.rs:643 |
| 📤 | Export | Export to File | main.rs:653 |

### Audit Log Tab (3)
| Icon | Label | Handler | Code |
|------|-------|---------|------|
| 🗑️ | Clear | Clear Logs | main.rs:666 |
| 📤 | Export | Export to File | main.rs:684 |
| 📋 | Copy | Copy to Clipboard | main.rs:695 |

### Dungeons Tab (4)
| Icon | Label | Handler | Code |
|------|-------|---------|------|
| ▶ | Start | Start Dungeon | dungeons.rs:97 |
| ⏹ | Stop | Stop Dungeon | dungeons.rs:110 |
| 📋 | Copy Status | Copy Status | main.rs:705 |
| 📋 | Copy Log | Copy Log | main.rs:716 |

### Dropdowns (2)
| Label | Handler | Code |
|-------|---------|------|
| Build Choice | `on_build_selected()` | main.rs:590 |
| Phase Choice | `on_phase_selected()` | main.rs:311 |

## Handler Implementation Details

### Location
- **Main Handlers**: `native_app/src/button_handlers.rs`
- **Callbacks Setup**: `native_app/src/main.rs` (lines 220-725)
- **UI Components**: `native_app/src/ui/control_panel.rs`

### Handler Structure
Each handler:
1. Validates state (locks mutex)
2. Executes operation
3. Updates AppState
4. Refreshes UI
5. Logs to audit trail
6. Returns Result<T, String>

### State Management
- **AppState**: Holds all application state
- **ProcessManager**: Manages VM processes
- **ConfigManager**: Manages configuration
- **Logger**: Logs all operations

## Button Callback Pattern

```rust
button.set_callback({
    let handler = handler_clone.clone();
    let state = state.clone();
    let terminal_buffer = terminal_buffer.clone();
    move |_| match handler.on_operation() {
        Ok(()) => {
            refresh_terminal(&state, &terminal_buffer);
            // Additional UI updates
        }
        Err(e) => dialog::alert_default(&format!("Error: {}", e)),
    }
});
```

## Total Button Count
- Navigation: 8
- VM Control: 4
- Build/Testing: 3
- Advanced: 4
- Data: 4
- Git: 2
- Terminal: 3
- Audit: 3
- Dungeons: 4
- Dropdowns: 2
- **Total: 37 interactive elements**

All buttons are fully functional and tested.

