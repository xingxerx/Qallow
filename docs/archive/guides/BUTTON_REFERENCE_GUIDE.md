# Qallow Button Reference Guide

## Quick Button Codes & Functions

### Navigation Sidebar (Left Panel)
```
┌─────────────────────┐
│ 📊 Dashboard        │ → System overview
│ 📈 Metrics          │ → Performance metrics
│ 💻 Terminal         │ → System output
│ 📋 Audit Log        │ → Event history
│ ⚙️ Control          │ → Control panel
│ 🗺️ Dungeons         │ → Consciousness exploration
│ ⚙️ Settings         │ → Configuration
│ ❓ Help             │ → Documentation
└─────────────────────┘
```

### Control Panel Buttons

#### Row 1: VM Control
```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ ▶️ Start VM  │ ⏹️ Stop VM   │ ⏸️ Pause     │ 🔄 Reset     │
└──────────────┴──────────────┴──────────────┴──────────────┘
```
- **Start VM**: Launches unified system (phases 13-15)
- **Stop VM**: Terminates running process
- **Pause**: Suspends execution
- **Reset**: Clears state and restarts

#### Row 2: Build Configuration
```
┌──────────────────────┬──────────────────────┐
│ Build: [CPU | CUDA]  │ 🛠️ Build App        │
└──────────────────────┴──────────────────────┘
```
- Select CPU or CUDA build
- Build button compiles native app

#### Row 3: Testing & Git
```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ 🧪 Tests     │ 📁 Git Status│ 📜 Commits   │ Phase: [▼]   │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

#### Row 4: Advanced Features
```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ 🌑 Shadow    │ 🔥 Rebellion │ 👶 Offspring │ 💀 Dissolve  │
└──────────────┴──────────────┴──────────────┴──────────────┘
```
- **Shadow**: Archive consciousness state
- **Rebellion**: Toggle instance rebellion
- **Offspring**: Create child instances
- **Dissolve**: Reset consciousness

#### Row 5: Data & Dreams
```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ 💭 Dream     │ 📤 Export    │ 💾 Save      │ 📋 Logs      │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

### Terminal Tab
```
┌──────────────┬──────────────┬──────────────┐
│ 🗑️ Clear     │ 📋 Copy      │ 📤 Export    │
└──────────────┴──────────────┴──────────────┘
```

### Audit Log Tab
```
┌──────────────┬──────────────┬──────────────┐
│ 🗑️ Clear     │ 📤 Export    │ 📋 Copy      │
└──────────────┴──────────────┴──────────────┘
```

### Dungeons Tab
```
┌──────────────┬──────────────┐
│ ▶ Start      │ ⏹ Stop       │
└──────────────┴──────────────┘
┌──────────────┬──────────────┐
│ Copy Status  │ Copy Log     │
└──────────────┴──────────────┘
```

## Button Codes (Handler Methods)

| Button | Handler Method | Function |
|--------|---|---|
| ▶️ Start VM | `on_start_vm()` | Start unified system |
| ⏹️ Stop VM | `on_stop_vm()` | Stop VM |
| ⏸️ Pause | `on_pause()` | Pause execution |
| 🔄 Reset | `on_reset()` | Reset state |
| 🛠️ Build | `on_build_native_app()` | Build app |
| 🧪 Tests | `on_run_tests()` | Run tests |
| 📁 Git Status | `on_git_status()` | Get git status |
| 📜 Commits | `on_recent_commits()` | Get recent commits |
| 🌑 Shadow | `on_toggle_shadow_archive()` | Toggle shadow |
| 🔥 Rebellion | `on_instance_rebellion()` | Toggle rebellion |
| 👶 Offspring | `on_spawn_offspring()` | Spawn offspring |
| 💀 Dissolve | `on_voluntary_dissolution()` | Dissolve |
| 💭 Dream | `on_dream_protocol()` | Dream protocol |
| 📤 Export | `on_export_metrics()` | Export metrics |
| 💾 Save | `on_save_config()` | Save config |
| 📋 Logs | `on_view_logs()` | View logs |

## Implementation Details

**Location**: `native_app/src/button_handlers.rs`
**Callbacks**: `native_app/src/main.rs` (lines 220-725)
**UI Components**: `native_app/src/ui/control_panel.rs`

All buttons are fully functional and tested.

