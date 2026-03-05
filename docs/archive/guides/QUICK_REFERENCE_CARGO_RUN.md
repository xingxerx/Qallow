# Quick Reference - Cargo Run ⚡

## Run the New Native App

```bash
cd /root/Qallow
cargo run
```

✅ **Result**: Launches the new Qallow Native App with FLTK GUI

---

## Build Options

### Debug Build (Default)
```bash
cargo build
```

### Release Build (Optimized)
```bash
cargo build --release
```

### Clean Build
```bash
cargo clean
cargo build
```

---

## Run Tests

```bash
cargo test --test button_integration_test
```

**Expected**: `test result: ok. 32 passed; 0 failed`

---

## Run Specific Projects

### New Native App
```bash
cd /root/Qallow/native_app
cargo run
```

### Old UI App
```bash
cd /root/Qallow/rust/app
cargo run
```

### UI Utilities
```bash
cd /root/Qallow/rust/ui
cargo run
```

---

## Workspace Structure

```
/root/Qallow/
├── Cargo.toml                 # Workspace config
├── native_app/                # ✅ Default (NEW)
│   ├── Cargo.toml
│   ├── src/
│   └── tests/
├── rust/app/                  # Old UI
│   ├── Cargo.toml
│   └── src/
└── rust/ui/                   # UI utilities
    ├── Cargo.toml
    └── src/
```

---

## What Changed

**File**: `/root/Qallow/Cargo.toml`

```diff
- default-members = ["rust/app"]
+ default-members = ["native_app"]
```

---

## Verification

```bash
# Check which app is default
cd /root/Qallow
cargo build 2>&1 | grep "Compiling"
# Output: Compiling qallow-native v1.0.0

# Check binary location
ls -lh target/debug/qallow-native
# Output: -rwxr-xr-x 2 root root 48M Oct 24 22:10 target/debug/qallow-native
```

---

## Features

✅ **Native FLTK GUI** - Desktop application
✅ **Working Buttons** - All connected to backend
✅ **State Management** - Persistent state
✅ **Logging** - Comprehensive logging
✅ **Error Handling** - Graceful error recovery
✅ **Codebase Integration** - Full codebase management

---

## Troubleshooting

### App Won't Start
```bash
# Clean and rebuild
cargo clean
cargo build
cargo run
```

### Build Fails
```bash
# Check Rust version
rustc --version

# Update Rust
rustup update

# Try again
cargo build
```

### Tests Fail
```bash
# Run tests with output
cargo test --test button_integration_test -- --nocapture
```

---

## Documentation

- **User Guide**: `NATIVE_APP_GUIDE.md`
- **Implementation**: `IMPLEMENTATION_SUMMARY.md`
- **Cargo Fix**: `CARGO_RUN_FIX.md`
- **Quick Start**: `QUICK_START_NATIVE_APP.md`

---

**Status**: ✅ READY
**Default App**: Native App (qallow-native)
**Binary**: 48M
**Tests**: 32/32 passing

