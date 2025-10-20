# Qallow SDL Telemetry UI

The SDL-based UI in `interface/qallow_ui.c` provides a minimal desktop interface for monitoring Qallow telemetry and launching phase runs directly from the keyboard.

## Features
- Live view of the most recent entry in `data/logs/telemetry_stream.csv`, including tick index and execution mode.
- Visual bar chart for up to 16 telemetry metrics with real-time updates (default every 750 ms).
- Keyboard shortcuts for executing phases 14–16 via the configured Qallow binary.

## Build
Ensure SDL2 and SDL2_ttf development packages are installed. On Arch Linux:

```bash
sudo pacman -S sdl2 sdl2_ttf
```

Then compile:

```bash
gcc -std=c11 -Wall -Wextra -O2 \
    -I/usr/include/SDL2 \
    interface/qallow_ui.c \
    -lSDL2 -lSDL2_ttf \
    -o build/qallow_ui
```

Adjust include/library paths as required for your distribution.

## Usage

```bash
./build/qallow_ui \
  --telemetry=data/logs/telemetry_stream.csv \
  --runner=./build/qallow_unified_cuda \
  --font=/usr/share/fonts/TTF/DejaVuSans.ttf \
  --refresh-ms=750
```

Command-line options are optional; sensible defaults are applied when paths are not provided. The UI attempts to locate a runnable Qallow binary automatically but will display phase-launch failures if none is found.

Keyboard controls:
- `1` — run phase 14
- `2` — run phase 15
- `3` — run phase 16
- `Esc` — exit the UI

The status banner at the bottom reflects the most recent phase command outcome.

