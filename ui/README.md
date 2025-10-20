# Qallow SDL Control UI

`interface/qallow_ui.c` now unifies the earlier Tkinter helper (`ui/qallow_monitor.py`) and the telemetry visualizer into a single SDL2 desktop application.

## Combined Features
- Live telemetry stream from `data/logs/telemetry_stream.csv` with bar visualisations for each metric.
- Pocket-dimension metrics parsed from `data/telemetry/pocket_metrics.json`.
- Command launcher with log console mirroring the Python monitor:
  - Build CUDA pipeline (`scripts/build_unified_cuda.sh`).
  - Run the selected Qallow binary (auto-detected or supplied via `--runner`).
  - Run the accelerator script (`scripts/run_auto.sh --watch <repo>`).
  - Launch phases 14–16 with captured stdout/stderr.
  - Stop button (or `S`) sends `SIGTERM` to the running command.
- Keyboard shortcuts: `B` build, `R` run binary, `A` accelerator, `1/2/3` phases, `S` stop, `Esc` quit.

## Build
Install SDL2 + SDL2_ttf development packages (Arch example):

```bash
sudo pacman -S sdl2 sdl2_ttf
```

Then build through CMake (the `qallow_ui` target is added automatically once the libraries are detected):

```bash
cmake -S . -B build_ninja -GNinja
cmake --build build_ninja --target qallow_ui
```

Or compile manually if desired:

```bash
gcc -std=c11 -Wall -Wextra -O2 \
    -I/usr/include/SDL2 \
    interface/qallow_ui.c \
    -lSDL2 -lSDL2_ttf \
    -o build/qallow_ui
```

## Usage

```bash
./build/qallow_ui \
  --telemetry=data/logs/telemetry_stream.csv \
  --pocket-json=data/telemetry/pocket_metrics.json \
  --runner=./build/qallow_unified_cuda \
  --repo-root=$(pwd)
```

All flags are optional; defaults cover the standard repository layout. The UI automatically refreshes telemetry (750 ms) and pocket metrics (1.2 s). Command output is captured in the right-hand log panel, and the status banner reflects the latest action taken.
