Qallow Native GUI — Quick Start

Requirements
- Linux desktop with X11 or Wayland
- OpenGL drivers (Mesa/NVIDIA/AMD)

Files
- qallow-native: Release binary
- LICENSE: Project license

Run
1) Make executable if needed:
   chmod +x ./qallow-native
2) Launch:
   ./qallow-native

Notes
- The app loads settings from qallow_config.json in the working directory when available.
- On headless systems, use xvfb-run to test: xvfb-run -a ./qallow-native
