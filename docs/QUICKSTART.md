# Qallow Quickstart

This guide consolidates environment preparation, dependency installation, and CUDA setup into one document. Use it alongside `README.md` for a smooth onboarding experience.

## System Requirements

- 64-bit Linux (Ubuntu 22.04+ recommended). macOS builds are experimental. Windows requires WSL2 with CUDA support.
- CPU: x86_64 with AVX2 support.
- GPU (optional): NVIDIA RTX 30xx/40xx, compute capability ≥ 8.0. The build scripts auto-detect CUDA availability.
- RAM: ≥ 16 GB for full phase pipeline, ≥ 8 GB for CPU-only research mode.
- Disk: ≥ 5 GB free for build artifacts, telemetry logs, and datasets.

## Required Dependencies

| Dependency | Minimum Version | Purpose |
| --- | --- | --- |
| CMake | 3.20 | Configure modular build targets |
| Ninja or GNU Make | Latest | Parallel builds |
| GCC | 11 | C11 compiler |
| CUDA Toolkit (optional) | 12.0 | GPU acceleration |
| Python | 3.10 | Telemetry collectors, scripts |
| Nsight Compute (optional) | 2023.1 | GPU profiling |
| `clang-format` | 15 | Code formatting |
| `markdownlint` | Latest | Documentation linting |
| `pip` packages | see below | Tooling |

Install prerequisite packages (Ubuntu example):

```bash
sudo apt update
sudo apt install -y build-essential ninja-build cmake python3 python3-pip clang-format
pip3 install markdownlint-cli cmake-format
```

### CUDA Toolkit

1. Verify GPU visibility:
   ```bash
   nvidia-smi
   ```
2. Install the CUDA Toolkit:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
   sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
   sudo apt update
   sudo apt install -y cuda-toolkit-12-4
   ```
3. Export environment variables (add to `.env` or shell profile):
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH="$CUDA_HOME/bin:$PATH"
   export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
   ```

## Environment Configuration

Copy `.env.example` to `.env` and adjust values:

```ini
QALLOW_ENABLE_CUDA=auto
QALLOW_LOG_DIR=data/logs
QALLOW_PROFILE=native
QALLOW_DEFAULT_PHASE=13
QALLOW_METRICS_FORMAT=csv,jsonl
```

The runtime reads `.env` at startup via `qallow_env_load()` (see `src/runtime/env.c`).

## Building

```bash
./scripts/build_all.sh              # configure + build
cmake --build build --target check  # static analysis
ctest --test-dir build              # unit/integration tests
```

`build_all.sh` performs the following:

1. Detect CUDA with `nvcc --version` (optional).
2. Configure CMake with `-DQALLOW_ENABLE_CUDA={ON/OFF}`.
3. Build primary binaries (`qallow`, `qallow_unified`, `qallow_examples`).
4. Generate compile commands for IDE tooling.

To force CPU-only mode: `./scripts/build_all.sh --cpu`.

## Running the VM

```bash
./build/qallow --phase=13 --ticks=400 --log=data/logs/phase13.csv
```

- `--phase` accepts `1` through `13`, `phase12`, `phase13`, or `vm` for the full loop.
- `--ticks` controls iterations for simulation-heavy phases.
- `--log` overrides default telemetry CSV path.
- Set `QALLOW_METRICS_FORMAT=jsonl` to emit newline-delimited JSON.

## Profiling and Logging

- Use `QALLOW_PROFILE=nsight` to emit NVTX ranges for Nsight Compute/Systems.
- Logs are stored in `data/logs/` by default. The telemetry schema is described in `docs/ARCHITECTURE_SPEC.md`.
- The logging subsystem (`src/runtime/logging.cpp`) uses `spdlog` for structured output.

## Examples and Benchmarks

Build and run:

```bash
cmake --build build --target qallow_examples
./build/examples/phase7_demo --ticks=100
./build/examples/benchmarks/throughput_bench --mode=cuda
```

Example outputs are stored in `examples/output/` to keep repo clean.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `nvcc: command not found` | Ensure CUDA Toolkit is installed and `CUDA_HOME` exported. Run `./scripts/build_all.sh --cpu` to continue without CUDA. |
| Missing `spdlog` headers | Run `cmake --build build --target qallow_runtime` (FetchContent will download `spdlog`). |
| `GLIBCXX` mismatch inside Docker | Use the provided `Dockerfile` and `docker-compose.yaml` for a consistent build environment. |
| Tests fail with file-permission errors | Ensure `data/logs/` exists or run `mkdir -p data/logs`. |

## Next Steps

- Read `docs/ARCHITECTURE_SPEC.md` for deep-dive architecture.
- Inspect `examples/phaseX_demo.cu` for per-phase CUDA kernels.
- Update `config/manifest.json` when you add new phases or change the API surface.
