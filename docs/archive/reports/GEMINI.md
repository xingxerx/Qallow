# Qallow Autonomous Intelligence Runtime

## Project Overview

Qallow is an experimental autonomous intelligence runtime that blends quantum computing, photonic simulation, and an ethics-first design. It is a research platform for AGI development, with a strong emphasis on ethics. The project is comprised of 20 research phases that can be executed from a single entry point, with deterministic telemetry for reproducible analysis.

The project is built with a mix of technologies:

*   **C/C++:** The core runtime, including the CPU and CUDA backends, is written in C/C++. It uses CMake for building.
*   **CUDA:** The project supports CUDA for GPU acceleration of the photonic and quantum simulations.
*   **Python:** Python is used for scripting, examples, and integration with cirq.
*   **Rust:** A native GUI application is built with Rust and the FLTK toolkit. There is also a `quantum_optimizer` crate, which provides lightweight, hybrid quantum-classical optimization helpers, including a simple simulator for QAOA.
*   **Docker:** The project includes a `Dockerfile` and `docker-compose.yaml` for building and running the application in a containerized environment.

## Building and Running

### Prerequisites

*   `cmake` >= 3.20
*   `gcc` >= 11 (or `clang` >= 15)
*   `python` >= 3.10
*   `ninja` or `make`
*   (Optional) CUDA Toolkit >= 12.0

### Building the Project

The project can be built using the provided shell script:

```bash
./scripts/build_all.sh
```

This script will auto-detect CUDA and build the project accordingly. You can also force a CPU or CUDA build with the `--cpu` or `--cuda` flags, respectively.

To build within a Docker container:

```bash
docker compose up --build
```

### Running Simulations

The main executable is `qallow`, located in the `build` directory. You can run a specific phase like this:

```bash
./build/qallow --phase=13 --ticks=400 --log=data/logs/phase13.csv
```

### Running the Native GUI

The Rust-based native GUI can be run with Cargo:

```bash
cargo run --package native_app
```

## Development Conventions

*   **Logging:** The project uses `spdlog` for logging in the C++ components and `env_logger` in the Rust components.
*   **Testing:** The C++ components are tested with CTest. The `scripts/build_all.sh` script runs the tests after building.
*   **Code Style:** The project uses `clang-format` for C/C++ code and `rustfmt` for Rust code.
*   **Contributions:** The `CONTRIBUTING.md` file (which I have not read) likely contains information about contribution guidelines.