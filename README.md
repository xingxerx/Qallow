# Qallow 

**Qallow Quantum-Photonic AGI System**

Qallow is a cutting-edge cognitive runtime and physiological bridge that creates a unified feedback loop between human biological states and artificial general intelligence reasoning. By leveraging advanced physiological metrics (HRV, EEG Beta Coherence, SpO2) through the VEYN signal bridge, Qallow dynamically gates and modulates its internal processing phases.

## Architecture

The system is organized into a hybrid C/Rust/Python stack:

- **Cognitive Engine (C)**: The core algorithmic workhorse. Features a fast semantic memory graph (LMDB-backed), cosine-similarity based temporal memory, and modular phase processors (Phases 12–15) that handle elasticity, harmonic convergence, and ethical coherence scoring.
- **VEYN Bridge (Rust)**: Ingests real-time physiological telemetry via websocket (`ws://localhost:7700/stream`). Maps biological indicators (e.g., HRV to energy, EEG Beta to risk) directly into the shared LMDB state. Detects REM sleep to trigger Øneiro dream-state snapshots.
- **Reasoning Engine (Python)**: Uses a local Ollama instance (Gemma 4) as the reasoning core. Injects real-time physiological context prefixes into all system prompts, ensuring the AGI's context window is physically coupled to the operator's biometric state.
- **Native Orchestrator (Rust)**: The `qallow-native` FLTK desktop application that monitors the unified loop, handling the phase transitions strictly gated by biometric thresholds and the CANON's ethical floor ($\kappa > 0.92$).

## Phase Architecture

Qallow's pipeline is conceptualized across 20 distinct processing phases. The core C engine explicitly implements **Phases 12 through 15**:

- **Phase 12 (Elasticity)**: Dynamic adjustment of system entropy and coherence.
- **Phase 13 (Harmonic)**: Alignment of internal reward trajectories with physiological baselines.
- **Phase 14 (Coherence)**: Synchronization of the semantic memory graph with the episodic SQLite logs.
- **Phase 15 (Convergence)**: Final stabilization of the cognitive state before unified execution.

*Note: Phases 1–11 represent lower-level sensory ingestion, raw data preprocessing, and initial boot sequences that are abstracted away or handled directly by the VEYN and Øneiro sensor bridges.*

## Setup

1. **Dependencies**: 
   - Ensure `lmdb` and `sqlite3` development headers are installed.
   - Install Rust and Cargo for the Native App and VEYN Bridge.
   - Install Ollama and pull the Gemma 4 model (`ollama run gemma4`).

2. **Build**:
   The native application can be built via standard cargo commands:
   ```bash
   cd native_app
   cargo build --release
   ```

3. **Run**:
   Start the VEYN websocket stream, run Ollama, and launch the Native App to orchestrate the unified phase loop.
