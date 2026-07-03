# Qallow

**A stateful, self-optimizing cognitive AGI runtime.**

Qallow is a multi-phase reasoning engine built as a hybrid C/Rust/Python stack. It maintains persistent internal state across sessions, manages a semantic memory graph, evolves agent populations, and gates all reasoning behind a hard ethics coherence invariant.

The local LLM (Gemma 4 via Ollama) is the final reasoning surface. The C engine, LMDB state store, ASIOS kernel, and swarm layer are the actual runtime — they handle state continuity, memory retrieval, parameter evolution, and phase orchestration that a raw LLM cannot maintain on its own.

VEYN integration is optional. Qallow runs fully on internal state defaults when no signal source is connected.

---

## Architecture

```
Optional Signal Input (VEYN WebSocket)
              │
              ▼
    qallow-veyn-bridge (Rust)
    WebSocket → LMDB state variables
    (energy / risk / reward_mod / autonomy)
              │
              ▼
       C Cognitive Engine
       ├── Phase 1: Elasticity
       ├── Phase 2: Harmonic
       ├── Phase 3: Coherence
       └── Phase 4: Convergence
              │
       LMDB Semantic Memory Graph
       SQLite Episodic History
       ASIOS Self-Optimization Kernel
       Swarm Evolution Layer
              │
              ▼
       FastAPI Reasoning Server (Python)
       Ollama → Gemma 4
       Optional state context injection
              │
              ▼
       MCP Server / Native FLTK App
```

---

## Phase Architecture

The C engine implements a 4-phase cognitive loop. Each phase is a distinct computational mode with specific entropy, coherence, and convergence characteristics.

| Phase | Name | Function |
|---|---|---|
| 1 | Elasticity | Dynamic adjustment of system entropy and coherence. High plasticity — wide state exploration. |
| 2 | Harmonic | Alignment of internal reward trajectories with state baselines. |
| 3 | Coherence | Synchronization of the semantic memory graph with episodic SQLite logs. |
| 4 | Convergence | Final stabilization of cognitive state before unified execution. Low entropy — commit mode. |

Phase transitions are gated by internal state thresholds. The ethics coherence score (κ) must remain ≥ 0.92 at all times — degradation below this floor triggers a Mandatory Logical Reset (MLR) enforced by the DGK-IES.

---

## ASIOS — Adaptive Self-Optimization Kernel

ASIOS accumulates inter-session learning. It observes session trajectories, identifies parameter configurations that produce favorable outcomes, and refines the runtime's internal weights across sessions.

- **Intra-session**: Adapts parameters in real time based on phase trajectory
- **Inter-session**: Writes refined parameters to persistent storage at session end
- **Session quality scoring**: Each session is scored against internal fitness metrics
- **Parameter evolution**: Winning configurations accumulate weight over time

The longer Qallow runs, the more precisely its parameters are tuned to the specific workload and signal environment it operates in.

---

## Swarm Evolution

Qallow can spawn a population of agent offspring — parameter variations on the current parent configuration. Each offspring is evaluated against a configurable fitness function. Winning configurations are merged back into the parent via weighted blending.

```
Parent Configuration
        │
   Spawn offspring (N variants)
        │
   Evaluate each against fitness function
        │
   Select winners
        │
   Blend weights back into parent
        │
   Parent evolves
```

Fitness functions are domain-configurable — evaluate against any measurable signal the system has access to.

---

## Semantic Memory

The C engine maintains a semantic memory graph backed by LMDB:

- **Cosine-similarity retrieval** — relevant past context surfaces automatically during reasoning
- **Temporal memory** — episodic records stored in SQLite, queryable by session and timestamp
- **No cloud vector DB** — all retrieval runs locally against the LMDB state

This gives Qallow genuine cross-session memory without external infrastructure.

---

## Cognitive State Variables

When VEYN is connected, the signal bridge maps incoming events to these internal state variables:

| Variable | Description | Range |
|---|---|---|
| `energy` | System activation level / overlay stability | [0.0, 1.0] |
| `risk` | Uncertainty and exposure modulation | [0.0, 1.0] |
| `reward_mod` | Reward trajectory alignment | [0.0, 1.0] |
| `autonomy` | Self-directed vs. supervised operation level | [0, 5] |
| `coherence` (κ) | Ethics coherence score — hard invariant | ≥ 0.92 |

These variables modulate phase transition thresholds and shape how context is injected into LLM prompts.

---

## Stack

| Layer | Technology |
|---|---|
| Cognitive engine | C |
| State store | LMDB |
| Session history | SQLite |
| VEYN bridge | Rust (Tokio / Axum) |
| Desktop orchestrator | Rust (FLTK) |
| Reasoning server | Python (FastAPI) |
| Local LLM | Ollama (Gemma 4) |
| MCP interface | Python (FastMCP) |

---

## Setup

**Dependencies:**
```bash
# System
apt install lmdb sqlite3   # Linux
brew install lmdb sqlite3  # macOS
# Windows: vcpkg or prebuilt headers

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma4:latest
```

**Build the native orchestrator:**
```bash
cd Qallow/native_app
cargo build --release
```

**Run the reasoning server:**
```bash
cd Qallow/python
uv sync
uv run uvicorn server:app --host 0.0.0.0 --port 5000
```

**Environment:**
```bash
QALLOW_OLLAMA_MODEL=gemma4:26b
OLLAMA_BASE_URL=http://localhost:11434
QALLOW_LLM_TIMEOUT=300
QALLOW_CONTEXT_INJECT=true   # inject state prefix into LLM prompts
```

---

## API

The FastAPI server exposes Qallow's full runtime over HTTP:

| Endpoint | Method | Description |
|---|---|---|
| `/metrics` | GET | Current cognitive state — phase, coherence, ethics score, biometrics |
| `/phase/{n}/start` | POST | Start phase 1–5 with tick budget |
| `/phase/stop` | POST | Stop current phase, trigger ASIOS adaptation step |
| `/chat` | POST | Send prompt to local LLM with optional state context prefix |
| `/logs` | GET | Audit log — phase transitions, chat calls, swarm events |
| `/export` | GET | Full state snapshot |
| `/swarm/spawn` | POST | Spawn agent offspring population |
| `/swarm/evolve` | POST | Evaluate offspring and blend winners into parent |
| `/swarm/status` | GET | Current swarm state |
| `/asios/state` | GET | Full ASIOS kernel state |
| `/asios/observe` | POST | Feed an observation into the ASIOS kernel |
| `/asios/adapt` | POST | Trigger intra-session adaptation |
| `/asios/end_session` | POST | Finalize session — write inter-session parameters |

---

## Core Invariants

- **κ ≥ 0.92** — Coherence below this threshold triggers MLR. Enforced by DGK-IES. Non-negotiable.
- **No `unwrap()` in production paths** — all error propagation via `anyhow::Context` or `thiserror`
- **Bounded channels** — backpressure handled at every async boundary
- **Clamped biometric entry points** — hardware spike protection via `clamp()` macros at all signal ingestion points

---

## License

MIT © XINGXERX / CGX
