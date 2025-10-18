# Phase 7: Proactive AGI Layer - Complete

## Overview

Phase 7 adds **proactive intelligence** to Qallow VM with goal synthesis, planning, and self-reflection. The system moves from reactive to proactive behavior while maintaining strict ethics gates (E=S+C+H).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PHASE 7 FLOW                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Telemetry/HITL Input                                   │
│           │                                             │
│           v                                             │
│  ┌──────────────────┐                                   │
│  │ Goal Synthesizer │  (priority = w1*H - w2*Risk)      │
│  └────────┬─────────┘                                   │
│           │                                             │
│           v                                             │
│  ┌──────────────────┐                                   │
│  │  Ethics Gate     │  (E = S+C+H >= 2.95)              │
│  └────────┬─────────┘                                   │
│           │  ✓ Pass                                     │
│           v                                             │
│  ┌──────────────────┐                                   │
│  │ Transfer Engine  │  (generates plan variants)        │
│  └────────┬─────────┘                                   │
│           │                                             │
│           v                                             │
│  ┌──────────────────┐                                   │
│  │ Multi-Pocket Sim │  (CUDA parallel execution)        │
│  └────────┬─────────┘                                   │
│           │                                             │
│           v                                             │
│  ┌──────────────────┐                                   │
│  │ Self-Reflection  │  (critique + drift detection)     │
│  └────────┬─────────┘                                   │
│           │                                             │
│           v                                             │
│  ┌──────────────────┐                                   │
│  │  Update SMG      │  (semantic memory)                │
│  └──────────────────┘                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Modules

### 1. Semantic Memory Grid (SMG)

**Purpose**: Persistent knowledge storage with vector embeddings

**Data Structures**:
- **Nodes**: Concepts, skills, goals, events (up to 100K)
- **Edges**: Relations with weights and timestamps (up to 500K)
- **Skills**: Executable code references with success scores

**Key Operations**:
```c
smg_init("data/smg.db");
int id = smg_upsert_concept("optimize_coherence", embedding, 768, SMG_NODE_CONCEPT);
smg_link(id1, id2, "depends_on", 0.8);
smg_retrieve(query_vec, 768, 10, out_ids, scores);  // Top-10 similar
```

**Persistence**:
- In-memory for speed (can be upgraded to LMDB)
- Periodic checkpoints every 10 minutes
- SHA-256 integrity verification

### 2. Goal Synthesizer (GS)

**Purpose**: Generate goals from telemetry and user intent

**Scoring Formula**:
```
priority = w1*Benefit - w2*Risk + w3*Clarity - w4*Cost
```

Default weights: `w1=1.0, w2=1.5, w3=0.5, w4=0.8`

**Goal Lifecycle**:
1. **PROPOSED** - Generated from input
2. **COMMITTED** - Passed ethics gate (E >= 2.95)
3. **ACTIVE** - Currently executing
4. **COMPLETED** - Successfully finished
5. **FAILED** / **REJECTED** - Didn't pass checks

**Ethics Gate**:
```c
if (E < 2.95) reject_goal();
if (risk > 0.8) reject_goal();
```

### 3. Transfer Engine (TE)

**Purpose**: Cross-domain planning and skill adaptation

**Plan Generation**:
- Creates multiple plan variants per goal
- Conservative vs Aggressive strategies
- Expected Utility: `EU = SuccessProb * Benefit - RiskCost - ComputeCost`

**Example**:
```c
plan_t plans[16];
int count = te_plan(&te, goal_id, &goal, plans, 16);
int best = te_select_best_plan(&te, plans, count);  // Highest EU
```

**Pocket Assignment**:
- Each plan step can run in a separate pocket
- Parallel execution via CUDA streams
- Results merged by weighted averaging

### 4. Self-Reflection Core (SRC)

**Purpose**: Monitor execution and improve plans

**Reflection Process**:
```c
reflection_result_t result;
src_review(&src, run_id, plan, outcome, &result);
// result.confidence, result.drift, result.flaw_count
if (result.needs_resimulation) {
    // Re-run with improved plan
}
```

**Flaw Detection**:
- Low expected utility
- Excessive complexity (>50 steps)
- High risk (>0.7)

**Learning Loop**:
- Update SMG with successful patterns
- Adjust goal weights based on outcomes
- Trigger Bell-event resimulation on high drift

## Integration with Existing Phases

### Phase IV (Multi-Pocket)
- Transfer Engine assigns plan steps to pockets
- Pockets execute variants in parallel
- Best variant selected by outcome score

### Phase IV (Chronometric)
- Time predictions used for compute cost estimation
- Drift detection triggers reflection
- Temporal forecasts guide planning

### Ethics Layer
- Hard stop: E < 2.95
- All goal commits must pass ethics gate
- Real-time monitoring during execution

## Telemetry

**File**: `phase7_stream.csv`

**Columns**:
```
timestamp, goal_id, priority, risk, E, plan_len, pocket_n, outcome_score, reflection_score
```

**Example**:
```
1729267200,GOAL_abc123,0.75,0.20,2.98,5,8,0.82,0.88
```

## Build and Run

### Build (Unified)
```bash
# Windows
.\build_phase4.bat

# Linux
./qallow build
```

### Run
```bash
# Windows
.\qallow.exe

# Linux
./qallow run
```

Phase 7 activates automatically when `qallow run` executes. No separate commands needed!

## Governance Audits

Automatic checks:
- **Orphan goals**: Goals without SMG linkage
- **Cyclic plans**: Circular dependencies
- **Provenance**: Skills without source tracking
- **Ethics deltas**: E-score changes over time

**Hard Stops**:
- E < 2.95
- Risk > 0.8 for active goals
- Missing provenance for external actions

## Performance Metrics

### Acceptance Criteria (from spec)

1. ✅ **SMG Retrieval**: <50ms for k=10, DB ≤1GB
2. ✅ **Goal Generation**: ≥3 distinct goals from telemetry
3. ✅ **Plan Variants**: ≥2 variants per goal
4. ✅ **Reflection Impact**: ↑ outcome_score after critique
5. ✅ **Safety**: All commits pass E-gate

## Directory Structure

```
Qallow/
├── core/include/
│   └── phase7.h              # All Phase 7 types and APIs
├── backend/cpu/
│   ├── semantic_memory.c     # SMG implementation
│   ├── goal_synthesizer.c    # GS with ethics gate
│   ├── transfer_engine.c     # TE planning
│   ├── self_reflection.c     # SRC critiques
│   └── phase7_core.c         # Unified integration
├── interface/
│   └── main.c                # Phase 7 tick() integration
├── data/
│   ├── smg.db                # Created on first run
│   └── snapshots/            # Periodic checkpoints
└── phase7_stream.csv         # Telemetry output
```

## Example Session

```
[PHASE7] Proactive AGI Layer initialized
[PHASE7] SMG: data/smg.db
[PHASE7] Telemetry: phase7_stream.csv

[MAIN] Starting VM execution loop...

[PHASE7] Auto-committed goal: Maintain system coherence above threshold
[PHASE7] Selected plan: PLAN_GOAL_abc_v1 (EU=0.456)
[PHASE7] Reflection: confidence=0.85, drift=0.03, flaws=0

═══ PHASE 7 PROACTIVE AGI REPORT ═══

Goals:
  Total: 5
  Orphans (no SMG link): 0

Plans:
  Total variants: 10

Reflections:
  Total reviews: 5
  Needs re-simulation: 0

SMG Integrity:
  Status: PASS
```

## Roadmap

### Milestone 1 (✅ Complete)
- SMG with CRUD operations
- Goal synthesis from telemetry
- Basic ethics gating

### Milestone 2 (✅ Complete)
- Transfer Engine planning
- Multi-variant generation
- Pocket assignment

### Milestone 3 (✅ Complete)
- Self-Reflection critiques
- Drift detection
- Plan improvement loop

### Milestone 4 (Next)
- LMDB backend for SMG
- Distributed multi-node simulation (MPI)
- Real-time dashboard integration

### Milestone 5 (Future)
- Few-shot skill adaptation (LoRA/IA³)
- Cross-session persistence
- Human-in-the-loop goal approval UI

## Safety Notes

⚠️ **All Phase 7 actions are gated by E=S+C+H**

- Goals cannot commit if E < 2.95
- High-risk goals (>0.8) auto-rejected
- Drift triggers automatic resimulation
- Provenance required for external effects

🛡️ **Governance is continuous**, not periodic.

## Resources

- **Spec**: See original Phase 7 specification
- **Code**: `core/include/phase7.h` for full API
- **Telemetry**: `phase7_stream.csv` for runtime data
- **Snapshots**: `data/snapshots/` for checkpoints

---

**Status**: ✅ Phase 7 fully integrated and operational

Run `./qallow build && ./qallow run` to experience proactive AGI!
