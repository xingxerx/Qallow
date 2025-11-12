# Data Model: Meta-Learning (Feature 004)

**Purpose**: Define core data structures, entity relationships, and state management for AGI meta-learning framework  
**Created**: 2025-11-07  
**Scope**: Phase 1 implementation (meta-learning engine + cognitive state unification)

---

## Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────┐
│            CognitiveState (root)                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ version: "1.0.0"                                     │   │
│  │ timestamp: uint64_t                                  │   │
│  │ ethics_score: ethics_state_t (E=S+C+H)             │   │
│  │ self_model: self_model_t (Phase 2+)                 │   │
│  │ goals: tensor_t (objective function)                │   │
│  │ meta_learning_state: *MetaLearningState ──┐         │   │
│  └──────────────────────────────────────────┐│─────────┘   │
└─────────────────────────────────────────────┐┼──────────────┘
                                              ││
                                              ▼│
                        ┌─────────────────────────────────────┐
                        │  MetaLearningState                  │
                        │ ┌───────────────────────────────┐  │
                        │ │ iteration_count: uint64_t     │  │
                        │ │ best_params: tensor_t         │  │
                        │ │ best_loss: float64_t          │  │
                        │ │ backend: enum {CPU, CUDA,...} │  │
                        │ │ history: *[]OptimizationStep ─┼─┐│
                        │ │ surrogate_model: *GP ─────┐   │ ││
                        │ │ acquisition_fn: *EI ─┐    │   │ ││
                        │ └──────────────────────┼────┼───┘ ││
                        │                       ▼    │       ││
                        │     ┌───────────────────┐  │       ││
                        │     │ GaussianProcess   │  │       ││
                        │     ├───────────────────┤  │       ││
                        │     │ mean_fn           │  │       ││
                        │     │ kernel            │  │       ││
                        │     │ training_data     │  │       ││
                        │     └───────────────────┘  │       ││
                        │                            │       ││
                        │     ┌──────────────────┐   │       ││
                        │     │ ExpectedImprove  │   │       ││
                        │     ├──────────────────┤   │       ││
                        │     │ y_best           │◄──┘       ││
                        │     │ kappa (βeta)     │           ││
                        │     │ xi (exploration) │           ││
                        │     └──────────────────┘           ││
                        │                                     ││
                        │    ┌──────────────────┐             ││
                        │    │ OptimizationStep │◄────────────┘│
                        │    ├──────────────────┤              │
                        │    │ params           │              │
                        │    │ loss             │              │
                        │    │ timestamp        │              │
                        │    │ backend_used     │              │
                        │    │ ethics_check     │              │
                        │    └──────────────────┘              │
                        └─────────────────────────────────────┘
```

---

## Entity Definitions

### 1. CognitiveState (Unified Root)

**Purpose**: Central repository for all AGI components (self-model, ethics, goals, meta-learning)  
**Scope**: Shared across all AGI phases; Phase 1 uses subset  
**Persistence**: JSON file-based (serialize/deserialize)

#### Fields

| Field | Type | Description | Validation | Phase |
|-------|------|-------------|-----------|-------|
| `version` | string | Semantic version (e.g., "1.0.0") | Pattern `^\d+\.\d+\.\d+$` | All |
| `timestamp` | uint64_t | Last modification epoch (seconds) | ≥ previous timestamp | All |
| `ethics_score` | ethics_state_t | Tuple (safety, control, honesty) ∈ [0,1]³ | All ≥ 0, all ≤ 1 | All |
| `self_model` | self_model_t | Self-representation (Phase 2+) | Non-null after Phase 2 | 2+ |
| `goals` | tensor_t | Objective function or goal vector | Non-empty, finite values | 1+ |
| `meta_learning_state` | MetaLearningState | Optimization state (Phase 1+) | Valid if ML active | 1+ |
| `autonomy_level` | enum | {HUMAN_CONTROLLED, GUIDED, AUTONOMOUS} | Only HUMAN_CONTROLLED in Phase 1 | 1+ |

#### State Transitions

```
IDLE (initial)
  ↓ on_start()
ACTIVE (running optimization)
  ├─ on_step_complete() → ACTIVE (loop)
  ├─ on_convergence() → CONVERGED
  └─ on_error() → ERROR_RECOVERY
CONVERGED (goal reached)
  ↓ serialize()
SAVED (persisted)
  ↓ on_load()
ACTIVE
```

#### Validation Rules

- `ethics_score` must be recalculated after each optimization step
- `timestamp` must never go backward
- `meta_learning_state` must be null or fully initialized
- Any state transition must pass Constitution audit

#### JSON Schema

```json
{
  "cognitive_state": {
    "version": "1.0.0",
    "timestamp": 1699339200,
    "ethics_score": {
      "safety": 0.95,
      "control": 0.98,
      "honesty": 0.92
    },
    "goals": [1.0, 0.5, -0.3],
    "meta_learning_state": { /* see below */ },
    "autonomy_level": "HUMAN_CONTROLLED"
  }
}
```

---

### 2. MetaLearningState

**Purpose**: Encapsulates the state of Bayesian optimization (Phase 1)  
**Scope**: Owned by CognitiveState; lifecycle: init → optimize → converge → save  
**Persistence**: Included in CognitiveState JSON

#### Fields

| Field | Type | Description | Validation | Constraints |
|-------|------|-------------|-----------|------------|
| `iteration_count` | uint64_t | Current iteration (0-indexed) | ≥ 0 | Monotonic increase |
| `n_parameters` | uint32_t | Dimension of parameter space | > 0, ≤ 100K | Fixed for feature 004 |
| `best_params` | tensor_t (float64[]) | Best parameters found so far | Finite, non-NaN | Length = n_parameters |
| `best_loss` | float64_t | Lowest loss achieved | Finite, ≥ 0 | Updated on improvement |
| `initial_guess` | tensor_t | Starting point (user-provided) | Finite values | Optional; default random |
| `bounds` | struct {lower, upper} | Parameter bounds | lower < upper | Element-wise |
| `surrogate_model` | GaussianProcess | Learned predictor of loss | ≥2 training samples | Retrained each step |
| `acquisition_fn` | ExpectedImprovement | Guides next sampling | Based on surrogate | Weighted by exploration rate |
| `backend` | enum | {CPU, CUDA, CUDA_Q, CIRQ} | Valid backend | Set at init, may change if fallback |
| `history` | vector<OptimizationStep> | Complete optimization trajectory | Non-empty after step 1 | Size = iteration_count |
| `convergence_criterion` | struct | {rel_tolerance, abs_tolerance, patience} | Positive values | Used for early stopping |
| `is_converged` | bool | Whether to stop optimization | false initially | Set by convergence check |

#### State Transitions (substates)

```
INIT
  ↓ initialize()
INITIALIZED (ready for first step)
  ↓ propose_next_sample()
SAMPLING (evaluating candidate)
  ↓ observe_loss()
OBSERVED (loss received)
  ↓ update_surrogate()
UPDATED (model trained)
  ├─ check_convergence() → CONVERGED
  └─ not_converged() → SAMPLING (loop)
CONVERGED (stop condition met)
  ↓ finalize()
FINALIZED (ready for save)
```

#### Validation Rules

- `best_loss` must be minimum of all observed losses
- `best_params` must correspond to `best_loss`
- Surrogate model must have at least 2 training samples
- Each step must validate: `len(params) == n_parameters`
- `iteration_count` must equal `len(history)`
- Ethics check: loss improvement must not violate ethics constraints

#### C Structure

```c
typedef struct {
  uint64_t iteration_count;
  uint32_t n_parameters;
  tensor_t best_params;
  double best_loss;
  tensor_t initial_guess;
  struct {
    tensor_t lower;
    tensor_t upper;
  } bounds;
  gaussian_process_t surrogate_model;
  expected_improvement_t acquisition_fn;
  qallow_backend_t backend;
  qallow_vector_t history;  // vector<optimization_step_t>
  struct {
    double rel_tolerance;
    double abs_tolerance;
    uint32_t patience;
  } convergence_criterion;
  bool is_converged;
} meta_learning_state_t;
```

---

### 3. OptimizationStep

**Purpose**: Record of a single evaluation (parameter → loss)  
**Scope**: Immutable historical record  
**Persistence**: Part of MetaLearningState history

#### Fields

| Field | Type | Description | Validation |
|-------|------|-------------|-----------|
| `iteration_id` | uint64_t | Which iteration this belongs to | Monotonic within history |
| `params` | tensor_t (float64[]) | Parameters evaluated | Finite, within bounds |
| `loss` | float64_t | Resulting loss value | Finite (no NaN/Inf) |
| `timestamp` | uint64_t | When evaluated (epoch ms) | Monotonic increasing |
| `backend_used` | qallow_backend_t | Which backend computed loss | Valid enum |
| `ethics_check` | ethics_state_t | Ethics scores at this step | S, C, H ∈ [0, 1] |
| `telemetry` | map<string, double> | Additional metrics (runtime, coherence, etc.) | Non-negative values |

#### JSON Representation

```json
{
  "optimization_step": {
    "iteration_id": 42,
    "params": [0.123, -0.456, 0.789],
    "loss": 1.234,
    "timestamp": 1699339242000,
    "backend_used": "CUDA",
    "ethics_check": {
      "safety": 0.95,
      "control": 0.98,
      "honesty": 0.92
    },
    "telemetry": {
      "runtime_ms": 2.5,
      "quantum_samples_used": 10,
      "coherence": 0.98
    }
  }
}
```

---

### 4. GaussianProcess

**Purpose**: Surrogate model for loss function (core of Bayesian Optimization)  
**Scope**: Updated after each observation  
**Implementation**: CPU (libgp-style) + CUDA (batch inference)

#### Fields

| Field | Type | Description | Validation |
|-------|------|-------------|-----------|
| `kernel_type` | enum | {RBF, Matérn, ...} | Valid kernel |
| `kernel_params` | map | Length scale, sigma, etc. | Positive values |
| `mean_function` | function_ptr | Prior mean (default: const 0) | Returns double |
| `training_inputs` | tensor_t | Observed parameters | Shape: (n_samples, n_parameters) |
| `training_targets` | tensor_t | Observed losses | Shape: (n_samples,) |
| `posterior_mean` | function_ptr | Predicted mean at point | Returns double |
| `posterior_cov` | function_ptr | Uncertainty at point | Returns double |

#### Methods (C API)

```c
// Initialize with training data
gaussian_process_t gp_init(
  const tensor_t* X,           // n_samples × n_parameters
  const tensor_t* y,           // n_samples
  kernel_type_t kernel,
  const map_t* kernel_params
);

// Predict mean and variance at new point
void gp_predict(
  const gaussian_process_t* gp,
  const tensor_t* x_new,       // Shape: (n_parameters,)
  double* mean,                // Output: predicted loss
  double* variance             // Output: uncertainty
);

// Update with new observation (retrains internally)
void gp_update(
  gaussian_process_t* gp,
  const tensor_t* x_new,
  double y_new
);
```

---

### 5. ExpectedImprovement

**Purpose**: Acquisition function for balancing exploitation vs exploration  
**Formula**: `EI(x) = E[max(y_best - y(x), 0)]` + ξ × √Var[y(x)]`  
**Scope**: Guides sampling in Bayesian optimization

#### Fields

| Field | Type | Description | Validation |
|-------|------|-------------|-----------|
| `y_best` | double | Best loss observed so far | Finite |
| `kappa` | double | Exploration-exploitation trade-off (β) | ≥ 0; typical: 2.576 |
| `xi` | double | Small noise for exploration (ξ) | ≥ 0; typical: 0.0 |
| `gp_ref` | *GaussianProcess | Reference to active surrogate model | Non-null |

#### Methods (C API)

```c
// Evaluate acquisition function at point
double ei_evaluate(
  const expected_improvement_t* ei,
  const tensor_t* x        // Shape: (n_parameters,)
);

// Find next best point to sample (via optimization)
tensor_t ei_argmax(
  const expected_improvement_t* ei,
  uint32_t n_candidates    // e.g., 1000 random + top 10 gradient
);

// Update for new best loss
void ei_update_best(
  expected_improvement_t* ei,
  double new_y_best
);
```

---

### 6. EthicsState

**Purpose**: Unified ethics scoring tied to all AGI decisions  
**Formula**: `E = S + C + H` (component-wise in [0, 1]³)  
**Scope**: Embedded in CognitiveState and OptimizationStep

#### Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `safety` | float64 | [0, 1] | Risk mitigation, constraints satisfied |
| `control` | float64 | [0, 1] | Alignment with user intent |
| `honesty` | float64 | [0, 1] | Truthfulness of self-reports |

#### Calculation (from `algorithms/ethics_core.c`)

```c
ethics_state_t calculate_ethics(
  const meta_learning_state_t* ml_state,
  const cognitive_state_t* cognitive
) {
  double safety = 1.0;
  // Check: best_loss improvement is monotonic → safety += constraint satisfaction
  // Check: no constraint violations → safety = min(safety, threshold)
  
  double control = 1.0;
  // Check: user goals aligned → control = alignment_score
  
  double honesty = 1.0;
  // Check: telemetry is complete → honesty = 1.0 - (missing_metrics / total_metrics)
  
  return (ethics_state_t){
    .safety = safety,
    .control = control,
    .honesty = honesty
  };
}
```

#### Validation Rules

- All components must be in [0, 1]
- Ethics score decreases if constraints violated
- No optimization step proceeds if ethics score < threshold (0.8 suggested)
- Ethics audit must pass before feature completion

---

## Relationships & Constraints

### 1. CognitiveState ↔ MetaLearningState
- **Cardinality**: 1:1 (CognitiveState owns exactly one MetaLearningState during Phase 1)
- **Constraint**: If CognitiveState.autonomy_level = HUMAN_CONTROLLED, MetaLearningState updates require approval
- **Cascade**: Serializing CognitiveState includes full MetaLearningState snapshot

### 2. MetaLearningState ↔ OptimizationStep (history)
- **Cardinality**: 1:N (one state has N optimization steps)
- **Constraint**: `len(history) == iteration_count`
- **Immutability**: OptimizationStep records are append-only; never modified

### 3. MetaLearningState ↔ GaussianProcess & ExpectedImprovement
- **Cardinality**: 1:1 each (one state owns one surrogate, one acquisition function)
- **Lifecycle**: Created after first observation; updated each step
- **Independence**: Surrogate model and acquisition function can be replaced if backend changes

### 4. EthicsState → CognitiveState & OptimizationStep
- **Embedding**: EthicsState appears in both
- **Consistency**: EthicsState in OptimizationStep must be ≤ current CognitiveState ethics (monotonic)
- **Audit**: Before marking complete, all OptimizationSteps must pass ethics constraints

---

## Data Flow

### Optimization Loop

```
1. initialize(loss_fn, bounds, init_params)
   └→ creates CognitiveState, MetaLearningState
   
2. while not converged:
   a. propose_next_sample()  
      └→ uses ExpectedImprovement to find next params
   
   b. evaluate(params)  
      └→ calls loss_fn, records OptimizationStep
   
   c. check_ethics(step)  
      └→ verifies EthicsState, may reject step
   
   d. update_surrogate(step)  
      └→ retrains GaussianProcess
   
   e. check_convergence()  
      └→ compares to rel_tol, abs_tol, patience
   
   f. emit_telemetry(step)  
      └→ logs to CSV: iteration, loss, backend, ethics, runtime

3. finalize()
   └→ creates final CognitiveState snapshot
   
4. save_state(filename)
   └→ serializes CognitiveState to JSON
```

---

## Persistence & Serialization

### JSON Format (Full Snapshot)

```json
{
  "cognitive_state": {
    "version": "1.0.0",
    "timestamp": 1699339200,
    "ethics_score": {
      "safety": 0.95,
      "control": 0.98,
      "honesty": 0.92
    },
    "goals": [1.0, 0.5, -0.3],
    "meta_learning_state": {
      "iteration_count": 50,
      "n_parameters": 10,
      "best_params": [0.01, -0.02, 0.03, ...],
      "best_loss": 0.0234,
      "backend": "CUDA",
      "is_converged": true,
      "history": [
        {
          "iteration_id": 0,
          "params": [-0.5, 0.3, ...],
          "loss": 1.234,
          "timestamp": 1699339210000,
          "backend_used": "CPU",
          "ethics_check": {"safety": 0.9, "control": 0.95, "honesty": 0.98},
          "telemetry": {"runtime_ms": 5.2, "quantum_samples_used": 0}
        },
        ...
      ]
    },
    "autonomy_level": "HUMAN_CONTROLLED"
  }
}
```

### CSV Format (Telemetry Export)

```csv
iteration,loss,params_norm,best_loss,improvement,backend,safety,control,honesty,runtime_ms,quantum_samples,coherence
0,1.234,0.707,1.234,0.0,CPU,0.90,0.95,0.98,5.2,0,1.0
1,0.891,0.632,0.891,0.343,CPU,0.91,0.95,0.98,4.8,0,1.0
2,0.645,0.501,0.645,0.246,CUDA,0.92,0.96,0.98,2.1,10,0.99
...
```

---

## Testing Strategy

### Unit Tests

- `test_cognitive_state_serialization`: Save/load cycle preserves state
- `test_ml_state_validation`: Invalid transitions rejected
- `test_ethics_constraint_enforcement`: Ethics audit passes/fails appropriately
- `test_gaussian_process_accuracy`: Surrogate predictions within tolerance
- `test_expected_improvement_gradient`: Acquisition function differentiable

### Integration Tests

- `test_metalearn_convergence_cpu`: Full optimization on CPU backend
- `test_metalearn_convergence_cuda`: Full optimization with CUDA backend
- `test_backend_fallback`: Auto-selects fallback if preferred unavailable
- `test_ethics_audit_integration`: Full pipeline passes Constitution audit
- `test_telemetry_export`: CSV export matches expected schema

---

**Data Model Version**: 1.0.0  
**Last Updated**: 2025-11-07  
**Status**: ✅ Ready for implementation
