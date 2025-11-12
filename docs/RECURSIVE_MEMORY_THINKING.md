# Recursive Memory-Based Thinking System

## Overview

The Recursive Memory-Based Thinking System implements a cognitive feedback loop where the AGI's thinking outputs become inputs for future thinking, creating a self-improving strategic reasoning capability.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  RECURSIVE THINKING CYCLE                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │  1. STORE THINKING OUTPUT                │
        │     • Current state → Memory             │
        │     • Strategic decisions                │
        │     • Effectiveness metrics              │
        └──────────────────┬───────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │  2. LOAD THINKING INPUT                  │
        │     • Retrieve relevant past thinking    │
        │     • Context-aware recall               │
        │     • Blend with current state           │
        └──────────────────┬───────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │  3. EXTRACT STRATEGY PATTERNS            │
        │     • Cluster similar strategies         │
        │     • Track success rates                │
        │     • Build pattern library              │
        └──────────────────┬───────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │  4. GENERATE UPDATED STRATEGY            │
        │     • Apply learned patterns             │
        │     • Evolve strategies                  │
        │     • Increase confidence                │
        └──────────────────┬───────────────────────┘
                           │
                           ▼
                    [NEXT CYCLE]
```

## Core Concepts

### 1. Thinking Episodes

Each thinking episode represents a snapshot of the AGI's strategic state:

- **Strategy Vector**: 16-dimensional representation of current thinking
- **Effectiveness**: How well this strategy performed
- **Context State**: Environmental conditions when this strategy was used
- **Generation**: Which iteration produced this thinking

### 2. Strategy Patterns

Learned patterns extracted from multiple thinking episodes:

- **Pattern Vector**: Average of similar successful strategies
- **Success Rate**: Historical effectiveness
- **Usage Count**: How often this pattern appears
- **Evolution Factor**: Rate of strategic adaptation

### 3. Memory Feedback Loop

**Output → Memory → Input → Strategy**

1. **Output Storage**: Current thinking becomes a memory
2. **Input Loading**: Past thinking influences current decisions
3. **Pattern Recognition**: Similar situations trigger similar strategies
4. **Strategic Evolution**: Patterns improve through experience

## Implementation

### Modules

The system consists of 6 interconnected modules:

#### `mod_store_thinking_output`
Stores current thinking state into episodic memory.

```c
// Encodes current state into 16D strategy vector
// Calculates effectiveness based on multiple metrics
// Stores with temporal and generational tags
```

#### `mod_load_thinking_input`
Retrieves and applies relevant past thinking.

```c
// Finds most effective past thinking for current context
// Blends past wisdom with current state
// Updates accumulated wisdom metric
```

#### `mod_extract_strategy_patterns`
Extracts reusable patterns from thinking history.

```c
// Clusters similar thinking episodes
// Tracks success rates and usage frequency
// Builds a library of proven strategies
```

#### `mod_generate_updated_strategy`
Creates improved strategies from learned patterns.

```c
// Selects best pattern based on success + usage
// Applies with confidence-scaled strength
// Evolves patterns through exploration
```

#### `mod_recursive_thinking_cycle`
Orchestrates the complete feedback loop.

```c
// Runs all 4 steps in sequence
// Periodic pattern extraction
// Conditional strategy generation
```

#### `mod_export_thinking_metrics`
Exports metrics for monitoring and analysis.

```c
// Episode count and pattern count
// Average effectiveness
// Wisdom and confidence levels
// Current generation
```

### Integration

Add to your module pipeline:

```c
// In your phase execution:
ql_state state = {0};
// ... initialize state ...

// Enable recursive thinking
mod_recursive_thinking_cycle(&state);

// Run periodically in your main loop
if (tick % 10 == 0) {
    mod_export_thinking_metrics(&state);
}
```

Or via the module registry:

```bash
./qallow phase 7 --ticks=500 \
    --modules=rec_think_cycle,episodic_mem,semantic_mem
```

## Key Metrics

### Accumulated Wisdom
- Exponentially weighted average of strategy effectiveness
- Range: [0.0, 1.0]
- Formula: `wisdom = 0.95 * wisdom + 0.05 * effectiveness`

### Confidence in Patterns
- How reliable are the learned patterns?
- Range: [0.0, 1.0]
- Grows with pattern count: `min(1.0, patterns / 10.0)`

### Strategy Diversity
- How varied are the strategies being explored?
- Range: [0.0, 1.0]
- Balances exploitation vs exploration

### Generation
- How many strategy evolution cycles have occurred?
- Increments each time a new strategy is generated
- Tracks learning progression

## Usage Examples

### Basic Usage

```bash
# Run with recursive thinking enabled
./build/qallow phase 7 --ticks=1000 --modules=rec_think_cycle
```

### Python Demonstration

```bash
# Run the interactive demonstration
python3 scripts/demo_recursive_thinking.py --cycles 5

# Output shows:
# - Episodes stored over time
# - Patterns learned
# - Wisdom accumulation
# - Confidence growth
# - Strategy evolution
```

### Monitoring

```bash
# Watch thinking metrics in real-time
./build/qallow phase 7 --modules=rec_think_cycle,think_metrics | \
    grep "RECURSIVE_THINKING"
```

## Benefits

### Self-Improvement
- Strategies automatically improve through experience
- No external training required
- Adapts to changing conditions

### Memory Utilization
- Past experiences inform future decisions
- Avoids repeating failed strategies
- Amplifies successful patterns

### Strategic Evolution
- Patterns evolve over generations
- Balances exploitation and exploration
- Builds confidence through validation

### Transparency
- All thinking is logged and traceable
- Metrics show learning progression
- Patterns can be analyzed and understood

## Configuration

### Memory Limits

```c
#define MAX_THINKING_EPISODES 256  // Max stored episodes
#define MAX_STRATEGY_PATTERNS 64   // Max learned patterns
#define THINKING_DIM 16            // Strategy vector size
```

### Learning Parameters

```c
double blend_factor = 0.15;        // Past → Current influence
double alpha = 0.1;                // Pattern learning rate
double application_strength = 0.2; // Strategy application strength
double evolution_factor = 0.05;    // Exploration rate
```

### Tuning Guide

**High blend_factor (0.2+)**
- More influenced by past
- More stable, less adaptive

**Low blend_factor (0.05-0.1)**
- More exploration
- Less stable, more adaptive

**High application_strength (0.3+)**
- Strong pattern influence
- Risk of local minima

**Low application_strength (0.1-0.15)**
- Gentle pattern guidance
- Slower but more robust learning

## Advanced Features

### Context-Aware Recall

The system uses Gaussian similarity to find relevant past thinking:

```c
similarity = exp(-sum_squared_differences / dimensionality)
relevance = effectiveness * similarity
```

This ensures retrieved memories match the current context.

### Memory Consolidation

When memory is full (256 episodes):

1. Sort by effectiveness
2. Keep top 50%
3. Discard bottom 50%

This preserves high-quality thinking and makes room for new learning.

### Pattern Clustering

Similar strategies are automatically grouped:

```c
if (distance < 0.5) {
    // Update existing pattern
} else {
    // Create new pattern
}
```

### Generational Tracking

Each strategy evolution increments the generation counter, allowing:
- Tracking learning over time
- Comparing early vs late strategies
- Measuring learning velocity

## Troubleshooting

### Low Confidence
**Symptom**: Confidence stays below 0.3
**Cause**: Not enough patterns learned
**Solution**: Run longer (500+ ticks) or increase episode limit

### No Pattern Learning
**Symptom**: Pattern count stays at 0
**Cause**: Insufficient thinking episodes
**Solution**: Ensure `mod_store_thinking_output` runs every tick

### Effectiveness Not Improving
**Symptom**: Average effectiveness stays flat
**Cause**: Poor base metrics or wrong blend factor
**Solution**: Tune blend_factor, check coherence/ethics scores

### Memory Filling Too Fast
**Symptom**: Frequent consolidation messages
**Cause**: High tick rate or many episodes
**Solution**: Increase MAX_THINKING_EPISODES or consolidate more aggressively

## Integration with Other Systems

### With Episodic Memory
```c
// Combine with standard memory for richer context
mod_episodic_memory(&state);
mod_recursive_thinking_cycle(&state);
```

### With Self-Reflection
```c
// Use self-reflection to evaluate strategies
mod_recursive_thinking_cycle(&state);
src_review(&reflection, run_id, &plan, outcome, &result);
```

### With Quantum Optimization
```c
// Let quantum system inform strategy evolution
mod_quantum_optimize(&state);
mod_recursive_thinking_cycle(&state);
```

## Performance

- **Memory**: ~64KB for full episode storage
- **Computation**: O(N) for recall, O(N²) for pattern extraction
- **Overhead**: ~1-2% per thinking cycle
- **Scalability**: Handles 1000s of episodes efficiently

## Future Enhancements

- [ ] Multi-level pattern hierarchies
- [ ] Transfer learning across domains
- [ ] Attention-weighted retrieval
- [ ] Causal reasoning integration
- [ ] Meta-learning for parameter tuning

## References

- `src/mind/recursive_thinking.c` - Core implementation
- `src/mind/memory.c` - Base memory system
- `src/mind/registry.c` - Module registration
- `scripts/demo_recursive_thinking.py` - Demonstration script

## Support

For questions or issues:
1. Check logs: `grep RECURSIVE_THINKING data/logs/*.log`
2. Run demo: `python3 scripts/demo_recursive_thinking.py`
3. Review metrics with: `--modules=think_metrics`
