# HallucinationShield Integration Guide

## Quick Integration

### Step 1: Import the Hook

```python
from python.hallucination_shield_hook import create_validated_pipeline
```

### Step 2: Wrap Your LLM Function

```python
# Your existing LLM function
def my_llm_inference(prompt: str) -> str:
    return llm_model.generate(prompt)

# Wrap it with HallucinationShield
validated_llm = create_validated_pipeline(
    my_llm_inference,
    threshold=0.8,
    enable_correction=True
)
```

### Step 3: Use It

```python
# Use exactly like your original function
output = validated_llm("What is quantum computing?")

# Or get validation details
output, validation_result = validated_llm(
    "What is quantum computing?",
    return_validation=True
)

print(f"Valid: {validation_result.is_valid}")
print(f"Confidence: {validation_result.confidence:.2f}")
```

## Integration Locations

### 1. AGI Integration (`python/qallow_agi_integration.py`)

```python
from python.hallucination_shield_hook import create_validated_pipeline

class QallowAGIIntegration:
    def __init__(self, ...):
        # Existing initialization
        ...
        
        # Add validation to inference
        self.validated_inference = create_validated_pipeline(
            self._raw_inference,
            threshold=0.8,
            enable_correction=True
        )
    
    def run_inference(self, prompt: str) -> str:
        # Use validated version
        return self.validated_inference(prompt)
```

### 2. Quantum Learning (`python/quantum_learning_system.py`)

```python
from python.hallucination_shield_hook import validate_and_correct

class QuantumLearningSystem:
    def generate_explanation(self, quantum_state) -> str:
        # Generate explanation
        explanation = self._generate_raw_explanation(quantum_state)
        
        # Validate and correct
        validated_explanation, result = validate_and_correct(
            explanation,
            context=f"Quantum state: {quantum_state}",
            threshold=0.85
        )
        
        return validated_explanation
```

### 3. Web API (`python/quantum/web_api.py`)

```python
from python.hallucination_shield_hook import ValidatedInferencePipeline

class QuantumWebAPI:
    def __init__(self):
        # Wrap API responses with validation
        self.shield_pipeline = ValidatedInferencePipeline(
            self._generate_response,
            shield_config={'threshold': 0.8},
            log_validations=True
        )
    
    @app.route('/api/query')
    def handle_query():
        prompt = request.json['prompt']
        
        # Validated response
        response = self.shield_pipeline(prompt)
        
        return {'response': response}
```

## Configuration Options

### Conservative (High Threshold)

```python
validated_llm = create_validated_pipeline(
    llm_function,
    threshold=0.95,              # Very strict
    enable_correction=True,
    max_correction_attempts=5,
    check_factual=True,
    check_contextual=True,
    check_semantic=True
)
```

### Balanced (Recommended)

```python
validated_llm = create_validated_pipeline(
    llm_function,
    threshold=0.8,               # Balanced
    enable_correction=True,
    max_correction_attempts=3
)
```

### Permissive (Low Threshold)

```python
validated_llm = create_validated_pipeline(
    llm_function,
    threshold=0.6,               # Lenient
    enable_correction=False,     # Detection only
    check_contextual=False       # Skip some checks
)
```

## Monitoring

### Get Statistics

```python
validated_llm = create_validated_pipeline(llm_function)

# After some inferences...
stats = validated_llm.get_statistics()

print(f"Total validations: {stats['total_validations']}")
print(f"Pass rate: {stats['passed_validations'] / stats['total_validations']:.2%}")
print(f"Average confidence: {stats['average_confidence']:.3f}")
```

### Log to Telemetry

```python
# In your telemetry collection
def log_validation_metrics():
    stats = validated_llm.get_statistics()
    
    telemetry.log_metric('hallucination_shield.total', stats['total_validations'])
    telemetry.log_metric('hallucination_shield.pass_rate', 
                        stats['passed_validations'] / stats['total_validations'])
    telemetry.log_metric('hallucination_shield.avg_confidence', 
                        stats['average_confidence'])
```

## Testing Your Integration

```python
import unittest
from python.hallucination_shield_hook import create_validated_pipeline

class TestMyIntegration(unittest.TestCase):
    def setUp(self):
        self.validated_llm = create_validated_pipeline(
            self.mock_llm,
            threshold=0.8
        )
    
    def mock_llm(self, prompt: str) -> str:
        return "Test response"
    
    def test_validation_works(self):
        output, result = self.validated_llm(
            "test prompt",
            return_validation=True
        )
        
        self.assertIsNotNone(output)
        self.assertIsInstance(result.confidence, float)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
```

## Troubleshooting

### Issue: All validations failing

**Possible causes:**
- Threshold too high for your use case
- LLM outputs contain uncertainty markers
- Context not being passed correctly

**Solutions:**
1. Lower threshold: `threshold=0.7`
2. Review LLM output patterns
3. Always pass context: `validated_llm(prompt, context=prompt)`

### Issue: Corrections not working

**Possible causes:**
- Correction disabled
- Output cannot be improved with current strategies
- Max attempts too low

**Solutions:**
1. Enable correction: `enable_correction=True`
2. Increase attempts: `max_correction_attempts=5`
3. Lower threshold to allow corrected outputs

### Issue: Performance overhead

**Solutions:**
1. Disable unused checks: `check_contextual=False`
2. Reduce max attempts: `max_correction_attempts=1`
3. Use higher threshold to skip corrections: `threshold=0.9`

## Performance Impact

| Configuration | Overhead per Request | Memory |
|--------------|---------------------|---------|
| Validation only | ~1-5ms | <1MB |
| With correction (1 attempt) | ~5-15ms | <1MB |
| With correction (3 attempts) | ~15-50ms | <1MB |

## See Also

- Full API Documentation: `docs/HALLUCINATION_SHIELD.md`
- Integration Examples: `examples/hallucination_shield_integration.py`
- Test Suite: `tests/unit/test_hallucination_shield.py`
- Hook Module: `python/hallucination_shield_hook.py`
