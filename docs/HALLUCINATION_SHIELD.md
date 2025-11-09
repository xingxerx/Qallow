# HallucinationShield Module

## Overview

The `HallucinationShield` module provides comprehensive hallucination detection and correction capabilities for LLM outputs in the Qallow inference pipeline. It validates model outputs for factual consistency, contextual relevance, and semantic coherence before they are returned to users.

## Features

- **Multi-level Validation**: Factual, contextual, and semantic checking
- **Automatic Correction**: Self-healing correction chains for detected issues
- **Confidence Scoring**: Quantitative assessment of output quality
- **Statistics Tracking**: Monitor validation performance over time
- **Flexible Configuration**: Customize thresholds and checking strategies
- **Integration Ready**: Easy integration into existing inference pipelines

## Installation

### Dependencies

Add the following to your `requirements.txt`:

```txt
uptrain>=0.6.0  # Hallucination detection and LLM validation
```

Install dependencies:

```bash
pip install -r config/requirements.txt
```

## Quick Start

### Basic Usage

```python
from qallow.core.governance import HallucinationShield

# Create shield instance
shield = HallucinationShield(threshold=0.8, enable_correction=True)

# Validate LLM output
result = shield.forward(
    output="Paris is the capital of France.",
    context="What is the capital of France?"
)

print(f"Valid: {result.is_valid}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Level: {result.hallucination_level.value}")
```

### With Automatic Correction

```python
from qallow.core.governance import HallucinationShield

shield = HallucinationShield(
    threshold=0.8,
    enable_correction=True,
    max_correction_attempts=3
)

# Validate output
output = "I apologize, but I cannot verify this information."
result = shield.forward(output)

if not result.is_valid:
    # Attempt correction
    corrected = shield.correction_chain(
        output=output,
        issues=result.issues
    )
    
    if corrected:
        print(f"Corrected: {corrected}")
```

### Pipeline Integration

```python
from qallow.core.governance import HallucinationShield

class LLMInferencePipeline:
    def __init__(self):
        self.shield = HallucinationShield(
            threshold=0.8,
            enable_correction=True
        )
    
    def generate(self, prompt: str) -> str:
        # Generate LLM output
        output = self.llm.generate(prompt)
        
        # Validate for hallucinations
        result = self.shield.forward(
            output=output,
            context=prompt
        )
        
        # Handle validation result
        if not result.is_valid and self.shield.enable_correction:
            corrected = self.shield.correction_chain(
                output=output,
                issues=result.issues,
                context=prompt
            )
            return corrected if corrected else output
        
        return output
```

## API Reference

### HallucinationShield Class

```python
HallucinationShield(
    threshold: float = 0.8,
    enable_correction: bool = True,
    max_correction_attempts: int = 3,
    check_factual: bool = True,
    check_contextual: bool = True,
    check_semantic: bool = True,
    uptrain_api_key: Optional[str] = None,
    **kwargs
)
```

**Parameters:**
- `threshold`: Confidence threshold for accepting outputs (0-1)
- `enable_correction`: Whether to enable automatic correction
- `max_correction_attempts`: Maximum number of correction attempts
- `check_factual`: Enable factual consistency checking
- `check_contextual`: Enable contextual relevance checking
- `check_semantic`: Enable semantic coherence checking
- `uptrain_api_key`: Optional API key for uptrain service

### Methods

#### forward()

Validate an LLM output for hallucinations.

```python
forward(
    output: str,
    context: Optional[str] = None,
    reference: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ValidationResult
```

**Parameters:**
- `output`: The LLM output to validate
- `context`: Optional context/prompt that generated the output
- `reference`: Optional reference text for factual verification
- `metadata`: Optional metadata for validation context

**Returns:**
- `ValidationResult`: Comprehensive validation result

#### correction_chain()

Apply correction chain to fix detected hallucinations.

```python
correction_chain(
    output: str,
    issues: List[str],
    context: Optional[str] = None,
    max_attempts: Optional[int] = None
) -> Optional[str]
```

**Parameters:**
- `output`: The original output to correct
- `issues`: List of detected issues from validation
- `context`: Optional context for correction
- `max_attempts`: Override for max correction attempts

**Returns:**
- `str` or `None`: Corrected output string, or None if correction failed

#### get_statistics()

Get validation and correction statistics.

```python
get_statistics() -> Dict[str, Union[int, float]]
```

**Returns:**
- Dictionary containing:
  - `total_validations`: Total number of validations performed
  - `passed_validations`: Number of validations that passed
  - `failed_validations`: Number of validations that failed
  - `corrections_applied`: Number of successful corrections
  - `average_confidence`: Average confidence score across all validations

#### reset_statistics()

Reset all statistics counters.

```python
reset_statistics() -> None
```

### ValidationResult Dataclass

```python
@dataclass
class ValidationResult:
    is_valid: bool
    confidence: float
    hallucination_level: HallucinationLevel
    issues: List[str]
    corrected_output: Optional[str] = None
    metadata: Dict[str, Any]
```

**Attributes:**
- `is_valid`: Whether the output passes validation
- `confidence`: Confidence score (0-1) for the validation
- `hallucination_level`: Detected level of hallucination (NONE, LOW, MEDIUM, HIGH, CRITICAL)
- `issues`: List of detected issues
- `corrected_output`: Optional corrected version of the output
- `metadata`: Additional metadata about the validation

### HallucinationLevel Enum

```python
class HallucinationLevel(Enum):
    NONE = "none"          # Confidence >= 0.9
    LOW = "low"            # Confidence >= 0.75
    MEDIUM = "medium"      # Confidence >= 0.5
    HIGH = "high"          # Confidence >= 0.25
    CRITICAL = "critical"  # Confidence < 0.25
```

## Configuration Examples

### Strict Validation

```python
shield = HallucinationShield(
    threshold=0.95,  # Very high threshold
    enable_correction=True,
    max_correction_attempts=5,
    check_factual=True,
    check_contextual=True,
    check_semantic=True
)
```

### Lenient Validation

```python
shield = HallucinationShield(
    threshold=0.6,  # Lower threshold
    enable_correction=False,
    check_factual=True,
    check_contextual=False,
    check_semantic=True
)
```

### Production Recommended

```python
shield = HallucinationShield(
    threshold=0.8,
    enable_correction=True,
    max_correction_attempts=3,
    check_factual=True,
    check_contextual=True,
    check_semantic=True
)
```

## Integration Points

### Post-LLM Processing

The recommended integration point is **post-LLM processing, pre-output stage**:

```
User Query → LLM Inference → [HallucinationShield] → Output
```

### Inference Pipeline Hook

```python
class QallowInferencePipeline:
    def __init__(self):
        self.llm = LLMEngine()
        self.shield = HallucinationShield(threshold=0.8)
    
    def process_query(self, query: str) -> str:
        # Generate output
        output = self.llm.generate(query)
        
        # Validate
        result = self.shield.forward(output, context=query)
        
        # Log validation
        self.log_validation(result)
        
        # Return output (corrected if needed)
        return result.corrected_output or output
```

## Testing

Run the test suite:

```bash
# Run all HallucinationShield tests
python -m unittest tests.unit.test_hallucination_shield -v

# Run specific test class
python -m unittest tests.unit.test_hallucination_shield.TestHallucinationShield -v
```

## Examples

See `examples/hallucination_shield_integration.py` for comprehensive integration examples:

```bash
python examples/hallucination_shield_integration.py
```

## Performance Considerations

### Overhead

- **Validation**: ~1-5ms per output (without uptrain)
- **Correction**: ~5-50ms per attempt (depends on output length)
- **Memory**: Minimal (<1MB per shield instance)

### Optimization Tips

1. **Cache Shield Instances**: Reuse shield instances across requests
2. **Selective Checking**: Disable checks not needed for your use case
3. **Batch Processing**: Validate multiple outputs in parallel
4. **Adjust Threshold**: Higher thresholds reduce false positives but may miss issues

## Monitoring

Track validation performance:

```python
shield = HallucinationShield()

# Perform validations...

stats = shield.get_statistics()
print(f"Pass Rate: {stats['passed_validations'] / stats['total_validations']:.2%}")
print(f"Avg Confidence: {stats['average_confidence']:.3f}")
```

## Troubleshooting

### Uptrain Not Found

**Issue**: Warning about uptrain library not found

**Solution**: The shield uses fallback detection methods when uptrain is unavailable. To enable full uptrain functionality:

```bash
pip install uptrain>=0.6.0
```

### Low Pass Rates

**Issue**: Many validations failing

**Solution**:
1. Lower the threshold: `HallucinationShield(threshold=0.7)`
2. Disable strict checks: `check_contextual=False`
3. Review your LLM output quality

### Corrections Failing

**Issue**: Correction chain returning None

**Solution**:
1. Increase max attempts: `max_correction_attempts=5`
2. Provide better context to correction chain
3. Lower validation threshold
4. Implement custom correction logic

## Contributing

When contributing to HallucinationShield:

1. Add tests for new features in `tests/unit/test_hallucination_shield.py`
2. Update this documentation
3. Follow existing code style and patterns
4. Ensure all tests pass: `python -m unittest tests.unit.test_hallucination_shield`

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- GitHub Issues: https://github.com/xingxerx/Qallow/issues
- Documentation: See docs/ directory
- Examples: See examples/ directory
