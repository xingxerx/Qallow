# Recursive Improvement Loop Guide

## Overview

The Recursive Improvement Loop is a continuous learning system that feeds build output back into code improvements, creating a self-improving codebase. It automatically:

1. **Builds** the project and captures output
2. **Analyzes** build errors, warnings, and metrics
3. **Scans** code for patterns and quality issues
4. **Generates** targeted improvement recommendations
5. **Reports** findings and suggestions
6. **Repeats** the cycle for continuous improvement

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│         Recursive Improvement Loop Orchestrator             │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
   ┌─────────┐         ┌──────────┐      ┌──────────────┐
   │  Build  │         │ Analyze  │      │ Code Pattern │
   │ Analyzer│         │ Output   │      │  Analyzer    │
   └─────────┘         └──────────┘      └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌───────▼────────┐
                    │ Improvement    │
                    │ Recommender    │
                    └────────────────┘
                            │
                    ┌───────▼────────┐
                    │ Report         │
                    │ Generator      │
                    └────────────────┘
```

## Components

### 1. Build Analyzer (`recursive_improvement_loop.py`)
- Executes CMake builds
- Captures stdout/stderr
- Parses errors and warnings
- Extracts compilation metrics
- Saves analysis to JSON

### 2. Code Pattern Analyzer (`code_improvement_engine.py`)
- Scans source files for patterns
- Detects unused variables
- Identifies missing includes
- Checks code quality metrics
- Categorizes issues by severity

### 3. Improvement Recommender
- Generates targeted suggestions
- Prioritizes by severity
- Groups by issue type
- Creates actionable recommendations

### 4. Unified Orchestrator (`unified_improvement_orchestrator.py`)
- Coordinates all phases
- Manages the complete cycle
- Generates comprehensive reports
- Tracks metrics over time

## Quick Start

### Basic Usage

```bash
# Run with default settings (5 iterations)
./scripts/run_recursive_improvement.sh

# Run with custom iterations
./scripts/run_recursive_improvement.sh --iterations 10

# Run CPU-only (no CUDA)
./scripts/run_recursive_improvement.sh --no-cuda
```

### Direct Python Usage

```bash
# Run unified orchestrator directly
python3 scripts/unified_improvement_orchestrator.py \
    --workspace /home/xing/Qallow \
    --iterations 5 \
    --analyze-code

# Run just build analysis
python3 scripts/recursive_improvement_loop.py

# Run just code analysis
python3 scripts/code_improvement_engine.py
```

## Output

All reports are saved to `improvement_reports/`:

```
improvement_reports/
├── cycle_20251111_120000.json          # Complete cycle data
├── cycle_20251111_120000.md            # Markdown summary
├── build_analysis_20251111_120000.json # Build analysis
├── recommendations_20251111_120000.json # Code recommendations
└── improvements_20251111_120000.md     # Improvement details
```

## Report Format

### Cycle Report (JSON)
```json
{
  "cycle_id": "20251111_120000",
  "start_time": "2025-11-11T12:00:00",
  "end_time": "2025-11-11T12:05:00",
  "duration_seconds": 300,
  "phases": {
    "build_analysis": {
      "success": true,
      "errors": 0,
      "warnings": 5,
      "issues": 5
    },
    "code_analysis": {
      "total_issues": 42,
      "by_severity": {
        "critical": 0,
        "high": 3,
        "medium": 12,
        "low": 27
      }
    },
    "recommendations": {
      "total": 42,
      "by_severity": {...},
      "by_type": {...}
    }
  }
}
```

### Markdown Summary
```markdown
# Recursive Improvement Cycle Report

**Cycle ID**: 20251111_120000
**Duration**: 300.5 seconds

## Build Analysis
- **Success**: true
- **Errors**: 0
- **Warnings**: 5
- **Issues**: 5

## Code Analysis
- **Total Issues**: 42
- **By Severity**: {...}

## Recommendations
- **Total**: 42
- **By Severity**: {...}
- **By Type**: {...}
```

## Configuration

Create `config/improvement_config.json`:

```json
{
  "max_iterations": 5,
  "build_timeout": 600,
  "analyze_code": true,
  "auto_fix": false,
  "report_format": "json",
  "source_dirs": [
    "src",
    "backend",
    "interface",
    "python"
  ],
  "exclude_patterns": [
    "build",
    "venv",
    "third_party"
  ]
}
```

## Environment Variables

```bash
# Set workspace root
export QALLOW_ROOT=/home/xing/Qallow

# Set max iterations
export QALLOW_MAX_ITERATIONS=10

# Enable code analysis
export QALLOW_ANALYZE_CODE=true

# Set build timeout (seconds)
export QALLOW_BUILD_TIMEOUT=600
```

## Workflow Examples

### Example 1: Quick Check
```bash
# Run 3 iterations with code analysis
./scripts/run_recursive_improvement.sh --iterations 3
```

### Example 2: Deep Analysis
```bash
# Run 10 iterations with full analysis
QALLOW_MAX_ITERATIONS=10 ./scripts/run_recursive_improvement.sh
```

### Example 3: CI/CD Integration
```bash
# Run in CI pipeline
python3 scripts/unified_improvement_orchestrator.py \
    --workspace . \
    --iterations 5 \
    --analyze-code
```

## Interpreting Results

### Build Analysis
- **Errors**: Critical issues preventing compilation
- **Warnings**: Non-critical issues to address
- **Issues**: Categorized problems by type

### Code Analysis
- **Unused Variables**: Remove for cleaner code
- **Missing Includes**: Add headers for functions
- **Code Quality**: Improve readability and maintainability

### Recommendations
- **Priority**: Critical > High > Medium > Low
- **Type**: Categorized by issue type
- **Action**: Specific suggestion for fix

## Best Practices

1. **Run regularly**: Execute after major changes
2. **Review reports**: Check recommendations carefully
3. **Prioritize**: Fix critical issues first
4. **Iterate**: Run multiple times for continuous improvement
5. **Track progress**: Monitor metrics over time

## Troubleshooting

### Build fails
- Check CMake configuration
- Verify dependencies installed
- Review build logs in `improvement_reports/`

### No recommendations generated
- Ensure source directories exist
- Check file permissions
- Verify Python dependencies

### Reports not saved
- Check `improvement_reports/` directory exists
- Verify write permissions
- Check disk space

## Integration with CI/CD

### GitHub Actions
```yaml
- name: Run Recursive Improvement
  run: |
    ./scripts/run_recursive_improvement.sh --iterations 5
    
- name: Upload Reports
  uses: actions/upload-artifact@v2
  with:
    name: improvement-reports
    path: improvement_reports/
```

### GitLab CI
```yaml
recursive_improvement:
  script:
    - ./scripts/run_recursive_improvement.sh --iterations 5
  artifacts:
    paths:
      - improvement_reports/
```

## Performance Metrics

The system tracks:
- Build time per iteration
- Error/warning trends
- Code quality metrics
- Improvement effectiveness
- Cycle duration

## Next Steps

1. Run your first cycle: `./scripts/run_recursive_improvement.sh`
2. Review the generated reports
3. Implement recommended improvements
4. Run again to verify progress
5. Integrate into your CI/CD pipeline

## Support

For issues or questions:
1. Check `improvement_reports/` for detailed logs
2. Review this guide
3. Check project documentation
4. Open an issue on GitHub

