# CI/Bootstrap Integration Guide

This document explains the integration of the bootstrap system with CI/CD pipelines.

## Overview

The bootstrap system (`bootstrap.sh`) provides:
- **Reproducible builds** across environments
- **Intelligent caching** for dependencies
- **Flexible configuration** for CPU/GPU builds
- **CI-friendly output** for automation

## CI/CD Architecture

### Three-Stage Pipeline

```
┌─────────────────────┐
│ Setup & Caching     │  Generate cache keys, setup environment
└──────────┬──────────┘
           │
    ┌──────┴──────────────────┐
    │                         │
┌───▼──────────┐    ┌────────▼────────┐
│ CPU Build    │    │ CUDA Build      │
│ (Required)   │    │ (Optional)      │
└───┬──────────┘    └────────┬────────┘
    │                        │
    └──────────┬─────────────┘
               │
          ┌────▼────────┐
          │ Quality     │
          │ Analysis    │
          └────┬────────┘
               │
          ┌────▼────────┐
          │ Summary     │
          │ Report      │
          └─────────────┘
```

### Key Features

1. **Parallel Builds**: CPU and CUDA builds run independently
2. **Caching Strategy**: 
   - Python venv cache (invalidates on requirements changes)
   - Asset download cache
3. **Fail-Safe**: CUDA build is optional; CPU build is required
4. **Fast Path**: CPU-only default for CI speed

## Cache Keys

### Python Virtual Environment Cache

```yaml
key: v1-Linux-venv-${{ hashFiles('requirements*.txt') }}
```

- Invalidates when any `requirements*.txt` changes
- Caches entire `.venv` directory
- Shared across all jobs in workflow

### Asset Download Cache

```yaml
key: v1-assets-${{ hashFiles('scripts/assets.json') }}
```

- Caches downloads to `~/.cache/qallow`
- Invalidates when `scripts/assets.json` changes
- Reduces bandwidth during bootstrap

## Bootstrap Flags for CI

### Recommended: CPU-only (Fast)

```bash
./bootstrap.sh --no-cuda --skip-tests
```

- Skips CUDA detection and setup
- ~10-15 minutes on typical CI runner
- Ideal for PR validation

### Optional: Full CUDA Support

```bash
./bootstrap.sh --cuda --skip-tests
```

- Attempts CUDA setup
- ~20-30 minutes on CI runner
- Useful for final release validation

### Local Development (Full Tests)

```bash
./bootstrap.sh
```

- Enables testing
- Longer runtime (~30-40 minutes)
- Validates complete pipeline

## Environment Variables

### CI-Specific

```bash
# Skip interactive prompts
export CI=true

# Set job name for logs
export CI_JOB_ID="build-cpu-ubuntu-22.04"

# Track build metadata
export CI_BUILD_REF="abc123def456"
```

### Python/Qiskit

```bash
# Enable Qiskit for Phase 11
export QALLOW_QISKIT=1

# Custom Python path
export PYTHON_EXECUTABLE="/usr/bin/python3.10"
```

## Workflow Configuration

### GitHub Actions Integration

The workflow file `.github/workflows/bootstrap-ci.yml` includes:

1. **Setup Job**
   - Generates cache keys
   - Reports cache strategy
   - Outputs for downstream jobs

2. **Build Jobs**
   - CPU: Always runs (required status check)
   - CUDA: Optional (continue-on-error: true)

3. **Quality Job**
   - Runs after CPU build
   - Checks Makefile consistency
   - Validates CI configuration

4. **Summary Job**
   - Always runs (cleanup/reporting)
   - Displays build status
   - Fails if CPU build failed

### Cache Hit/Miss Rates

Expected behavior:

- **First run**: All caches miss (full setup)
- **PR #1 + PR #2**: venv cache hits (same requirements)
- **After requirements update**: venv cache miss (rebuilds)
- **After assets change**: Only assets cache miss

## Troubleshooting

### Issue: Cache not working

**Symptoms**: Each CI run takes full time, downloading all dependencies

**Solution**:
1. Verify `requirements*.txt` hasn't changed
2. Check cache key generation in "Generate cache keys" step
3. Ensure branch is not in cache cleanup (GitHub default)

### Issue: "No CUDA toolkit found"

**Symptoms**: CUDA build fails with missing libraries

**Solution**:
1. CUDA build has `continue-on-error: true` (safe to fail)
2. Check CPU build still passes
3. For CUDA support, run locally: `./bootstrap.sh --cuda`

### Issue: Disk space errors

**Symptoms**: "No space left on device" during setup

**Solution**:
1. "Free disk space" step removes ~20GB of unused tools
2. Bootstrap already optimized for CI runners
3. Check if artifacts folder is too large

### Issue: Bootstrap timeout

**Symptoms**: "timed out after 15 minutes"

**Solution**:
1. Increase `timeout-minutes` in workflow (currently 15)
2. Ensure venv cache is working (hit rate)
3. Check CI runner resources (CPU cores, I/O)

## Performance Metrics

### Expected Times (GitHub Actions)

| Build Type | Cold Cache | Warm Cache | Notes |
|-----------|-----------|-----------|-------|
| CPU-only  | 12-15m    | 4-6m      | Recommended for PR validation |
| CPU+CUDA  | 20-30m    | 8-12m     | For release validation |
| Full test | 30-40m    | 15-20m    | Local dev only |

### Optimization Tips

1. **Pre-populate cache**: Run workflow on main branch first
2. **Use scheduled builds**: Daily CUDA validation (separate job)
3. **Parallel jobs**: CPU and CUDA run simultaneously
4. **Artifact cleanup**: Set retention to 5 days (default 30)

## Integration Examples

### Pull Request Validation

```yaml
on:
  pull_request:
    branches: [ main, develop ]
```

- Runs on every PR
- CPU-only for speed
- Required status check

### Nightly Full Build

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily
```

Add separate job:
```yaml
build-nightly-cuda:
  if: github.event_name == 'schedule'
  runs-on: ubuntu-22.04
  steps:
    # ... use full bootstrap with CUDA ...
```

### Release Build

```yaml
on:
  push:
    tags: [ 'v*' ]
```

Adds validation steps:
```yaml
validate-release:
  steps:
    - run: ./bootstrap.sh --cuda  # Full validation
    - run: cd build && ctest --verbose
```

## Local CI Simulation

To test CI behavior locally:

```bash
# Simulate CI environment
export CI=true
export CI_JOB_ID="test-local"

# Clear caches (simulate cold start)
rm -rf .venv ~/.cache/qallow

# Run bootstrap as CI would
./bootstrap.sh --no-cuda --skip-tests

# Check results
cd build
ctest --output-on-failure
```

## Migration Checklist

When migrating from old CI system:

- [ ] Update workflow triggers to match old system
- [ ] Add required branch protection rules
- [ ] Configure status checks (require CPU build pass)
- [ ] Set artifact retention policy (5-30 days)
- [ ] Add notification hooks if needed
- [ ] Test on feature branch before merging to main
- [ ] Update documentation (CI_DOCUMENTATION_INDEX.md)
- [ ] Archive old CI configuration
- [ ] Document any special secrets/tokens

## Related Documentation

- [ARCHITECTURE_SPEC.md](docs/ARCHITECTURE_SPEC.md) - System architecture
- [README.md](README.md) - Build instructions
- [bootstrap.sh](bootstrap.sh) - Bootstrap script
- [.github/workflows/](./github/workflows/) - Workflow configurations

## Support & Questions

For CI/bootstrap issues:

1. Check logs in GitHub Actions UI
2. Test locally with: `export CI=true && ./bootstrap.sh --no-cuda --skip-tests`
3. Review this guide's Troubleshooting section
4. Open issue with: workflow logs + local test results

---

**Last Updated**: 2024
**Bootstrap Version**: 2.0+
**Status**: Active & Maintained
