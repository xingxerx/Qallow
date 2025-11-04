# Qallow Project Constitution

## Core Principles

This document establishes the non-negotiable principles for Qallow development using Spec-Driven Development.

### 1. **Spec-First Development**
- Every feature begins with a clear specification using `/specify` command
- Specifications define requirements before implementation
- Technical planning follows specification through `/plan` command
- Implementation details are derived from specs, not the reverse

### 2. **Quality and Coherence**
- All code maintains 1.0 coherence score (perfect state tracking)
- Zero-crash guarantee: production code runs without errors
- 100% test coverage for critical paths
- Performance validated with multi-scenario testing

### 3. **Architecture Consistency**
- **Core Runtime**: C/CUDA orchestration with Python bridging
- **Phase Flow**: Sequential execution of ethics → quantum → elasticity → lattice phases
- **Telemetry**: All metrics funnel through centralized telemetry system
- **Storage**: Network integration via Samba shares with real-time sync

### 4. **Production Readiness**
- All code changes must pass:
  - Syntax validation
  - Multi-scenario testing (minimum 3 configurations)
  - Performance benchmarking
  - Documentation verification
  
- Deployment approval requires:
  - Zero test failures
  - 100% success rate across all scenarios
  - Clear git commit with spec reference
  - Status file updated with metrics

### 5. **Error Handling**
- Bugs discovered during implementation must be:
  - Root-cause analyzed immediately
  - Fixed with test-driven approach
  - Verified across all scenarios
  - Documented in commit messages

### 6. **Documentation**
- Every major feature includes:
  - Feature specification (SPEC_*.md)
  - Architecture documentation
  - Usage examples
  - Performance metrics
  - Deployment checklist

### 7. **Cross-Platform Integration**
- Support Linux (primary) and WSL with Windows interop
- Network storage syncing to Windows shares (Z:\)
- Status files updated in real-time for monitoring
- Python/C/CUDA code maintains compatibility

### 8. **AI-Native Development**
- Specs written for Copilot with `$ARGUMENTS` expansion
- Commands exposed: `/specify`, `/plan`, `/tasks`, `/implement`
- Prompts stored in `.github/prompts/speckit.*.prompt.md`
- MCP memory service available for context persistence

## Feature Development Workflow

1. **Specification** (`/specify`)
   - Describe what needs to be built
   - Focus on "what" and "why", not implementation details
   - Result: `specs/{number}-{short-name}/spec.md`

2. **Planning** (`/plan`)
   - Define tech stack and architecture choices
   - Establish constraints and requirements
   - Result: `specs/{number}-{short-name}/plan.md`

3. **Task Breakdown** (`/tasks`)
   - Create actionable, granular tasks
   - Identify dependencies and test criteria
   - Result: `specs/{number}-{short-name}/tasks.md`

4. **Implementation** (`/implement`)
   - Execute tasks following the plan
   - Test each component immediately
   - Update status file with progress

5. **Verification**
   - Run full test suite
   - Validate against spec requirements
   - Generate final status report

## Code Quality Standards

### Python (`.github/workflows/Driver.py`)
- Type hints required (torch, numpy types)
- Docstrings for all classes and methods
- Error handling with meaningful messages
- Logging via `qallow_log_*` functions

### C/CUDA (`backend/{cpu,cuda}/`)
- Defensive programming with bounds checking
- Telemetry around hot paths with `QALLOW_PROFILE_SCOPE`
- Consistent struct naming with `*_t` suffix
- Phase separation: one file per phase

### Build System
- CMake as primary build tool
- `./scripts/build_all.sh` for standard builds
- `ctest` for validation
- Parallel compilation enabled by default

## Testing Requirements

- **Unit Tests**: `ctest --test-dir build`
- **Integration Tests**: Multi-scenario framework
- **Performance Tests**: `tests/sequential_phase_benchmark.sh`
- **Regression Tests**: Compare against baseline metrics

### Success Criteria
- All 3+ test scenarios pass with 100% success rate
- Coherence maintained at 1.0 across runs
- Zero crashes or hangs
- Performance within 5% of baseline

## Telemetry and Monitoring

- **Log Location**: `data/logs/`
- **Status File**: `/home/xing/share/status.txt` (Windows: `Z:\status.txt`)
- **Metrics**: Exported as CSV and JSON summaries
- **Real-time Updates**: Sync to Windows on each execution

## Deployment Checklist

- [ ] Spec created and reviewed
- [ ] Plan documented with architecture
- [ ] All tasks completed and tested
- [ ] Multi-scenario test suite passes (100%)
- [ ] Performance benchmarked
- [ ] Documentation updated
- [ ] Status file synchronized
- [ ] Git commit with spec reference
- [ ] Production approval verified

## Emergency Procedures

### If Test Failures Occur
1. Isolate failing scenario
2. Capture full error trace
3. Root-cause analysis
4. Implement fix with new test
5. Verify fix across all scenarios

### If Production Crashes
1. Check telemetry logs in `data/logs/`
2. Verify status file for last known state
3. Review MCP memory service context
4. Restore from previous known-good state
5. Create fix specification

## Continuous Improvement

- Review test results after each deployment
- Monitor performance trends in metrics
- Update CONSTITUTION.md when new patterns emerge
- Share learnings in documentation
- Maintain backward compatibility where possible

---

**Version**: 2.2  
**Last Updated**: November 4, 2025  
**Status**: Active  

This constitution is binding for all Qallow development using Spec-Driven Development and GitHub Copilot integration.
