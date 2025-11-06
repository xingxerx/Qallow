# Implementation Tasks: Organize Codebase by File Type

**Feature**: 002-organize-codebase  
**Created**: November 6, 2025  
**Status**: Ready for Execution  
**Plan Reference**: [plan.md](./plan.md)  
**Specification Reference**: [spec.md](./spec.md)

---

## Task Overview

Total Tasks: 8  
Total Effort: ~30 minutes  
Sequencing: Sequential (dependencies noted)  
Success Criteria: All tasks complete + 10/10 specification success criteria met + 7/7 validation checks pass

---

## Task 1: Review & Validate Specification & Plan

**Objective**: Ensure understanding of spec and plan before execution

**Duration**: 5 minutes

**Prerequisites**: None

**Acceptance Criteria**:
- [x] Specification (spec.md) reviewed - understand all 3 user stories
- [x] Plan (plan.md) reviewed - understand all 5 phases
- [x] 10 success criteria understood
- [x] Constitution § IV alignment verified (CONSTITUTION_AUDIT.md)
- [x] No ambiguities or questions remaining

**Key Checkpoints**:
1. User Story P1: Developer discovers clear structure → P2: Maintainer verifies compliance → P3: Git history preserved
2. Phase sequence: Inventory → Categorize → Execute → Commit → Validate
3. Success criteria mapping: 10 SC mapped to 5 implementation phases
4. Risk mitigation: 5 risks with contingencies understood

**Success Indicator**: ✅ Proceed to Task 2

---

## Task 2: Inventory Current Loose Files

**Objective**: Document current state before reorganization (baseline for comparison)

**Duration**: 2 minutes

**Prerequisites**: Task 1 complete

**Acceptance Criteria**:
- [x] Root directory scanned with `find . -maxdepth 1 -type f`
- [x] Current loose file count recorded
- [x] File list captured in loose_files_baseline.txt
- [x] Baseline saved for post-reorganization comparison

**Implementation**:
```bash
cd /home/xing/Qallow
find . -maxdepth 1 -type f ! -name '.git*' | sort > loose_files_baseline.txt
wc -l loose_files_baseline.txt  # Record count (expect ~18 from feature 001)
```

**Success Indicator**: ✅ baseline file created with all loose files listed

---

## Task 3: Create Pre-Reorganization Git Tag

**Objective**: Enable rollback if needed

**Duration**: 1 minute

**Prerequisites**: Task 2 complete

**Acceptance Criteria**:
- [x] Git tag created: `pre-002-reorganization`
- [x] Tag points to current HEAD
- [x] Tag verified with `git tag -l`

**Implementation**:
```bash
cd /home/xing/Qallow
git tag pre-002-reorganization HEAD
git tag -l | grep pre-002  # Verify
```

**Success Indicator**: ✅ Git tag created and visible

---

## Task 4: Review & Update File Categorization Mapping

**Objective**: Ensure mapping is accurate for all files

**Duration**: 3 minutes

**Prerequisites**: Task 1 complete (plan.md review)

**Acceptance Criteria**:
- [x] Mapping table from plan.md reviewed (matches feature 001 + enhancements)
- [x] Special cases confirmed (canon dirs, root files to keep)
- [x] Unknown extensions identified and assigned to misc/ (if needed)
- [x] No ambiguities in categorization

**Key Mapping Reminders** (from plan.md):
```
.js, .ts, .tsx, .jsx → src/
.py → scripts/
.md, .txt → docs/
.json, .yaml, .yml → config/
.sh → scripts/
.c, .h → backend/cpu/misc/ (or backend/cuda/misc/)
.deb, .tar.gz, .zip → deploy/
.png, .jpg, .jpeg, .svg, .gif → public/assets/
* (unknown) → misc/

KEEP IN ROOT:
CMakeLists.txt, Cargo.toml, Makefile, Dockerfile, 
bootstrap.sh, setup.bat, README.md, LICENSE, 
Qallow.code-workspace
```

**Success Indicator**: ✅ Mapping confirmed, no ambiguities

---

## Task 5: Run Reorganization in Dry-Run Mode

**Objective**: Preview changes without executing

**Duration**: 2 minutes

**Prerequisites**: Task 4 complete

**Acceptance Criteria**:
- [x] reorganize.sh executed with `--dry-run --verbose` flags
- [x] Output shows all proposed moves (no actual moves)
- [x] Output captured to dry_run_output.txt
- [x] Output reviewed for correctness
- [x] No errors or warnings in output

**Implementation**:
```bash
cd /home/xing/Qallow
bash scripts/reorganize.sh --dry-run --verbose 2>&1 | tee dry_run_output.txt

# Review output
cat dry_run_output.txt | grep -E "MOVE|ERROR|WARNING"
```

**Verification Checklist**:
- [ ] All loose files shown in categorization
- [ ] Target directories identified correctly
- [ ] No errors (exit code 0)
- [ ] Output makes sense

**If Issues Found**:
- Review mapping in plan.md
- Check for permission issues
- Verify scripts exist (reorganize.sh, validate.sh)
- Compare with feature 001 execution

**Success Indicator**: ✅ Dry-run shows correct moves, no errors

---

## Task 6: Execute Reorganization

**Objective**: Actually move files to proper directories

**Duration**: 5 seconds (execution) + 2 minutes (monitoring)

**Prerequisites**: Task 5 complete (dry-run validated)

**Acceptance Criteria**:
- [x] reorganize.sh executed without --dry-run flag
- [x] All files moved to target directories (Phase 3 complete)
- [x] Audit trail logged to reorg.log
- [x] Git commit created (Phase 4 complete)
- [x] Exit code: 0 (success)

**Implementation**:
```bash
cd /home/xing/Qallow
bash scripts/reorganize.sh --verbose 2>&1 | tee reorganization_execution.txt

# Check result
tail -30 reorganization_execution.txt
echo "Exit code: $?"
```

**Verification Checklist**:
- [x] Output shows "Move" operations completing
- [x] No "ERROR" messages in output
- [x] Git commit created (check `git log --oneline -5`)
- [x] reorg.log exists and contains entries
- [x] File count matches dry-run prediction

**If Issues Encountered**:
- Check error messages in output
- Verify permissions: `ls -l /home/xing/Qallow/ | grep '^d'` (directories writable)
- Check available disk space: `df -h /home/xing/Qallow`
- Rollback if needed: `git reset --hard pre-002-reorganization`

**Success Indicator**: ✅ All files moved, commit created, exit code 0

---

## Task 7: Run Validation & Generate Report

**Objective**: Verify reorganization success per specification success criteria

**Duration**: 2 seconds (execution) + 2 minutes (review)

**Prerequisites**: Task 6 complete (reorganization executed)

**Acceptance Criteria**:
- [x] validate.sh executed
- [x] All 7 validation checks PASS (expect exit code 0)
- [x] reorganization_report.txt generated
- [x] Report shows:
  - 0 loose files in root ✓ (SC-002, SC-007)
  - All required directories exist ✓ (FR-003)
  - Files in correct categories ✓ (FR-002)
  - Script permissions preserved ✓ (FR-005)
  - Git status clean ✓ (FR-008)
  - No broken symlinks ✓ (SC-008)
  - reorg.log complete ✓ (FR-006)

**Implementation**:
```bash
cd /home/xing/Qallow
bash scripts/validate.sh --verbose 2>&1 | tee validation_output.txt

# Check result
tail -50 validation_output.txt
echo "Exit code: $?"
```

**Validation Report Interpretation**:
```
Expected output format:
✓ Check 1: Loose Files [PASS - 0 violations]
✓ Check 2: Directory Existence [PASS - 8 required dirs present]
✓ Check 3: File Organization [PASS - 0 misplaced files]
✓ Check 4: File Permissions [PASS - 55/55 scripts executable]
✓ Check 5: Git Status [PASS - clean tree, commits found]
✓ Check 6: Broken Symlinks [PASS - 0 broken links]
✓ Check 7: Reorg Log [PASS - 17+ operations recorded]

OVERALL: [PASS] - All 7 checks successful
```

**Success Criteria Mapping**:
- [x] SC-001: 100% of files moved → Check 3 (0 misplaced)
- [x] SC-002: Zero loose files → Check 1 (0 violations)
- [x] SC-003: Audit trail complete → Check 7 (log valid)
- [ ] SC-004: Permissions preserved → Check 4 (permissions OK)
- [ ] SC-005: Sub-5-second execution → Validation runs in 2 sec ✓
- [ ] SC-006: Git history maintained → Check 5 (commits exist)
- [ ] SC-007: Violations detected → Check 1 (detection works)
- [x] SC-008: No broken symlinks → Check 6 (0 broken)
- [x] SC-009: Build system unaffected → Build files in root ✓
- [x] SC-010: Documentation explains → Commit messages + audit ✓

**If Validation Fails**:
- Review specific check failure in output
- Check reorg.log for operation errors
- Verify move operations completed (check file locations)
- If critical failure, rollback: `git reset --hard pre-002-reorganization`

**Success Indicator**: ✅ All 7 checks PASS, exit code 0

---

## Task 8: Update Documentation

**Objective**: Record reorganization and document directory structure

**Duration**: 5 minutes

**Prerequisites**: Task 7 complete (validation passed)

**Acceptance Criteria**:
- [x] README.md updated with Constitution § IV reference
- [x] Directory structure section added to README
- [x] Links to relevant documentation added
- [x] Changes committed to git

**Implementation**:

### 8a. Update README.md

Edit README.md to add Directory Structure section:

```markdown
## Project Structure (Constitution § IV)

This project follows Qallow Constitution § IV: Modular Directory Structure.

### Directory Organization

- **backend/{cpu|cuda}/**: CPU and CUDA-optimized phase implementations
- **src/**: CLI tools, runtime services, and distributed coordination
- **python/**: Python quantum bridge and utilities
- **docs/specs/**: Specification-driven design artifacts
- **scripts/**: Build utilities and development scripts
- **config/**: Configuration files (.json, .yaml)
- **deploy/**: Deployment artifacts and container configs
- **tests/**: Unit and integration test suites

For detailed structure, see [ARCHITECTURE_SPEC.md](docs/ARCHITECTURE_SPEC.md).

### Adding New Files

When adding new files, follow Constitution § IV:
1. Identify file type/purpose
2. Place in appropriate canonical directory (not root)
3. Verify no loose files remain: `bash scripts/validate.sh`

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.
```

### 8b. Update CONTRIBUTING.md (or create if missing)

Add file placement guidelines:

```markdown
### File Placement Guidelines (Constitution § IV)

When adding new files to Qallow, ensure they go in the appropriate directory:

| File Type | Directory | Examples |
|-----------|-----------|----------|
| C/CUDA source | backend/{cpu\|cuda}/{module}/ | phase_*.c, *.h |
| Python modules | python/{module}/ | quantum_bridge.py, utils.py |
| Python scripts | scripts/ | run_phase11.py, test_runner.py |
| CLI tools | src/cli/ | CLI entry points |
| Runtime services | src/runtime/ | telemetry_outputs.c, execution.c |
| Configuration | config/ | *.json, *.yaml, .env |
| Documentation | docs/ | *.md, roadmaps, specifications |
| Tests | tests/{module}/ | test_*.py, test_*.c |
| Deployment | deploy/ | Dockerfile, k8s manifests |
| Build scripts | scripts/ | *.sh, CMakeLists.txt helpers |

**NO LOOSE FILES IN ROOT** (except: CMakeLists.txt, Cargo.toml, Makefile, 
Dockerfile, bootstrap.sh, setup.bat, README.md, LICENSE, Qallow.code-workspace)

Verify compliance: `bash scripts/validate.sh`
```

### 8c. Commit Documentation Changes

```bash
cd /home/xing/Qallow
git add README.md CONTRIBUTING.md
git commit -m "docs: add Constitution § IV structure guidelines

- Add directory structure section to README
- Add file placement guidelines to CONTRIBUTING
- Reference canonical directory organization
- Link to validation scripts for compliance checking

This enables future developers to follow Constitution § IV when adding files."
```

**Success Indicator**: ✅ Documentation updated and committed

---

## Task Sequencing & Dependencies

```
Task 1: Review Spec & Plan [START]
  ↓
Task 2: Inventory Current State [depends on 1]
  ↓
Task 3: Create Git Tag [depends on 2]
  ↓
Task 4: Review Mapping [depends on 1]
  ↓
Task 5: Dry-Run [depends on 4]
  ↓
Task 6: Execute [depends on 5]
  ↓
Task 7: Validate [depends on 6]
  ↓
Task 8: Update Docs [depends on 7] → COMPLETE ✅
```

---

## Success Criteria Verification

### Specification Success Criteria Mapping

| SC # | Criterion | Task | Verification |
|------|-----------|------|--------------|
| SC-001 | 100% of categorizable files moved | 6, 7 | Check 3 (0 misplaced) |
| SC-002 | Zero loose files in root | 7 | Check 1 (0 violations) |
| SC-003 | Complete audit trail | 6, 7 | Check 7 (log valid) |
| SC-004 | Content/permissions preserved | 6, 7 | Check 4 (permissions OK) |
| SC-005 | Sub-5-second execution | 6, 7 | Total ~2 sec ✓ |
| SC-006 | Git history maintained | 6, 7 | Check 5 (commits exist) |
| SC-007 | Violations detected | 7 | Check 1 (detection works) |
| SC-008 | No broken symlinks | 7 | Check 6 (0 broken) |
| SC-009 | Build system unaffected | 6 | Build files in root ✓ |
| SC-010 | Documentation explains | 6, 8 | Commits + README ✓ |

### User Story Acceptance Scenarios

**P1: Developer Discovers Clear Structure**
- [ ] Task 2: Baseline shows loose files scattered
- [ ] Task 6: Reorganization groups by type
- [ ] Task 7: Validation confirms organization
- [ ] Task 8: README documents new structure
- **Result**: ✅ Clear structure discoverable

**P2: Maintainer Verifies Compliance**
- [ ] Task 7: validate.sh runs successfully
- [ ] Task 7: All 7 checks pass
- [ ] Task 7: Zero violations reported
- **Result**: ✅ Compliance verified automatically

**P3: Git History Preserved**
- [ ] Task 5: Dry-run shows files will be moved
- [ ] Task 6: Git commit created with rationale
- [ ] Task 7: Check 5 confirms git status clean
- [ ] Task 8: Documentation notes history preservation
- **Result**: ✅ History preserved (git log --follow works)

---

## Effort Estimation Breakdown

| Task | Duration | Parallelizable | Notes |
|------|----------|-----------------|-------|
| 1 | 5 min | No | Sequential (foundation) |
| 2 | 2 min | No | Depends on 1 |
| 3 | 1 min | No | Depends on 2 |
| 4 | 3 min | Yes (parallel to 3) | Depends on 1 |
| 5 | 2 min | No | Depends on 4 |
| 6 | 2 min | No | Depends on 5 |
| 7 | 2 min | No | Depends on 6 |
| 8 | 5 min | No | Depends on 7 |
| **TOTAL** | **22 min** | Sequential | Can't parallelize (git state) |

**Realistic (with buffer)**: 30 minutes

---

## Troubleshooting Guide

### Issue: Scripts not found (reorganize.sh, validate.sh)

**Cause**: Scripts from feature 001 missing  
**Solution**: 
- Check: `ls -la scripts/reorganize.sh scripts/validate.sh`
- If missing, copy from previous implementation or create from scratch

### Issue: Permission denied when moving files

**Cause**: File ownership issues  
**Solution**:
```bash
# Check ownership
ls -la <file>
# Fix if needed
sudo chown $USER:$USER /path/to/file
# Retry
bash scripts/reorganize.sh
```

### Issue: Git commit fails

**Cause**: Git configuration incomplete or uncommitted changes  
**Solution**:
```bash
# Check git status
git status
# Configure if needed
git config user.email "your-email@example.com"
git config user.name "Your Name"
# Retry
bash scripts/reorganize.sh
```

### Issue: Validation fails on Check 1 (loose files remain)

**Cause**: Some files didn't move successfully  
**Solution**:
- Review reorg.log for errors
- Manually move remaining files
- Commit changes: `git add && git commit`
- Re-run validation

### Issue: Need to rollback

**Solution**:
```bash
# Option 1: Soft reset (keep files moved)
git reset --hard HEAD~1

# Option 2: Full rollback to pre-reorganization tag
git reset --hard pre-002-reorganization

# Manually restore if git approach fails
# Use loose_files_baseline.txt to see original state
```

---

## Sign-Off & Next Steps

### Definition of Done (DoD)

Task sequence is complete when:
- ✅ Task 1: Spec & plan reviewed and understood
- ✅ Task 2: Baseline captured (loose_files_baseline.txt)
- ✅ Task 3: Git tag created (pre-002-reorganization)
- ✅ Task 4: Mapping reviewed and confirmed
- ✅ Task 5: Dry-run successful (dry_run_output.txt)
- ✅ Task 6: Reorganization executed (reorg.log created, git commit made)
- ✅ Task 7: Validation passed (all 7 checks ✓, reorganization_report.txt)
- ✅ Task 8: Documentation updated (README.md, CONTRIBUTING.md committed)

### Success Indicators

- ✅ All 8 tasks completed in sequence
- ✅ All 10 specification success criteria met
- ✅ All 7 validation checks pass
- ✅ Git history clean (2 commits: reorganization + documentation)
- ✅ No loose files in root (except canonicals)
- ✅ Zero validation violations

### What's Next

After tasks complete:
1. **Code Review**: Have team review commits and validation report
2. **CI/CD Integration** (future): Add validate.sh to pipeline
3. **CONTRIBUTING Update** (future): Ensure team sees file placement guidelines
4. **Monitoring** (ongoing): Watch for new loose files in root

---

**Tasks Created**: November 6, 2025  
**Owner**: GitHub Copilot (via speckit workflow)  
**Status**: ✅ READY FOR EXECUTION

Start with Task 1. Proceed sequentially. Refer to plan.md for technical details.
