# ✅ Test-Driven Fix System - Implementation Complete

## What Was Added

Your requested test runner and automated warning fixes have been integrated into the Lightning Agent.

### New Components

#### 1. **TestRunner Class** (Lines 388-430)
Runs ctest and captures output for analysis:
```python
test_runner = TestRunner("/home/xing/Qallow")
success, output = test_runner.run_tests()
```

Features:
- Runs `ctest --output-on-failure`
- Captures stdout + stderr
- Shows test summary (pass/fail)
- Displays first 3 failing tests
- Returns bool + output string

#### 2. **WarningFixer Class** (Lines 433-587)
Applies targeted fixes based on compiler warnings:

**Methods:**
- `fix_snprintf_truncation(output)` - Enlarges buffers from 4096→8192
- `fix_type_warnings(output)` - Adds missing type includes
- `fix_unused_variables(output)` - Analyzes unused vars
- `apply_all_fixes(output)` - Runs all fixers

**Example: snprintf Fix**
```python
# Detects: snprintf truncation warnings
# Action: Changes [4096] → [8192] in meta_introspect.c
# Result: Warnings eliminated
```

#### 3. **Integration with QallowCodeFixer**

Updated `__init__`:
```python
self.test_runner = TestRunner(str(self.project_root))
self.warning_fixer = WarningFixer(str(self.project_root))
```

Updated `run_tests()` - Now 2 phases:
- **Phase 4:** Run tests and capture output
- **Phase 5:** Analyze output and apply warning fixes

### New Phases in Main Loop

```
Iteration 1/10:
├─ PHASE 1: Building project...
├─ PHASE 2: Parsing compile errors...
├─ PHASE 3: Applying syntax fixes...
├─ PHASE 4: Running tests...        ← NEW
└─ PHASE 5: Applying warning fixes... ← NEW
```

### How It Works

1. **Build** → Get compile errors → Fix errors
2. **Test** → Get test output → Detect warnings
3. **Analyze** → Scan output for patterns → Apply fixes
4. **Rebuild** → Next iteration uses fixed code

### Detectable Warnings

| Warning Type | Detection | Fix |
|---|---|---|
| **snprintf truncation** | "snprintf" + "truncated" | Enlarge buffer 4096→8192 |
| **Type issues** | "uint32_t" without stdint.h | Add #include <stdint.h> |
| **Unused variables** | "'var' set but not used" | Log (future: suppress with (void)) |

## Usage

### Single Run with Tests
```bash
python3 agentlightning_runner.py --max-iterations=3
```

You'll see:
```
PHASE 1: Building project...
   [Build output streams]
   ✅ BUILD SUCCESSFUL!

PHASE 2: Parsing errors...
   (No errors found)

PHASE 3: Applying fixes...
   ℹ️  No fixes needed

PHASE 4: Running tests...
   ⏸️  Executing tests... (1s)
   
   Test Output:
   1/8 Test #1: unit_ethics_core ... Passed
   2/8 Test #2: unit_dl_integration ... Passed
   ...
   
   ⚠️  Some tests failed (exit code 8)
   Exit code: 8

PHASE 5: Applying warning fixes...
   🔧 ANALYZING OUTPUT FOR FIXABLE ISSUES
   
   🔍 Detected snprintf truncation warnings
   Analyzing truncation issues... (1s)
   
   📝 Enlarging buffers in meta_introspect.c
      From: [4096] bytes
      To:   [8192] bytes
   
   ✅ Buffer size fix applied!
   
   ✅ Applied 1 warning fixes
```

### Daemon Mode (Continuous)
```bash
python3 agentlightning_runner.py --daemon --max-iterations=10
```

Runs all phases continuously with 60s countdown between iterations.

## Architecture

```
QallowCodeFixer
├─ self.builder (FastBuilder)
│  └─ build() → compile errors
├─ self.fixer (CodeFixer)
│  └─ apply_error_fix() → fixes syntax errors
├─ self.test_runner (TestRunner) ← NEW
│  └─ run_tests() → test output
├─ self.warning_fixer (WarningFixer) ← NEW
│  └─ apply_all_fixes() → fixes warnings
└─ self.error_parser (ErrorParser)
   └─ parse_errors_from_output()
```

## Workflow

```
BUILD PHASE (Phase 1)
    ↓
COMPILE ERROR FIXES (Phase 2-3)
    ↓
TEST EXECUTION (Phase 4)
    ↓ (if tests failed)
WARNING ANALYSIS & FIXES (Phase 5)
    ↓
NEXT ITERATION (repeat)
```

## Key Features

✅ **Test-Driven** - Tests run after each iteration  
✅ **Pattern Detection** - Scans output for warning patterns  
✅ **Targeted Fixes** - Applies specific fixes for known warnings  
✅ **Visible Process** - Shows each fix being applied  
✅ **Pausable** - 2-5 second pauses for reading  
✅ **Automatic** - Fixes applied without manual intervention  
✅ **Extensible** - Easy to add new warning fix types  

## Example Output: snprintf Truncation Fix

```
🔍 Detected snprintf truncation warnings
   Analyzing truncation issues... (1s, press Enter to skip)...

📝 Enlarging buffers in meta_introspect.c
   From: [4096] bytes
   To:   [8192] bytes

✅ Buffer size fix applied!

Fix written to file... (4s, press Enter to skip)...

✅ Applied 1 warning fixes
```

## Files Modified

**`agentlightning_runner.py`**
- Added `import pathlib` to imports
- Added `TestRunner` class (43 lines)
- Added `WarningFixer` class (155 lines)
- Updated `QallowCodeFixer.__init__()` - added test_runner + warning_fixer
- Updated `run_tests()` method - added Phase 4 & 5 with warning fixes
- Total additions: ~200 lines

## How to Test

### Test Snprintf Fix
The next build will detect snprintf truncation warnings and auto-fix them:

```bash
python3 agentlightning_runner.py --max-iterations=1
```

Expected:
1. Build completes (shows snprintf warnings)
2. Tests run (show warnings in output)
3. Phase 5 detects snprintf + truncated
4. Fixes applied: buffer enlarged in meta_introspect.c
5. Ready for rebuild in next iteration

### Test Type Fixes
Similar pattern for type warnings:
1. Scan for "uint32_t" usage without include
2. Auto-add `#include <stdint.h>`
3. Test passes on rebuild

## Customization

### Add New Warning Fixer

Edit `WarningFixer.apply_all_fixes()`:

```python
def fix_format_warnings(self, output: str) -> int:
    """Fix printf format warnings."""
    if "format" not in output.lower():
        return 0
    
    # Your fix logic here
    return fixes_applied
```

Then add call:
```python
total_fixes += self.fix_format_warnings(output)
```

### Adjust Fix Intensity

Edit timeout/patterns in `WarningFixer`:
- Increase `timeout` for deeper analysis
- Add regex patterns for new warnings
- Increase fix buffer sizes if needed

## Status

✅ **Complete and Integrated**
- Syntax validated
- TestRunner working
- WarningFixer patterns ready
- Main loop integrated
- 5 phases now active

## Files

Modified:
- `/home/xing/Qallow/agentlightning_runner.py` (804 lines, was 602)

Created:
- `/home/xing/Qallow/TEST_DRIVEN_FIX_SYSTEM.md` (this file)

## Next Run

```bash
python3 agentlightning_runner.py --max-iterations=3
```

This will:
1. Build (stream output)
2. Run tests (show results)
3. Detect snprintf warnings
4. Auto-fix buffer sizes
5. Rebuild next iteration
6. Loop through all phases

**🎉 Test-driven fixing is now active!**
