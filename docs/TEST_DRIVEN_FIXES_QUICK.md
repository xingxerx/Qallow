# 🧪 Test-Driven Fixes - Quick Start

## What's New

Your AgentLightning Runner now has **Phase 4 & 5** after tests fail:
- **Phase 4:** Run ctest → Capture output
- **Phase 5:** Analyze output → Apply warning fixes

## Try It Now

```bash
python3 agentlightning_runner.py --max-iterations=1
```

You'll see snprintf truncation warnings detected and fixed automatically.

## New Features

### 1. TestRunner Class
```python
test_runner = TestRunner()
success, output = test_runner.run_tests()
# Returns: (bool, output_string)
```

### 2. WarningFixer Class
```python
fixer = WarningFixer()
fixes = fixer.apply_all_fixes(test_output)
# Returns: number of fixes applied
```

### 3. Automatic Fixes
- **snprintf truncation** → Enlarge buffers (4096→8192)
- **Type warnings** → Add missing includes
- **Unused variables** → Log detection

## Example Output

```
🧪 PHASE 4: Running tests to verify fixes...

Test Output:
1/8 Test #1: unit_ethics_core ... Passed
2/8 Test #2: unit_dl_integration ... Passed
...
⚠️  Some tests failed (exit code 8)

🔧 PHASE 5: Applying warning fixes...

🔍 Detected snprintf truncation warnings
📝 Enlarging buffers in meta_introspect.c
   From: [4096] bytes
   To:   [8192] bytes
✅ Buffer size fix applied!

✅ Applied 1 warning fixes
```

## How It Works

1. Tests run → Output captured
2. Pattern matching on output
3. If "snprintf" + "truncated" found → Fix applied
4. Buffer sizes enlarged automatically
5. Next iteration rebuilds with fix

## Supported Warnings

| Pattern | Fix |
|---------|-----|
| `snprintf` + `truncated` | Enlarge buffer |
| `type` issues | Add includes |
| `unused` variables | Log (analyze only) |

## Add Custom Fixes

Edit `WarningFixer.apply_all_fixes()` to add:

```python
total_fixes += self.fix_my_warning(output)
```

Then add method:
```python
def fix_my_warning(self, output: str) -> int:
    if "my pattern" in output:
        # Apply fix
        return fixes_count
    return 0
```

## Run Options

```bash
# Single iteration with tests & fixes
python3 agentlightning_runner.py --max-iterations=1

# 3 iterations with all phases
python3 agentlightning_runner.py --max-iterations=3

# Daemon mode (continuous)
python3 agentlightning_runner.py --daemon
```

## Workflow Diagram

```
Iteration Loop:
┌─ BUILD → Compile errors
├─ PARSE ERRORS → Detect problems
├─ FIX ERRORS → Syntax fixes
├─ TEST → Run ctest
├─ ANALYZE → Pattern matching
└─ FIX WARNINGS → Auto-fix issues
   └─ REPEAT
```

## Status

✅ Fully implemented and integrated
✅ 5-phase workflow active
✅ 200+ lines added
✅ Ready to use

---

**Run it:** `python3 agentlightning_runner.py --max-iterations=3`
