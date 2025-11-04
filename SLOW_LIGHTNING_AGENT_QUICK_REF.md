# 🐢 SLOW Lightning Agent - Quick Reference

## Start Here

The Lightning Agent is now **SLOW and READABLE**. Every fix shows before/after. Every error displays with context. You can read everything.

## Run It

```bash
# Single run: 3 iterations (takes ~2 min)
python3 agentlightning_runner.py --max-iterations=3

# Daemon: Continuous (30s per iteration + pauses)
python3 agentlightning_runner.py --daemon
```

## What You'll See

### Build Phase
```
[BUILD OUTPUT STREAMS HERE AS IT HAPPENS]
   ✓ Compiling...
   ❌ error: missing semicolon at file.c:42
```

### Error Parsing Phase
```
   ✅ Found 3 ERRORS to fix:

   Error 1/3:
   ╔════════════════════════════════════════╗
   ║ file.c:42:10: error: missing semicolon║
   ╚════════════════════════════════════════╝
   [PAUSE 2 SECONDS]
```

### Fix Phase
```
   💡 Fix 1: file.c:42
      missing semicolon
   [PAUSE 2 SECONDS]
   
   🔧 Attempting fix...
   ✏️  Adding missing semicolon
   
   BEFORE: int x = 5
   AFTER:  int x = 5;
   
   ✅ FIX APPLIED!
   [PAUSE 4 SECONDS]
```

## Key Features

| What | How |
|------|-----|
| **Skip pause** | Press **Enter** during any pause |
| **Stop agent** | Press **Ctrl+C** anytime |
| **Adjust speed** | Edit constants at top of file (lines 37-40) |
| **Read output** | Will WAIT 2-5 seconds between each step |
| **See changes** | Before/after code shown for each fix |

## Pause Times

- ⏸️ **2 seconds** - Before fixing an error
- ⏸️ **3 seconds** - When showing code context  
- ⏸️ **4 seconds** - Between individual fixes
- ⏸️ **5 seconds** - Between iterations
- ⏸️ **60 seconds** - Between daemon runs

**All pauses are skippable** by pressing Enter!

## Daemon Mode Countdown

When running daemon, you'll see:
```
────────────────────────────────────────────────────────────────────
⏱️  Daemon sleeping for 60 seconds before next run...
   (Press Ctrl+C to stop)
────────────────────────────────────────────────────────────────────
   60 seconds remaining...
   50 seconds remaining...
   40 seconds remaining...
```

Press **Ctrl+C** anytime to stop the daemon.

## Tweak Speed

Edit `/home/xing/Qallow/agentlightning_runner.py` lines 37-40:

```python
PAUSE_BEFORE_FIX = 2          # ← Change these numbers
PAUSE_SHOW_CODE = 3           # ← to adjust timing
PAUSE_BETWEEN_FIXES = 4       # ← (in seconds)
PAUSE_BETWEEN_ITERATIONS = 5  # ← Keep >= 2 for readability
```

## Output Example

```
======================================================================
🐢 SLOW Lightning Agent - Readable Code Fixer
======================================================================
   Mode: SINGLE RUN
   Max iterations: 3
======================================================================

Starting agent...
[2-second pause]

======================================================================
Iteration 1/3
======================================================================

────────────────────────────────────────────────────────────────────
📝 PHASE 1: Building project...
────────────────────────────────────────────────────────────────────
About to build...
[1-second pause]

   ✅ Build succeeded!

────────────────────────────────────────────────────────────────────
🔍 PHASE 2: Parsing errors...
────────────────────────────────────────────────────────────────────

   ✅ Found 1 ERRORS to fix:

   Error 1/1:
   ╔═══════════════════════════════════════════════════╗
   ║ gray.cpp:42:5: error: conflicting types for 'gray2int'║
   ╚═══════════════════════════════════════════════════╝
[2-second pause]

────────────────────────────────────────────────────────────────────
🔧 PHASE 3: Applying fixes...
────────────────────────────────────────────────────────────────────
Ready to apply fixes...
[2-second pause]

   💡 Fix 1: gray.cpp:42
      conflicting types for 'gray2int'
   About to attempt fix...
   [2-second pause]

🔍 Analyzing imports in gray.cpp...
[1-second pause]

   ✅ Removed unused imports

   💡 Fix 1: gray.cpp:42
      
   ✏️  Fixing function signature...
   
   BEFORE: unsigned int gray2int(unsigned int g)
   AFTER:  int gray2int(uint32_t g)
   
   ✅ Fix applied!
   [4-second pause]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ✅ ITERATION COMPLETE: Applied 1 fix
   📊 Total fixes so far: 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Iteration done. Ready for next...
[5-second pause]

[Iterations 2-3 repeat...]

======================================================================
Agent Finished: 3 iterations, 3 total fixes
======================================================================
Done!
```

## Troubleshooting

### "Can't skip pause" 
The pause timeout might be processing. Press **Enter** harder or wait for timeout.

### "Daemon won't stop"
Press **Ctrl+C** (Ctrl+Break on Windows). Agent will catch it and exit gracefully.

### "Going too fast"
Edit the pause constants (see "Tweak Speed" above) and increase numbers.

### "Going too slow"
Edit the pause constants and decrease numbers. Minimum 1 second recommended.

### "Nothing happens"
1. Make sure CMake is installed: `cmake --version`
2. Make sure build directory exists: `mkdir -p build`
3. Try: `python3 agentlightning_runner.py --max-iterations=1`

## Features

✅ Slow readable output  
✅ Before/after code comparison  
✅ Pause between each step (2-5 seconds)  
✅ Skip pauses with Enter  
✅ Stop anytime with Ctrl+C  
✅ Color-coded build output  
✅ Error context display  
✅ Daemon mode with countdown  
✅ No silent failures  

## Status

- ✅ Converted from FAST to SLOW
- ✅ Added display helpers
- ✅ Added pause constants
- ✅ Updated all phases
- ✅ Syntax checked
- ⏳ Ready to test!

---

**Run it now:** `python3 agentlightning_runner.py --max-iterations=3`
