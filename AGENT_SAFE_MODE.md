# Lightning Agent - Safe Operation Guide

**Status:** ✅ SAFE MODE ENABLED - NO AUTO-PUSH

## How It Works Now

### Agent Behavior
✅ **STAGES changes for human review**
❌ **NO automatic commits**
❌ **NO automatic pushes to GitHub**

### Workflow

1. **Start the daemon:**
   ```bash
   QALLOW_CIRQ=1 QALLOW_ENABLE_CUDA=ON python3 agentlightning_runner.py \
     --fast --use-cuda --daemon --max-iterations=500
   ```

2. **Agent analyzes code and stages improvements:**
   ```
   📋 ✅ STAGED 6 improvement(s) for review
   📋 To review: git diff --cached
   📋 To commit: git commit -m 'Refactor: Code quality improvements - 6 fixes'
   📋 To push: git push origin main
   📋 To discard: git reset HEAD .
   ```

3. **You review the staged changes:**
   ```bash
   # See what the agent found
   git diff --cached
   
   # See file-by-file diffs
   git diff --cached --stat
   
   # See specific file
   git diff --cached path/to/file.py
   ```

4. **Decide what to do:**
   
   **Option A: Accept all changes**
   ```bash
   git commit -m "Refactor: Code quality improvements - 6 fixes applied"
   git push origin main
   ```
   
   **Option B: Accept some changes**
   ```bash
   # Unstage files you don't want
   git reset HEAD path/to/file.py
   
   # Commit what's left
   git commit -m "Refactor: Code quality improvements - 4 fixes applied"
   git push origin main
   ```
   
   **Option C: Reject all changes**
   ```bash
   git reset HEAD .
   git checkout .
   ```

5. **Monitor daemon progress:**
   ```bash
   # Watch in real-time
   tail -f agent_daemon.log
   
   # Check staged changes
   git status
   
   # See what's staged
   git diff --cached --stat
   ```

## What The Agent Improves

✅ **Safe improvements (ENABLED):**
- Code style: Removes trailing whitespace, fixes blank lines
- Performance: Detects anti-patterns
- Variable naming: Improves clarity
- Excessive blank lines: Cleans up formatting

❌ **Dangerous features (DISABLED):**
- Unused imports: Removes necessary imports even with false positives
- Dead code: Too aggressive, removes valid code
- Function complexity: Adds useless comments

## Safety Features

| Feature | Status | Why |
|---------|--------|-----|
| Auto-commit | ❌ Disabled | Requires human review |
| Auto-push | ❌ Disabled | Requires human review |
| Staging | ✅ Enabled | Changes reviewed before commit |
| Import removal | ❌ Disabled | False positives broke CI |
| Dead code removal | ❌ Disabled | Removed valid code |
| Function complexity | ❌ Disabled | Added useless comments |

## Useful Commands

```bash
# View staged changes
git diff --cached

# View staged changes by file
git diff --cached --stat

# View specific staged file
git diff --cached path/to/file.py

# Unstage specific file
git reset HEAD path/to/file.py

# Unstage all
git reset HEAD .

# Discard all staged changes
git checkout .

# Show git status
git status

# Commit staged changes
git commit -m "Your message"

# Push to GitHub
git push origin main

# View recent commits
git log --oneline -10

# Stop daemon
pkill -f "agentlightning_runner.py"
```

## Example Session

```bash
# 1. Start daemon
$ QALLOW_CIRQ=1 QALLOW_ENABLE_CUDA=ON python3 agentlightning_runner.py --daemon

# 2. Wait for improvements (check logs)
$ tail -f agent_daemon.log

# 3. Check what was staged
$ git status
On branch main
Changes to be committed:
  (use "git restore --cached <file>..." to unstage)
    modified:   python/quantum/adaptive_agent.py
    modified:   python/quantum/web_api.py
    modified:   sections/Compute.py

# 4. Review changes
$ git diff --cached --stat
 python/quantum/adaptive_agent.py | 2 -
 python/quantum/web_api.py        | 3 -
 sections/Compute.py              | 4 -

# 5. Accept and push
$ git commit -m "Refactor: Code quality improvements - 3 fixes"
$ git push origin main
```

## Important Notes

⚠️ **The agent is now SAFE because:**
- Changes must be reviewed before committing
- You control all commits
- You control all pushes
- No surprise GitHub changes

⚠️ **You must manually:**
- Review staged changes: `git diff --cached`
- Commit changes: `git commit -m "..."`
- Push to GitHub: `git push origin main`

## Troubleshooting

**Q: Agent is staging but I don't see changes?**
```bash
# Check git status
git status

# Check staged diff
git diff --cached
```

**Q: I want to accept some but not all changes?**
```bash
# Unstage files you don't want
git reset HEAD path/to/unwanted/file.py

# Commit what's left
git commit -m "Refactor: Code quality improvements - 2 fixes"
```

**Q: How do I stop the agent?**
```bash
pkill -f "agentlightning_runner.py"
```

**Q: How do I see what the agent did?**
```bash
git diff --cached               # Staged changes
git diff --cached path/file.py  # Specific file
git diff --cached --stat        # Summary
```

---

**Status:** ✅ Safe for Production  
**Last Updated:** November 3, 2025  
**Mode:** HUMAN REVIEW REQUIRED - NO AUTO-PUSH
