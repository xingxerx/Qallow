# Spec-Kit Quick Reference Card

## One-Line Commands

### Create a Feature Specification
```
/specify Describe what you want to build (focus on what and why, not how)
```

### Plan the Implementation  
```
/plan Describe your tech stack and architecture choices
```

### Break Into Actionable Tasks
```
/tasks
```

### Execute Implementation with Testing
```
/implement
```

## Example: Complete Workflow

```
1. /specify Add a metrics dashboard that displays real-time phase coherence scores

2. /plan Use Python with PyTorch, sync data to /home/xing/share/ via Samba, 
          display metrics every 5 seconds

3. /tasks

4. /implement
```

## What Each Command Does

| Command | Creates | Output |
|---------|---------|--------|
| `/specify` | Feature spec | `specs/{N}-{name}/spec.md` |
| `/plan` | Technical plan | `specs/{N}-{name}/plan.md` |
| `/tasks` | Task list | `specs/{N}-{name}/tasks.md` |
| `/implement` | Code + tests | Implementation directory + report |

## Success Criteria

Every implementation must achieve:

- ✅ 100% test success rate (3+ scenarios)
- ✅ Coherence maintained at 1.0
- ✅ Zero crashes or errors
- ✅ Performance within 5% of baseline
- ✅ Documentation updated
- ✅ Status file synchronized

## Key Files

| Path | Purpose |
|------|---------|
| `CONSTITUTION.md` | Project principles |
| `.specify/templates/` | Spec templates |
| `.github/prompts/` | Copilot integration prompts |
| `specs/` | Feature specifications directory |
| `/home/xing/share/status.txt` | Live status sync (Windows: Z:\) |

## Testing Commands

```bash
# Run full test suite
/home/xing/Qallow/.venv/bin/python .github/workflows/Driver.py

# View status
cat /home/xing/share/status.txt

# Check git history
git log --oneline -5
```

## Environment

```bash
# Python executable
/home/xing/Qallow/.venv/bin/python

# Spec-kit CLI
/home/xing/Qallow/.venv/bin/uvx --from git+https://github.com/xingxerx/spec-kit.git specify
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Commands not available | Open Copilot Chat → Select Agent mode → Enable tools |
| Git branches not created | Run `git fetch --all --prune` |
| Templates missing | Verify `.specify/` directory exists |
| Status not syncing | Check `/home/xing/share/` permissions |

## Pro Tips

1. **Write concise specifications** - Focus on business value, not implementation details
2. **Let Copilot break down tasks** - Don't over-engineer the task list
3. **Test incrementally** - Run `/implement` which handles testing automatically
4. **Keep CONSTITUTION.md updated** - Reflect new patterns as you discover them
5. **Use MCP memory service** - It persists context across sessions for long-running features

## Quick Start

1. Open Copilot Chat (Ctrl+Shift+I)
2. Type: `/specify Your feature description here`
3. Type: `/plan Your technical decisions here`
4. Type: `/tasks`
5. Type: `/implement`
6. Done! Feature is implemented and tested.

---

**Version**: 2.2  
**Integrated with**: Qallow v2.2 + GitHub Copilot + MCP Memory Service
