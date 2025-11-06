# Spec-Kit Integration Setup Guide

**Project**: Qallow  
**Date**: November 4, 2025  
**Status**: ✅ COMPLETE

## Overview

Spec-Kit has been successfully installed and configured for the Qallow project with GitHub Copilot integration. This enables Spec-Driven Development, where specifications define requirements before implementation.

## Installation Summary

### What Was Installed

```
✅ spec-kit CLI (via uv)
✅ GitHub Copilot prompts (.github/prompts/)
✅ Specification templates (.specify/templates/)
✅ Build scripts (.specify/scripts/bash/)
✅ Memory system (.specify/memory/)
✅ CONSTITUTION.md (project principles)
```

### Key Directories

```
/home/xing/Qallow/
├── .specify/                          # Spec-kit configuration
│   ├── templates/                     # Template files for specs
│   ├── scripts/bash/                  # Helper scripts
│   └── memory/                        # Memory system
├── .github/prompts/                   # Copilot integration prompts
│   ├── speckit.specify.prompt.md      # Create specifications
│   ├── speckit.plan.prompt.md         # Technical planning
│   ├── speckit.tasks.prompt.md        # Task breakdown
│   ├── speckit.implement.prompt.md    # Implementation guidance
│   └── [other utility prompts]
└── CONSTITUTION.md                    # Project principles
```

## Using Spec-Kit with GitHub Copilot

### Quick Start Commands

Use these commands in GitHub Copilot Chat (within VS Code) to guide development:

#### 1. **Create a Specification** (`/specify`)

```
/specify Build a real-time log monitoring dashboard that tracks ethics scores across phases 12-15
```

**Result**: Creates a feature spec in `specs/{number}-{short-name}/spec.md`

**What it does**:
- Generates a concise branch name
- Creates feature specification with requirements
- Sets up specification directory structure
- Provides implementation checklist

#### 2. **Plan Architecture** (`/plan`)

```
/plan The dashboard uses PyTorch for metric aggregation, Matplotlib for visualization, 
and syncs to /home/xing/share/status.txt for real-time monitoring on Windows (Z:\)
```

**Result**: Creates technical plan in `specs/{number}-{short-name}/plan.md`

**What it does**:
- Documents tech stack decisions
- Establishes architectural constraints
- Identifies required dependencies
- Defines integration points

#### 3. **Break Down into Tasks** (`/tasks`)

```
/tasks
```

**Result**: Creates task list in `specs/{number}-{short-name}/tasks.md`

**What it does**:
- Decomposes specification into actionable items
- Identifies task dependencies
- Defines success criteria
- Estimates complexity

#### 4. **Implement** (`/implement`)

```
/implement
```

**Result**: Provides implementation guidance with code examples

**What it does**:
- Walks through each task systematically
- Generates code following project patterns
- Runs tests on completion
- Creates verification reports

### Example Workflow

#### Step 1: Create Feature Spec

```
User: /specify Add adaptive phase timeout adjustment based on coherence scores

Copilot Response:
✅ Branch: 03-adaptive-phase-timeout
✅ Created: specs/03-adaptive-phase-timeout/spec.md
✅ Ready for: Planning with /plan command
```

#### Step 2: Plan Implementation

```
User: /plan Use the existing coherence tracking system, add a new timeout algorithm 
to backend/cpu/phases/phase13.c, update telemetry exports

Copilot Response:
✅ Tech Stack: C (backend) + Python (telemetry)
✅ Dependencies: Existing phase system
✅ Integration: Telemetry pipeline
✅ Created: specs/03-adaptive-phase-timeout/plan.md
```

#### Step 3: Create Tasks

```
User: /tasks

Copilot Response:
📋 Task 1: Analyze coherence decay patterns
📋 Task 2: Design timeout algorithm
📋 Task 3: Implement in phase13.c
📋 Task 4: Update telemetry exports
📋 Task 5: Add performance tests
📋 Task 6: Validate across 3+ scenarios
```

#### Step 4: Implement and Test

```
User: /implement

Copilot Response:
[Implements each task progressively]
✅ Task 1: Complete
✅ Task 2: Complete
✅ Task 3: Complete (with code review)
✅ Task 4: Complete (with telemetry verification)
✅ Task 5: Complete (3 new test scenarios)
✅ Task 6: Complete (100% pass rate)

📊 Final Report:
- Success: 3/3 scenarios
- Coherence: 1.0
- Performance: +2% improvement
```

## Project Structure

### Specifications Directory

After creating a spec, the directory structure looks like:

```
specs/
├── 01-user-auth/
│   ├── spec.md          # Feature specification
│   ├── plan.md          # Technical plan
│   ├── tasks.md         # Task breakdown
│   └── implementation/   # Implementation files
│
├── 02-dashboard-ui/
│   └── ...
│
└── 03-adaptive-timeout/
    └── ...
```

### Available Templates

Templates are in `.specify/templates/`:

- **spec-template.md** - Feature specification format
- **plan-template.md** - Technical planning structure
- **tasks-template.md** - Task breakdown format
- **checklist-template.md** - Implementation checklist
- **agent-file-template.md** - Agent interaction log

## Copilot Chat Setup

### Enabling Spec-Kit Commands

1. **Open VS Code** in `/home/xing/Qallow`
2. **Open Copilot Chat** (Ctrl+Shift+I or Cmd+Shift+I)
3. **Select Agent Mode** (dropdown in chat header)
4. **Tools Icon** → Enable MCP memory service

### Available Prompts

The spec-kit prompts automatically available in Copilot:

| Command | Prompt File | Purpose |
|---------|------------|---------|
| `/specify` | speckit.specify.prompt.md | Create feature specifications |
| `/plan` | speckit.plan.prompt.md | Technical architecture planning |
| `/tasks` | speckit.tasks.prompt.md | Task decomposition |
| `/implement` | speckit.implement.prompt.md | Implementation execution |
| `/clarify` | speckit.clarify.prompt.md | Resolve ambiguities |
| `/analyze` | speckit.analyze.prompt.md | Analyze specifications |
| `/checklist` | speckit.checklist.prompt.md | Verify completion |

## Integration with Qallow Project

### Connected Systems

#### 1. **MCP Memory Service** (`/home/xing/Qallow/mcp-memory-service/`)
- Provides persistent context across sessions
- Stores specification decisions and rationales
- Enables long-running feature development
- Configuration: Port 8000, SQLite-vec backend

#### 2. **Network Storage** (`/home/xing/share/`)
- Real-time status syncing
- Windows interop via Samba
- Live metrics export
- Cross-platform monitoring

#### 3. **Telemetry System** (`data/logs/`)
- Automatic metric collection
- CSV and JSON exports
- Phase-level performance tracking
- Integration with specifications

#### 4. **Build Pipeline** (`.specify/scripts/bash/`)
- Automated project setup
- Dependency management
- Test execution
- Status reporting

### Environment Configuration

```bash
# Python environment (already configured)
/home/xing/Qallow/.venv/bin/python

# uv package manager (installed)
/home/xing/Qallow/.venv/bin/uv

# Spec-kit CLI
/home/xing/Qallow/.venv/bin/uvx --from git+https://github.com/xingxerx/spec-kit.git specify
```

## Usage Examples

### Example 1: Add New Phase Feature

```
User: /specify Add a new phase 16 for distributed consensus validation that 
       combines results from phases 12-15

Copilot: ✅ Branch: 04-phase16-consensus
         Specification created in specs/04-phase16-consensus/spec.md

User: /plan Phase 16 runs after lattice convergence (phase 15), uses aggregated state 
      from Redis cache, validates against ethics constraints, updates telemetry

Copilot: ✅ Technical plan created
         Key decisions documented

User: /tasks
Copilot: 📋 6 tasks created:
         - Design consensus algorithm
         - Implement core logic
         - Add Redis integration
         - Create validation layer
         - Add telemetry exports
         - Run multi-scenario tests

User: /implement
Copilot: [Executes all tasks with testing]
         ✅ All 6 tasks complete
         📊 3/3 scenarios pass
         Status: Ready for production
```

### Example 2: Fix Performance Issue

```
User: /specify Phase 13 is taking 30% longer than phase 12, need to optimize 
      the quantum bridge computation

Copilot: ✅ Branch: 05-phase13-optimization
         Spec created

User: /plan Profile the code first using NVIDIA NSight, identify hotspots,
      optimize memory access patterns in the CUDA kernels

Copilot: ✅ Plan documented with profiling approach

User: /tasks
Copilot: 📋 4 tasks:
         - Profile with NSight
         - Analyze memory patterns
         - Implement optimizations
         - Benchmark improvements

User: /implement
Copilot: [Executes with testing]
         ✅ Performance improved by 35%
         ✅ All tests passing
         Status: 3/3 scenarios successful
```

## Best Practices

### 1. **Write Clear Specifications**
✅ DO: "Add real-time metric aggregation that combines ethics scores from the current phase"  
❌ DON'T: "Make it faster"

### 2. **Plan Before Coding**
- Always use `/plan` before `/tasks`
- Document architecture decisions
- Identify integration points

### 3. **Break Down Large Features**
- Use `/tasks` to decompose complexity
- Aim for 5-10 actionable tasks
- Define clear success criteria for each

### 4. **Test Continuously**
- Run tests after each task
- Validate across multiple scenarios
- Keep coherence at 1.0

### 5. **Document Decisions**
- Explanation in specs
- Technical rationale in plans
- Implementation notes in code comments

## Troubleshooting

### Issue: Commands Not Available

**Solution**: 
1. Open Copilot Chat (Ctrl+Shift+I)
2. Select "Agent" mode from dropdown
3. Click "Tools" icon
4. Ensure MCP memory service is enabled

### Issue: Git Branches Not Created

**Solution**:
1. Verify Git is installed: `git --version`
2. Check repo is initialized: `git status`
3. Run: `git fetch --all --prune`

### Issue: Templates Not Found

**Solution**:
- Templates at: `/home/xing/Qallow/.specify/templates/`
- Verify directory exists: `ls -la .specify/`
- Recreate if needed: `rm -rf .specify && specify init --here --ai copilot`

## Next Steps

### 1. **Create First Spec** 
Use `/specify` to create a new feature specification for your next task.

### 2. **Review CONSTITUTION.md**
Understand the project principles before development.

### 3. **Run Full Test Suite**
```bash
cd /home/xing/Qallow
/home/xing/Qallow/.venv/bin/python .github/workflows/Driver.py
```

### 4. **Explore Prompts**
Review the spec-kit prompts in `.github/prompts/` to understand available commands.

## Commands Reference

### Quick Copy-Paste Commands

```bash
# Check system setup
cd /home/xing/Qallow && /home/xing/Qallow/.venv/bin/uvx --from git+https://github.com/xingxerx/spec-kit.git specify check

# Verify Copilot setup
/home/xing/Qallow/.venv/bin/python --version
cat /home/xing/Qallow/.vscode/mcp.json

# Test execution
/home/xing/Qallow/.venv/bin/python .github/workflows/Driver.py

# View status
cat /home/xing/share/status.txt

# Git status
git log --oneline -10
```

## Resources

- **Spec-Kit Repository**: https://github.com/xingxerx/spec-kit
- **GitHub Copilot Docs**: https://docs.github.com/en/copilot
- **Qallow Repository**: https://github.com/xingxerx/Qallow
- **MCP Memory Service**: `/home/xing/Qallow/mcp-memory-service/`

## Support

- Review CONSTITUTION.md for project principles
- Check `.github/prompts/` for available commands
- View existing specs in `specs/` directory
- Consult README.md for technical details

---

**Setup Complete** ✅

You can now use spec-kit with GitHub Copilot to drive development of the Qallow project. Start with `/specify` to create your first feature specification!
