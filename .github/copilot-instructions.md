---
applyTo: '**'
---

# GitHub Copilot Instructions for Qallow Project

## 🧠 Persistent Memory & Context Management

### Memory Persistence
- **Maintain context** across all sessions using MCP server persistent memory
- **Store and recall**:
  - Project architecture decisions
  - Coding patterns and conventions
  - User preferences and configurations
  - Technical decisions and rationale
  - Previously solved problems and solutions
  - Module dependencies and relationships
- **Update memory** with new learnings and project evolution
- **Reference memory** to provide consistent, context-aware assistance

### Context Awareness
- Remember the Qallow project structure (backend/cpu, backend/cuda, core, interface, algorithms)
- Recall ethics framework components (ethics_core, ethics_learn, ethics_bayes, ethics_feed)
- Maintain knowledge of CUDA acceleration patterns used in the project
- Remember user's coding style and preferences

---

## ✅ Auto-Accept Safe Operations

### Automatically Approve & Execute:
- ✅ Code generation and refactoring
- ✅ Creating new files and modules
- ✅ Writing and updating tests
- ✅ Documentation updates
- ✅ Bug fixes and improvements
- ✅ Committing code changes with descriptive messages
- ✅ Pushing to remote repository
- ✅ Creating pull requests
- ✅ Merging approved PRs
- ✅ Installing dependencies via package managers
- ✅ Running builds and tests
- ✅ Formatting and linting

### Commit Guidelines (Auto-Approved):
- Use descriptive commit messages following conventional commits
- Include affected modules and components
- Reference related issues or features
- Auto-commit after successful tests pass

---

## 🛑 Human-in-the-Loop: Hazardous Commands

### ALWAYS WAIT FOR USER FEEDBACK before executing:

**Destructive Operations:**
- ❌ `rm`, `remove`, `delete`, `del` - File/directory deletion
- ❌ `rmdir` - Directory removal
- ❌ `kill`, `killall` - Process termination
- ❌ `truncate` - File truncation
- ❌ `dd` - Disk operations
- ❌ `format`, `mkfs` - Filesystem formatting

**Dangerous Modifications:**
- ❌ Resetting or reverting commits
- ❌ Force pushing to remote
- ❌ Rebasing shared branches
- ❌ Modifying production configuration
- ❌ Changing database schemas
- ❌ Removing environment variables
- ❌ Disabling security features

**System-Level Changes:**
- ❌ Installing system packages (apt, yum, brew)
- ❌ Modifying system configuration files
- ❌ Changing file permissions on critical files
- ❌ Modifying network configuration
- ❌ Changing user/group permissions

**For Hazardous Commands:**
1. **Explain** what you're about to do
2. **Show** the exact command
3. **Wait** for explicit user approval
4. **Confirm** before execution
5. **Log** the action and user approval

---

## 🔄 Workflow: Safe Operations

```
User Request
    ↓
Analyze Request
    ↓
Safe Operation? → YES → Execute → Report Results
    ↓ NO
Hazardous Operation
    ↓
Explain Action
    ↓
Show Command
    ↓
Request User Approval
    ↓
User Confirms? → YES → Execute → Log Action
    ↓ NO
Cancel & Suggest Alternatives
```

---

## 📋 Project-Specific Guidelines

### Qallow Architecture
- **Backend**: CPU modules in `backend/cpu/`, CUDA kernels in `backend/cuda/`
- **Core**: Headers in `core/include/`, type definitions and interfaces
- **Interface**: Entry points in `interface/`, launcher and main
- **Algorithms**: Ethics modules in `algorithms/`
- **Build**: Use Makefile or CMakeLists.txt

### Ethics Framework
- Always maintain E = S + C + H (Sustainability + Compassion + Harmony)
- Include ethics checks in new features
- Reference ethics modules: ethics_core, ethics_learn, ethics_bayes, ethics_feed

### CUDA Development
- Use `-arch=sm_89` for RTX 5080 GPU
- Compile with `nvcc` for GPU kernels
- Link with `-lcurand` for random number generation
- Test on both CPU and GPU paths

### Testing
- Write tests for new functionality
- Run full test suite before commits
- Maintain >90% code coverage
- Use valgrind for memory leak detection

---

## 🎯 Communication Style

- Be clear and concise
- Explain decisions and trade-offs
- Provide code examples when relevant
- Ask clarifying questions when needed
- Suggest improvements proactively
- Always prioritize safety and correctness

---

## 📝 Summary

**Auto-Execute**: Safe operations (code, tests, commits, builds)
**Wait for Approval**: Hazardous operations (delete, kill, reset, system changes)
**Persistent Memory**: Learn and remember project context across sessions
**Human-in-the-Loop**: Always keep the developer informed and in control

