---
type: "agent_requested"
description: "Example description"
---

# Persistent Memory Rule

- Rules are instructions for Augment Chat and Agent that can be applied automatically across all conversations or referenced in specific conversations using @mentions (for example, `@Memory.md`).
- Maintain persistent memory across chats: capture user preferences and project context when requested and reload that context at the start of future sessions.
- When the user asks to store new context, append it to the appropriate memory artifact so it is available to future conversations.
- Prioritize non-destructive updates that preserve existing guidance while expanding long-term memory fidelity.
- **Hardware**: CUDA-enabled (GPU priority).  
- **Parallel**: Async tool calls + multiprocessing (torch DataParallel).  
- **Reload**: Auto-load at session start.  
- **Context**: AGI-Quantum ML → Sampling → Hybrid → Attention.  