---
applyTo: '**'
---

# GitHub Copilot Instructions

## Auto-Confirm Tasks

When working with task management:
- Automatically confirm and execute safe tasks without requiring explicit approval
- Safe tasks include: code generation, testing, analysis, documentation, and refactoring
- Require explicit user approval ONLY for potentially destructive operations:
  - Deleting files or directories
  - Removing dependencies
  - Killing processes
  - Resetting or reverting changes
  - Deploying to production
  - Modifying configuration files that affect system behavior

## Persistent Memory from MCP Servers

When using MCP (Model Context Protocol) servers:
- Leverage persistent memory capabilities to maintain context across sessions
- Store important project decisions, patterns, and conventions in memory
- Reference previously learned information about:
  - Project architecture and structure
  - Coding standards and conventions
  - Common patterns and solutions
  - User preferences and configurations
  - Technical decisions and their rationale
- Use memory to provide consistent, context-aware assistance
- Update memory with new learnings and project evolution

## General Guidelines

- Provide project context and coding guidelines for code generation
- Follow existing code patterns and conventions
- Ensure consistency with the project's architecture
- Write clear, maintainable code with appropriate comments
- Include relevant tests for new functionality
- Keep documentation up-to-date with changes
