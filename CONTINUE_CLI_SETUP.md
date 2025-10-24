# Continue CLI - Setup & Usage Guide

## What is Continue CLI?

The Continue CLI (`cn`) is a command-line interface for Continue.dev that lets you use AI directly from the terminal.

## Installation

```bash
npm i -g @continuedev/cli
```

## Login

```bash
cn login
```

This will:
1. Generate an authentication code
2. Open a browser for you to confirm
3. Save your authentication token

## Configuration

The Continue CLI uses the same config as the VS Code extension:
- **Location**: `~/.continue/config.json`
- **Current Setup**: Ollama (no API keys needed)

## Using Continue CLI

### Start the CLI
```bash
cn
```

### Chat with AI
```
● Hello, what can you do?
● Explain the quantum phase 13 implementation
● How does QAOA work?
```

### Use Context
```
● @codebase How does QAOA work?
● @file src/quantum/qaoa.py What does this do?
```

### Shell Mode
```
! ls -la
! npm run build
```

### Exit
```
exit
```

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Send Message | Enter |
| New Line | Shift+Enter |
| Exit | Ctrl+C or type `exit` |

## Configuration

**File**: `~/.continue/config.json`

**Current Setup**:
```json
{
  "models": [
    {
      "title": "Ollama Local",
      "provider": "ollama",
      "model": "llama2",
      "apiBase": "http://localhost:11434"
    }
  ]
}
```

## Requirements

### Ollama Must Be Running

Before using the CLI, start Ollama:
```bash
ollama serve
```

In another terminal, download a model:
```bash
ollama pull llama2
```

## Troubleshooting

### "Failed to connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve

# Verify it's working
curl http://localhost:11434/api/tags
```

### "Model not found"
```bash
# Download a model
ollama pull llama2
```

### "Authentication failed"
```bash
# Re-login
cn login
```

### "No index file found"
This is normal on first run. The CLI will create an index of your codebase.

## Usage Examples

### Example 1: Ask a Question
```
$ cn
● What does the QAOA algorithm do?

The QAOA (Quantum Approximate Optimization Algorithm) is...
```

### Example 2: Use Codebase Context
```
$ cn
● @codebase How is the quantum circuit implemented?

Looking at your codebase, the quantum circuit is implemented in...
```

### Example 3: Reference a File
```
$ cn
● @file src/quantum/qaoa.py Explain this function

This function implements the QAOA algorithm by...
```

### Example 4: Shell Commands
```
$ cn
! npm run build
! ls -la src/
```

## Tips

1. **Keep Ollama running** - Start `ollama serve` in a separate terminal
2. **Use context** - Use `@codebase` and `@file` for better answers
3. **Shell mode** - Use `!` prefix for shell commands
4. **Multi-line** - Use Shift+Enter for multi-line messages

## Comparison: CLI vs VS Code Extension

| Feature | CLI | VS Code |
|---------|-----|---------|
| Chat | ✅ | ✅ |
| Code Editing | ❌ | ✅ |
| Terminal Integration | ✅ | ✅ |
| Codebase Search | ✅ | ✅ |
| File References | ✅ | ✅ |
| Keyboard Shortcuts | Limited | Full |

## Next Steps

1. **Start Ollama**: `ollama serve`
2. **Download Model**: `ollama pull llama2`
3. **Start CLI**: `cn`
4. **Start Chatting**: Type your question

---

**Status**: ✅ Ready to Use
**Configuration**: ✅ Ollama
**Authentication**: ✅ Logged In
**Next**: Run `ollama serve` then `cn`

