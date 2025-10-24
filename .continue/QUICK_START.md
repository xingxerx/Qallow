# Continue.dev Quick Start

## 1-Minute Setup

```bash
# From /root/Qallow directory:
chmod +x setup_continue.sh
./setup_continue.sh

# Follow the prompts to choose your AI provider
```

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Open Chat | Ctrl+L (Cmd+L) |
| Edit Code | Ctrl+K (Cmd+K) |
| Terminal | Ctrl+Shift+` |
| Accept Suggestion | Tab |
| Reject Suggestion | Esc |

## Quick Commands

### Chat Examples

```
"Explain this function"
"How do I use the quantum phase 13 API?"
"Write a test for this code"
"Fix the bug in this code"
"Refactor this for performance"
```

### With Context

```
"@codebase How is the QAOA algorithm implemented?"
"@file src/quantum/qaoa.py What does this do?"
"@memory What was the last issue we discussed?"
```

## Providers

### Ollama (Recommended for Testing)
- ✅ Free
- ✅ Local (no internet needed)
- ✅ Private
- ⚠️ Slower than cloud models

**Setup**:
```bash
ollama serve
ollama pull llama2
```

### Gemini (Free Tier)
- ✅ Free tier available
- ✅ Very fast
- ✅ Good quality
- ⚠️ Requires API key

**Setup**:
```bash
# Get key: https://aistudio.google.com/app/apikey
# Add to .env: GEMINI_API_KEY=your-key
source .env
```

### Claude (Paid)
- ✅ Excellent quality
- ✅ Good for complex tasks
- ⚠️ Requires paid account

**Setup**:
```bash
# Get key: https://console.anthropic.com/
# Add to .env: ANTHROPIC_API_KEY=your-key
source .env
```

### GPT-4 (Paid)
- ✅ Excellent quality
- ✅ Very capable
- ⚠️ Requires paid account

**Setup**:
```bash
# Get key: https://platform.openai.com/api-keys
# Add to .env: OPENAI_API_KEY=your-key
source .env
```

## Troubleshooting

### "Model not found" error
1. Check `.env` has your API key
2. Run: `source .env`
3. Restart VS Code

### Ollama not connecting
1. Make sure Ollama is running: `ollama serve`
2. Check: `curl http://localhost:11434/api/tags`
3. Restart Continue

### API key not working
1. Verify key is correct
2. Check it hasn't expired
3. Try creating a new one
4. Restart VS Code

## Files

- **Setup Script**: `setup_continue.sh`
- **Config**: `.continue/config.json`
- **Environment**: `.env` (not in git)
- **Full Guide**: `CONTINUE_SETUP_GUIDE.md`
- **MCP Servers**: `.continue/mcpServers/`

## Next Steps

1. ✅ Run `./setup_continue.sh`
2. ✅ Choose your provider
3. ✅ Restart VS Code
4. ✅ Press Ctrl+L to start chatting!

## Tips

- Use `@codebase` to search the entire project
- Use `@file` to reference specific files
- Use `@memory` to access persistent memory
- Highlight code before pressing Ctrl+K for edits
- Check VS Code Output panel for error messages

---

For full documentation, see `CONTINUE_SETUP_GUIDE.md`

