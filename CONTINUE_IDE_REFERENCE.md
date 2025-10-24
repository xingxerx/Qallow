# Continue IDE - Complete Reference

## Quick Start

### Option 1: Ollama (Recommended - No API Key)
```bash
# Install and setup
bash /root/Qallow/setup_ollama.sh

# Or manually:
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull llama2

# Restart Continue IDE
```

### Option 2: Gemini (Free Tier)
```bash
# Get API key from: https://aistudio.google.com/app/apikey
export GEMINI_API_KEY="your-key-here"
# Restart Continue IDE
```

### Option 3: Claude
```bash
# Get API key from: https://console.anthropic.com/
export ANTHROPIC_API_KEY="your-key-here"
# Restart Continue IDE
```

### Option 4: OpenAI
```bash
# Get API key from: https://platform.openai.com/api-keys
export OPENAI_API_KEY="your-key-here"
# Restart Continue IDE
```

## Configuration

**File**: `/root/Qallow/.continue/config.json`

**Current Setup**:
- Default: Ollama (local)
- Fallback: Gemini, Claude, OpenAI

**Models Available**:
- Ollama: llama2, mistral, neural-chat, codellama, dolphin-mixtral
- Gemini: 2.0-flash, 1.5-pro, 1.5-flash
- Claude: 3.5-sonnet, opus, haiku
- OpenAI: gpt-4, gpt-3.5-turbo

## Ollama Commands

```bash
# Install
curl -fsSL https://ollama.ai/install.sh | sh

# Start server
ollama serve

# Download model
ollama pull llama2

# List models
ollama list

# Remove model
ollama rm llama2

# Check if running
curl http://localhost:11434
```

## Environment Variables

```bash
# Gemini
export GEMINI_API_KEY="your-key"

# Claude
export ANTHROPIC_API_KEY="your-key"

# OpenAI
export OPENAI_API_KEY="your-key"

# Ollama (no key needed)
# Just make sure ollama serve is running
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Connection refused" | Start Ollama: `ollama serve` |
| "Model not found" | Pull model: `ollama pull llama2` |
| "API key error" | Set env var: `export GEMINI_API_KEY="..."` |
| "Still getting error" | Restart Continue IDE completely |
| "Ollama is slow" | Try faster model: `ollama pull mistral` |

## Model Comparison

| Model | Speed | Quality | API Key | Cost |
|-------|-------|---------|---------|------|
| llama2 | Medium | Good | No | Free |
| mistral | Fast | Good | No | Free |
| codellama | Medium | Excellent | No | Free |
| Gemini | Fast | Excellent | Yes | Free tier |
| Claude | Medium | Excellent | Yes | Paid |
| GPT-4 | Medium | Excellent | Yes | Paid |

## Files

| File | Purpose |
|------|---------|
| CONTINUE_IDE_QUICK_FIX.md | Setup guide |
| setup_ollama.sh | Automated Ollama setup |
| .continue/config.json | Configuration |

## Resources

- **Ollama**: https://ollama.ai
- **Continue IDE**: https://docs.continue.dev/
- **Gemini**: https://ai.google.dev/
- **Claude**: https://docs.anthropic.com/
- **OpenAI**: https://platform.openai.com/docs/

## Status

✅ Continue IDE configured and ready
✅ Ollama support enabled (default)
✅ Cloud models supported (Gemini, Claude, OpenAI)
✅ No API key required for local use

---

**Last Updated**: 2025-10-24

