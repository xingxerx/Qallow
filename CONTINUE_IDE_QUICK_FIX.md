# Continue IDE - Quick Fix (No API Key Required!)

## The Problem

Continue IDE is showing an error about missing `GEMINI_API_KEY` and trying to use their cloud proxy service.

## The Solution

**Good news!** We've configured Continue IDE to work with **Ollama** (local AI) by default - **no API key required!**

## Option 1: Use Ollama (Recommended - No API Key Needed)

### Step 1: Install Ollama

**On Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Or download from: https://ollama.ai

**On macOS:**
```bash
brew install ollama
```

**On Windows:**
Download from: https://ollama.ai/download

### Step 2: Start Ollama

```bash
ollama serve
```

This starts the Ollama server on `http://localhost:11434`

### Step 3: Pull a Model

In a new terminal:
```bash
ollama pull llama2
```

Other available models:
```bash
ollama pull mistral          # Faster, good for coding
ollama pull neural-chat      # Good for conversations
ollama pull codellama        # Specialized for code
ollama pull dolphin-mixtral  # More capable
```

### Step 4: Restart Continue IDE

Close and reopen Continue IDE. It should now work with Ollama!

## Option 2: Use Gemini (Free Tier Available)

If you want to use Google Gemini instead:

### Step 1: Get API Key

1. Go to: https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy your key

### Step 2: Set Environment Variable

Add to `~/.bashrc` or `~/.zshrc`:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Then reload:
```bash
source ~/.bashrc
```

### Step 3: Restart Continue IDE

Close and reopen Continue IDE.

## Option 3: Use Claude (Anthropic)

### Step 1: Get API Key

1. Go to: https://console.anthropic.com/
2. Create an API key
3. Copy your key

### Step 2: Set Environment Variable

Add to `~/.bashrc` or `~/.zshrc`:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Then reload:
```bash
source ~/.bashrc
```

### Step 3: Restart Continue IDE

Close and reopen Continue IDE.

## Option 4: Use OpenAI (GPT-4)

### Step 1: Get API Key

1. Go to: https://platform.openai.com/api-keys
2. Create an API key
3. Copy your key

### Step 2: Set Environment Variable

Add to `~/.bashrc` or `~/.zshrc`:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Then reload:
```bash
source ~/.bashrc
```

### Step 3: Restart Continue IDE

Close and reopen Continue IDE.

## Current Configuration

Your Continue IDE is now configured with:

1. **Ollama (Default)** - Local, no API key needed
2. **Gemini** - Free tier available
3. **Claude** - Requires API key
4. **GPT-4** - Requires API key

**Location**: `/root/Qallow/.continue/config.json`

## Recommended Setup

### For Immediate Use (No Setup):
1. Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
2. Start Ollama: `ollama serve`
3. Pull a model: `ollama pull llama2`
4. Restart Continue IDE
5. Done! ✓

### For Best Performance:
1. Use Gemini (free tier) or Claude
2. Get API key from their website
3. Set environment variable
4. Restart Continue IDE

## Troubleshooting

### "Connection refused" error
- Make sure Ollama is running: `ollama serve`
- Check it's on port 11434: `curl http://localhost:11434`

### "Model not found" error
- Pull the model: `ollama pull llama2`
- List available models: `ollama list`

### Still getting GEMINI_API_KEY error
1. Make sure you've restarted Continue IDE completely
2. Check the config file: `/root/Qallow/.continue/config.json`
3. Try switching to Ollama (no API key needed)

### Ollama is slow
- Try a faster model: `ollama pull mistral`
- Or use Gemini/Claude (cloud-based, faster)

## Quick Commands

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Pull a model
ollama pull llama2

# List models
ollama list

# Remove a model
ollama rm llama2

# Check Ollama is running
curl http://localhost:11434
```

## Model Recommendations

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| llama2 | Medium | Good | General purpose |
| mistral | Fast | Good | Coding, fast responses |
| neural-chat | Medium | Good | Conversations |
| codellama | Medium | Excellent | Code generation |
| dolphin-mixtral | Slow | Excellent | Complex tasks |

## Next Steps

1. **Choose your option** (Ollama recommended for immediate use)
2. **Follow the setup steps** for your chosen option
3. **Restart Continue IDE**
4. **Start coding!**

## Support

- **Ollama Docs**: https://ollama.ai
- **Continue IDE Docs**: https://docs.continue.dev/
- **Gemini API**: https://ai.google.dev/
- **Claude API**: https://docs.anthropic.com/
- **OpenAI API**: https://platform.openai.com/docs/

---

**Status**: ✅ Configuration updated
**Default Model**: Ollama (local, no API key)
**Last Updated**: 2025-10-24

