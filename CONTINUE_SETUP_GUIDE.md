# Continue.dev Setup Guide for Qallow

This guide helps you set up Continue.dev with the Qallow project for AI-assisted development.

## Quick Start (Automated)

```bash
# Make the setup script executable
chmod +x setup_continue.sh

# Run the setup script
./setup_continue.sh
```

The script will guide you through:
1. Creating/updating `.env` file
2. Choosing your AI model provider
3. Configuring API keys (if needed)
4. Setting up MCP servers

## Manual Setup

### Step 1: Install Continue Extension

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X / Cmd+Shift+X)
3. Search for "Continue"
4. Install the official Continue extension

### Step 2: Configure Environment Variables

Edit `.env` file:

```bash
cp .env.example .env
nano .env
```

Choose ONE provider and add your API key:

#### Option A: Ollama (Recommended for Testing)
**No API key needed - runs locally!**

```bash
# In .env:
OLLAMA_API_BASE=http://localhost:11434
```

Then install and run Ollama:
```bash
# Install from https://ollama.ai
ollama serve

# In another terminal, pull a model:
ollama pull llama2
```

#### Option B: Google Gemini (Free Tier Available)

```bash
# Get free API key: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your-api-key-here
```

#### Option C: Anthropic Claude

```bash
# Get API key: https://console.anthropic.com/
ANTHROPIC_API_KEY=your-api-key-here
```

#### Option D: OpenAI GPT-4

```bash
# Get API key: https://platform.openai.com/api-keys
OPENAI_API_KEY=your-api-key-here
```

### Step 3: Load Environment Variables

```bash
source .env
```

Verify it worked:
```bash
echo $GEMINI_API_KEY  # or your chosen provider
```

### Step 4: Restart VS Code

1. Close VS Code completely
2. Reopen it
3. Continue should now be configured

## Configuration Files

### `.continue/config.json`
Main Continue configuration with model definitions and MCP servers.

**Location**: `/root/Qallow/.continue/config.json`

**Models configured**:
- Ollama Local (no API key)
- Gemini 2.0 Flash
- Claude 3.5 Sonnet
- GPT-4

### `.env` / `.env.example`
Environment variables for API keys and configuration.

**Location**: `/root/Qallow/.env` (not committed to git)

**Security**: `.env` is in `.gitignore` - your API keys won't be committed!

### MCP Servers
Model Context Protocol servers for extended functionality.

**Location**: `/root/Qallow/.continue/mcpServers/`

**Available**:
- `qallow-memory.yaml` - Persistent memory service

## Using Continue in VS Code

### Basic Usage

1. **Chat**: Ctrl+L (or Cmd+L on Mac)
   - Ask questions about your code
   - Get explanations and suggestions

2. **Edit**: Ctrl+K (or Cmd+K on Mac)
   - Highlight code and ask for edits
   - Continue will suggest changes

3. **Terminal**: Ctrl+Shift+` (backtick)
   - Ask Continue to run commands
   - Get help with terminal tasks

### With Qallow Context

Use `@` mentions to reference:
- `@codebase` - Search the entire codebase
- `@file` - Reference specific files
- `@memory` - Access persistent memory

Example:
```
@codebase How does the quantum phase 13 implementation work?
```

## Troubleshooting

### Error: "GEMINI_API_KEY not found"

**Solution**:
```bash
# Make sure you've set the environment variable
echo $GEMINI_API_KEY

# If empty, reload your shell
source .env

# Restart VS Code
```

### Error: "Model not found"

**Solution**:
1. Verify your API key is correct
2. Check the model name is spelled correctly
3. Try a different model from the config
4. Restart Continue IDE

### Ollama Connection Failed

**Solution**:
```bash
# Make sure Ollama is running
ollama serve

# In another terminal, verify it's working
curl http://localhost:11434/api/tags
```

### MCP Server Not Loading

**Solution**:
1. Check the YAML file exists: `.continue/mcpServers/qallow-memory.yaml`
2. Verify Python path is correct in the YAML
3. Check logs in VS Code Output panel
4. Restart VS Code

## Security Best Practices

⚠️ **IMPORTANT**: Never commit API keys to git!

✅ **DO**:
- Store API keys in `.env` (which is in `.gitignore`)
- Use environment variables
- Rotate keys regularly
- Use separate keys for development/production

❌ **DON'T**:
- Hardcode API keys in config files
- Commit `.env` to git
- Share API keys in chat/messages
- Use production keys for testing

## Model Comparison

| Model | Speed | Quality | Cost | Setup |
|-------|-------|---------|------|-------|
| Ollama (llama2) | Fast | Good | Free | Local |
| Gemini 2.0 Flash | Very Fast | Excellent | Free tier | API key |
| Claude 3.5 Sonnet | Medium | Excellent | Paid | API key |
| GPT-4 | Medium | Excellent | Paid | API key |

## Advanced Configuration

### Custom Models

Edit `.continue/config.json` to add custom models:

```json
{
  "title": "My Custom Model",
  "provider": "openai",
  "model": "gpt-4-turbo",
  "apiKey": "${OPENAI_API_KEY}"
}
```

### Context Length

Adjust context window in `.continue/config.json`:

```json
{
  "contextLength": 8000,
  "maxTokens": 4096
}
```

### Disable Telemetry

Already configured in `.continue/config.json`:

```json
{
  "allowAnonymousTelemetry": false
}
```

## Resources

- **Continue Docs**: https://docs.continue.dev/
- **Gemini API**: https://ai.google.dev/
- **Claude API**: https://docs.anthropic.com/
- **OpenAI API**: https://platform.openai.com/docs/
- **Ollama**: https://ollama.ai/

## Support

For issues:
1. Check the troubleshooting section above
2. Review Continue documentation: https://docs.continue.dev/
3. Check VS Code Output panel for error messages
4. Restart VS Code and try again

---

**Status**: ✅ Ready to use
**Last Updated**: 2025-10-24

