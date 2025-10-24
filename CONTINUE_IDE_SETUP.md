# Continue IDE Setup Guide

## Problem

You were seeing this error:
```
Model/deployment not found for: continuedev/default-assistant/gemini/gemini-2.5-pro
We couldn't find a secret with the name "GEMINI_API_KEY"
```

This error occurs because:
1. The Continue IDE is trying to use a model that's not configured
2. The API key is not set up properly

## Solution

### Step 1: Get a Google Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key

### Step 2: Set the Environment Variable

Add your API key to your shell profile:

**For Bash** (add to `~/.bashrc`):
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**For Zsh** (add to `~/.zshrc`):
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**For Fish** (add to `~/.config/fish/config.fish`):
```bash
set -gx GEMINI_API_KEY "your-api-key-here"
```

Then reload your shell:
```bash
source ~/.bashrc  # or ~/.zshrc or reload fish
```

### Step 3: Verify the Configuration

The Continue IDE configuration at `/root/Qallow/.continue/config.json` is already set up to use environment variables:

```json
{
  "models": [
    {
      "title": "Gemini 2.0 Flash",
      "provider": "gemini",
      "model": "gemini-2.0-flash",
      "apiKey": "${GEMINI_API_KEY}"
    }
  ]
}
```

### Step 4: Restart Continue IDE

1. Close Continue IDE completely
2. Reopen it
3. The error should be gone

## Alternative: Use a Different Model

If you don't have a Gemini API key, you can use other models:

### Claude (Anthropic)
```json
{
  "title": "Claude 3.5 Sonnet",
  "provider": "anthropic",
  "model": "claude-3-5-sonnet-20241022",
  "apiKey": "${ANTHROPIC_API_KEY}"
}
```

Get API key: https://console.anthropic.com/

### OpenAI
```json
{
  "title": "GPT-4",
  "provider": "openai",
  "model": "gpt-4",
  "apiKey": "${OPENAI_API_KEY}"
}
```

Get API key: https://platform.openai.com/api-keys

### Ollama (Local)
```json
{
  "title": "Ollama Local",
  "provider": "ollama",
  "model": "llama2"
}
```

No API key needed - runs locally!

## Configuration File

**Location**: `/root/Qallow/.continue/config.json`

**Current Configuration**:
- Gemini 2.0 Flash (recommended)
- Gemini 1.5 Pro
- Gemini 1.5 Flash

All use the `${GEMINI_API_KEY}` environment variable.

## Troubleshooting

### Error: "GEMINI_API_KEY not found"
- Make sure you've set the environment variable
- Restart your terminal/IDE
- Check: `echo $GEMINI_API_KEY` (should show your key)

### Error: "Invalid API key"
- Verify your API key is correct
- Check it hasn't expired
- Try creating a new one at https://aistudio.google.com/app/apikey

### Error: "Model not found"
- Make sure the model name is correct
- Check your API key has access to that model
- Try a different model (e.g., gemini-2.0-flash)

### Continue IDE not picking up the environment variable
- Restart Continue IDE completely
- Make sure you've reloaded your shell after setting the variable
- Try setting it directly in the config file (not recommended for security)

## Security Best Practices

⚠️ **IMPORTANT**: Never commit API keys to version control!

1. **Use environment variables** (recommended)
   - Set in `~/.bashrc`, `~/.zshrc`, etc.
   - Not committed to git

2. **Use `.env` files** (with `.gitignore`)
   - Create `.env` file with your keys
   - Add `.env` to `.gitignore`
   - Load with: `source .env`

3. **Use secret management tools**
   - 1Password, LastPass, etc.
   - Inject at runtime

## Quick Setup Script

```bash
#!/bin/bash

# Set your API key here
read -p "Enter your Gemini API key: " API_KEY

# Add to bashrc
echo "export GEMINI_API_KEY='$API_KEY'" >> ~/.bashrc

# Reload
source ~/.bashrc

echo "✓ API key configured!"
echo "✓ Restart Continue IDE to apply changes"
```

## Supported Models

### Gemini
- `gemini-2.0-flash` (recommended - fastest)
- `gemini-1.5-pro` (most capable)
- `gemini-1.5-flash` (balanced)

### Claude
- `claude-3-5-sonnet-20241022` (recommended)
- `claude-3-opus-20240229`
- `claude-3-haiku-20240307`

### OpenAI
- `gpt-4-turbo`
- `gpt-4`
- `gpt-3.5-turbo`

### Ollama (Local)
- `llama2`
- `mistral`
- `neural-chat`
- Any model you've pulled

## Next Steps

1. ✅ Get an API key (Gemini, Claude, or OpenAI)
2. ✅ Set the environment variable
3. ✅ Restart Continue IDE
4. ✅ Start using it!

## Resources

- **Continue IDE Docs**: https://docs.continue.dev/
- **Gemini API**: https://ai.google.dev/
- **Claude API**: https://docs.anthropic.com/
- **OpenAI API**: https://platform.openai.com/docs/

---

**Status**: ✅ Configuration updated
**Last Updated**: 2025-10-24

