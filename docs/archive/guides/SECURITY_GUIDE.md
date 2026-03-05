# Security Guide - API Keys & Secrets

## ⚠️ Critical: Never Commit API Keys to Git

API keys are **credentials** - treat them like passwords. If exposed, anyone can use your API quota and potentially incur charges.

## Quick Setup

### 1. Create Your `.env` File

```bash
# Copy the example file
cp .env.example .env

# Edit it with your actual API keys
nano .env
```

### 2. Add Your API Keys

Edit `.env` and add your actual keys:

```bash
# Google Gemini
GEMINI_API_KEY=your-actual-key-here

# Or Anthropic Claude
ANTHROPIC_API_KEY=your-actual-key-here

# Or OpenAI
OPENAI_API_KEY=your-actual-key-here
```

### 3. Load the Environment Variables

```bash
# Load in current session
source .env

# Or add to ~/.bashrc for permanent setup
echo "source /root/Qallow/.env" >> ~/.bashrc
source ~/.bashrc
```

### 4. Verify It's Protected

```bash
# Check that .env is in .gitignore
grep "^\.env$" .gitignore

# Verify .env won't be committed
git status .env  # Should show "not tracked"
```

## Security Best Practices

### ✓ DO

- ✅ Use environment variables for API keys
- ✅ Keep `.env` in `.gitignore`
- ✅ Rotate API keys regularly
- ✅ Use key restrictions (IP whitelist, API limits)
- ✅ Monitor API usage for unusual activity
- ✅ Use different keys for different projects
- ✅ Store keys in secure password managers
- ✅ Review git history before pushing

### ✗ DON'T

- ❌ Commit `.env` to git
- ❌ Hardcode API keys in source code
- ❌ Share API keys in chat, email, or Slack
- ❌ Use the same key across multiple projects
- ❌ Leave API keys in shell history
- ❌ Store keys in comments or documentation
- ❌ Push keys to public repositories
- ❌ Use keys in logs or error messages

## Files Protected by `.gitignore`

```
.env                    # Environment variables
.env.local              # Local overrides
.env.*.local            # Environment-specific
.continue/config.json   # Continue IDE config
claude_desktop_config.json
*_api_key               # Any file with API key
*_secret                # Any file with secrets
*.key, *.pem            # Cryptographic keys
credentials.json        # Service account credentials
```

## If You Accidentally Expose an API Key

### Immediate Actions

1. **Revoke the key immediately:**
   - Go to your provider's console
   - Find and delete the exposed key
   - Create a new key

2. **Update your `.env` file:**
   ```bash
   # Edit .env with new key
   nano .env
   
   # Reload
   source .env
   ```

3. **Check git history:**
   ```bash
   # Search for the exposed key
   git log -p | grep "AIzaSy"
   
   # If found, you may need to rewrite history
   # (Advanced - contact your team lead)
   ```

4. **Monitor for abuse:**
   - Check your API usage dashboard
   - Look for unusual activity
   - Set up billing alerts

## Setting Up API Keys

### Google Gemini (Recommended - Free Tier)

1. Go to: https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key
4. Add to `.env`:
   ```bash
   GEMINI_API_KEY=your-key-here
   ```

### Anthropic Claude

1. Go to: https://console.anthropic.com/
2. Create an API key
3. Copy the key
4. Add to `.env`:
   ```bash
   ANTHROPIC_API_KEY=your-key-here
   ```

### OpenAI

1. Go to: https://platform.openai.com/api-keys
2. Create an API key
3. Copy the key
4. Add to `.env`:
   ```bash
   OPENAI_API_KEY=your-key-here
   ```

### Ollama (Local - No Key Needed!)

```bash
# Install
curl -fsSL https://ollama.ai/install.sh | sh

# Start
ollama serve

# Download model
ollama pull llama2

# No API key needed!
```

## Checking for Exposed Keys

### Before Committing

```bash
# Check for common patterns
git diff --cached | grep -i "api_key\|apikey\|secret\|token"

# Check for specific patterns
git diff --cached | grep -E "AIzaSy|sk-|AKIA"
```

### In Git History

```bash
# Search entire history
git log -p | grep -i "api_key"

# Search for specific key
git log -p | grep "AIzaSyDtYCqQj1XmFe1QsmGP8Yf"
```

## Environment Variable Usage

### In Continue IDE

```json
{
  "models": [
    {
      "title": "Gemini",
      "provider": "gemini",
      "model": "gemini-2.0-flash",
      "apiKey": "${GEMINI_API_KEY}"
    }
  ]
}
```

### In Python

```python
import os

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set")
```

### In Bash

```bash
# Load from .env
source .env

# Use in script
echo "Using API key: ${GEMINI_API_KEY:0:10}..."
```

## Git Configuration

### Prevent Accidental Commits

```bash
# Add pre-commit hook to check for secrets
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
if git diff --cached | grep -E "AIzaSy|sk-|AKIA|api_key|apiKey"; then
    echo "ERROR: Potential API key detected in staged changes!"
    echo "Please remove the key and try again."
    exit 1
fi
EOF

chmod +x .git/hooks/pre-commit
```

## Monitoring & Alerts

### Set Up Billing Alerts

- **Google Cloud**: https://console.cloud.google.com/billing
- **Anthropic**: https://console.anthropic.com/account/billing
- **OpenAI**: https://platform.openai.com/account/billing/overview

### Monitor API Usage

```bash
# Check recent API calls
# (Provider-specific - see your console)

# Set up alerts for unusual activity
# (Provider-specific - see your console)
```

## Summary

| Action | Status |
|--------|--------|
| `.env` in `.gitignore` | ✅ Protected |
| `.env.example` created | ✅ Safe template |
| API keys in environment variables | ✅ Recommended |
| Pre-commit hooks | ⚠️ Optional but recommended |
| Monitoring setup | ⚠️ Recommended |

## Quick Commands

```bash
# Setup
cp .env.example .env
nano .env
source .env

# Verify
echo $GEMINI_API_KEY
git status .env

# Check for leaks
git log -p | grep -i "api_key"
```

---

**Remember**: API keys are credentials. Treat them like passwords!

**Last Updated**: 2025-10-24

