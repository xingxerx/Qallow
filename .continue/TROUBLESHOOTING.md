# Continue.dev Troubleshooting Guide

## Common Issues & Solutions

### Issue 1: "GEMINI_API_KEY not found"

**Symptoms**:
- Error message about missing GEMINI_API_KEY
- Continue won't load

**Solution**:
1. Close VS Code completely
2. Reopen VS Code
3. Try again

**Why**: VS Code needs to reload the configuration file.

---

### Issue 2: "Model not found"

**Symptoms**:
- Error about model not being available
- Chat doesn't work

**Solution**:
1. Check your API key is correct
2. Verify it hasn't expired
3. Try a different model:
   - Select "Ollama Local" (no API key needed)
   - Or select "Gemini 2.0 Flash"

---

### Issue 3: Ollama not connecting

**Symptoms**:
- "Failed to connect to Ollama"
- Ollama model won't work

**Solution**:
```bash
# Make sure Ollama is running
ollama serve

# In another terminal, verify it's working
curl http://localhost:11434/api/tags

# If that works, restart VS Code
```

---

### Issue 4: Continue not responding

**Symptoms**:
- Chat interface opens but no response
- Spinning wheel that never stops

**Solution**:
1. Check VS Code Output panel:
   - View → Output
   - Select "Continue" from dropdown
   - Look for error messages

2. Try a different model:
   - Click model selector
   - Choose "Ollama Local" or "Gemini 2.0 Flash"

3. Restart VS Code

---

### Issue 5: "Failed to connect to MCP server"

**Symptoms**:
- Error about MCP server
- cmd.exe not found

**Solution**:
This has been fixed in the latest configuration.
- Restart VS Code
- The error should be gone

---

### Issue 6: Code editing (Ctrl+K) not working

**Symptoms**:
- Ctrl+K opens something else
- Code editing doesn't work

**Solution**:
1. Make sure you have code highlighted
2. Press Ctrl+K
3. Type your request
4. Continue should suggest edits

**Note**: Some VS Code extensions may override Ctrl+K. Try:
- Disabling other extensions
- Or use the Continue menu instead

---

### Issue 7: Chat (Ctrl+L) not working

**Symptoms**:
- Ctrl+L opens something else
- Chat doesn't open

**Solution**:
1. Check VS Code keybindings:
   - File → Preferences → Keyboard Shortcuts
   - Search for "Continue"
   - Make sure Ctrl+L is mapped to "Continue: Open Chat"

2. If not, manually set it:
   - Click the pencil icon next to the command
   - Press Ctrl+L
   - Press Enter

---

### Issue 8: API key not working

**Symptoms**:
- "Invalid API key"
- "Unauthorized"
- "Authentication failed"

**Solution**:
1. Verify your API key is correct:
   - Copy it again from the provider
   - Make sure there are no extra spaces

2. Check it hasn't expired:
   - Log into your provider account
   - Create a new API key if needed

3. Update the config:
   - Edit `.continue/config.json`
   - Replace the API key
   - Restart VS Code

---

### Issue 9: Slow responses

**Symptoms**:
- Continue takes a long time to respond
- Chat is very slow

**Solution**:
1. Try a faster model:
   - Gemini 2.0 Flash (very fast)
   - Ollama Local (fast, local)

2. Check your internet connection:
   - If using cloud models, verify internet is working

3. Check your computer:
   - Close other applications
   - Free up RAM

---

### Issue 10: "No such file or directory"

**Symptoms**:
- Error about missing files
- Configuration errors

**Solution**:
1. Verify the config file exists:
   ```bash
   ls -la /root/Qallow/.continue/config.json
   ```

2. Verify it's valid JSON:
   ```bash
   python3 -m json.tool /root/Qallow/.continue/config.json
   ```

3. If there are errors, restore from backup:
   ```bash
   # The config should be correct now
   # If not, contact support
   ```

---

## Quick Fixes

### Restart Everything
```bash
# 1. Close VS Code
# 2. Reopen VS Code
# 3. Try again
```

### Reset to Defaults
```bash
# The config is already set up correctly
# Just restart VS Code
```

### Check Configuration
```bash
# Verify the config is valid
python3 -m json.tool /root/Qallow/.continue/config.json
```

### Check Logs
```bash
# In VS Code:
# View → Output
# Select "Continue" from dropdown
# Look for error messages
```

---

## Getting Help

1. **Check this guide** - Most issues are covered above
2. **Check VS Code Output** - View → Output → Continue
3. **Restart VS Code** - Close and reopen
4. **Try Ollama** - It's local and doesn't need API keys
5. **Check Continue Docs** - https://docs.continue.dev/

---

## Still Having Issues?

1. Check the error message carefully
2. Look for it in this guide
3. Try the suggested solution
4. Restart VS Code
5. Try again

If you're still stuck:
- Check Continue documentation: https://docs.continue.dev/
- Review the configuration: `.continue/config.json`
- Try a different model

---

**Last Updated**: 2025-10-24
**Status**: ✅ All Known Issues Fixed

