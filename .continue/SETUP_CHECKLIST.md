# Continue.dev Setup Checklist

Use this checklist to ensure Continue.dev is properly set up for Qallow.

## Pre-Setup

- [ ] VS Code is installed
- [ ] Continue extension is installed in VS Code
- [ ] You're in the `/root/Qallow` directory
- [ ] You have internet access (for API keys)

## Automated Setup

- [ ] Run: `make setup-continue`
- [ ] Choose your AI provider:
  - [ ] Ollama (local, free)
  - [ ] Gemini (free tier)
  - [ ] Claude (paid)
  - [ ] GPT-4 (paid)
- [ ] Setup script completed successfully

## Manual Configuration (if needed)

- [ ] Edit `.env` file: `nano .env`
- [ ] Add API key for your chosen provider:
  - [ ] `GEMINI_API_KEY=your-key`
  - [ ] `ANTHROPIC_API_KEY=your-key`
  - [ ] `OPENAI_API_KEY=your-key`
  - [ ] `OLLAMA_API_BASE=http://localhost:11434`
- [ ] Save and exit

## Environment Setup

- [ ] Load environment: `source .env`
- [ ] Verify: `echo $GEMINI_API_KEY` (or your provider)
- [ ] Should show your API key (or be empty for Ollama)

## Verification

- [ ] Run: `make verify-continue`
- [ ] All checks pass (or only warnings)
- [ ] No critical failures

## Provider-Specific Setup

### If Using Ollama
- [ ] Install Ollama: https://ollama.ai
- [ ] Start Ollama: `ollama serve`
- [ ] Pull a model: `ollama pull llama2`
- [ ] Verify: `curl http://localhost:11434/api/tags`

### If Using Gemini
- [ ] Get API key: https://aistudio.google.com/app/apikey
- [ ] Add to `.env`: `GEMINI_API_KEY=your-key`
- [ ] Load: `source .env`
- [ ] Verify: `echo $GEMINI_API_KEY`

### If Using Claude
- [ ] Get API key: https://console.anthropic.com/
- [ ] Add to `.env`: `ANTHROPIC_API_KEY=your-key`
- [ ] Load: `source .env`
- [ ] Verify: `echo $ANTHROPIC_API_KEY`

### If Using GPT-4
- [ ] Get API key: https://platform.openai.com/api-keys
- [ ] Add to `.env`: `OPENAI_API_KEY=your-key`
- [ ] Load: `source .env`
- [ ] Verify: `echo $OPENAI_API_KEY`

## VS Code Setup

- [ ] Close VS Code completely
- [ ] Reopen VS Code
- [ ] Continue extension loads without errors
- [ ] Check VS Code Output panel for errors

## First Use

- [ ] Press Ctrl+L to open chat
- [ ] Type a test message: "Hello, what can you do?"
- [ ] Continue responds with capabilities
- [ ] Chat works correctly

## Code Editing

- [ ] Highlight some code
- [ ] Press Ctrl+K
- [ ] Ask for a change: "Add error handling"
- [ ] Continue suggests edits
- [ ] Code editing works correctly

## Advanced Features

- [ ] Try `@codebase` mention in chat
- [ ] Try `@file` mention for specific files
- [ ] Try `@memory` for persistent memory
- [ ] Context features work correctly

## Security Check

- [ ] `.env` file exists
- [ ] `.env` is NOT committed to git
- [ ] `.gitignore` includes `.env`
- [ ] No API keys in config files
- [ ] No API keys in documentation
- [ ] Security is properly configured

## Final Verification

- [ ] Run: `make verify-continue`
- [ ] All checks pass
- [ ] No critical errors
- [ ] Setup is complete

## Troubleshooting

If something doesn't work:

1. [ ] Check the error message in VS Code Output panel
2. [ ] Run `make verify-continue` to diagnose
3. [ ] Review `CONTINUE_SETUP_GUIDE.md` troubleshooting section
4. [ ] Check Continue documentation: https://docs.continue.dev/
5. [ ] Restart VS Code and try again

## Common Issues

### "Model not found"
- [ ] Run: `source .env`
- [ ] Restart VS Code
- [ ] Check API key is correct

### Ollama not connecting
- [ ] Run: `ollama serve`
- [ ] Check: `curl http://localhost:11434/api/tags`
- [ ] Verify Ollama is running

### API key not working
- [ ] Verify key is correct
- [ ] Check it hasn't expired
- [ ] Create a new one
- [ ] Restart VS Code

## Documentation

- [ ] Read `CONTINUE_SETUP_GUIDE.md` for full details
- [ ] Read `.continue/QUICK_START.md` for quick reference
- [ ] Bookmark Continue docs: https://docs.continue.dev/

## Ready to Use!

Once all items are checked:

✅ Continue.dev is ready to use with Qallow
✅ You can start chatting with AI
✅ You can edit code with AI assistance
✅ You have persistent memory
✅ Your API keys are secure

## Next Steps

1. Open VS Code
2. Press Ctrl+L to start chatting
3. Ask questions about your code
4. Use Ctrl+K to edit code
5. Enjoy AI-assisted development!

---

**Setup Date**: _______________
**Provider Used**: _______________
**Status**: ✅ Complete

For help, see `CONTINUE_SETUP_GUIDE.md`

