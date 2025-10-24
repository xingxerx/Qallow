# MCP Server Resources & Documentation

## Quick Navigation

### 🚀 Getting Started
- **Start Here**: `MCP_RESOLUTION_SUMMARY.md` - Overview of the fix
- **Quick Start**: `MCP_QUICK_REFERENCE.md` - Common commands and tasks
- **Setup Guide**: `MCP_SETUP_COMPLETE.md` - Detailed setup information

### 🔧 Setup & Configuration
- **Setup Script**: `setup_mcp_linux.sh` - Automated setup (run this first!)
- **Verification Script**: `verify_mcp_setup.sh` - Verify installation
- **VS Code Config**: `.vscode/mcp.json` - MCP server configuration

### 📚 Documentation
- **Troubleshooting**: `MCP_FIX_GUIDE.md` - Detailed troubleshooting guide
- **This File**: `MCP_RESOURCES.md` - Resource index

### 🔗 External Resources
- **MCP Service**: `mcp-memory-service/README.md` - MCP Memory Service documentation
- **IDE Compatibility**: `mcp-memory-service/docs/ide-compatability.md` - IDE setup guides

## File Locations

### Configuration Files
```
/root/Qallow/.vscode/mcp.json              # VS Code MCP configuration
~/.config/claude/claude_desktop_config.json # Claude Desktop config (if exists)
```

### Scripts
```
/root/Qallow/setup_mcp_linux.sh            # Setup script (executable)
/root/Qallow/verify_mcp_setup.sh           # Verification script (executable)
```

### Documentation
```
/root/Qallow/MCP_RESOLUTION_SUMMARY.md     # Problem & solution overview
/root/Qallow/MCP_SETUP_COMPLETE.md         # Complete setup details
/root/Qallow/MCP_QUICK_REFERENCE.md        # Quick reference guide
/root/Qallow/MCP_FIX_GUIDE.md              # Troubleshooting guide
/root/Qallow/MCP_RESOURCES.md              # This file
```

### MCP Service
```
/root/Qallow/mcp-memory-service/           # MCP Memory Service directory
/root/Qallow/mcp-memory-service/.venv/     # Virtual environment
/root/Qallow/mcp-memory-service/src/       # Source code
```

### Data & Logs
```
~/.local/share/mcp-memory/                 # Database directory
/tmp/mcp_memory_service.log                # Server logs
```

## Common Tasks

### First Time Setup
```bash
cd /root/Qallow
bash setup_mcp_linux.sh
```

### Verify Installation
```bash
bash /root/Qallow/verify_mcp_setup.sh
```

### Start Server Manually
```bash
cd /root/Qallow/mcp-memory-service
./.venv/bin/python -m src.mcp_memory_service.server
```

### Check Server Status
```bash
ps aux | grep mcp_memory_service
curl http://localhost:8000/health
```

### View Logs
```bash
tail -f /tmp/mcp_memory_service.log
```

### Restart IDE
- **VS Code**: `Ctrl+Shift+P` → "Developer: Reload Window"
- **Claude Desktop**: Restart application
- **Other IDEs**: Restart normally

## Documentation by Topic

### Setup & Installation
- `MCP_RESOLUTION_SUMMARY.md` - Overview
- `MCP_SETUP_COMPLETE.md` - Detailed setup
- `setup_mcp_linux.sh` - Automated setup

### Configuration
- `.vscode/mcp.json` - VS Code configuration
- `MCP_QUICK_REFERENCE.md` - Configuration examples
- `mcp-memory-service/docs/ide-compatability.md` - IDE-specific setup

### Troubleshooting
- `MCP_FIX_GUIDE.md` - Troubleshooting guide
- `MCP_QUICK_REFERENCE.md` - Common issues & fixes
- `verify_mcp_setup.sh` - Verification script

### Reference
- `MCP_QUICK_REFERENCE.md` - Quick commands
- `MCP_RESOURCES.md` - This file
- `mcp-memory-service/README.md` - MCP service docs

## IDE-Specific Guides

### VS Code
1. Configuration: `.vscode/mcp.json`
2. Reload: `Ctrl+Shift+P` → "Developer: Reload Window"
3. Verify: Check for MCP server connection

### Claude Desktop
1. Edit: `~/.config/claude/claude_desktop_config.json`
2. Add: Memory server configuration
3. Restart: Close and reopen Claude Desktop

### Cursor IDE
1. Create: `.cursor/mcp.json`
2. Add: Memory server configuration
3. Restart: Close and reopen Cursor

### WindSurf
1. Open: MCP configuration
2. Add: Memory server configuration
3. Restart: Close and reopen WindSurf

## Troubleshooting Quick Links

| Issue | Solution |
|-------|----------|
| "Failed to connect to MCP server" | See `MCP_FIX_GUIDE.md` → "Issue: Failed to connect" |
| "Module not found: mcp" | See `MCP_FIX_GUIDE.md` → "Issue: Module not found" |
| "Permission denied" | See `MCP_FIX_GUIDE.md` → "Issue: Permission denied" |
| "Port 8000 already in use" | See `MCP_FIX_GUIDE.md` → "Issue: Port already in use" |
| Setup failed | Run: `bash /root/Qallow/setup_mcp_linux.sh` |
| Verification failed | Run: `bash /root/Qallow/verify_mcp_setup.sh` |

## Environment Variables

```bash
MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
MCP_MEMORY_SQLITE_VEC_PATH=/root/.local/share/mcp-memory
PYTHONPATH=/root/Qallow/mcp-memory-service/src
LOG_LEVEL=INFO
```

## Key Commands

```bash
# Setup
bash /root/Qallow/setup_mcp_linux.sh

# Verify
bash /root/Qallow/verify_mcp_setup.sh

# Start server
cd /root/Qallow/mcp-memory-service
./.venv/bin/python -m src.mcp_memory_service.server

# Check status
ps aux | grep mcp_memory_service
curl http://localhost:8000/health

# View logs
tail -f /tmp/mcp_memory_service.log

# Stop server
pkill -f mcp_memory_service
```

## Support

### Documentation
- **Quick Start**: `MCP_QUICK_REFERENCE.md`
- **Detailed Guide**: `MCP_FIX_GUIDE.md`
- **Setup Info**: `MCP_SETUP_COMPLETE.md`
- **Overview**: `MCP_RESOLUTION_SUMMARY.md`

### Scripts
- **Setup**: `setup_mcp_linux.sh`
- **Verify**: `verify_mcp_setup.sh`

### External
- **MCP Service**: `mcp-memory-service/README.md`
- **IDE Compatibility**: `mcp-memory-service/docs/ide-compatability.md`

## Status

✅ **All systems operational**

- Python 3.13.7 installed
- Virtual environment configured
- All dependencies installed
- VS Code configuration ready
- Database directory created
- Server tested and working
- Documentation complete

## Next Steps

1. **Restart IDE** - Close and reopen your IDE
2. **Verify Connection** - MCP server should connect automatically
3. **Check Logs** - If issues, check `/tmp/mcp_memory_service.log`
4. **Reference Docs** - Use `MCP_QUICK_REFERENCE.md` for common tasks

---

**Last Updated**: 2025-10-24
**Status**: ✅ Complete
**System**: Linux (Arch-based)

