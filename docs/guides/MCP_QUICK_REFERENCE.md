# MCP Server Quick Reference

## Quick Start

### 1. Run Setup (First Time)
```bash
cd /root/Qallow
bash setup_mcp_linux.sh
```

### 2. Restart IDE
- **VS Code**: `Ctrl+Shift+P` → "Developer: Reload Window"
- **Claude Desktop**: Restart application
- **Other IDEs**: Restart normally

### 3. Done!
The MCP server should now connect automatically.

## Manual Server Control

### Start Server
```bash
cd /root/Qallow/mcp-memory-service
./.venv/bin/python -m src.mcp_memory_service.server
```

### Stop Server
```bash
# Find the process
ps aux | grep mcp_memory_service

# Kill it
kill -9 <PID>
```

### Check Server Status
```bash
# Is it running?
ps aux | grep mcp_memory_service

# Is it responding?
curl http://localhost:8000/health

# Check logs
tail -f /tmp/mcp_memory_service.log
```

## Configuration Files

### VS Code MCP Config
**Location**: `/root/Qallow/.vscode/mcp.json`

**To use in VS Code**:
1. Open Command Palette: `Ctrl+Shift+P`
2. Search: "Preferences: Open Settings (JSON)"
3. Add to user settings:
```json
"mcp.servers": {
  "memory": {
    "command": "/root/Qallow/mcp-memory-service/.venv/bin/python",
    "args": ["-m", "src.mcp_memory_service.server"],
    "cwd": "/root/Qallow/mcp-memory-service",
    "env": {
      "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
      "MCP_MEMORY_SQLITE_VEC_PATH": "/root/.local/share/mcp-memory",
      "PYTHONPATH": "/root/Qallow/mcp-memory-service/src",
      "LOG_LEVEL": "INFO"
    }
  }
}
```

### Claude Desktop Config
**Location**: `~/.config/claude/claude_desktop_config.json`

**Configuration**:
```json
{
  "mcpServers": {
    "memory": {
      "command": "/root/Qallow/mcp-memory-service/.venv/bin/python",
      "args": ["-m", "src.mcp_memory_service.server"],
      "cwd": "/root/Qallow/mcp-memory-service",
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "MCP_MEMORY_SQLITE_VEC_PATH": "/root/.local/share/mcp-memory",
        "PYTHONPATH": "/root/Qallow/mcp-memory-service/src",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

## Common Issues & Fixes

### Error: "Failed to connect to MCP server"
```bash
# 1. Check if server is running
ps aux | grep mcp_memory_service

# 2. If not running, start it
cd /root/Qallow/mcp-memory-service
./.venv/bin/python -m src.mcp_memory_service.server

# 3. Restart IDE
```

### Error: "Module not found: mcp"
```bash
# Reinstall dependencies
cd /root/Qallow/mcp-memory-service
./.venv/bin/pip install mcp sqlite-vec sentence-transformers
```

### Error: "Permission denied"
```bash
# Create and fix permissions
mkdir -p ~/.local/share/mcp-memory
chmod 755 ~/.local/share/mcp-memory
```

### Error: "Port 8000 already in use"
```bash
# Find and kill the process
lsof -i :8000
kill -9 <PID>
```

## Environment Variables

```bash
# Storage backend (use sqlite_vec for best performance)
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec

# Database location
export MCP_MEMORY_SQLITE_VEC_PATH=/root/.local/share/mcp-memory

# Python module path
export PYTHONPATH=/root/Qallow/mcp-memory-service/src

# Logging level (DEBUG, INFO, WARNING, ERROR)
export LOG_LEVEL=INFO
```

## Useful Commands

### Check Python Version
```bash
/root/Qallow/mcp-memory-service/.venv/bin/python --version
```

### List Installed Packages
```bash
/root/Qallow/mcp-memory-service/.venv/bin/pip list
```

### Check Database
```bash
ls -lh ~/.local/share/mcp-memory/
```

### View Server Logs
```bash
tail -f /tmp/mcp_memory_service.log
```

### Test Server Startup
```bash
cd /root/Qallow/mcp-memory-service
timeout 5 ./.venv/bin/python -m src.mcp_memory_service.server || true
```

### Reinstall Everything
```bash
cd /root/Qallow
bash setup_mcp_linux.sh
```

## File Locations

| Item | Location |
|------|----------|
| MCP Service | `/root/Qallow/mcp-memory-service` |
| Virtual Env | `/root/Qallow/mcp-memory-service/.venv` |
| Database | `~/.local/share/mcp-memory` |
| VS Code Config | `/root/Qallow/.vscode/mcp.json` |
| Setup Script | `/root/Qallow/setup_mcp_linux.sh` |
| Logs | `/tmp/mcp_memory_service.log` |
| Documentation | `/root/Qallow/MCP_FIX_GUIDE.md` |

## IDE-Specific Instructions

### VS Code
1. Install MCP extension (if needed)
2. Open `.vscode/mcp.json` or add to settings
3. Reload window: `Ctrl+Shift+P` → "Developer: Reload Window"

### Claude Desktop
1. Edit `~/.config/claude/claude_desktop_config.json`
2. Add memory server configuration
3. Restart Claude Desktop

### Cursor IDE
1. Create `.cursor/mcp.json` in project root
2. Add memory server configuration
3. Restart Cursor

### WindSurf
1. Open MCP configuration
2. Add memory server configuration
3. Restart WindSurf

## Performance Tips

1. **Use SQLite-Vec**: Fastest option for local development
2. **Increase Cache**: Set `MCP_MEMORY_SQLITE_PRAGMAS="cache_size=20000"`
3. **Batch Operations**: Group multiple operations together
4. **Monitor Logs**: Check `/tmp/mcp_memory_service.log` for bottlenecks

## Getting Help

- **Setup Issues**: See `MCP_FIX_GUIDE.md`
- **Detailed Info**: See `MCP_SETUP_COMPLETE.md`
- **MCP Service Docs**: `/root/Qallow/mcp-memory-service/README.md`
- **IDE Compatibility**: `/root/Qallow/mcp-memory-service/docs/ide-compatability.md`

---

**Last Updated**: 2025-10-24
**Status**: ✅ All systems operational

