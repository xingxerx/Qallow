# MCP Server Configuration Fix for Linux

## Problem
The IDE was trying to connect to an MCP server using `cmd.exe`, which is a Windows-specific command interpreter. This error occurs because the MCP server configuration was pointing to Windows commands instead of Linux-compatible commands.

## Solution

### 1. VS Code Configuration (Recommended)

A proper Linux-compatible MCP configuration has been created at `.vscode/mcp.json`.

To use it in VS Code:

**Option A: Using VS Code Settings UI**
1. Open VS Code Command Palette: `Ctrl+Shift+P`
2. Search for "Preferences: Open Settings (JSON)"
3. Add the following to your user settings:
```json
"[mcp]": {
  "editor.defaultFormatter": "esbenp.prettier-vscode"
},
"mcp.servers": {
  "memory": {
    "command": "python",
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

**Option B: Using Workspace Settings**
The `.vscode/mcp.json` file is already configured. VS Code should automatically detect it.

### 2. Claude Desktop Configuration

If using Claude Desktop, update `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
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

### 3. Start the MCP Server

Run the setup script to start the MCP server:

```bash
cd /root/Qallow
bash setup_mcp_linux.sh
```

Or manually:

```bash
cd /root/Qallow/mcp-memory-service
python -m src.mcp_memory_service.server
```

### 4. Verify Installation

Check if the MCP server is running:

```bash
curl http://localhost:8000/health 2>/dev/null || echo "Server not running"
```

## Key Changes from Windows to Linux

| Aspect | Windows | Linux |
|--------|---------|-------|
| Command | `cmd.exe` | `python` or `bash` |
| Path Separator | `\` | `/` |
| Home Directory | `%USERPROFILE%` | `~` or `$HOME` |
| Config Location | `AppData\Roaming` | `.config` |
| Database Path | `%USERPROFILE%\.mcp_memory` | `~/.local/share/mcp-memory` |

## Troubleshooting

### Error: "Failed to connect to MCP server"
- Ensure Python is installed: `python --version`
- Check if the server is running: `ps aux | grep mcp_memory_service`
- Check logs: `tail -f /tmp/mcp_memory_service.log`

### Error: "Module not found"
- Install dependencies: `cd /root/Qallow/mcp-memory-service && pip install -e .`
- Verify PYTHONPATH: `echo $PYTHONPATH`

### Error: "Permission denied"
- Ensure database directory exists: `mkdir -p ~/.local/share/mcp-memory`
- Fix permissions: `chmod 755 ~/.local/share/mcp-memory`

## Next Steps

1. Restart your IDE (VS Code, Claude Desktop, etc.)
2. The MCP server should now connect successfully
3. You can now use the memory service features

For more information, see:
- MCP Memory Service: `/root/Qallow/mcp-memory-service/README.md`
- IDE Compatibility: `/root/Qallow/mcp-memory-service/docs/ide-compatability.md`

