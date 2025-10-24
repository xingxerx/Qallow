Why this folder exists
---------------------

This `.continue` folder contains configuration snippets used by the Continue/Agent extension and helpers for running local MCP servers during development. When working across Windows and WSL there are two common ways to make the repository's MCP server YAML visible to the Windows-side Continue extension:

1. Copy the YAML into the Windows user's `.continue` folder (simple, robust).
2. Reference the file via the WSL UNC path `\\wsl$\<DistroName>\...` (no copy, single source-of-truth).

How to configure (examples)
---------------------------

A. Windows-local copy (already supported by many setups)

Edit your Windows Continue config (typically `C:\Users\<you>\.continue\config.yaml`) and add an entry that points at the Windows path where the server YAML lives. Example:

```yaml
mcpServers:
  - name: New MCP server
    path: C:\\Users\\there\\.continue\\mcpServers\\new-mcp-server.yaml
```

Make sure the file exists on Windows. From WSL you can copy the repo version into the Windows folder with:

```bash
mkdir -p /mnt/c/Users/there/.continue/mcpServers
cp /root/Qallow/.continue/mcpServers/new-mcp-server.yaml /mnt/c/Users/there/.continue/mcpServers/new-mcp-server.yaml
```

B. UNC (\wsl$) reference — single source without copying

If you prefer not to copy files, point the Windows config at the WSL UNC path. Example (replace `ArchLinux` with your distro name):

```yaml
mcpServers:
  - name: New MCP server
    path: \\\\wsl$\\ArchLinux\\root\\Qallow\\.continue\\mcpServers\\new-mcp-server.yaml
```

You can check your distro name with `wsl -l -v` on Windows.

C. Reopen folder in WSL (recommended when actively developing inside WSL)

If you open the repository directly in VS Code's WSL session (Remote-WSL: Reopen Folder in WSL) the Continue extension will run inside WSL and can access `/root/Qallow/.continue/mcpServers/new-mcp-server.yaml` directly — no Windows-side configuration is needed.

Validation and troubleshooting
-----------------------------

- Ensure the path in `C:\Users\<you>\.continue\config.yaml` points to an existing file (use Explorer to confirm or `ls` from WSL under `/mnt/c/Users/<you>/.continue`).
- If Continue still errors with a `vscode-remote:` URI: reload VS Code (Developer: Reload Window) after saving `config.yaml` so the extension re-reads the file.
- If referencing `\\wsl$` and permissions or access are blocked, try the Windows-local copy approach instead.

Contact
-------
If other contributors hit this, keep this README up-to-date or add a small shell helper to copy the file automatically from the repo into the Windows `.continue` folder.
