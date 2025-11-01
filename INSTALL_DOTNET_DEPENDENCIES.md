# .NET Dependencies Installation Guide

## Current System Status

**OS**: Arch Linux (running in Flatpak environment)  
**Existing .NET Runtime**: 6.0.36 (VSCode extension)  
**Location**: `/home/xing/.var/app/com.visualstudio.code/config/Code/User/globalStorage/ms-dotnettools.vscode-dotnet-runtime/.dotnet/6.0.36~x64/dotnet`

## Dependency Check Results

✅ **No missing libraries detected** - The existing .NET runtime has all required dependencies!

```bash
ldd /home/xing/.var/app/com.visualstudio.code/config/Code/User/globalStorage/ms-dotnettools.vscode-dotnet-runtime/.dotnet/6.0.36~x64/dotnet | grep "not found"
# Output: (empty - all libraries found)
```

## Install .NET SDK on Arch Linux (Host System)

Since you're on Arch Linux, here are the commands to install .NET SDK and dependencies:

### Option 1: Install from Official Arch Repositories

```bash
# Install .NET SDK 8.0 (latest LTS)
sudo pacman -S dotnet-sdk

# Or install specific version
sudo pacman -S dotnet-sdk-8.0

# Install runtime only (if you don't need SDK)
sudo pacman -S dotnet-runtime
```

### Option 2: Install Required Dependencies Only

If you just want to ensure all .NET dependencies are present:

```bash
# Install common .NET dependencies on Arch
sudo pacman -S --needed icu zlib openssl curl krb5 lttng-ust
```

### Option 3: Install from Microsoft Repository (Alternative)

```bash
# Download Microsoft package repository
wget https://packages.microsoft.com/config/arch/packages-microsoft-prod.pkg.tar.zst

# Install the repository package
sudo pacman -U packages-microsoft-prod.pkg.tar.zst

# Update package database
sudo pacman -Sy

# Install .NET SDK
sudo pacman -S dotnet-sdk-8.0
```

## Verify Installation

After installation, verify .NET is working:

```bash
# Check dotnet version
dotnet --info

# Check for missing libraries
ldd $(which dotnet) | grep "not found"

# Should output nothing if all dependencies are satisfied
```

## For Other Linux Distributions

### Debian/Ubuntu/Pop!_OS/Linux Mint

```bash
# Update package list
sudo apt-get update

# Install .NET dependencies
sudo apt-get install -y \
    libc6 \
    libgcc1 \
    libgssapi-krb5-2 \
    libicu-dev \
    liblttng-ust1 \
    libssl3 \
    libstdc++6 \
    zlib1g \
    libcurl4 \
    libkrb5-3 \
    libbrotli1

# Or install Microsoft's dependency package
wget https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install -y dotnet-runtime-deps-8.0
```

### Fedora/RHEL/CentOS/AlmaLinux/Rocky Linux

```bash
# Install .NET dependencies
sudo dnf install -y \
    compat-openssl10 \
    libicu \
    libuuid \
    krb5-libs \
    zlib \
    libcurl \
    libunwind
```

### Arch-based Distributions (Manjaro, EndeavourOS, etc.)

```bash
# Install .NET dependencies
sudo pacman -S --needed \
    icu \
    zlib \
    openssl \
    curl \
    krb5 \
    lttng-ust
```

## Current Qallow Project Status

The Qallow project currently has:
- ✅ .NET runtime available (6.0.36)
- ✅ All required libraries present
- ✅ No missing dependencies

However, **no .NET SDK is installed**, which means:
- ❌ Cannot compile C# code
- ❌ Cannot run `dotnet build`
- ❌ Cannot create new .NET projects

## Recommended Action

For full .NET development capabilities in Qallow:

```bash
# On your Arch Linux host system, run:
sudo pacman -S dotnet-sdk

# This will install:
# - .NET 8.0 SDK (latest)
# - All required dependencies
# - dotnet CLI tools
```

## Verification Commands

After installing, verify everything works:

```bash
# 1. Check dotnet is in PATH
which dotnet

# 2. Check version and SDKs
dotnet --info

# 3. Check for missing libraries
ldd $(which dotnet) | grep "not found"

# 4. Create a test project
dotnet new console -n TestApp
cd TestApp
dotnet run
```

## Integration with Qallow AGI

Once .NET SDK is installed, you can:

1. **Build C# components** for Qallow
2. **Create .NET agents** that integrate with Agent Lightning
3. **Use C# for high-performance** AGI modules
4. **Interop with C/C++** quantum code via P/Invoke

## Notes

- The existing .NET runtime (6.0.36) is sufficient for **running** .NET applications
- You need the **SDK** to **develop** .NET applications
- All dependencies are already satisfied on your system
- No library installation is required unless you install a newer .NET version

---

**Status**: ✅ Dependencies satisfied  
**Action Required**: Install .NET SDK for development (optional)  
**Command**: `sudo pacman -S dotnet-sdk`

