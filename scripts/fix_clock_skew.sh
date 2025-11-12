#!/bin/bash
# Fix clock skew warnings in build system
# Clock skew occurs when system clock is out of sync or files have future timestamps

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================"
echo "Qallow Clock Skew Fix"
echo "================================"
echo ""

# Check if running with sudo for NTP sync
if [ "$1" = "--sync-ntp" ]; then
    echo "[1/2] Syncing system clock with NTP..."
    if command -v ntpdate &> /dev/null; then
        sudo ntpdate pool.ntp.org || echo "⚠️  ntpdate failed (may require internet)"
    elif command -v timedatectl &> /dev/null; then
        sudo timedatectl set-ntp true || echo "⚠️  timedatectl failed"
    else
        echo "⚠️  No NTP sync tool found. Install ntp or use systemd-timesyncd"
    fi
    echo ""
fi

echo "[2/2] Touching all build artifacts to current time..."
echo "This resolves 'Clock skew detected' warnings"
echo ""

# Find and touch all files in build directory
if [ -d "$PROJECT_ROOT/build" ]; then
    echo "Updating timestamps in build directory..."
    find "$PROJECT_ROOT/build" -type f -exec touch {} \; 2>/dev/null || true
    echo "✅ Build directory timestamps updated"
else
    echo "ℹ️  No build directory found (will be created on next build)"
fi

# Touch CMake cache
if [ -f "$PROJECT_ROOT/CMakeCache.txt" ]; then
    touch "$PROJECT_ROOT/CMakeCache.txt"
    echo "✅ CMakeCache.txt updated"
fi

# Touch CMakeLists.txt and other source files
echo "Updating source file timestamps..."
find "$PROJECT_ROOT" -maxdepth 1 -name "CMakeLists.txt" -exec touch {} \;
find "$PROJECT_ROOT" -name "*.c" -o -name "*.cpp" -o -name "*.cu" -o -name "*.h" | xargs touch 2>/dev/null || true

echo ""
echo "================================"
echo "✅ Clock skew fix complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "  1. Run: cmake --build build"
echo "  2. If warnings persist, try: $0 --sync-ntp"
echo ""

