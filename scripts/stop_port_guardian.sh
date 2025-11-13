#!/bin/bash
# Stop Port Guardian service

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/../data/logs/port_guardian.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Port Guardian is not running (no PID file found)"
    exit 0
fi

PID=$(cat "$PID_FILE")
if ps -p "$PID" > /dev/null 2>&1; then
    echo "Stopping Port Guardian (PID: $PID)..."
    kill "$PID"
    rm -f "$PID_FILE"
    echo "Port Guardian stopped"
else
    echo "Port Guardian is not running (stale PID file)"
    rm -f "$PID_FILE"
fi
