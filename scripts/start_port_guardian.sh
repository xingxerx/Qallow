#!/bin/bash
# Start Port Guardian as a background service

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/../data/logs/port_guardian.pid"

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Port Guardian is already running (PID: $PID)"
        exit 0
    else
        echo "Removing stale PID file"
        rm -f "$PID_FILE"
    fi
fi

# Start in background
echo "Starting Port Guardian..."
nohup python3 "$SCRIPT_DIR/port_guardian.py" > /dev/null 2>&1 &
echo $! > "$PID_FILE"

echo "Port Guardian started (PID: $(cat $PID_FILE))"
echo "Logs: $SCRIPT_DIR/../data/logs/port_guardian.log"
echo ""
echo "To stop: kill $(cat $PID_FILE)"
