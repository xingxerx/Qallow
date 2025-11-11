#!/bin/bash
# stop_mock_server.sh
# Finds and stops the mock_roblox_server.py process.

echo "Searching for mock_roblox_server.py process..."
PID=$(ps aux | grep '[m]ock_roblox_server.py' | awk '{print $2}')

if [ -z "$PID" ]; then
  echo "Mock server is not running."
else
  echo "Found mock server process with PID: $PID"
  kill $PID
  echo "Mock server process stopped."
fi
