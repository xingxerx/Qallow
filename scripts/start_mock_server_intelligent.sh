#!/bin/bash
# start_mock_server_intelligent.sh
# Checks if the mock server is running on port 8745 and starts it if not.

PORT=8745
echo "Checking for process on port $PORT..."

# Use lsof to check for a process listening on the specified TCP port.
# The -t option outputs only the PID.
PID=$(lsof -t -i:TCP:$PORT -sTCP:LISTEN)

if [ -z "$PID" ]; then
  echo "No process found on port $PORT."
  echo "Starting mock_roblox_server.py..."
  # Activate virtual environment and start the server in the background
  source .venv/bin/activate
  nohup python3 scripts/mock_roblox_server.py > mock_server.log 2>&1 &
  echo "Mock server started in the background. Logs are in mock_server.log."
else
  echo "Process with PID $PID is already running on port $PORT."
  echo "No action needed."
fi
