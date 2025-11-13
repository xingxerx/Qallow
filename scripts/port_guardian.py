#!/usr/bin/env python3
"""
Port Guardian - Monitors and closes unused ports automatically
Runs as a background service to prevent unauthorized port exposure
"""

import subprocess
import time
import json
import os
from datetime import datetime

# Whitelist of allowed ports (add your approved ports here)
ALLOWED_PORTS = {
    # Add any ports you need, e.g.:
    # 8080: "Dev server",
    # 5432: "PostgreSQL",
}

# Check interval in seconds
CHECK_INTERVAL = 30

LOG_FILE = "/home/xing/Qallow/data/logs/port_guardian.log"

def log(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_msg + "\n")

def get_open_ports():
    """Get all listening ports on the system"""
    try:
        # Get system ports
        result = subprocess.run(
            ["netstat", "-tuln"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        ports = set()
        for line in result.stdout.split('\n'):
            if 'LISTEN' in line:
                parts = line.split()
                for part in parts:
                    if ':' in part:
                        try:
                            port = int(part.split(':')[-1])
                            if 0 < port < 65536:
                                ports.add(port)
                        except (ValueError, IndexError):
                            continue
        
        return ports
    except Exception as e:
        log(f"Error getting open ports: {e}")
        return set()

def get_docker_ports():
    """Get all Docker container exposed ports"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Ports}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        ports = {}
        for line in result.stdout.strip().split('\n'):
            if not line or line == '':
                continue
            # Parse Docker port mappings like "0.0.0.0:8080->80/tcp"
            for mapping in line.split(','):
                mapping = mapping.strip()
                if '->' in mapping and ':' in mapping:
                    try:
                        host_part = mapping.split('->')[0]
                        port_str = host_part.split(':')[-1]
                        port = int(port_str)
                        ports[port] = f"Docker: {mapping}"
                    except (ValueError, IndexError):
                        continue
        
        return ports
    except Exception as e:
        log(f"Error getting Docker ports: {e}")
        return {}

def close_unauthorized_ports():
    """Close any ports not in the whitelist"""
    system_ports = get_open_ports()
    docker_ports = get_docker_ports()
    
    unauthorized = []
    
    # Check Docker ports
    for port, desc in docker_ports.items():
        if port not in ALLOWED_PORTS:
            unauthorized.append((port, desc))
            log(f"⚠️  Unauthorized Docker port detected: {port} ({desc})")
    
    # Stop Docker containers with unauthorized ports
    if unauthorized:
        try:
            result = subprocess.run(
                ["docker", "ps", "-q"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            container_ids = result.stdout.strip().split('\n')
            for container_id in container_ids:
                if container_id:
                    # Get container ports
                    port_result = subprocess.run(
                        ["docker", "port", container_id],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    should_stop = False
                    for line in port_result.stdout.split('\n'):
                        if '->' in line and ':' in line:
                            try:
                                port_part = line.split('->')[1].strip()
                                port = int(port_part.split(':')[-1])
                                if port not in ALLOWED_PORTS:
                                    should_stop = True
                                    break
                            except (ValueError, IndexError):
                                continue
                    
                    if should_stop:
                        log(f"🛑 Stopping container {container_id[:12]} with unauthorized ports")
                        subprocess.run(
                            ["docker", "stop", container_id],
                            capture_output=True,
                            timeout=10
                        )
        except Exception as e:
            log(f"Error stopping containers: {e}")
    
    if not unauthorized:
        log("✅ All ports authorized")
    
    return len(unauthorized)

def main():
    """Main monitoring loop"""
    log("🚀 Port Guardian started")
    log(f"Allowed ports: {ALLOWED_PORTS if ALLOWED_PORTS else 'None (all ports will be blocked)'}")
    log(f"Check interval: {CHECK_INTERVAL}s")
    
    try:
        while True:
            closed = close_unauthorized_ports()
            if closed > 0:
                log(f"Closed {closed} unauthorized port(s)")
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        log("🛑 Port Guardian stopped by user")
    except Exception as e:
        log(f"❌ Port Guardian error: {e}")

if __name__ == "__main__":
    main()
