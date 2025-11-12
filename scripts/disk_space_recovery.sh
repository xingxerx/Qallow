#!/bin/bash
# Disk Space Recovery Script
# Automatically frees up disk space when running low
# Usage: source disk_space_recovery.sh

set -e

# Color codes for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Configuration
CRITICAL_THRESHOLD=1048576  # 1GB in KB
WARNING_THRESHOLD=2097152   # 2GB in KB
SAFE_THRESHOLD=5242880      # 5GB in KB

# Function to get available disk space
get_available_space() {
    df / | awk 'NR==2 {print $4}'
}

# Function to print disk status
print_disk_status() {
    local available=$(get_available_space)
    echo "Available disk space: ${available}KB"
    
    if [ "$available" -lt "$CRITICAL_THRESHOLD" ]; then
        echo -e "${RED}CRITICAL: Less than 1GB available${NC}"
        return 2
    elif [ "$available" -lt "$WARNING_THRESHOLD" ]; then
        echo -e "${YELLOW}WARNING: Less than 2GB available${NC}"
        return 1
    elif [ "$available" -lt "$SAFE_THRESHOLD" ]; then
        echo -e "${YELLOW}INFO: Less than 5GB available${NC}"
        return 0
    else
        echo -e "${GREEN}OK: Sufficient disk space${NC}"
        return 0
    fi
}

# Function to perform cleanup
cleanup_disk_space() {
    echo "Performing disk space cleanup..."
    
    # Remove Docker images
    echo "Removing unused Docker images..."
    docker image prune -af --filter "until=24h" 2>/dev/null || true
    
    # Remove build caches
    echo "Removing build caches..."
    rm -rf ~/.ccache 2>/dev/null || true
    rm -rf ~/.cargo/registry 2>/dev/null || true
    rm -rf ~/.cargo/git 2>/dev/null || true
    
    # Remove temporary files
    echo "Removing temporary files..."
    rm -rf /tmp/* 2>/dev/null || true
    rm -rf /var/tmp/* 2>/dev/null || true
    
    # Clean package manager caches
    echo "Cleaning package manager caches..."
    sudo apt-get clean 2>/dev/null || true
    sudo rm -rf /var/lib/apt/lists/* 2>/dev/null || true
    
    # Remove old log files
    echo "Removing old log files..."
    find /var/log -type f -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    echo "Cleanup completed"
}

# Function to ensure minimum disk space
ensure_disk_space() {
    local available=$(get_available_space)
    
    if [ "$available" -lt "$CRITICAL_THRESHOLD" ]; then
        echo -e "${RED}CRITICAL: Disk space critically low!${NC}"
        cleanup_disk_space
        available=$(get_available_space)
        
        if [ "$available" -lt "$CRITICAL_THRESHOLD" ]; then
            echo -e "${RED}ERROR: Unable to free sufficient disk space${NC}"
            return 1
        fi
    elif [ "$available" -lt "$WARNING_THRESHOLD" ]; then
        echo -e "${YELLOW}WARNING: Disk space low, performing cleanup${NC}"
        cleanup_disk_space
    fi
    
    return 0
}

# Main execution
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    echo "=== Disk Space Recovery Tool ==="
    print_disk_status
    ensure_disk_space
fi

