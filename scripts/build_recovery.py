#!/usr/bin/env python3
"""
Build Recovery System
Automatically detects and recovers from build failures due to disk space or resource issues.
"""

import os
import subprocess
import shutil
import sys
from pathlib import Path

class BuildRecovery:
    def __init__(self, build_dir="build"):
        self.build_dir = Path(build_dir)
        self.critical_threshold = 1 * 1024 * 1024 * 1024  # 1GB
        self.warning_threshold = 2 * 1024 * 1024 * 1024   # 2GB
        
    def get_available_space(self):
        """Get available disk space in bytes"""
        stat = shutil.disk_usage("/")
        return stat.free
    
    def print_disk_status(self):
        """Print current disk usage"""
        available = self.get_available_space()
        total = shutil.disk_usage("/").total
        used = shutil.disk_usage("/").used
        
        percent = (used / total) * 100
        print(f"Disk Usage: {used / (1024**3):.1f}GB / {total / (1024**3):.1f}GB ({percent:.1f}%)")
        print(f"Available: {available / (1024**3):.1f}GB")
        
        if available < self.critical_threshold:
            print("⚠️  CRITICAL: Less than 1GB available")
            return "critical"
        elif available < self.warning_threshold:
            print("⚠️  WARNING: Less than 2GB available")
            return "warning"
        return "ok"
    
    def cleanup_build_artifacts(self):
        """Remove build artifacts to free space"""
        print("Cleaning build artifacts...")
        
        cleanup_paths = [
            self.build_dir / "CMakeFiles",
            self.build_dir / "*.o",
            Path.home() / ".ccache",
            Path("/tmp"),
        ]
        
        for path in cleanup_paths:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"  Removed: {path}")
            except Exception as e:
                print(f"  Failed to remove {path}: {e}")
    
    def cleanup_docker(self):
        """Clean Docker images"""
        print("Cleaning Docker images...")
        try:
            subprocess.run(["docker", "image", "prune", "-af"], 
                         capture_output=True, timeout=30)
            print("  Docker cleanup completed")
        except Exception as e:
            print(f"  Docker cleanup failed: {e}")
    
    def cleanup_package_cache(self):
        """Clean package manager caches"""
        print("Cleaning package caches...")
        try:
            subprocess.run(["sudo", "apt-get", "clean"], 
                         capture_output=True, timeout=30)
            subprocess.run(["sudo", "rm", "-rf", "/var/lib/apt/lists/*"], 
                         capture_output=True, timeout=30, shell=True)
            print("  Package cache cleanup completed")
        except Exception as e:
            print(f"  Package cache cleanup failed: {e}")
    
    def perform_recovery(self):
        """Perform full recovery"""
        print("\n=== Build Recovery System ===\n")
        
        status = self.print_disk_status()
        
        if status == "critical":
            print("\nPerforming aggressive cleanup...")
            self.cleanup_build_artifacts()
            self.cleanup_docker()
            self.cleanup_package_cache()
            
            print("\nDisk status after cleanup:")
            self.print_disk_status()
        elif status == "warning":
            print("\nPerforming cleanup...")
            self.cleanup_build_artifacts()
            self.cleanup_docker()
        
        return self.get_available_space() > self.critical_threshold

if __name__ == "__main__":
    recovery = BuildRecovery()
    success = recovery.perform_recovery()
    sys.exit(0 if success else 1)

