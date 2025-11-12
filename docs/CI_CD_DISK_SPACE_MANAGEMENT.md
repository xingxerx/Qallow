# CI/CD Disk Space Management

## Overview

The Qallow CI/CD pipeline has been enhanced with comprehensive disk space management to prevent build failures due to insufficient storage on GitHub Actions runners.

## Problem Statement

GitHub Actions runners have limited disk space (~14GB). Large builds (especially CUDA builds) can exhaust available space, causing:
- Build failures with "No space left on device" errors
- Incomplete artifact generation
- Workflow runner crashes

## Solution Architecture

### 1. Aggressive Initial Cleanup

Before any build starts, the workflow performs aggressive cleanup:
- Removes Android SDK (~5GB)
- Removes .NET runtime (~3GB)
- Removes GHC toolchain (~2GB)
- Removes hosted toolcache (~2GB)
- Cleans Docker images
- Cleans package manager caches

**Result**: Frees ~15GB of space

### 2. Runtime Disk Monitoring

During builds, the system continuously monitors available space:
- Checks before CMake configuration
- Checks before compilation
- Triggers cleanup if space drops below 2GB
- Aborts if space drops below 1GB

### 3. Automatic Recovery

If a build fails due to disk space:
1. Cleanup is triggered automatically
2. Build is retried with reduced parallelism (-j1)
3. Temporary files are removed
4. Docker system is pruned

### 4. Build Parallelism Optimization

- **CPU builds**: Use full parallelism (-j$(nproc))
- **CUDA builds**: Use reduced parallelism (-j2) to save memory
- **Recovery builds**: Use single job (-j1) to minimize resource usage

## Scripts

### disk_space_recovery.sh

Bash script for disk space management:

```bash
# Check disk status
source scripts/disk_space_recovery.sh
print_disk_status

# Ensure minimum space
ensure_disk_space
```

**Thresholds**:
- Critical: < 1GB
- Warning: < 2GB
- Safe: > 5GB

### build_recovery.py

Python script for automated build recovery:

```bash
python3 scripts/build_recovery.py
```

**Features**:
- Disk usage reporting
- Automatic cleanup
- Docker image pruning
- Package cache cleaning

## Workflow Configuration

### CPU Build Job

- Runs on every push/PR
- Uses full parallelism
- Includes recovery mechanism
- Cleans up after completion

### CUDA Build Job

- Runs on every push/PR (non-blocking)
- Uses reduced parallelism (-j2)
- Includes aggressive recovery
- Optional (continue-on-error: true)

## Monitoring

Each job includes disk space monitoring:

```yaml
- name: Monitor disk space
  if: always()
  run: |
    df -h
    du -sh /* | sort -rh | head -10
```

## Best Practices

1. **Always cleanup after builds**: Use `if: always()` to ensure cleanup runs
2. **Monitor disk space**: Check before and after major operations
3. **Use reduced parallelism for large builds**: CUDA builds use -j2
4. **Implement recovery**: Retry with -j1 if initial build fails
5. **Clean Docker regularly**: Prune unused images

## Troubleshooting

### Build still fails with disk space error

1. Check if cleanup scripts are running
2. Verify Docker is installed and running
3. Check for large files in /tmp or /var/log
4. Consider splitting build into smaller jobs

### Slow builds after recovery

This is expected - recovery builds use -j1 for stability. Once space is freed, subsequent builds will use full parallelism.

### Cleanup doesn't free enough space

Check for:
- Large Docker images (docker system df)
- Old log files (du -sh /var/log)
- Build artifacts (du -sh build/)
- Cargo cache (du -sh ~/.cargo)

## Future Improvements

1. Implement incremental builds to reduce space usage
2. Use ccache for faster rebuilds
3. Split large builds across multiple jobs
4. Implement artifact caching strategy
5. Monitor and alert on disk usage trends

