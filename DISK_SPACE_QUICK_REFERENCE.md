# Disk Space Management - Quick Reference

## Problem
GitHub Actions workflows fail with "No space left on device" errors during builds.

## Solution
Enhanced CI/CD pipeline with automatic disk space management and recovery.

## What Changed

### Workflow File: `.github/workflows/internal-ci.yml`

**Before**: Simple build without disk management
**After**: 
- Aggressive initial cleanup (~15GB freed)
- Runtime disk monitoring
- Automatic recovery on failure
- Separate CUDA build job

### New Scripts

1. **`scripts/disk_space_recovery.sh`**
   - Bash script for disk management
   - Usage: `source scripts/disk_space_recovery.sh && ensure_disk_space`

2. **`scripts/build_recovery.py`**
   - Python script for automated recovery
   - Usage: `python3 scripts/build_recovery.py`

### New Documentation

1. **`docs/CI_CD_DISK_SPACE_MANAGEMENT.md`** - Comprehensive guide
2. **`CI_CD_IMPROVEMENTS_SUMMARY.md`** - Implementation summary
3. **`DISK_SPACE_QUICK_REFERENCE.md`** - This file

## Key Improvements

| Aspect | Before | After |
|--------|--------|-------|
| Initial cleanup | None | ~15GB freed |
| Disk monitoring | None | Continuous |
| Recovery | None | Automatic retry |
| CUDA builds | Same job | Separate job |
| Parallelism | Full | Adaptive |

## Disk Space Thresholds

- **Critical**: < 1GB → Abort build
- **Warning**: < 2GB → Trigger cleanup
- **Safe**: > 5GB → Normal operation

## Build Parallelism

- **CPU builds**: `-j$(nproc)` (full parallelism)
- **CUDA builds**: `-j2` (reduced to save memory)
- **Recovery builds**: `-j1` (single job for stability)

## Cleanup Actions

1. Remove Android SDK (~5GB)
2. Remove .NET runtime (~3GB)
3. Remove GHC toolchain (~2GB)
4. Remove hosted toolcache (~2GB)
5. Clean Docker images
6. Clean package caches
7. Remove temporary files

## Recovery Process

If build fails:
1. Cleanup triggered automatically
2. Disk space freed
3. Build retried with reduced parallelism
4. If still fails, retry with -j1

## Monitoring

Each job includes:
- Initial disk space report
- Pre-build disk check
- Post-build disk report
- Largest directories listing

## Testing Locally

```bash
# Check disk status
source scripts/disk_space_recovery.sh
print_disk_status

# Perform recovery
python3 scripts/build_recovery.py

# Build with recovery
cmake -S . -B build/CPU -DQALLOW_ENABLE_CUDA=OFF
cmake --build build/CPU -- -j$(nproc)
```

## Expected Workflow Results

✅ CPU builds complete successfully
✅ CUDA builds complete successfully
✅ No "No space left on device" errors
✅ Automatic recovery on transient failures
✅ Detailed disk space reporting

## Troubleshooting

**Build still fails?**
- Check cleanup scripts are running
- Verify Docker is installed
- Check for large files in /tmp

**Slow builds?**
- Normal after recovery (uses -j1)
- Subsequent builds will use full parallelism

**Cleanup doesn't free enough?**
- Check Docker images: `docker system df`
- Check logs: `du -sh /var/log`
- Check build artifacts: `du -sh build/`

## Files Modified

1. `.github/workflows/internal-ci.yml` - Enhanced workflow
2. `scripts/disk_space_recovery.sh` - New helper script
3. `scripts/build_recovery.py` - New recovery script
4. `docs/CI_CD_DISK_SPACE_MANAGEMENT.md` - New documentation
5. `CI_CD_IMPROVEMENTS_SUMMARY.md` - Implementation summary

## Next Steps

1. Commit and push changes
2. Monitor workflow execution
3. Verify no disk space errors
4. Adjust thresholds if needed
5. Document any issues found

## Support

For detailed information, see:
- `docs/CI_CD_DISK_SPACE_MANAGEMENT.md` - Full documentation
- `CI_CD_IMPROVEMENTS_SUMMARY.md` - Implementation details
- `.github/workflows/internal-ci.yml` - Workflow configuration

