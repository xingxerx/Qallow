# How to Copy Qallow to Linux and Build

## Option 1: Using Git (Recommended)

If you have git on both machines:

```bash
# On Windows (in D:\Qallow)
git init
git add .
git commit -m "Qallow unified system"

# On Linux
git clone <your-repo-url> /root/Qallow
cd /root/Qallow
chmod +x build_simple.sh
./build_simple.sh
```

---

## Option 2: Using tar + scp

```bash
# On Windows (in D:\Qallow)
tar czf qallow.tar.gz backend/ interface/ io/ core/ build_simple.sh

# On Linux
scp user@windows:/path/to/qallow.tar.gz .
tar xzf qallow.tar.gz
chmod +x build_simple.sh
./build_simple.sh
```

---

## Option 3: Manual Copy (If scp not available)

### Step 1: Create a tar file on Windows

```bash
# In PowerShell on Windows (D:\Qallow)
tar -czf qallow.tar.gz backend/ interface/ io/ core/ build_simple.sh
```

### Step 2: Transfer the file

- Use USB drive
- Use cloud storage (Google Drive, Dropbox, etc.)
- Use FTP/SFTP
- Use email

### Step 3: Extract on Linux

```bash
# On Linux
tar xzf qallow.tar.gz
chmod +x build_simple.sh
./build_simple.sh
```

---

## Option 4: Direct File Copy (If you have shared network)

```bash
# On Linux, mount Windows share
sudo mount -t cifs //windows-ip/share /mnt/windows -o username=user,password=pass

# Copy files
cp -r /mnt/windows/Qallow/* /root/Qallow/

# Build
cd /root/Qallow
chmod +x build_simple.sh
./build_simple.sh
```

---

## Quick Build (After Files Are Copied)

```bash
# 1. Go to directory
cd /root/Qallow

# 2. Make script executable
chmod +x build_simple.sh

# 3. Build
./build_simple.sh

# 4. Run
./qallow_unified run
./qallow_unified bench
./qallow_unified verify
./qallow_unified live
```

---

## What Gets Built

```
qallow_unified  ← Single executable with all 7 commands
```

---

## Troubleshooting

### "gcc: command not found"

```bash
sudo apt-get install -y build-essential
```

### "No such file or directory"

```bash
# Check files are in right place
ls -la backend/cpu/qallow_kernel.c
ls -la interface/launcher.c
ls -la core/include/qallow_kernel.h

# Check you're in right directory
pwd
# Should show: /root/Qallow
```

### Build fails with CUDA error

```bash
# Just use CPU-only (CUDA is optional)
# The script will auto-detect and use CPU if CUDA fails
./build_simple.sh
```

---

## Files You Need to Copy

```
backend/
├── cpu/
│   ├── qallow_kernel.c
│   ├── overlay.c
│   ├── ethics.c
│   ├── ppai.c
│   ├── qcp.c
│   ├── pocket_dimension.c
│   ├── telemetry.c
│   ├── adaptive.c
│   ├── pocket.c
│   ├── govern.c
│   ├── ingest.c
│   ├── verify.c
│   ├── semantic_memory.c
│   ├── goal_synthesizer.c
│   ├── transfer_engine.c
│   ├── self_reflection.c
│   ├── phase7_core.c
│   ├── chronometric.c
│   ├── multi_pocket.c
│   └── (any other .c files)
├── cuda/
│   ├── ppai_kernels.cu
│   ├── qcp_kernels.cu
│   ├── photonic.cu
│   ├── quantum.cu
│   └── pocket.cu
interface/
├── launcher.c
└── main.c
io/
└── adapters/
    ├── net_adapter.c
    └── sim_adapter.c
core/
└── include/
    ├── qallow_kernel.h
    ├── ppai.h
    ├── qcp.h
    ├── ethics.h
    ├── overlay.h
    ├── sandbox.h
    ├── telemetry.h
    ├── pocket.h
    ├── phase7.h
    ├── ingest.h
    └── verify.h
build_simple.sh
```

---

## Status

✅ Build script created: `build_simple.sh`
✅ Ready to copy to Linux
✅ Supports CPU and CUDA
✅ All 7 commands included

---

**Version**: Phase 8-10 Complete
**Last Updated**: 2025-10-18

