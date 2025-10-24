# 🔨 Qallow Native App - Build Guide

Complete guide for building the Qallow Native Desktop Application from source.

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS 10.13+, or Windows 10+
- **RAM**: 4 GB minimum, 8 GB recommended
- **Disk Space**: 2 GB for build artifacts
- **Internet**: Required for downloading dependencies

### Required Software

1. **Rust Toolchain** (1.70.0+)
2. **C/C++ Compiler**
3. **CMake** (for FLTK)
4. **Git**

## Installation Steps

### Step 1: Install Rust

#### Linux/macOS

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustup update
```

#### Windows

Download from https://rustup.rs/ and run the installer.

### Step 2: Install Build Tools

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libx11-dev \
    libxext-dev \
    libxft-dev \
    libxinerama-dev \
    libxrandr-dev \
    libxcursor-dev \
    libxfixes-dev \
    libxkbcommon-dev
```

#### macOS

```bash
brew install cmake
xcode-select --install
```

#### Windows

- Install Visual Studio Build Tools
- Install CMake from https://cmake.org/download/

### Step 3: Clone Repository

```bash
cd /root/Qallow
git clone https://github.com/xingxerx/Qallow.git
cd Qallow/native_app
```

## Building

### Development Build

```bash
cd /root/Qallow/native_app
cargo build
```

**Time**: 3-5 minutes (first time)  
**Output**: `target/debug/qallow-native`

### Release Build (Optimized)

```bash
cd /root/Qallow/native_app
cargo build --release
```

**Time**: 5-10 minutes (first time)  
**Output**: `target/release/qallow-native`  
**Size**: ~15-20 MB (stripped)

### Optimized Release Build

```bash
cd /root/Qallow/native_app
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

**Performance**: 10-15% faster  
**Build Time**: 10-15 minutes

## Platform-Specific Builds

### Linux (x86_64)

```bash
cargo build --release --target x86_64-unknown-linux-gnu
```

### Linux (ARM64)

```bash
rustup target add aarch64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu
```

### macOS (Intel)

```bash
cargo build --release --target x86_64-apple-darwin
```

### macOS (Apple Silicon)

```bash
rustup target add aarch64-apple-darwin
cargo build --release --target aarch64-apple-darwin
```

### Windows (MSVC)

```bash
cargo build --release --target x86_64-pc-windows-msvc
```

### Windows (GNU)

```bash
rustup target add x86_64-pc-windows-gnu
cargo build --release --target x86_64-pc-windows-gnu
```

## Building with Features

### Minimal Build

```bash
cargo build --release --no-default-features
```

### Full Build

```bash
cargo build --release --all-features
```

## Troubleshooting Build Issues

### "FLTK not found"

```bash
# Install FLTK development files
sudo apt-get install -y libfltk1.3-dev

# Or build FLTK from source
cargo build --release --features fltk-bundled
```

### "CMake not found"

```bash
# Install CMake
sudo apt-get install -y cmake

# Or on macOS
brew install cmake
```

### "Linker error"

```bash
# Update Rust
rustup update

# Clean and rebuild
cargo clean
cargo build --release
```

### "Out of memory during build"

```bash
# Use single-threaded build
cargo build --release -j 1

# Or increase swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Testing the Build

### Run Tests

```bash
cargo test
```

### Run Application

```bash
./target/release/qallow-native
```

### Check Binary

```bash
file target/release/qallow-native
ldd target/release/qallow-native  # Linux
otool -L target/release/qallow-native  # macOS
```

## Creating Distribution Packages

### Linux (AppImage)

```bash
# Install appimagetool
wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x appimagetool-x86_64.AppImage

# Create AppImage
./appimagetool-x86_64.AppImage target/release/qallow-native Qallow-Native.AppImage
```

### Linux (Debian Package)

```bash
# Create debian package structure
mkdir -p debian/usr/bin
cp target/release/qallow-native debian/usr/bin/

# Build package
dpkg-deb --build debian qallow-native_1.0.0_amd64.deb
```

### macOS (DMG)

```bash
# Create app bundle
mkdir -p Qallow.app/Contents/MacOS
cp target/release/qallow-native Qallow.app/Contents/MacOS/

# Create DMG
hdiutil create -volname "Qallow" -srcfolder Qallow.app -ov -format UDZO Qallow.dmg
```

### Windows (Installer)

```bash
# Use NSIS or WiX to create installer
# Copy target/release/qallow-native.exe to installer
```

## Continuous Integration

### GitHub Actions

Create `.github/workflows/build.yml`:

```yaml
name: Build

on: [push, pull_request]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - run: cargo build --release
      - run: cargo test
```

## Performance Optimization

### Link-Time Optimization (LTO)

Add to `Cargo.toml`:

```toml
[profile.release]
lto = true
codegen-units = 1
```

### Strip Binary

```bash
strip target/release/qallow-native
```

### Reduce Binary Size

```bash
cargo build --release -Z build-std=std,panic_abort --target x86_64-unknown-linux-gnu
```

## Cleaning Up

### Remove Build Artifacts

```bash
cargo clean
```

### Remove Specific Target

```bash
cargo clean --target x86_64-unknown-linux-gnu
```

## Build Statistics

### Typical Build Times

| Configuration | Time | Size |
|---------------|------|------|
| Debug | 3-5 min | 50-80 MB |
| Release | 5-10 min | 15-20 MB |
| Optimized | 10-15 min | 12-18 MB |
| LTO | 15-20 min | 10-15 MB |

### Typical Binary Sizes

| Platform | Size |
|----------|------|
| Linux x86_64 | 18 MB |
| macOS x86_64 | 20 MB |
| macOS ARM64 | 19 MB |
| Windows x86_64 | 22 MB |

## Next Steps

1. **Run the Application**: `./target/release/qallow-native`
2. **Read Documentation**: See README.md
3. **Try Examples**: Run different phases
4. **Contribute**: Submit improvements

---

**Build Complete!** 🎉

Your Qallow Native Desktop Application is ready to use!

