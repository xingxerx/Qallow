#!/bin/bash

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║         🚀 INSTALLING QISKIT C API + HPC FOR REAL QUANTUM 🚀              ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "❌ Cannot detect OS"
    exit 1
fi

echo "📋 Detected OS: $OS"
echo ""

# Step 1: Install system dependencies
echo "═══════════════════════════════════════════════════════════════════════════"
echo "STEP 1: Installing system dependencies..."
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    echo "🔧 Installing Ubuntu/Debian packages..."
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        python3.11 \
        python3.11-dev \
        libopenblas-dev \
        libopenmpi-dev \
        libeigen3-dev \
        libboost-all-dev \
        curl \
        pkg-config
    echo "✅ Ubuntu/Debian packages installed"
elif [ "$OS" = "rhel" ] || [ "$OS" = "centos" ] || [ "$OS" = "fedora" ]; then
    echo "🔧 Installing RHEL/CentOS packages..."
    sudo yum install -y \
        gcc-c++ \
        cmake \
        git \
        python311 \
        python311-devel \
        openblas-devel \
        openmpi-devel \
        eigen3-devel \
        boost-devel \
        curl \
        pkgconfig
    echo "✅ RHEL/CentOS packages installed"
else
    echo "⚠️  Unsupported OS: $OS"
    echo "Please install dependencies manually:"
    echo "  - build-essential/gcc-c++"
    echo "  - cmake"
    echo "  - python3.11+"
    echo "  - libopenblas-dev"
    echo "  - libopenmpi-dev"
    echo "  - libeigen3-dev"
    echo "  - libboost-all-dev"
fi

echo ""

# Step 2: Install Rust
echo "═══════════════════════════════════════════════════════════════════════════"
echo "STEP 2: Installing Rust..."
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

if command -v rustc &> /dev/null; then
    echo "✅ Rust already installed: $(rustc --version)"
else
    echo "🔧 Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
    rustup update
    echo "✅ Rust installed: $(rustc --version)"
fi

echo ""

# Step 3: Clone Qiskit C API Demo
echo "═══════════════════════════════════════════════════════════════════════════"
echo "STEP 3: Cloning Qiskit C API Demo..."
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

if [ -d "/root/Qallow/qiskit-c-api-demo" ]; then
    echo "⚠️  Directory already exists, skipping clone"
else
    echo "🔧 Cloning repository..."
    cd /root/Qallow
    git clone https://github.com/qiskit-community/qiskit-c-api-demo.git
    cd qiskit-c-api-demo
    git submodule update --init --recursive
    echo "✅ Repository cloned"
fi

echo ""

# Step 4: Build Qiskit C Extension
echo "═══════════════════════════════════════════════════════════════════════════"
echo "STEP 4: Building Qiskit C Extension..."
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

cd /root/Qallow/qiskit-c-api-demo/deps/qiskit
echo "🔧 Building Qiskit C extension (this may take 5-10 minutes)..."
make c 2>&1 | tail -20
echo "✅ Qiskit C extension built"

echo ""

# Step 5: Build QRMI
echo "═══════════════════════════════════════════════════════════════════════════"
echo "STEP 5: Building QRMI (Quantum Resource Management Interface)..."
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

cd /root/Qallow/qiskit-c-api-demo/deps/qrmi
echo "🔧 Building QRMI (this may take 10-15 minutes)..."
cargo build --release 2>&1 | tail -20
echo "✅ QRMI built"

echo ""

# Step 6: Build Demo Application
echo "═══════════════════════════════════════════════════════════════════════════"
echo "STEP 6: Building C API Demo Application..."
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

cd /root/Qallow/qiskit-c-api-demo
mkdir -p build
cd build
echo "🔧 Running CMake..."
cmake .. 2>&1 | tail -10
echo "🔧 Building application (this may take 5-10 minutes)..."
make 2>&1 | tail -20
echo "✅ Demo application built"

echo ""

# Step 7: Verify installation
echo "═══════════════════════════════════════════════════════════════════════════"
echo "STEP 7: Verifying Installation..."
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

if [ -f "/root/Qallow/qiskit-c-api-demo/build/c-api-demo" ]; then
    echo "✅ c-api-demo executable found"
    ls -lh /root/Qallow/qiskit-c-api-demo/build/c-api-demo
else
    echo "❌ c-api-demo executable not found"
    exit 1
fi

echo ""

# Step 8: Create setup script
echo "═══════════════════════════════════════════════════════════════════════════"
echo "STEP 8: Creating IBM Quantum Setup Script..."
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

cat > /root/Qallow/setup_ibm_quantum.sh << 'IBM_SETUP'
#!/bin/bash

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║              🔑 IBM QUANTUM CREDENTIALS SETUP 🔑                           ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

echo "📋 INSTRUCTIONS:"
echo "1. Go to https://quantum.ibm.com/"
echo "2. Sign up (free account)"
echo "3. Go to Account settings"
echo "4. Copy your API key"
echo "5. Copy your CRN (Cloud Resource Name)"
echo ""

read -p "Enter your IBM Quantum API key: " API_KEY
read -p "Enter your IBM Quantum CRN: " CRN

echo ""
echo "Setting environment variables..."

export QISKIT_IBM_TOKEN="$API_KEY"
export QISKIT_IBM_INSTANCE="$CRN"

echo "export QISKIT_IBM_TOKEN=\"$API_KEY\"" >> ~/.bashrc
echo "export QISKIT_IBM_INSTANCE=\"$CRN\"" >> ~/.bashrc

echo ""
echo "✅ Credentials saved to ~/.bashrc"
echo ""
echo "To use immediately, run:"
echo "  source ~/.bashrc"
echo ""
IBM_SETUP

chmod +x /root/Qallow/setup_ibm_quantum.sh
echo "✅ Setup script created: /root/Qallow/setup_ibm_quantum.sh"

echo ""

# Final summary
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║                    ✅ INSTALLATION COMPLETE! ✅                            ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

echo "📋 NEXT STEPS:"
echo ""
echo "1. Setup IBM Quantum credentials:"
echo "   bash /root/Qallow/setup_ibm_quantum.sh"
echo ""
echo "2. Run on real quantum hardware:"
echo "   cd /root/Qallow/qiskit-c-api-demo/build"
echo "   ./c-api-demo \\"
echo "     --fcidump ../data/fcidump_Fe4S4_MO.txt \\"
echo "     -v \\"
echo "     --tolerance 1.0e-3 \\"
echo "     --max_time 600 \\"
echo "     --recovery 1 \\"
echo "     --number_of_samples 300 \\"
echo "     --num_shots 1000 \\"
echo "     --backend_name ibm_kingston"
echo ""
echo "3. Run distributed (MPI):"
echo "   mpirun -np 96 ./c-api-demo \\"
echo "     --fcidump ../data/fcidump_Fe4S4_MO.txt \\"
echo "     -v \\"
echo "     --tolerance 1.0e-3 \\"
echo "     --max_time 600 \\"
echo "     --recovery 1 \\"
echo "     --number_of_samples 2000 \\"
echo "     --num_shots 10000 \\"
echo "     --backend_name ibm_kingston"
echo ""
echo "📚 Documentation:"
echo "   - GitHub: https://github.com/qiskit-community/qiskit-c-api-demo"
echo "   - Qiskit C API: https://quantum.cloud.ibm.com/docs/en/api/qiskit-c"
echo "   - IBM Quantum: https://quantum.ibm.com/"
echo ""

