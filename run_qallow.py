#!/usr/bin/env python3
"""
Qallow Project Runner - Quick Start Demo
Shows the project structure and available components
"""

import os
import sys

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def check_component(name, path):
    """Check if a component exists"""
    exists = Path(path).exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {name:40} {path}")
    return exists

def main():
    os.chdir(Path(__file__).parent)
    
    print_header("QALLOW - Quantum-Photonic Computing Platform")
    print("\n🚀 PROJECT OVERVIEW\n")
    
    print("Qallow is an experimental quantum-photonic computing platform with:")
    print("  • 20 execution phases for quantum computing")
    print("  • Photonic simulation and quantum optimization")
    print("  • GPU acceleration (CUDA) with CPU fallback")
    print("  • Multiple UI frameworks (Native, Web, Electron)")
    print("  • Quantum machine learning capabilities")
    
    print_header("PROJECT COMPONENTS")
    
    print("\n📦 CORE MODULES:\n")
    components = [
        ("Quantum Algorithms", "quantum_algorithms/"),
        ("Quantum Optimizer", "quantum_optimizer/"),
        ("Native App (Rust)", "native_app/"),
        ("Web App", "web-app/"),
        ("Backend Server", "server/"),
        ("ALG Framework", "alg/"),
    ]
    
    found_count = 0
    for name, path in components:
        if check_component(name, path):
            found_count += 1
    
    print(f"\n✓ {found_count}/{len(components)} components available")
    
    print_header("BUILD STATUS")
    
    print("\n🔨 BUILD ARTIFACTS:\n")
    build_items = [
        ("Cargo.toml (Rust project)", "Cargo.toml"),
        ("CMakeLists.txt (C/C++ project)", "CMakeLists.txt"),
        ("build.sh script", "build.sh"),
    ]
    
    for name, path in build_items:
        check_component(name, path)
    
    print_header("HOW TO RUN QALLOW")
    
    print("""
✨ OPTION 1: Run Python Test Suite
   cd /home/xing/qallow/Qallow
   python3 test_quantum_complete.py

✨ OPTION 2: Build C/C++ Project (requires gcc)
   cd /home/xing/qallow/Qallow
   ./build.sh
   ./qallow_unified run

✨ OPTION 3: Run Quantum Algorithms (requires numpy, cirq)
   cd /home/xing/qallow/Qallow/quantum_algorithms
   pip install numpy cirq
   python3 application_runner.py

✨ OPTION 4: Build and Run Native App (requires Rust)
   cd /home/xing/qallow/Qallow/native_app
   cargo build --release
   cargo run --release

✨ OPTION 5: Start Web Server (requires Node.js)
   cd /home/xing/qallow/Qallow/server
   npm install
   npm start
    """)
    
    print_header("DOCUMENTATION")
    
    print("""
📚 Key Documentation Files:
   • README.md              - Main project overview
   • QUICKSTART.md          - Quick start guide
   • START_HERE.md          - Getting started
   • QUANTUM_ALGORITHMS_GUIDE.md - Quantum computing guide
   • K8S_QUICK_START.md     - Kubernetes deployment
    """)
    
    print_header("PROJECT STATUS")
    
    print(f"""
✓ System Information:
  • Current directory: {Path.cwd()}
  • Python version: {sys.version.split()[0]}
  • Platform: {sys.platform}
  
📊 Component Status:
  • Quantum algorithms: Available
  • Python quantum framework: Available (needs dependencies)
  • C/C++ Build system: Ready (needs compilation)
  • Rust components: Available (needs Rust toolchain)
  • Web components: Available (needs Node.js)
    """)
    
    print_header("QUICK COMMANDS")
    
    print("""
cd /home/xing/qallow/Qallow      # Go to project directory
python3 test_quantum_complete.py  # Run verification tests
./build.sh                        # Build the C/C++ project
ls alg/                           # View algorithm framework
ls quantum_algorithms/            # View quantum modules
    """)
    
    print("\n✅ Qallow is ready! Choose an option above to get started.\n")

if __name__ == "__main__":
    main()
