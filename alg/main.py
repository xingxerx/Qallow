#!/usr/bin/env python3
"""
ALG - Quantum Algorithm Optimizer for Qallow
Unified command-line interface for QAOA + SPSA parameter tuning
"""

import sys
import os

# Add core module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import build, run, test, verify


def print_usage():
    """Print usage information"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  ALG - Quantum Algorithm Optimizer for Qallow                 ║
║  QAOA + SPSA Parameter Tuning System                          ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

Usage: alg <command> [options]

Commands:
  build       Compile and prepare dependencies (Qiskit, CUDA libs)
  run         Execute quantum optimizer (QAOA + SPSA)
  test        Run internal test (8-node ring model)
  verify      Validate results and JSON integrity

Examples:
  alg build
  alg run --config=/var/qallow/ising_spec.json
  alg test
  alg verify

For more information, visit: https://github.com/xingxerx/Qallow
""")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    cmd = sys.argv[1]
    
    try:
        if cmd == "build":
            build.exec(sys.argv[2:])
        elif cmd == "run":
            run.exec(sys.argv[2:])
        elif cmd == "test":
            test.exec(sys.argv[2:])
        elif cmd == "verify":
            verify.exec(sys.argv[2:])
        elif cmd in ["-h", "--help", "help"]:
            print_usage()
            sys.exit(0)
        elif cmd in ["-v", "--version", "version"]:
            print("ALG version 1.0.0")
            sys.exit(0)
        else:
            print(f"[ALG ERROR] Unknown command: {cmd}")
            print_usage()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[ALG] Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"[ALG ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

