# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] #!/usr/bin/env python3
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] """
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] ALG - Unified Quantum Algorithm Framework for Qallow
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] Orchestrates all quantum algorithms: QAOA, Grover's, Shor's, VQE, Bell State, Deutsch
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] Plus QAOA + SPSA parameter tuning for coherence-lattice integration
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] """
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # Add core module to path
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] def print_usage():
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     """Print usage information"""
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     print("""
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] ╔════════════════════════════════════════════════════════════════╗
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] ║                                                                ║
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] ║  ALG - Unified Quantum Algorithm Framework for Qallow         ║
║  All Quantum Algorithms + QAOA + SPSA Parameter Tuning        ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

Usage: alg <command> [options]

Commands:
  build       Install dependencies (Qiskit, Cirq, NumPy, SciPy)
  run         Execute all quantum algorithms + QAOA optimizer
  test        Run validation suite (Bell, Grover, VQE)
  verify      Validate results and JSON integrity

Options:
  --export=PATH       Export results to JSON file
  --quick             Skip long-running algorithms
  --noise             Enable noise simulation
  --scale N           System size (default: 8)

Examples:
  alg build
  alg run --export=/var/qallow/quantum_report.json
  alg test --quick
  alg verify

Output Files:
  /var/qallow/quantum_report.json      Raw results
  /var/qallow/quantum_report.md        Human-readable summary

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

