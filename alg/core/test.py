"""
ALG Test Module
Runs validation suite: unified framework tests + QAOA tests
"""

import json
import os
import sys


def run_unified_framework_tests():
    """Run subset of unified framework algorithms for validation"""
    print("[ALG TEST] Running unified framework validation (Bell, Grover, VQE)...\n")

    try:
        sys.path.insert(0, "/root/Qallow")
        from quantum_algorithms.unified_quantum_framework import QuantumAlgorithmFramework

        framework = QuantumAlgorithmFramework(verbose=False)

        results = {}
        tests = [
            ("Bell State", framework.run_bell_state),
            ("Grover's Algorithm", framework.run_grover),
            ("VQE", framework.run_vqe)
        ]

        passed = 0
        for test_name, test_func in tests:
            try:
                result = test_func()
                if hasattr(result, 'success') and result.success:
                    print(f"[ALG TEST] ✓ {test_name}")
                    passed += 1
                else:
                    print(f"[ALG TEST] ✗ {test_name}")
                results[test_name] = result
            except Exception as e:
                print(f"[ALG TEST] ✗ {test_name}: {e}")
                results[test_name] = None

        print(f"\n[ALG TEST] Framework tests: {passed}/3 passed")
        return passed == 3, results

    except Exception as e:
        print(f"[ALG TEST] ✗ Framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def run_qaoa_test():
    """Run test on 8-node ring Hamiltonian"""
    print("\n[ALG TEST] Running QAOA + SPSA test (8-node ring)...")

    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from qaoa_spsa import run_qaoa_optimizer

        # Create test config
        test_config = {
            "N": 8,
            "p": 1,  # Shallow circuit for fast test
            "csv_j": None,  # Use default ring
            "alpha_min": 0.001,
            "alpha_max": 0.01,
            "spsa_iterations": 10,  # Few iterations for speed
            "spsa_a": 0.1,
            "spsa_c": 0.1
        }

        # Create temporary config file
        test_config_path = "/tmp/alg_test_config.json"
        with open(test_config_path, "w") as f:
            json.dump(test_config, f)

        # Run optimizer
        result = run_qaoa_optimizer(test_config_path)

        # Validate result
        assert "energy" in result, "Missing 'energy' in result"
        assert "alpha_eff" in result, "Missing 'alpha_eff' in result"
        assert 0 < result["alpha_eff"] < 0.02, "alpha_eff out of bounds"

        print(f"[ALG TEST] ✓ QAOA test passed")
        print(f"[ALG TEST]   Energy: {result['energy']:.6f}")
        print(f"[ALG TEST]   Alpha_eff: {result['alpha_eff']:.6f}")

        return True

    except Exception as e:
        print(f"[ALG TEST] ✗ QAOA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_results_file():
    """Check if results file exists and is valid"""
    results_path = "/var/qallow/quantum_report.json"

    if not os.path.exists(results_path):
        print(f"\n[ALG TEST] No results file: {results_path}")
        print("[ALG TEST] Run 'alg run' first to generate results")
        return False

    try:
        with open(results_path, "r") as f:
            data = json.load(f)

        print(f"\n[ALG TEST] Results file: {results_path}")
        summary = data.get("summary", {})
        print(f"[ALG TEST]   Success rate: {summary.get('success_rate', 'N/A')}")
        print(f"[ALG TEST]   QAOA Energy: {data.get('qaoa_optimizer', {}).get('energy', 'N/A')}")
        print(f"[ALG TEST]   QAOA Alpha_eff: {data.get('qaoa_optimizer', {}).get('alpha_eff', 'N/A')}")

        return True

    except Exception as e:
        print(f"[ALG TEST] ERROR reading results: {e}")
        return False


def exec(args=None):
    """Execute test"""
    if args is None:
        args = []

    print("\n" + "="*70)
    print("ALG TEST - Validation Suite")
    print("="*70 + "\n")

    # Check if --quick flag is set
    quick_mode = "--quick" in args

    if quick_mode:
        print("[ALG TEST] Quick mode: checking existing results only\n")
        success = check_results_file()
    else:
        print("[ALG TEST] Full mode: running validation tests\n")

        # Run unified framework tests
        framework_success, _ = run_unified_framework_tests()

        # Run QAOA test
        qaoa_success = run_qaoa_test()

        success = framework_success and qaoa_success

        if success:
            print("\n[ALG TEST] Checking results file...")
            success = check_results_file()

    print("\n" + "="*70)
    if success:
        print("TEST COMPLETE - All checks passed ✓")
    else:
        print("TEST FAILED - See errors above")
    print("="*70 + "\n")

    return 0 if success else 1

