"""
ALG Verify Module
Validates unified quantum report and QAOA results
"""

import json
import os
import sys


def verify_quantum_report_structure(data):
    """Verify quantum report has required fields"""
    required_fields = ["timestamp", "version", "quantum_algorithms", "qaoa_optimizer", "summary"]

    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Check QAOA optimizer fields
    qaoa = data.get("qaoa_optimizer", {})
    qaoa_fields = ["energy", "alpha_eff"]
    for field in qaoa_fields:
        if field not in qaoa:
            raise ValueError(f"Missing QAOA field: {field}")

    return True


def verify_qaoa_value_ranges(data):
    """Verify QAOA values are within expected ranges"""
    qaoa = data.get("qaoa_optimizer", {})

    checks = [
        ("alpha_eff", 0.0, 0.02, "Control gain out of bounds"),
        ("energy", -100.0, 0.0, "Energy should be negative or zero"),
        ("iterations", 1, 1000, "Iterations out of range"),
    ]

    for field, min_val, max_val, msg in checks:
        if field in qaoa:
            val = qaoa[field]
            if not (min_val <= val <= max_val):
                raise ValueError(f"{msg}: {field}={val}")

    return True


def verify_algorithm_success_rates(data):
    """Verify algorithm success rates meet thresholds"""
    summary = data.get("summary", {})
    success_rate_str = summary.get("success_rate", "0%")

    # Parse success rate
    try:
        success_rate = float(success_rate_str.rstrip("%"))
    except:
        raise ValueError(f"Invalid success rate format: {success_rate_str}")

    # Check minimum threshold (95%)
    if success_rate < 95.0:
        raise ValueError(f"Success rate {success_rate}% below threshold (95%)")

    return True


def verify_config_consistency(config_path, results_path):
    """Verify results match config"""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        with open(results_path, "r") as f:
            results = json.load(f)

        # Check alpha bounds match
        if "alpha_min" in config and "alpha_max" in config:
            qaoa = results.get("qaoa_optimizer", {})
            alpha_eff = qaoa.get("alpha_eff", 0)
            if not (config["alpha_min"] <= alpha_eff <= config["alpha_max"]):
                raise ValueError(
                    f"alpha_eff {alpha_eff} outside config bounds "
                    f"[{config['alpha_min']}, {config['alpha_max']}]"
                )

        return True

    except FileNotFoundError:
        print("[ALG VERIFY] Config file not found (optional check)")
        return True


def exec(args=None):
    """Execute verification"""
    if args is None:
        args = []

    print("\n" + "="*70)
    print("ALG VERIFY - Quantum Report Validation")
    print("="*70 + "\n")

    results_path = "/var/qallow/quantum_report.json"
    config_path = "/var/qallow/ising_spec.json"

    # Check if results file exists
    if not os.path.exists(results_path):
        print(f"[ALG VERIFY] ERROR: Results file not found: {results_path}")
        print("[ALG VERIFY] Run 'alg run' first to generate results")
        sys.exit(1)

    print(f"[ALG VERIFY] Checking: {results_path}\n")

    try:
        # Load results
        with open(results_path, "r") as f:
            results = json.load(f)
        print("[ALG VERIFY] ✓ JSON is valid")

        # Verify structure
        verify_quantum_report_structure(results)
        print("[ALG VERIFY] ✓ All required fields present")

        # Verify QAOA value ranges
        verify_qaoa_value_ranges(results)
        print("[ALG VERIFY] ✓ QAOA values within expected ranges")

        # Verify algorithm success rates
        verify_algorithm_success_rates(results)
        print("[ALG VERIFY] ✓ Algorithm success rates meet threshold (≥95%)")

        # Verify consistency with config
        verify_config_consistency(config_path, results_path)
        print("[ALG VERIFY] ✓ Results consistent with config")

        # Print summary
        print("\n[ALG VERIFY] Quantum Report Summary:")
        summary = results.get("summary", {})
        print(f"  Total Algorithms:  {summary.get('total_algorithms', 'N/A')}")
        print(f"  Successful:        {summary.get('successful', 'N/A')}")
        print(f"  Success Rate:      {summary.get('success_rate', 'N/A')}")

        qaoa = results.get("qaoa_optimizer", {})
        print(f"\n[ALG VERIFY] QAOA Optimizer Results:")
        print(f"  Energy:            {qaoa.get('energy', 'N/A'):.6f}")
        print(f"  Alpha_eff:         {qaoa.get('alpha_eff', 'N/A'):.6f}")
        print(f"  Iterations:        {qaoa.get('iterations', 'N/A')}")

        if "timestamp" in results:
            print(f"  Timestamp:         {results['timestamp']}")

        print("\n" + "="*70)
        print("VERIFICATION COMPLETE - All checks passed ✓")
        print("="*70 + "\n")

    except json.JSONDecodeError as e:
        print(f"[ALG VERIFY] ERROR: Invalid JSON: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ALG VERIFY] ERROR: Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ALG VERIFY] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

