"""
ALG Verify Module
Validates results and JSON integrity
"""

import json
import os
import sys


def verify_json_structure(data):
    """Verify JSON has required fields"""
    required_fields = ["energy", "alpha_eff", "iterations"]
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    return True


def verify_value_ranges(data):
    """Verify values are within expected ranges"""
    checks = [
        ("alpha_eff", 0.0, 0.02, "Control gain out of bounds"),
        ("energy", -100.0, 0.0, "Energy should be negative or zero"),
        ("iterations", 1, 1000, "Iterations out of range"),
    ]
    
    for field, min_val, max_val, msg in checks:
        if field in data:
            val = data[field]
            if not (min_val <= val <= max_val):
                raise ValueError(f"{msg}: {field}={val}")
    
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
            alpha_eff = results.get("alpha_eff", 0)
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
    print("ALG VERIFY - Results Validation")
    print("="*70 + "\n")
    
    results_path = "/var/qallow/qaoa_gain.json"
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
        verify_json_structure(results)
        print("[ALG VERIFY] ✓ All required fields present")
        
        # Verify value ranges
        verify_value_ranges(results)
        print("[ALG VERIFY] ✓ All values within expected ranges")
        
        # Verify consistency with config
        verify_config_consistency(config_path, results_path)
        print("[ALG VERIFY] ✓ Results consistent with config")
        
        # Print summary
        print("\n[ALG VERIFY] Results Summary:")
        print(f"  Energy:      {results.get('energy', 'N/A'):.6f}")
        print(f"  Alpha_eff:   {results.get('alpha_eff', 'N/A'):.6f}")
        print(f"  Iterations:  {results.get('iterations', 'N/A')}")
        
        if "timestamp" in results:
            print(f"  Timestamp:   {results['timestamp']}")
        
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

