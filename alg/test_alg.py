#!/usr/bin/env python3
"""
ALG Test Suite
Comprehensive testing for quantum optimizer
"""

import sys
import os
import json
import tempfile
import subprocess

# Add alg to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported"""
    print("\n[TEST] Testing imports...")
    try:
        from core import build, run, test, verify
        from qaoa_spsa import run_qaoa_optimizer
        print("[TEST] ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"[TEST] ✗ Import failed: {e}")
        return False


def test_build():
    """Test build module"""
    print("\n[TEST] Testing build module...")
    try:
        from core import build
        # Just check that functions exist
        assert hasattr(build, 'check_python_version')
        assert hasattr(build, 'check_package')
        assert hasattr(build, 'install_package')
        print("[TEST] ✓ Build module OK")
        return True
    except Exception as e:
        print(f"[TEST] ✗ Build test failed: {e}")
        return False


def test_config_creation():
    """Test configuration file creation"""
    print("\n[TEST] Testing config creation...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")
            
            config = {
                "N": 4,
                "p": 1,
                "csv_j": None,
                "alpha_min": 0.001,
                "alpha_max": 0.01,
                "spsa_iterations": 5,
                "spsa_a": 0.1,
                "spsa_c": 0.1
            }
            
            with open(config_path, "w") as f:
                json.dump(config, f)
            
            # Verify it can be read back
            with open(config_path, "r") as f:
                loaded = json.load(f)
            
            assert loaded["N"] == 4
            print("[TEST] ✓ Config creation OK")
            return True
    except Exception as e:
        print(f"[TEST] ✗ Config test failed: {e}")
        return False


def test_ising_energy():
    """Test Ising energy calculation"""
    print("\n[TEST] Testing Ising energy calculation...")
    try:
        from qaoa_spsa import ising_energy
        import numpy as np
        
        # Simple 2-node system
        J = np.array([[0, 1], [1, 0]])
        z = np.array([1, -1])
        
        energy = ising_energy(z, J)
        expected = -1 * 1 * 1 * (-1)  # -J[0,1] * z[0] * z[1]
        
        assert abs(energy - expected) < 1e-6
        print(f"[TEST] ✓ Ising energy OK (energy={energy:.6f})")
        return True
    except Exception as e:
        print(f"[TEST] ✗ Ising energy test failed: {e}")
        return False


def test_verify_module():
    """Test verify module"""
    print("\n[TEST] Testing verify module...")
    try:
        from core import verify
        
        # Test JSON structure verification
        test_data = {
            "energy": -5.0,
            "alpha_eff": 0.005,
            "iterations": 50
        }
        
        verify.verify_json_structure(test_data)
        verify.verify_value_ranges(test_data)
        
        print("[TEST] ✓ Verify module OK")
        return True
    except Exception as e:
        print(f"[TEST] ✗ Verify test failed: {e}")
        return False


def test_cli_help():
    """Test CLI help output"""
    print("\n[TEST] Testing CLI help...")
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        assert result.returncode == 0
        assert "ALG" in result.stdout
        assert "build" in result.stdout
        assert "run" in result.stdout
        
        print("[TEST] ✓ CLI help OK")
        return True
    except Exception as e:
        print(f"[TEST] ✗ CLI help test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("ALG TEST SUITE")
    print("="*70)
    
    tests = [
        test_imports,
        test_build,
        test_config_creation,
        test_ising_energy,
        test_verify_module,
        test_cli_help,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"[TEST] ✗ Unexpected error: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*70 + "\n")
    
    return all(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

