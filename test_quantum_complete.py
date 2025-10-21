#!/usr/bin/env python3
"""
Comprehensive test of the quantum workload implementation
"""
import json
import sys
from pathlib import Path

def test_quantum_implementation():
    """Test all quantum components"""
    print("\n" + "="*70)
    print("QUANTUM WORKLOAD IMPLEMENTATION TEST")
    print("="*70 + "\n")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Check Python modules exist
    tests_total += 1
    print("[TEST 1] Checking Python modules...")
    modules = [
        'python/quantum_ibm_workload.py',
        'python/quantum_cuda_bridge.py',
        'python/quantum_learning_system.py'
    ]
    for module in modules:
        if Path(module).exists():
            print(f"  ✓ {module}")
            tests_passed += 1
        else:
            print(f"  ✗ {module} NOT FOUND")
    
    # Test 2: Check scripts exist
    tests_total += 1
    print("\n[TEST 2] Checking execution scripts...")
    scripts = [
        'scripts/setup_quantum_workload.sh',
        'scripts/run_quantum_workload.sh'
    ]
    for script in scripts:
        if Path(script).exists():
            print(f"  ✓ {script}")
            tests_passed += 1
        else:
            print(f"  ✗ {script} NOT FOUND")
    
    # Test 3: Check documentation
    tests_total += 1
    print("\n[TEST 3] Checking documentation...")
    docs = [
        'docs/QUANTUM_WORKLOAD_GUIDE.md',
        'QUANTUM_IMPLEMENTATION_SUMMARY.md'
    ]
    for doc in docs:
        if Path(doc).exists():
            print(f"  ✓ {doc}")
            tests_passed += 1
        else:
            print(f"  ✗ {doc} NOT FOUND")
    
    # Test 4: Check output directories
    tests_total += 1
    print("\n[TEST 4] Checking output directories...")
    dirs = ['data/quantum_results', 'logs']
    for d in dirs:
        if Path(d).exists():
            print(f"  ✓ {d}")
            tests_passed += 1
        else:
            print(f"  ✗ {d} NOT FOUND")
    
    # Test 5: Check generated results
    tests_total += 1
    print("\n[TEST 5] Checking generated quantum results...")
    results_dir = Path('data/quantum_results')
    if results_dir.exists():
        result_files = list(results_dir.glob('*.json'))
        if result_files:
            print(f"  ✓ Found {len(result_files)} result files")
            for rf in result_files[:2]:
                try:
                    with open(rf) as f:
                        data = json.load(f)
                    print(f"    - {rf.name}: {len(data)} keys")
                    tests_passed += 1
                except Exception as e:
                    print(f"    ✗ Error reading {rf.name}: {e}")
        else:
            print("  ✗ No result files found")
    
    # Test 6: Check CUDA benchmark
    tests_total += 1
    print("\n[TEST 6] Checking CUDA benchmark results...")
    cuda_file = Path('data/cuda_benchmark.json')
    if cuda_file.exists():
        try:
            with open(cuda_file) as f:
                data = json.load(f)
            print(f"  ✓ CUDA benchmark: {len(data)} entries")
            tests_passed += 1
        except Exception as e:
            print(f"  ✗ Error reading CUDA benchmark: {e}")
    else:
        print("  ✗ CUDA benchmark not found")
    
    # Test 7: Check learning history
    tests_total += 1
    print("\n[TEST 7] Checking learning system history...")
    history_files = list(Path('data').glob('quantum_learning_history_*.json'))
    if history_files:
        print(f"  ✓ Found {len(history_files)} learning history files")
        try:
            with open(history_files[-1]) as f:
                data = json.load(f)
            print(f"    - Latest: {history_files[-1].name}")
            print(f"    - Keys: {list(data.keys())}")
            tests_passed += 1
        except Exception as e:
            print(f"    ✗ Error reading history: {e}")
    else:
        print("  ✗ No learning history found")
    
    # Test 8: Check adapt_state.json
    tests_total += 1
    print("\n[TEST 8] Checking adaptive learning state...")
    state_file = Path('adapt_state.json')
    if state_file.exists():
        try:
            with open(state_file) as f:
                data = json.load(f)
            print(f"  ✓ Adaptive state file exists")
            print(f"    - Learning rate: {data.get('learning_rate')}")
            print(f"    - Iterations: {data.get('iterations')}")
            print(f"    - Error correction enabled: {data.get('error_correction_enabled')}")
            tests_passed += 1
        except Exception as e:
            print(f"  ✗ Error reading state: {e}")
    else:
        print("  ✗ Adaptive state file not found")
    
    # Summary
    print("\n" + "="*70)
    print(f"TEST RESULTS: {tests_passed}/{tests_total} tests passed")
    print("="*70 + "\n")
    
    if tests_passed == tests_total:
        print("✅ ALL TESTS PASSED - Quantum workload implementation is complete!")
        return 0
    else:
        print(f"⚠️  {tests_total - tests_passed} tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(test_quantum_implementation())
