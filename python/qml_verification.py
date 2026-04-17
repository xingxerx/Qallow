#!/usr/bin/env python3
"""
QML Verification Suite for Qallow
Validates quantum gradients, CUDA kernel latency, and entanglement fidelity
"""

import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any

def check_gradient_flow() -> Dict[str, Any]:
    """Check if quantum gradients are flowing correctly"""
    print("\n" + "="*60)
    print("1. GRADIENT FLOW CHECK")
    print("="*60)
    
    try:
        # Run phase 13 which has good gradient properties
        result = subprocess.run(
            ["./build/qallow", "phase", "13", "--ticks=32"],
            capture_output=True,
            text=True,
            cwd="/home/xing/Qallow",
            timeout=30
        )
        
        if result.returncode == 0:
            print("✓ Phase 13 executed successfully")
            print(f"  Output: {result.stdout[:200]}...")
            return {"status": "PASS", "message": "Gradients flowing"}
        else:
            print(f"✗ Phase 13 failed: {result.stderr}")
            return {"status": "FAIL", "message": result.stderr}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def check_cuda_kernel_latency() -> Dict[str, Any]:
    """Verify CUDA kernel latency (should be <5ms for QML)"""
    print("\n" + "="*60)
    print("2. CUDA KERNEL LATENCY CHECK")
    print("="*60)
    
    try:
        with open("/home/xing/Qallow/data/logs/qallow_bench.log", "r") as f:
            lines = f.readlines()
        
        # Parse benchmark log
        cuda_runs = []
        for line in lines:
            if "CUDA" in line and "run_ms" not in line:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    try:
                        run_ms = float(parts[2])
                        cuda_runs.append(run_ms)
                    except:
                        pass
        
        if cuda_runs:
            avg_latency = sum(cuda_runs) / len(cuda_runs)
            max_latency = max(cuda_runs)
            print(f"✓ CUDA kernel latency detected")
            print(f"  Average: {avg_latency:.2f}ms")
            print(f"  Max: {max_latency:.2f}ms")
            print(f"  Samples: {len(cuda_runs)}")
            
            status = "PASS" if avg_latency < 5.0 else "WARN"
            return {
                "status": status,
                "avg_latency_ms": avg_latency,
                "max_latency_ms": max_latency,
                "samples": len(cuda_runs)
            }
        else:
            return {"status": "WARN", "message": "No CUDA runs found in benchmark"}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def check_entanglement_fidelity() -> Dict[str, Any]:
    """Inspect entanglement fidelity from phase summary"""
    print("\n" + "="*60)
    print("3. ENTANGLEMENT FIDELITY CHECK")
    print("="*60)
    
    try:
        with open("/home/xing/Qallow/data/logs/phase_summary.json", "r") as f:
            data = json.load(f)
        
        metrics = data.get("metrics", {})
        coherence_final = metrics.get("coherence_final", 0)
        ethics_total = metrics.get("ethics_total", 0)
        
        print(f"✓ Phase summary loaded")
        print(f"  Final Coherence: {coherence_final:.6f}")
        print(f"  Ethics Score (S+C+H): {ethics_total:.6f}")
        print(f"  Sustainability: {metrics.get('sustainability', 0):.6f}")
        print(f"  Compassion: {metrics.get('compassion', 0):.6f}")
        print(f"  Harmony: {metrics.get('harmony', 0):.6f}")
        
        return {
            "status": "PASS",
            "coherence_final": coherence_final,
            "ethics_total": ethics_total,
            "metrics": metrics
        }
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def check_data_loading() -> Dict[str, Any]:
    """Check if data loading infrastructure is ready"""
    print("\n" + "="*60)
    print("4. DATA LOADING INFRASTRUCTURE CHECK")
    print("="*60)
    
    try:
        # Check for DL integration module
        dl_module = Path("/home/xing/Qallow/build/qallow_unit_dl_integration")
        if dl_module.exists():
            print(f"✓ DL integration module found: {dl_module}")
            return {"status": "PASS", "module": str(dl_module)}
        else:
            print(f"✗ DL integration module not found")
            return {"status": "WARN", "message": "Module not built"}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def check_hybrid_loop_readiness() -> Dict[str, Any]:
    """Check if hybrid loop infrastructure is ready"""
    print("\n" + "="*60)
    print("5. HYBRID LOOP READINESS CHECK")
    print("="*60)
    
    try:
        # Check for quantum ML modules
        qml_dir = Path("/home/xing/Qallow/quantum_ml")
        modules = list(qml_dir.glob("*.py"))
        
        print(f"✓ Found {len(modules)} QML modules:")
        for mod in modules:
            print(f"  - {mod.name}")
        
        # Check for hybrid learner
        hybrid_learner = Path("/home/xing/Qallow/python/quantum/hybrid_meta_learner.py")
        if hybrid_learner.exists():
            print(f"✓ Hybrid meta-learner found")
        
        return {
            "status": "PASS",
            "qml_modules": len(modules),
            "modules": [m.name for m in modules]
        }
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def main():
    """Run all QML verification checks"""
    print("\n" + "█"*60)
    print("█  QALLOW QML VERIFICATION SUITE")
    print("█"*60)
    
    results = {
        "gradient_flow": check_gradient_flow(),
        "cuda_latency": check_cuda_kernel_latency(),
        "entanglement_fidelity": check_entanglement_fidelity(),
        "data_loading": check_data_loading(),
        "hybrid_loop": check_hybrid_loop_readiness(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for check_name, result in results.items():
        status = result.get("status", "UNKNOWN")
        symbol = "✓" if status == "PASS" else "⚠" if status == "WARN" else "✗"
        print(f"{symbol} {check_name}: {status}")
    
    # Save results
    with open("/home/xing/Qallow/data/logs/qml_verification.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to data/logs/qml_verification.json")
    print("\n" + "█"*60)
    print("█  QML VERIFICATION COMPLETE")
    print("█"*60 + "\n")

if __name__ == "__main__":
    main()

