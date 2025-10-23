"""
ALG Run Module
Executes all quantum algorithms + QAOA + SPSA optimizer
Integrates unified quantum framework with QAOA parameter tuning
"""

import json
import os
import sys
from datetime import datetime


def get_export_path(args):
    """Extract export path from arguments"""
    for arg in args:
        if arg.startswith("--export="):
            return arg.split("=", 1)[1]
    return "/var/qallow/quantum_report.json"


def get_quick_mode(args):
    """Check if --quick flag is set"""
    return "--quick" in args


def create_default_config():
    """Create default Ising model configuration"""
    config_path = "/var/qallow/ising_spec.json"
    csv_path = "/var/qallow/ring8.csv"

    # Create default 8-node ring configuration
    config = {
        "N": 8,
        "p": 2,
        "csv_j": csv_path,
        "alpha_min": 0.001,
        "alpha_max": 0.01,
        "spsa_iterations": 50,
        "spsa_a": 0.1,
        "spsa_c": 0.1
    }

    os.makedirs("/var/qallow", exist_ok=True)

    # Write config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[ALG RUN] Created default config: {config_path}")

    # Write ring topology (8-node ring with unit coupling)
    with open(csv_path, "w") as f:
        f.write("# node_i,node_j,coupling_J\n")
        for i in range(8):
            j = (i + 1) % 8
            f.write(f"{i},{j},1.0\n")
    print(f"[ALG RUN] Created default topology: {csv_path}")

    return config_path


def run_unified_framework(quick_mode=False):
    """Run all quantum algorithms from unified framework"""
    print("\n" + "="*70)
    print("PHASE 1: UNIFIED QUANTUM ALGORITHMS")
    print("="*70 + "\n")

    try:
        # Add quantum_algorithms to path
        sys.path.insert(0, "/root/Qallow")
        from quantum_algorithms.unified_quantum_framework import QuantumAlgorithmFramework

        framework = QuantumAlgorithmFramework(verbose=True)

        # Run algorithms
        results = {}

        print("[ALG] Running Hello Quantum...")
        results['hello_quantum'] = framework.run_hello_quantum()

        print("\n[ALG] Running Bell State...")
        results['bell_state'] = framework.run_bell_state()

        print("\n[ALG] Running Deutsch Algorithm...")
        results['deutsch'] = framework.run_deutsch()

        if not quick_mode:
            print("\n[ALG] Running Grover's Algorithm...")
            results['grover'] = framework.run_grover()

            print("\n[ALG] Running Shor's Algorithm...")
            results['shor'] = framework.run_shor()

            print("\n[ALG] Running VQE...")
            results['vqe'] = framework.run_vqe()

        return results

    except Exception as e:
        print(f"[ALG ERROR] Failed to run unified framework: {e}")
        import traceback
        traceback.print_exc()
        return {}


def run_qaoa_optimizer(config_path):
    """Run QAOA + SPSA optimizer"""
    print("\n" + "="*70)
    print("PHASE 2: QAOA + SPSA OPTIMIZER")
    print("="*70 + "\n")

    try:
        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"[ALG] Loaded config: N={config['N']}, p={config['p']}")

        # Import and run the optimizer
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from qaoa_spsa import run_qaoa_optimizer as qaoa_run

        print("[ALG] Starting QAOA + SPSA optimization...")
        result = qaoa_run(config_path)

        print(f"[ALG] ✓ Optimization complete")
        print(f"[ALG] Energy: {result['energy']:.6f}")
        print(f"[ALG] Alpha_eff: {result['alpha_eff']:.6f}")

        return result

    except Exception as e:
        print(f"[ALG ERROR] QAOA optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def generate_report(framework_results, qaoa_result, export_path):
    """Generate comprehensive report"""
    print("\n" + "="*70)
    print("GENERATING REPORT")
    print("="*70 + "\n")

    os.makedirs("/var/qallow", exist_ok=True)

    # Prepare report data
    report = {
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "quantum_algorithms": {},
        "qaoa_optimizer": qaoa_result,
        "summary": {}
    }

    # Add framework results
    for algo_name, result in framework_results.items():
        if hasattr(result, '__dict__'):
            report["quantum_algorithms"][algo_name] = result.__dict__
        else:
            report["quantum_algorithms"][algo_name] = result

    # Calculate summary statistics
    successful = sum(1 for r in framework_results.values()
                    if hasattr(r, 'success') and r.success)
    total = len(framework_results)

    report["summary"] = {
        "total_algorithms": total,
        "successful": successful,
        "success_rate": f"{100 * successful / total:.1f}%" if total > 0 else "0%",
        "qaoa_energy": qaoa_result.get("energy", 0),
        "qaoa_alpha_eff": qaoa_result.get("alpha_eff", 0)
    }

    # Write JSON report
    with open(export_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[ALG] ✓ Report written to: {export_path}")

    # Write markdown summary
    md_path = export_path.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write("# Quantum Algorithm Report\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        f.write(f"## Summary\n")
        f.write(f"- Total Algorithms: {total}\n")
        f.write(f"- Successful: {successful}/{total}\n")
        f.write(f"- Success Rate: {report['summary']['success_rate']}\n\n")
        f.write(f"## QAOA Optimizer Results\n")
        f.write(f"- Energy: {qaoa_result.get('energy', 'N/A')}\n")
        f.write(f"- Alpha_eff: {qaoa_result.get('alpha_eff', 'N/A')}\n")
        f.write(f"- Iterations: {qaoa_result.get('iterations', 'N/A')}\n\n")
        f.write(f"## Algorithm Results\n")
        for algo_name, result in framework_results.items():
            success = "✓" if (hasattr(result, 'success') and result.success) else "✗"
            f.write(f"- {algo_name}: {success}\n")

    print(f"[ALG] ✓ Summary written to: {md_path}")


def exec(args=None):
    """Execute all quantum algorithms + QAOA optimizer"""
    if args is None:
        args = []

    print("\n" + "="*70)
    print("ALG RUN - Unified Quantum Algorithm Framework")
    print("="*70 + "\n")

    export_path = get_export_path(args)
    quick_mode = get_quick_mode(args)

    print(f"[ALG] Export path: {export_path}")
    print(f"[ALG] Quick mode: {quick_mode}")

    # Create default config if missing
    config_path = "/var/qallow/ising_spec.json"
    if not os.path.exists(config_path):
        config_path = create_default_config()

    # Run unified framework
    framework_results = run_unified_framework(quick_mode)

    # Run QAOA optimizer
    qaoa_result = run_qaoa_optimizer(config_path)

    # Generate report
    generate_report(framework_results, qaoa_result, export_path)

    print("\n" + "="*70)
    print("RUN COMPLETE")
    print("="*70 + "\n")

