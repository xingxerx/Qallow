"""
ALG Run Module
Executes QAOA + SPSA quantum optimizer
"""

import subprocess
import json
import os
import sys


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


def get_config_path(args):
    """Extract config path from arguments"""
    for arg in args:
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return "/var/qallow/ising_spec.json"


def exec(args=None):
    """Execute quantum optimizer"""
    if args is None:
        args = []
    
    print("\n" + "="*70)
    print("ALG RUN - Quantum Optimizer (QAOA + SPSA)")
    print("="*70 + "\n")
    
    config_path = get_config_path(args)
    output_path = "/var/qallow/qaoa_gain.json"
    
    # Create default config if missing
    if not os.path.exists(config_path):
        print(f"[ALG RUN] Config not found: {config_path}")
        config_path = create_default_config()
    
    print(f"[ALG RUN] Using config: {config_path}")
    print(f"[ALG RUN] Output: {output_path}")
    
    # Load config
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"[ALG RUN] Loaded config: N={config['N']}, p={config['p']}")
    except Exception as e:
        print(f"[ALG RUN] ERROR: Failed to load config: {e}")
        sys.exit(1)
    
    # Run QAOA optimizer
    print("\n[ALG RUN] Starting quantum optimization...")
    print("[ALG RUN] This may take 1-5 minutes depending on system size...\n")
    
    try:
        # Import and run the optimizer
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from qaoa_spsa import run_qaoa_optimizer
        
        result = run_qaoa_optimizer(config_path)
        
        # Write output
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\n[ALG RUN] ✓ Optimization complete")
        print(f"[ALG RUN] Energy: {result['energy']:.6f}")
        print(f"[ALG RUN] Alpha_eff: {result['alpha_eff']:.6f}")
        print(f"[ALG RUN] Output: {output_path}")
        
    except Exception as e:
        print(f"[ALG RUN] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*70)
    print("RUN COMPLETE")
    print("="*70 + "\n")

