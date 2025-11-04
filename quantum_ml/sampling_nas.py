# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # quantum_ml/sampling_nas.py
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] class QuantumNASExplorer:
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     def __init__(self, qallow_binary="/root/Qallow/build/qallow"):
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         self.binary = qallow_binary
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         print(f"Using binary: {self.binary}") 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         print(f"Checking binary exists: {os.path.exists(self.binary)}")
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         if not os.path.exists(self.binary):
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]             raise FileNotFoundError(f"Qallow binary not found at {self.binary}")
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         print("Binary exists: OK")
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         print("QuantumNASExplorer initialized")
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         print("=========================================")
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     def generate_architectures(self, n_samples=100):
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         print(f"Running Phase 11 with {n_samples} samples...")
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]         # Call Phase 11 to get quantum states
        cmd = [self.binary, "phase", "11", "--ticks", str(n_samples)]
        print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"Return code: {result.returncode}")

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            raise RuntimeError(f"Phase 11 failed with return code {result.returncode}")

        # Parse JSON from stdout, skipping debug lines
        lines = result.stdout.strip().split('\n')
        json_str = '\n'.join([line for line in lines if not line.startswith('[')])

        try:
            quantum_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON. Output was:\n{result.stdout}")
            raise

        return [self._decode_architecture(state) for state in quantum_data["states"]]
    
    def _decode_architecture(self, state):
        # Convert quantum state to architecture spec
        return {
            'layer_type': 'conv' if state > 0 else 'dense',
            'neurons': abs(state) * 64,
            'activation': 'relu'
        }

if __name__ == "__main__":
    explorer = QuantumNASExplorer()
    architectures = explorer.generate_architectures(10)
    print(f"\nGenerated {len(architectures)} architectures:")
    for i, arch in enumerate(architectures):
        print(f"  {i+1}: {arch}")
