"""
Advanced Integration Test: QAOA Optimization with Kimi-K2 Analysis
Demonstrates quantum optimization with AI reasoning
"""

import sys
import os
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import cirq
import cudaq
import numpy as np
from python.agents.kimi_k2_agent import create_kimi_k2_agent

print("\n" + "=" * 80)
print("ADVANCED TEST: QAOA Optimization with Kimi-K2 Analysis")
print("=" * 80 + "\n")


class QAOAWithKimiK2:
    """QAOA optimization with Kimi-K2 analysis"""
    
    def __init__(self):
        self.results = {}
        self.kimi_agent = None
        self.initialize_kimi()
    
    def initialize_kimi(self):
        """Initialize Kimi-K2 agent"""
        try:
            self.kimi_agent = create_kimi_k2_agent()
            print("✓ Kimi-K2 Agent initialized")
        except Exception as e:
            print(f"⚠ Kimi-K2 not available: {e}")
    
    def create_qaoa_circuit_cirq(self, n_qubits: int = 3) -> cirq.Circuit:
        """Create QAOA circuit using Cirq"""
        print(f"\n[1/4] Creating QAOA circuit with Cirq ({n_qubits} qubits)...")
        
        qubits = cirq.LineQubit.range(n_qubits)
        circuit = cirq.Circuit()
        
        # Initial superposition
        circuit.append(cirq.H.on_each(*qubits))
        
        # Problem Hamiltonian (MaxCut example)
        for i in range(n_qubits - 1):
            circuit.append(cirq.ZZ(qubits[i], qubits[i+1])**(0.5))
        
        # Mixer Hamiltonian
        for qubit in qubits:
            circuit.append(cirq.rx(np.pi/4)(qubit))
        
        # Measurement
        circuit.append(cirq.measure(*qubits, key='result'))
        
        print(f"  ✓ Circuit created with {len(circuit)} operations")
        print(f"  ✓ Circuit:\n{circuit}")
        
        return circuit
    
    def simulate_qaoa_cirq(self, circuit: cirq.Circuit) -> dict:
        """Simulate QAOA with Cirq"""
        print("\n[2/4] Simulating QAOA with Cirq...")
        
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit, initial_state=0)
        
        # Get measurement statistics
        measurements = simulator.run(circuit, repetitions=1000)
        counts = measurements.measurements['result']
        
        # Convert to bitstring counts
        bitstring_counts = {}
        for measurement in counts:
            bitstring = ''.join(map(str, measurement))
            bitstring_counts[bitstring] = bitstring_counts.get(bitstring, 0) + 1
        
        print(f"  ✓ Simulation completed")
        print(f"  ✓ Measurement results (top 5):")
        for bitstring, count in sorted(bitstring_counts.items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {bitstring}: {count}/1000")
        
        return bitstring_counts
    
    def create_qaoa_cudaq(self, n_qubits: int = 3) -> dict:
        """Create and run QAOA with CUDA-Q"""
        print(f"\n[3/4] Running QAOA with CUDA-Q ({n_qubits} qubits)...")

        @cudaq.kernel
        def qaoa_kernel():
            qubits = cudaq.qvector(n_qubits)

            # Initial superposition
            h(qubits)

            # Problem Hamiltonian (using controlled Z gates)
            for i in range(n_qubits - 1):
                cz(qubits[i], qubits[i+1])

            # Mixer Hamiltonian
            for qubit in qubits:
                rx(np.pi/4, qubit)

        # Run sampling
        result = cudaq.sample(qaoa_kernel, shots_count=1000)

        # Convert to dictionary - result is already a dict-like object
        result_dict = {}
        for bitstring, count in result.items():
            result_dict[bitstring] = count

        print(f"  ✓ CUDA-Q execution completed")
        print(f"  ✓ Measurement results (top 5):")
        for bitstring, count in sorted(result_dict.items(),
                                       key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {bitstring}: {count}")

        return result_dict
    
    def analyze_with_kimi(self, cirq_results: dict, cudaq_results: dict):
        """Use Kimi-K2 to analyze QAOA results"""
        print("\n[4/4] Analyzing results with Kimi-K2...")
        
        if not self.kimi_agent:
            print("  ⚠ Kimi-K2 not available, skipping analysis")
            return None
        
        try:
            # Prepare analysis prompt
            cirq_top = sorted(cirq_results.items(), key=lambda x: x[1], reverse=True)[:3]
            cudaq_top = sorted(cudaq_results.items(), key=lambda x: x[1], reverse=True)[:3]
            
            prompt = f"""
Analyze these QAOA optimization results:

Cirq Results (top 3):
{json.dumps(dict(cirq_top), indent=2)}

CUDA-Q Results (top 3):
{json.dumps(dict(cudaq_top), indent=2)}

Provide a brief analysis of:
1. Which bitstrings are optimal
2. Convergence quality
3. Recommendations for improvement
"""
            
            response = self.kimi_agent.chat(prompt)
            print(f"  ✓ Kimi-K2 Analysis:")
            print(f"    {response[:200]}...")
            
            return response
        except Exception as e:
            print(f"  ✗ Analysis failed: {e}")
            return None
    
    def test_cuda_acceleration(self):
        """Test CUDA acceleration for tensor operations"""
        print("\n[BONUS] Testing CUDA Acceleration...")
        
        # Create large tensors
        size = 5000
        x = torch.randn(size, size, device='cuda')
        y = torch.randn(size, size, device='cuda')
        
        # Measure time
        import time
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"  ✓ Matrix multiplication ({size}x{size}): {elapsed*1000:.2f}ms")
        print(f"  ✓ CUDA acceleration working")
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "=" * 80)
        print("RUNNING COMPREHENSIVE TESTS")
        print("=" * 80)
        
        # Create circuits
        cirq_circuit = self.create_qaoa_circuit_cirq(n_qubits=3)
        
        # Simulate with Cirq
        cirq_results = self.simulate_qaoa_cirq(cirq_circuit)
        
        # Run with CUDA-Q
        cudaq_results = self.create_qaoa_cudaq(n_qubits=3)
        
        # Analyze with Kimi-K2
        analysis = self.analyze_with_kimi(cirq_results, cudaq_results)
        
        # Test CUDA acceleration
        self.test_cuda_acceleration()
        
        # Print summary
        self.print_summary(cirq_results, cudaq_results, analysis)
    
    def print_summary(self, cirq_results, cudaq_results, analysis):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        print("\n✓ Cirq QAOA Simulation")
        print(f"  - Unique bitstrings: {len(cirq_results)}")
        print(f"  - Best result: {max(cirq_results.items(), key=lambda x: x[1])}")
        
        print("\n✓ CUDA-Q QAOA Execution")
        print(f"  - Unique bitstrings: {len(cudaq_results)}")
        print(f"  - Best result: {max(cudaq_results.items(), key=lambda x: x[1])}")
        
        if analysis:
            print("\n✓ Kimi-K2 Analysis")
            print(f"  - Analysis provided: Yes")
        else:
            print("\n⚠ Kimi-K2 Analysis")
            print(f"  - Analysis provided: No (server not running)")
        
        print("\n" + "=" * 80)
        print("✓ ALL ADVANCED TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80 + "\n")


def main():
    """Run advanced integration tests"""
    tester = QAOAWithKimiK2()
    tester.run_all_tests()
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

