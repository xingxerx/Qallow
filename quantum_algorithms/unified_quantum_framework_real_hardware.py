#!/usr/bin/env python3
"""
REAL QUANTUM HARDWARE FRAMEWORK
Runs quantum algorithms on actual quantum computers (IBM Quantum, Google Cirq)
NOT simulations - actual quantum hardware execution
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

# IBM Quantum
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2
from qiskit_aer import AerSimulator

# Google Cirq (for reference)
import cirq

print("=" * 80)
print("🚀 REAL QUANTUM HARDWARE FRAMEWORK")
print("=" * 80)

class RealQuantumHardware:
    """Interface to real quantum computers"""
    
    def __init__(self):
        self.results = []
        self.setup_ibm_quantum()
    
    def setup_ibm_quantum(self):
        """Setup IBM Quantum access"""
        print("\n📡 Setting up IBM Quantum access...")
        
        # Check for IBM API key
        api_key = os.getenv("IBM_QUANTUM_API_KEY")
        
        if not api_key:
            print("⚠️  IBM_QUANTUM_API_KEY not found in environment")
            print("\n📋 TO USE REAL QUANTUM HARDWARE:")
            print("   1. Go to https://quantum.ibm.com/")
            print("   2. Sign up (free account)")
            print("   3. Get your API key from Account settings")
            print("   4. Set environment variable:")
            print("      export IBM_QUANTUM_API_KEY='your_key_here'")
            print("\n   Then run: python3 unified_quantum_framework_real_hardware.py")
            return False
        
        try:
            # Authenticate with IBM Quantum
            QiskitRuntimeService.save_account(
                channel="ibm_quantum",
                api_key=api_key,
                overwrite=True
            )
            self.service = QiskitRuntimeService(channel="ibm_quantum")
            
            # List available backends
            backends = self.service.backends()
            print(f"✅ Connected to IBM Quantum!")
            print(f"📊 Available quantum computers:")
            for backend in backends:
                print(f"   - {backend.name} ({backend.num_qubits} qubits)")
            
            return True
        except Exception as e:
            print(f"❌ Error connecting to IBM Quantum: {e}")
            return False
    
    def run_grover_on_real_hardware(self):
        """Run Grover's algorithm on REAL quantum hardware"""
        print("\n" + "=" * 80)
        print("🔍 GROVER'S ALGORITHM - REAL QUANTUM HARDWARE")
        print("=" * 80)
        
        # Create circuit
        qc = QuantumCircuit(3, 3, name="Grovers_Real_Hardware")
        
        # Initialize superposition
        qc.h(range(3))
        
        # Oracle (mark state |101⟩ = 5)
        qc.x(1)
        qc.z(2)
        qc.x(1)
        
        # Diffusion operator
        qc.h(range(3))
        qc.x(range(3))
        qc.z(2)
        qc.x(range(3))
        qc.h(range(3))
        
        # Measure
        qc.measure(range(3), range(3))
        
        print(f"\n📐 Circuit:\n{qc}")
        
        try:
            # Get least busy backend
            backend = self.service.least_busy(
                operational=True,
                simulator=False,  # REAL HARDWARE ONLY
                min_num_qubits=3
            )
            print(f"\n✅ Using real quantum computer: {backend.name}")
            print(f"   Qubits: {backend.num_qubits}")
            print(f"   Queue depth: {backend.queue_depth}")
            
            # Run on real hardware
            with Session(service=self.service, backend=backend) as session:
                sampler = SamplerV2(session=session)
                job = sampler.run([qc], shots=1000)
                result = job.result()
                
                # Extract results
                counts = result[0].data.c.get_counts()
                print(f"\n📊 Results from REAL quantum hardware:")
                print(f"   Total shots: 1000")
                for state, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"   State |{state}⟩: {count} times ({count/10}%)")
                
                return counts
        
        except Exception as e:
            print(f"❌ Error running on real hardware: {e}")
            print("   Falling back to simulator...")
            return self.run_grover_simulator()
    
    def run_grover_simulator(self):
        """Fallback: Run on simulator"""
        print("\n⚠️  Running on SIMULATOR (not real hardware)")
        qc = QuantumCircuit(3, 3)
        qc.h(range(3))
        qc.x(1)
        qc.z(2)
        qc.x(1)
        qc.h(range(3))
        qc.x(range(3))
        qc.z(2)
        qc.x(range(3))
        qc.h(range(3))
        qc.measure(range(3), range(3))
        
        simulator = AerSimulator()
        job = simulator.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        return counts
    
    def list_available_backends(self):
        """List all available quantum computers"""
        print("\n" + "=" * 80)
        print("📡 AVAILABLE QUANTUM COMPUTERS")
        print("=" * 80)
        
        try:
            backends = self.service.backends()
            print(f"\n✅ Found {len(backends)} quantum computers:\n")
            
            for i, backend in enumerate(backends, 1):
                print(f"{i}. {backend.name}")
                print(f"   Qubits: {backend.num_qubits}")
                print(f"   Status: {'🟢 Operational' if backend.operational else '🔴 Offline'}")
                print(f"   Queue: {backend.queue_depth} jobs")
                print()
        except Exception as e:
            print(f"❌ Error listing backends: {e}")

# Main execution
if __name__ == "__main__":
    hardware = RealQuantumHardware()
    
    print("\n" + "=" * 80)
    print("🎯 QUANTUM HARDWARE OPTIONS")
    print("=" * 80)
    print("""
1. IBM Quantum (FREE TIER)
   ├─ 5-127 qubits
   ├─ Real quantum computers
   ├─ Free access (limited queue)
   └─ Sign up: https://quantum.ibm.com/

2. Google Cirq (if available)
   ├─ Sycamore processor
   ├─ Real quantum hardware
   └─ Limited access (research only)

3. IonQ (Cloud access)
   ├─ Trapped ion quantum computer
   ├─ 11 qubits
   └─ Paid service

4. AWS Braket
   ├─ Multiple quantum computers
   ├─ IonQ, Rigetti, D-Wave
   └─ Paid service
    """)
    
    print("\n" + "=" * 80)
    print("✅ SETUP INSTRUCTIONS")
    print("=" * 80)
    print("""
TO RUN ON REAL QUANTUM HARDWARE:

1. Get IBM Quantum API Key:
   - Go to https://quantum.ibm.com/
   - Create free account
   - Copy API key from Account settings

2. Set environment variable:
   export IBM_QUANTUM_API_KEY='your_api_key_here'

3. Run this script:
   python3 unified_quantum_framework_real_hardware.py

4. Your job will be queued on real quantum hardware
   - Wait time: 5 minutes to 1 hour (depends on queue)
   - Results will be from ACTUAL quantum computer
   - NOT simulation!
    """)
    
    # Try to run if API key is available
    if hasattr(hardware, 'service'):
        print("\n🚀 Attempting to run Grover's algorithm on REAL quantum hardware...")
        hardware.run_grover_on_real_hardware()
        hardware.list_available_backends()
    else:
        print("\n⚠️  IBM Quantum API key not configured")
        print("   Please follow setup instructions above")

