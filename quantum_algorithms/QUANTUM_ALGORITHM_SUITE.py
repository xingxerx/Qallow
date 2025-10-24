#!/usr/bin/env python3
"""
QALLOW QUANTUM ALGORITHM SUITE
Complete collection of quantum algorithms for the Qallow engine
"""

import sys
import json
from datetime import datetime
from pathlib import Path

# Import all algorithm modules
sys.path.insert(0, str(Path(__file__).parent / "algorithms"))

from unified_quantum_framework import QuantumAlgorithmFramework
from my_quantum_search import QuantumDatabaseSearch
from quantum_optimization import QuantumMaxCut, QuantumTravelingSalesman
from quantum_ml import QuantumClassifier, QuantumClustering
from quantum_simulation import QuantumHarmonicOscillator, QuantumMolecularSimulation

import numpy as np


class QuantumAlgorithmSuite:
    """Master suite for all quantum algorithms"""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
    
    def run_all(self):
        """Run all quantum algorithms"""
        print("\n" + "█"*80)
        print("█" + " QALLOW QUANTUM ALGORITHM SUITE - COMPLETE EXECUTION ".center(78) + "█")
        print("█"*80)
        
        # 1. Unified Framework (6 algorithms)
        print("\n" + "="*80)
        print("PHASE 1: UNIFIED QUANTUM ALGORITHMS")
        print("="*80)
        self.run_unified_framework()
        
        # 2. Quantum Search
        print("\n" + "="*80)
        print("PHASE 2: QUANTUM SEARCH ALGORITHMS")
        print("="*80)
        self.run_quantum_search()
        
        # 3. Quantum Optimization
        print("\n" + "="*80)
        print("PHASE 3: QUANTUM OPTIMIZATION ALGORITHMS")
        print("="*80)
        self.run_quantum_optimization()
        
        # 4. Quantum Machine Learning
        print("\n" + "="*80)
        print("PHASE 4: QUANTUM MACHINE LEARNING")
        print("="*80)
        self.run_quantum_ml()
        
        # 5. Quantum Simulation
        print("\n" + "="*80)
        print("PHASE 5: QUANTUM SIMULATION")
        print("="*80)
        self.run_quantum_simulation()
        
        # Print summary
        self.print_summary()
    
    def run_unified_framework(self):
        """Run unified quantum framework"""
        framework = QuantumAlgorithmFramework(verbose=False)
        results = framework.run_all_algorithms()
        
        self.results["unified_framework"] = {
            "algorithms": len(results),
            "passed": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "algorithms_list": [r.algorithm for r in results]
        }
        
        print(f"✅ Ran {len(results)} algorithms")
        print(f"   Passed: {self.results['unified_framework']['passed']}")
        print(f"   Failed: {self.results['unified_framework']['failed']}")
    
    def run_quantum_search(self):
        """Run quantum search algorithms"""
        search = QuantumDatabaseSearch(database_size=16, target_value=11)
        result = search.run()

        # Extract best solution from result
        best_sol = result.get("best_solution") if isinstance(result, dict) else result.best_solution
        metrics = result.get("metrics") if isinstance(result, dict) else result.metrics

        self.results["quantum_search"] = {
            "algorithm": "Quantum Database Search",
            "database_size": 16,
            "target_value": 11,
            "best_solution": best_sol,
            "metrics": metrics
        }

        print(f"✅ Quantum Database Search")
        print(f"   Target: {best_sol}")
        print(f"   Metrics: {metrics}")
    
    def run_quantum_optimization(self):
        """Run quantum optimization algorithms"""
        # MaxCut
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        maxcut = QuantumMaxCut(graph_edges=edges, n_qubits=4, p=2)
        result1 = maxcut.run(gamma=0.5, beta=0.5)
        
        self.results["maxcut"] = {
            "algorithm": "QAOA-MaxCut",
            "best_cut_size": result1.best_energy,
            "approximation_ratio": result1.metrics["approximation_ratio"]
        }
        
        print(f"✅ QAOA-MaxCut")
        print(f"   Best cut: {result1.best_energy}")
        print(f"   Ratio: {result1.metrics['approximation_ratio']:.2%}")
        
        # TSP
        distances = np.array([
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ])
        
        tsp = QuantumTravelingSalesman(distance_matrix=distances, n_qubits=4, p=2)
        result2 = tsp.run(gamma=0.5, beta=0.5)
        
        self.results["tsp"] = {
            "algorithm": "QAOA-TSP",
            "best_distance": result2.best_energy,
            "n_cities": 4
        }
        
        print(f"✅ QAOA-TSP")
        print(f"   Best distance: {result2.best_energy:.2f}")
    
    def run_quantum_ml(self):
        """Run quantum machine learning algorithms"""
        # Classifier
        classifier = QuantumClassifier(n_qubits=3, n_layers=2)
        X_train = np.random.randn(10, 3) * 0.5
        y_train = np.random.randint(0, 2, 10)
        
        training_result = classifier.train(X_train, y_train, epochs=3)
        
        self.results["quantum_classifier"] = {
            "algorithm": "Quantum Classifier",
            "final_accuracy": training_result["final_accuracy"],
            "n_qubits": 3
        }
        
        print(f"✅ Quantum Classifier")
        print(f"   Accuracy: {training_result['final_accuracy']:.2%}")
        
        # Clustering
        clustering = QuantumClustering(n_qubits=3, n_clusters=2)
        X = np.random.randn(6, 3) * 0.3
        
        clustering_result = clustering.cluster(X)
        
        self.results["quantum_clustering"] = {
            "algorithm": "Quantum Clustering",
            "n_clusters": 2,
            "n_points": 6
        }
        
        print(f"✅ Quantum Clustering")
        print(f"   Clusters: {clustering_result['n_clusters']}")
    
    def run_quantum_simulation(self):
        """Run quantum simulation algorithms"""
        # Harmonic Oscillator
        oscillator = QuantumHarmonicOscillator(n_qubits=3, omega=1.0)
        result1 = oscillator.simulate(n_states=4)
        
        self.results["harmonic_oscillator"] = {
            "algorithm": "Quantum Harmonic Oscillator",
            "ground_state_energy": result1.ground_state_energy,
            "n_states": 4
        }
        
        print(f"✅ Quantum Harmonic Oscillator")
        print(f"   Ground state: {result1.ground_state_energy:.4f}")
        
        # Molecular Simulation
        molecule = QuantumMolecularSimulation(n_qubits=4, n_electrons=2)
        result2 = molecule.optimize(n_iterations=10)
        
        self.results["molecular_simulation"] = {
            "algorithm": "Quantum Molecular Simulation",
            "ground_state_energy": result2.ground_state_energy,
            "n_qubits": 4
        }
        
        print(f"✅ Quantum Molecular Simulation")
        print(f"   Ground state: {result2.ground_state_energy:.6f}")
    
    def print_summary(self):
        """Print execution summary"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "█"*80)
        print("█" + " EXECUTION SUMMARY ".center(78) + "█")
        print("█"*80)
        
        print(f"\n📊 RESULTS:")
        print(f"   Total algorithms: {sum(len(v) if isinstance(v, dict) else 1 for v in self.results.values())}")
        print(f"   Execution time: {elapsed:.2f}s")
        print(f"   Timestamp: {datetime.now().isoformat()}")
        
        print(f"\n🎯 ALGORITHM CATEGORIES:")
        for category, data in self.results.items():
            print(f"   ✅ {category}: {data}")
        
        # Export results
        self.export_results()
    
    def export_results(self):
        """Export results to JSON"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "execution_time": (datetime.now() - self.start_time).total_seconds(),
            "results": self.results
        }
        
        output_file = "quantum_algorithm_suite_results.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n💾 Results exported to: {output_file}")


if __name__ == "__main__":
    suite = QuantumAlgorithmSuite()
    suite.run_all()
    
    print("\n" + "█"*80)
    print("█" + " QUANTUM ALGORITHM SUITE COMPLETE ".center(78) + "█")
    print("█"*80)
    print("\n🚀 Ready to integrate with Qallow phases!")
    print("   Run: ./build/qallow phase 14 --ticks=500")
    print("   Monitor: cargo run")

