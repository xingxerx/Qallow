#!/usr/bin/env python3
"""
Quantum Optimization Algorithms
QAOA (Quantum Approximate Optimization Algorithm) for solving optimization problems
"""

import cirq
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class OptimizationResult:
    """Result from optimization algorithm"""
    algorithm: str
    problem: str
    timestamp: str
    best_solution: int
    best_energy: float
    all_energies: List[float]
    circuit: str
    metrics: Dict[str, Any]


class QuantumMaxCut:
    """
    QAOA for MaxCut Problem
    Find the maximum cut in a graph
    """
    
    def __init__(self, graph_edges: List[Tuple[int, int]], n_qubits: int = 4, p: int = 1):
        """
        Args:
            graph_edges: List of edges (u, v)
            n_qubits: Number of qubits
            p: QAOA depth (number of layers)
        """
        self.graph_edges = graph_edges
        self.n_qubits = n_qubits
        self.p = p
        self.simulator = cirq.Simulator()
    
    def build_circuit(self, gamma: float, beta: float) -> cirq.Circuit:
        """Build QAOA circuit for MaxCut"""
        qubits = cirq.LineQubit.range(self.n_qubits)
        circuit = cirq.Circuit()
        
        # Initial superposition
        for q in qubits:
            circuit.append(cirq.H(q))
        
        # QAOA layers
        for layer in range(self.p):
            # Cost Hamiltonian: e^{-i*gamma*H_C}
            for u, v in self.graph_edges:
                if u < self.n_qubits and v < self.n_qubits:
                    circuit.append(cirq.ZZ(qubits[u], qubits[v]) ** (gamma / np.pi))
            
            # Mixer Hamiltonian: e^{-i*beta*H_M}
            for q in qubits:
                circuit.append(cirq.rx(2 * beta)(q))
        
        # Measure
        circuit.append(cirq.measure(*qubits, key='result'))
        return circuit
    
    def evaluate_cut(self, bitstring: int) -> int:
        """Evaluate cut size for a bitstring"""
        cut_size = 0
        for u, v in self.graph_edges:
            bit_u = (bitstring >> u) & 1
            bit_v = (bitstring >> v) & 1
            if bit_u != bit_v:
                cut_size += 1
        return cut_size
    
    def run(self, gamma: float = 0.5, beta: float = 0.5, shots: int = 1000) -> OptimizationResult:
        """Run QAOA for MaxCut"""
        print(f"\n{'='*70}")
        print(f"QAOA - MaxCut Problem")
        print(f"{'='*70}")
        print(f"Graph edges: {self.graph_edges}")
        print(f"Qubits: {self.n_qubits}, Depth: {self.p}")
        print(f"Parameters: gamma={gamma:.4f}, beta={beta:.4f}")
        
        circuit = self.build_circuit(gamma, beta)
        print(f"\nCircuit:\n{circuit}")
        
        # Run simulation
        result = self.simulator.run(circuit, repetitions=shots)
        histogram = result.histogram(key='result')
        
        # Evaluate all solutions
        energies = []
        for bitstring, count in histogram.items():
            cut_size = self.evaluate_cut(int(bitstring))
            energies.extend([cut_size] * count)
        
        best_solution = max(histogram.items(), key=lambda x: self.evaluate_cut(int(x[0])))[0]
        best_energy = self.evaluate_cut(int(best_solution))
        
        print(f"\nBest cut size: {best_energy}")
        print(f"Best solution: {best_solution:0{self.n_qubits}b}")
        
        metrics = {
            "graph_edges": len(self.graph_edges),
            "best_cut_size": best_energy,
            "average_cut_size": np.mean(energies),
            "max_possible_cut": len(self.graph_edges),
            "approximation_ratio": best_energy / len(self.graph_edges)
        }
        
        return OptimizationResult(
            algorithm="QAOA-MaxCut",
            problem=f"MaxCut on {self.n_qubits}-qubit graph",
            timestamp=datetime.now().isoformat(),
            best_solution=int(best_solution),
            best_energy=float(best_energy),
            all_energies=energies,
            circuit=str(circuit),
            metrics=metrics
        )


class QuantumTravelingSalesman:
    """
    QAOA for Traveling Salesman Problem (TSP)
    Find shortest route visiting all cities
    """
    
    def __init__(self, distance_matrix: np.ndarray, n_qubits: int = 4, p: int = 1):
        """
        Args:
            distance_matrix: Distance between cities
            n_qubits: Number of qubits
            p: QAOA depth
        """
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        self.n_qubits = n_qubits
        self.p = p
        self.simulator = cirq.Simulator()
    
    def build_circuit(self, gamma: float, beta: float) -> cirq.Circuit:
        """Build QAOA circuit for TSP"""
        qubits = cirq.LineQubit.range(self.n_qubits)
        circuit = cirq.Circuit()
        
        # Initial superposition
        for q in qubits:
            circuit.append(cirq.H(q))
        
        # QAOA layers
        for layer in range(self.p):
            # Cost Hamiltonian
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    if i < self.n_cities and j < self.n_cities:
                        cost = self.distance_matrix[i][j]
                        circuit.append(cirq.ZZ(qubits[i], qubits[j]) ** (gamma * cost / np.pi))
            
            # Mixer
            for q in qubits:
                circuit.append(cirq.rx(2 * beta)(q))
        
        circuit.append(cirq.measure(*qubits, key='result'))
        return circuit
    
    def evaluate_tour(self, bitstring: int) -> float:
        """Evaluate tour length"""
        tour = []
        for i in range(self.n_cities):
            if (bitstring >> i) & 1:
                tour.append(i)
        
        if len(tour) < 2:
            return float('inf')
        
        distance = 0
        for i in range(len(tour)):
            distance += self.distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
        return distance
    
    def run(self, gamma: float = 0.5, beta: float = 0.5, shots: int = 1000) -> OptimizationResult:
        """Run QAOA for TSP"""
        print(f"\n{'='*70}")
        print(f"QAOA - Traveling Salesman Problem")
        print(f"{'='*70}")
        print(f"Cities: {self.n_cities}, Qubits: {self.n_qubits}")
        
        circuit = self.build_circuit(gamma, beta)
        result = self.simulator.run(circuit, repetitions=shots)
        histogram = result.histogram(key='result')
        
        # Evaluate solutions
        distances = []
        for bitstring, count in histogram.items():
            dist = self.evaluate_tour(int(bitstring))
            if dist != float('inf'):
                distances.extend([dist] * count)
        
        best_solution = min(histogram.items(), key=lambda x: self.evaluate_tour(int(x[0])))[0]
        best_distance = self.evaluate_tour(int(best_solution))
        
        print(f"\nBest tour distance: {best_distance:.2f}")
        print(f"Average distance: {np.mean(distances):.2f}")
        
        metrics = {
            "n_cities": self.n_cities,
            "best_distance": float(best_distance),
            "average_distance": float(np.mean(distances)) if distances else 0,
            "min_distance": float(np.min(distances)) if distances else 0,
            "max_distance": float(np.max(distances)) if distances else 0
        }
        
        return OptimizationResult(
            algorithm="QAOA-TSP",
            problem=f"TSP with {self.n_cities} cities",
            timestamp=datetime.now().isoformat(),
            best_solution=int(best_solution),
            best_energy=best_distance,
            all_energies=distances,
            circuit=str(circuit),
            metrics=metrics
        )


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("█" + " QUANTUM OPTIMIZATION ALGORITHMS ".center(68) + "█")
    print("█"*70)
    
    # Example 1: MaxCut
    print("\n" + "="*70)
    print("EXAMPLE 1: MaxCut Problem")
    print("="*70)
    
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    maxcut = QuantumMaxCut(graph_edges=edges, n_qubits=4, p=2)
    result1 = maxcut.run(gamma=0.5, beta=0.5)
    
    print(f"\nMetrics: {result1.metrics}")
    
    # Example 2: TSP
    print("\n" + "="*70)
    print("EXAMPLE 2: Traveling Salesman Problem")
    print("="*70)
    
    # 4 cities distance matrix
    distances = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])
    
    tsp = QuantumTravelingSalesman(distance_matrix=distances, n_qubits=4, p=2)
    result2 = tsp.run(gamma=0.5, beta=0.5)
    
    print(f"\nMetrics: {result2.metrics}")
    
    print("\n" + "█"*70)
    print("█" + " OPTIMIZATION COMPLETE ".center(68) + "█")
    print("█"*70)

