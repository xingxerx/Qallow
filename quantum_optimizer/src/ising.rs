//! Basic data structures for Ising cost Hamiltonians used by QAOA/VQE routines.

/// Represents a pair-wise Ising coupling `J_{ij}` between qubits `i` and `j`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coupling {
    /// First qubit index
    pub i: usize,
    /// Second qubit index
    pub j: usize,
    /// Coupling strength (positive for ferromagnetic, negative for antiferro)
    pub weight: f64,
}

/// Simple Ising Hamiltonian `H = Σ_i h_i Z_i + Σ_{i<j} J_{ij} Z_i Z_j`.
#[derive(Debug, Clone, PartialEq)]
pub struct IsingProblem {
    pub num_qubits: usize,
    pub biases: Vec<f64>,
    pub couplings: Vec<Coupling>,
}

impl IsingProblem {
    /// Construct a new problem, validating that indices fall within range.
    pub fn new(num_qubits: usize, biases: Vec<f64>, couplings: Vec<Coupling>) -> Self {
        assert_eq!(biases.len(), num_qubits, "Bias vector must match number of qubits");
        for coupling in &couplings {
            assert!(coupling.i < num_qubits && coupling.j < num_qubits,
                "Coupling indices out of range");
            assert!(coupling.i != coupling.j, "Self-couplings are not supported");
        }
        Self {
            num_qubits,
            biases,
            couplings,
        }
    }

    /// Energy of a basis configuration described by spins `±1` (length `num_qubits`).
    pub fn energy_from_spins(&self, spins: &[f64]) -> f64 {
        debug_assert_eq!(spins.len(), self.num_qubits);
        let field_term: f64 = self
            .biases
            .iter()
            .zip(spins.iter())
            .map(|(h, s)| h * s)
            .sum();
        let coupling_term: f64 = self
            .couplings
            .iter()
            .map(|c| c.weight * spins[c.i] * spins[c.j])
            .sum();
        field_term + coupling_term
    }

    /// Energy of a computational basis state labelled by the integer `basis`.
    pub fn energy_from_basis(&self, basis: u64) -> f64 {
        let mut spins = vec![1.0; self.num_qubits];
        for q in 0..self.num_qubits {
            let bit = (basis >> q) & 1;
            spins[q] = if bit == 0 { 1.0 } else { -1.0 };
        }
        self.energy_from_spins(&spins)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_energy() {
        let problem = IsingProblem::new(
            2,
            vec![1.0, -0.5],
            vec![Coupling { i: 0, j: 1, weight: 0.25 }],
        );
        // |00⟩ => spins [+1, +1]
        let e00 = problem.energy_from_basis(0);
        // |01⟩ => [+1, -1]
        let e01 = problem.energy_from_basis(1);
        assert!(e00 < e01);
    }
}
