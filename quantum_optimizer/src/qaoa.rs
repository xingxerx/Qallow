//! Minimal state-vector QAOA simulator for small Ising instances.

use crate::ising::IsingProblem;
use num_complex::Complex64;

/// Parameters describing a depth-`p` QAOA circuit.
#[derive(Debug, Clone, PartialEq)]
pub struct QaoaParams {
    pub gammas: Vec<f64>,
    pub betas: Vec<f64>,
}

impl QaoaParams {
    pub fn new(gammas: Vec<f64>, betas: Vec<f64>) -> Result<Self, QaoaError> {
        if gammas.is_empty() || betas.is_empty() || gammas.len() != betas.len() {
            return Err(QaoaError::InvalidDepth);
        }
        Ok(Self { gammas, betas })
    }

    pub fn depth(&self) -> usize {
        self.gammas.len()
    }
}

/// Output of a QAOA evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct QaoaResult {
    pub expected_energy: f64,
    pub variance: f64,
    pub amplitudes: Vec<Complex64>,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum QaoaError {
    #[error("circuit depth must be > 0 and gamma/beta lengths must match")]
    InvalidDepth,
    #[error("problem size must be between 1 and 12 qubits for the reference simulator")]
    ProblemTooLarge,
}

/// Runs QAOA with the provided parameters, returning the energy expectation value.
pub fn run_qaoa(problem: &IsingProblem, params: &QaoaParams) -> Result<QaoaResult, QaoaError> {
    if problem.num_qubits == 0 || problem.num_qubits > 12 {
        return Err(QaoaError::ProblemTooLarge);
    }

    let p = params.depth();
    let dim = 1_usize << problem.num_qubits;
    let norm = (dim as f64).sqrt();
    let mut state = vec![Complex64::new(1.0 / norm, 0.0); dim];

    for layer in 0..p {
        apply_cost_layer(&mut state, problem, params.gammas[layer]);
        apply_mixer_layer(&mut state, problem.num_qubits, params.betas[layer]);
    }

    let mut expected = 0.0;
    let mut second_moment = 0.0;
    for (basis, amp) in state.iter().enumerate() {
        let prob = amp.norm_sqr();
        let energy = problem.energy_from_basis(basis as u64);
        expected += prob * energy;
        second_moment += prob * energy * energy;
    }
    let variance = (second_moment - expected * expected).max(0.0);

    Ok(QaoaResult {
        expected_energy: expected,
        variance,
        amplitudes: state,
    })
}

/// Naïve grid-search heuristic exploring `samples` equispaced angles in `[0, π]`.
pub fn grid_search_qaoa(
    problem: &IsingProblem,
    depth: usize,
    samples: usize,
) -> Result<(QaoaParams, QaoaResult), QaoaError> {
    if depth == 0 || samples == 0 {
        return Err(QaoaError::InvalidDepth);
    }
    let mut best_params = None;
    let mut best_result = None;
    let step = std::f64::consts::PI / samples as f64;
    let mut angles = Vec::with_capacity(depth * 2);

    fn search_layer(
        layer: usize,
        depth: usize,
        step: f64,
        samples: usize,
        problem: &IsingProblem,
        current: &mut Vec<f64>,
        best_params: &mut Option<QaoaParams>,
        best_result: &mut Option<QaoaResult>,
    ) -> Result<(), QaoaError> {
        if layer == depth {
            let gammas = current[..depth].to_vec();
            let betas = current[depth..].to_vec();
            let params = QaoaParams::new(gammas, betas)?;
            let result = run_qaoa(problem, &params)?;
            match best_result {
                None => {
                    *best_result = Some(result.clone());
                    *best_params = Some(params);
                }
                Some(existing) => {
                    if result.expected_energy < existing.expected_energy {
                        *best_result = Some(result.clone());
                        *best_params = Some(params);
                    }
                }
            }
            return Ok(());
        }

        for idx in 0..samples {
            let angle = idx as f64 * step;
            current[layer] = angle;
            current[layer + depth] = angle;
            search_layer(
                layer + 1,
                depth,
                step,
                samples,
                problem,
                current,
                best_params,
                best_result,
            )?;
        }
        Ok(())
    }

    angles.resize(depth * 2, 0.0);
    search_layer(
        0,
        depth,
        step,
        samples,
        problem,
        &mut angles,
        &mut best_params,
        &mut best_result,
    )?;

    Ok((
        best_params.expect("at least one candidate"),
        best_result.expect("at least one candidate"),
    ))
}

fn apply_cost_layer(state: &mut [Complex64], problem: &IsingProblem, gamma: f64) {
    let phase_factor = |energy: f64| Complex64::from_polar(1.0, -gamma * energy);
    for (basis, amp) in state.iter_mut().enumerate() {
        let energy = problem.energy_from_basis(basis as u64);
        *amp *= phase_factor(energy);
    }
}

fn apply_mixer_layer(state: &mut [Complex64], num_qubits: usize, beta: f64) {
    let cos = beta.cos();
    let minus_i_sin = Complex64::new(0.0, -beta.sin());
    let dim = state.len();

    for qubit in 0..num_qubits {
        let stride = 1 << qubit;
        let period = stride << 1;
        for base in (0..dim).step_by(period) {
            for offset in 0..stride {
                let i0 = base + offset;
                let i1 = i0 + stride;
                let a0 = state[i0];
                let a1 = state[i1];
                state[i0] = a0 * cos + a1 * minus_i_sin;
                state[i1] = a1 * cos + a0 * minus_i_sin;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ising::{Coupling, IsingProblem};

    #[test]
    fn single_qubit_cost() {
        let problem = IsingProblem::new(1, vec![-1.0], vec![]);
        let params = QaoaParams::new(vec![1.2], vec![0.8]).unwrap();
        let result = run_qaoa(&problem, &params).unwrap();
        assert!(result.expected_energy.is_finite());
        assert!(result.variance >= 0.0);
    }

    #[test]
    fn grid_search_depth1() {
        let problem = IsingProblem::new(
            2,
            vec![0.0, 0.0],
            vec![Coupling {
                i: 0,
                j: 1,
                weight: -1.0,
            }],
        );
        let (params, result) = grid_search_qaoa(&problem, 1, 3).unwrap();
        assert_eq!(params.depth(), 1);
        assert!(result.expected_energy <= -0.5);
    }
}
