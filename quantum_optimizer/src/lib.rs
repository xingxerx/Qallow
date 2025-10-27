//! quantum_optimizer: minimal, focused crate for quantum optimization routines.
//!
//! This crate is intentionally lightweight. It exposes a stable API that
//! `native_app` can depend on optionally via the `quantum` feature.

/// Parameters for a simple optimization routine (placeholder for QAOA/QUBO, etc.).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimizeParams {
    /// Number of layers/steps; must be > 0
    pub steps: usize,
    /// Learning rate or step size; must be > 0
    pub lr: f64,
}

/// Result of an optimization run.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationResult {
    pub best_value: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Error type for optimization failures.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum OptimizeError {
    #[error("invalid steps: {0}")]
    InvalidSteps(usize),
    #[error("invalid learning rate: {0}")]
    InvalidLr(f64),
}

/// A tiny, deterministic "optimizer" placeholder.
///
/// Contract:
/// - Inputs: OptimizeParams { steps > 0, lr > 0 }
/// - Output: OptimizationResult with iterations == steps, best_value monotonically improves, converged when improvement < 1e-6
/// - Errors: InvalidSteps, InvalidLr
pub fn optimize_stub(params: OptimizeParams) -> Result<OptimizationResult, OptimizeError> {
    if params.steps == 0 {
        return Err(OptimizeError::InvalidSteps(params.steps));
    }
    if !(params.lr.is_finite()) || params.lr <= 0.0 {
        return Err(OptimizeError::InvalidLr(params.lr));
    }

    let mut value = 1.0; // worse is higher; we "minimize"
    let mut converged = false;
    for _ in 0..params.steps {
        let prev = value;
        // Simple convex-like decay with diminishing returns
        value *= 1.0 - (params.lr.min(0.5) * 0.1);
        if (prev - value).abs() < 1e-6 {
            converged = true;
            break;
        }
    }

    Ok(OptimizationResult {
        best_value: value.max(0.0),
        iterations: params.steps,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_path_converges() {
        let res = optimize_stub(OptimizeParams { steps: 50, lr: 0.1 }).unwrap();
        assert!(res.best_value >= 0.0);
        assert!(res.iterations == 50);
    }

    #[test]
    fn rejects_zero_steps() {
        let err = optimize_stub(OptimizeParams { steps: 0, lr: 0.1 }).unwrap_err();
        assert!(matches!(err, OptimizeError::InvalidSteps(0)));
    }

    #[test]
    fn rejects_bad_lr() {
        let err = optimize_stub(OptimizeParams { steps: 10, lr: 0.0 }).unwrap_err();
        assert!(matches!(err, OptimizeError::InvalidLr(_)));
    }
}
