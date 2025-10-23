use clap::{Parser, Subcommand};
use qallow_quantum_rust::{
    phase14_run, phase15_run, QAOAConfig,
    quantum_optimizer::{
        build_optimized_circuit, compute_gate_sequence_optimized, estimate_circuit_metrics,
        get_qaoa_optimal_params,
        get_vqe_params, get_phase_estimation_params, get_trotter_params,
    },
};
use serde_json::json;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run Phase 14 (coherence lattice integration) with QAOA tuner
    Phase14 {
        /// Number of ticks
        #[arg(long, default_value = "500")]
        ticks: usize,

        /// Lattice nodes
        #[arg(long, default_value = "256")]
        nodes: usize,

        /// Target fidelity threshold
        #[arg(long, default_value = "0.981")]
        target_fidelity: f64,

        /// Explicit alpha override
        #[arg(long)]
        alpha: Option<f64>,

        /// Enable QAOA tuner
        #[arg(long)]
        tune_qaoa: bool,

        /// QAOA problem size
        #[arg(long, default_value = "16")]
        qaoa_n: usize,

        /// QAOA depth p
        #[arg(long, default_value = "2")]
        qaoa_p: usize,

        /// Load alpha_eff from JSON file
        #[arg(long)]
        gain_json: Option<PathBuf>,

        /// Export JSON results
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Run Phase 15 (convergence & lock-in)
    Phase15 {
        /// Phase 14 fidelity to use as prior (if no file)
        #[arg(long, default_value = "0.95")]
        phase14_fidelity: f64,

        /// Number of ticks
        #[arg(long, default_value = "400")]
        ticks: usize,

        /// Convergence tolerance
        #[arg(long, default_value = "0.000005")]
        eps: f64,

        /// Export JSON results
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Run full unified pipeline: Phase 14 → Phase 15
    Pipeline {
        /// Number of ticks for Phase 14
        #[arg(long, default_value = "600")]
        phase14_ticks: usize,

        /// Lattice nodes
        #[arg(long, default_value = "256")]
        nodes: usize,

        /// Target fidelity for Phase 14
        #[arg(long, default_value = "0.981")]
        target_fidelity: f64,

        /// Enable QAOA tuner
        #[arg(long)]
        tune_qaoa: bool,

        /// QAOA problem size
        #[arg(long, default_value = "16")]
        qaoa_n: usize,

        /// QAOA depth p
        #[arg(long, default_value = "2")]
        qaoa_p: usize,

        /// Number of ticks for Phase 15
        #[arg(long, default_value = "800")]
        phase15_ticks: usize,

        /// Convergence tolerance
        #[arg(long, default_value = "0.000005")]
        phase15_eps: f64,

        /// Export Phase 14 JSON results
        #[arg(long)]
        export_phase14: Option<PathBuf>,

        /// Export Phase 15 JSON results
        #[arg(long)]
        export_phase15: Option<PathBuf>,

        /// Export combined pipeline JSON
        #[arg(long)]
        export_pipeline: Option<PathBuf>,
    },

    /// Quantum Circuit Optimizer - Hardcoded VQE/QAOA circuit generation
    CircuitOptimize {
        /// Number of qubits
        #[arg(long, default_value = "16")]
        qubits: usize,

        /// Circuit depth
        #[arg(long, default_value = "2")]
        depth: usize,

        /// Export circuit to JSON
        #[arg(long)]
        export_circuit: Option<PathBuf>,

        /// Export metrics to JSON
        #[arg(long)]
        export_metrics: Option<PathBuf>,
    },

    /// Generate hardcoded QAOA parameters for problem optimization
    QAOAParams {
        /// Problem size
        #[arg(long, default_value = "16")]
        problem_size: usize,

        /// QAOA depth
        #[arg(long, default_value = "2")]
        depth: usize,

        /// Export to JSON
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Generate VQE parameters for variational quantum eigensolver
    VQEParams {
        /// Problem size
        #[arg(long, default_value = "16")]
        problem_size: usize,

        /// Export to JSON
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Generate phase estimation parameters for quantum phase estimation
    PhaseEst {
        /// Precision bits
        #[arg(long, default_value = "4")]
        precision: usize,

        /// Export to JSON
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Generate Trotter decomposition parameters
    Trotter {
        /// Time steps
        #[arg(long, default_value = "10")]
        time_steps: usize,

        /// Trotter order (2, 3, or 4)
        #[arg(long, default_value = "2")]
        order: usize,

        /// Export to JSON
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Show help and examples
    Help,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Phase14 {
            ticks,
            nodes,
            target_fidelity,
            alpha,
            tune_qaoa,
            qaoa_n,
            qaoa_p,
            gain_json,
            export,
        } => {
            println!("[PHASE14] Coherence-lattice integration");
            println!("[PHASE14] nodes={} ticks={} target_fidelity={:.3}", nodes, ticks, target_fidelity);

            let qaoa_cfg = if tune_qaoa {
                Some(QAOAConfig {
                    n: qaoa_n,
                    p: qaoa_p,
                    max_iters: 50,
                    gain_min: 0.001,
                    gain_max: 0.01,
                })
            } else {
                None
            };

            let gain_json_str = if let Some(path) = gain_json {
                fs::read_to_string(&path).ok()
            } else {
                None
            };

            let result = phase14_run(
                ticks,
                nodes,
                target_fidelity,
                tune_qaoa,
                qaoa_cfg,
                alpha,
                gain_json_str,
            );

            if let Some(export_path) = export {
                let json = json!({
                    "fidelity": result.fidelity,
                    "target": result.target_fidelity,
                    "ticks": result.ticks,
                    "alpha_base": result.alpha_base,
                    "alpha_used": result.alpha_used,
                });
                fs::write(&export_path, serde_json::to_string_pretty(&json)?)?;
                println!("[PHASE14] Exported to {}", export_path.display());
            }
        }

        Commands::Phase15 {
            phase14_fidelity,
            ticks,
            eps,
            export,
        } => {
            println!("[PHASE15] Starting convergence & lock-in");
            let result = phase15_run(phase14_fidelity, ticks, eps);

            if let Some(export_path) = export {
                let json = json!({
                    "score": result.score,
                    "stability": result.stability,
                    "convergence_tick": result.convergence_tick,
                    "ticks_run": result.ticks_run,
                });
                fs::write(&export_path, serde_json::to_string_pretty(&json)?)?;
                println!("[PHASE15] Exported to {}", export_path.display());
            }
        }

        Commands::Pipeline {
            phase14_ticks,
            nodes,
            target_fidelity,
            tune_qaoa,
            qaoa_n,
            qaoa_p,
            phase15_ticks,
            phase15_eps,
            export_phase14,
            export_phase15,
            export_pipeline,
        } => {
            println!("╔════════════════════════════════════════════════╗");
            println!("║ QALLOW QUANTUM UNIFIED PIPELINE (Rust)          ║");
            println!("║ Phase 14 (Coherence) → Phase 15 (Convergence)  ║");
            println!("╚════════════════════════════════════════════════╝\n");

            // Phase 14
            let qaoa_cfg = if tune_qaoa {
                Some(QAOAConfig {
                    n: qaoa_n,
                    p: qaoa_p,
                    max_iters: 50,
                    gain_min: 0.001,
                    gain_max: 0.01,
                })
            } else {
                None
            };

            let result14 = phase14_run(
                phase14_ticks,
                nodes,
                target_fidelity,
                tune_qaoa,
                qaoa_cfg,
                None,
                None,
            );

            if let Some(export_path) = export_phase14 {
                let json = json!({
                    "fidelity": result14.fidelity,
                    "target": result14.target_fidelity,
                    "ticks": result14.ticks,
                    "alpha_base": result14.alpha_base,
                    "alpha_used": result14.alpha_used,
                });
                fs::write(&export_path, serde_json::to_string_pretty(&json)?)?;
                println!("[PIPELINE] Phase 14 exported to {}\n", export_path.display());
            }

            // Phase 15 uses Phase 14 fidelity as prior
            let result15 = phase15_run(result14.fidelity, phase15_ticks, phase15_eps);

            if let Some(export_path) = export_phase15 {
                let json = json!({
                    "score": result15.score,
                    "stability": result15.stability,
                    "convergence_tick": result15.convergence_tick,
                    "ticks_run": result15.ticks_run,
                });
                fs::write(&export_path, serde_json::to_string_pretty(&json)?)?;
                println!("[PIPELINE] Phase 15 exported to {}\n", export_path.display());
            }

            // Combined pipeline export
            if let Some(export_path) = export_pipeline {
                let json = json!({
                    "pipeline": {
                        "phase14": {
                            "fidelity": result14.fidelity,
                            "target": result14.target_fidelity,
                            "ticks": result14.ticks,
                            "alpha_base": result14.alpha_base,
                            "alpha_used": result14.alpha_used,
                        },
                        "phase15": {
                            "score": result15.score,
                            "stability": result15.stability,
                            "convergence_tick": result15.convergence_tick,
                            "ticks_run": result15.ticks_run,
                        },
                        "success": result14.fidelity >= result14.target_fidelity && result15.stability >= 0.0,
                    }
                });
                fs::write(&export_path, serde_json::to_string_pretty(&json)?)?;
                println!("[PIPELINE] Combined export to {}\n", export_path.display());
            }

            println!("╔════════════════════════════════════════════════╗");
            println!("║ PIPELINE COMPLETE                              ║");
            println!("║ Phase 14 fidelity: {:.6} [{}]                 ║", result14.fidelity, if result14.fidelity >= result14.target_fidelity { "OK" } else { "WARN" });
            println!("║ Phase 15 stability: {:.6} [{}]                ║", result15.stability, if result15.stability >= 0.0 { "OK" } else { "WARN" });
            println!("╚════════════════════════════════════════════════╝");
        }

        Commands::CircuitOptimize {
            qubits,
            depth,
            export_circuit,
            export_metrics,
        } => {
            println!("╔════════════════════════════════════════════════╗");
            println!("║ QUANTUM CIRCUIT OPTIMIZER                      ║");
            println!("║ Hardcoded VQE/QAOA Circuit Generation           ║");
            println!("╚════════════════════════════════════════════════╝\n");

            let circuit = build_optimized_circuit(qubits, depth);
            let metrics = estimate_circuit_metrics(&circuit);
            let gate_sequence = compute_gate_sequence_optimized(&circuit);

            println!("[CIRCUIT] Qubits: {}", circuit.qubits);
            println!("[CIRCUIT] Total gates: {}", circuit.gates.len());
            println!("[CIRCUIT] Circuit depth: {}", circuit.depth);
            println!("[CIRCUIT] Fidelity: {:.6}", circuit.fidelity);
            println!("[CIRCUIT] Gate count: {}", gate_sequence.len());
            println!("[CIRCUIT] Estimated runtime: {:.4} µs", metrics.estimated_runtime_us);
            println!("[CIRCUIT] Memory footprint: {:.6} MB", metrics.memory_footprint_mb);
            println!("[CIRCUIT] Gate fidelity: {:.6}\n", metrics.gate_fidelity);

            if let Some(export_path) = export_circuit {
                let json = json!({
                    "qubits": circuit.qubits,
                    "gates": circuit.gates,
                    "fidelity": circuit.fidelity,
                    "depth": circuit.depth,
                    "gate_sequence": gate_sequence,
                });
                fs::write(&export_path, serde_json::to_string_pretty(&json)?)?;
                println!("[CIRCUIT] Exported circuit to {}", export_path.display());
            }

            if let Some(export_path) = export_metrics {
                let json = json!({
                    "estimated_runtime_us": metrics.estimated_runtime_us,
                    "memory_footprint_mb": metrics.memory_footprint_mb,
                    "gate_fidelity": metrics.gate_fidelity,
                    "total_depth": metrics.total_depth,
                });
                fs::write(&export_path, serde_json::to_string_pretty(&json)?)?;
                println!("[CIRCUIT] Exported metrics to {}\n", export_path.display());
            }

            println!("╔════════════════════════════════════════════════╗");
            println!("║ CIRCUIT OPTIMIZATION COMPLETE                  ║");
            println!("╚════════════════════════════════════════════════╝");
        }

        Commands::QAOAParams {
            problem_size,
            depth,
            export,
        } => {
            println!("[QAOA] Generating optimal parameters...");
            let params = get_qaoa_optimal_params(problem_size, depth);

            println!("[QAOA] Problem size: {}", problem_size);
            println!("[QAOA] Depth: {}", depth);
            println!("[QAOA] Gamma angles: {} values", params.gamma.len());
            println!("[QAOA] Beta angles: {} values", params.beta.len());
            println!("[QAOA] Expected energy: {:.6}\n", params.expected_energy);

            if let Some(export_path) = export {
                let json = json!({
                    "problem_size": problem_size,
                    "depth": depth,
                    "gamma": params.gamma,
                    "beta": params.beta,
                    "expected_energy": params.expected_energy,
                });
                fs::write(&export_path, serde_json::to_string_pretty(&json)?)?;
                println!("[QAOA] Exported to {}", export_path.display());
            }
        }

        Commands::VQEParams {
            problem_size,
            export,
        } => {
            println!("[VQE] Generating VQE parameters...");
            let params = get_vqe_params(problem_size);

            println!("[VQE] Problem size: {}", problem_size);
            println!("[VQE] Total parameters: {}", params.num_params);
            println!("[VQE] Learning rate: {:.6}", params.learning_rate);
            println!("[VQE] Parameter bounds: [0, 2π]\n");

            if let Some(export_path) = export {
                let json = json!({
                    "problem_size": problem_size,
                    "num_params": params.num_params,
                    "initial_params": params.initial_params,
                    "learning_rate": params.learning_rate,
                });
                fs::write(&export_path, serde_json::to_string_pretty(&json)?)?;
                println!("[VQE] Exported to {}", export_path.display());
            }
        }

        Commands::PhaseEst { precision, export } => {
            println!("[PHASE_EST] Generating phase estimation parameters...");
            let params = get_phase_estimation_params(precision);

            println!("[PHASE_EST] Precision bits: {}", params.precision_bits);
            println!("[PHASE_EST] Feedback angles: {} values\n", params.feedback_angles.len());

            if let Some(export_path) = export {
                let json = json!({
                    "precision_bits": params.precision_bits,
                    "feedback_angles": params.feedback_angles,
                });
                fs::write(&export_path, serde_json::to_string_pretty(&json)?)?;
                println!("[PHASE_EST] Exported to {}", export_path.display());
            }
        }

        Commands::Trotter {
            time_steps,
            order,
            export,
        } => {
            println!("[TROTTER] Generating Trotter decomposition parameters...");
            let params = get_trotter_params(time_steps, order);

            println!("[TROTTER] Time steps: {}", params.time_steps);
            println!("[TROTTER] Order: {}", params.order);
            println!("[TROTTER] Coefficients: {} values\n", params.coefficients.len());

            if let Some(export_path) = export {
                let json = json!({
                    "time_steps": params.time_steps,
                    "order": params.order,
                    "coefficients": params.coefficients,
                });
                fs::write(&export_path, serde_json::to_string_pretty(&json)?)?;
                println!("[TROTTER] Exported to {}", export_path.display());
            }
        }

        Commands::Help => {
            println!("QALLOW Quantum Rust - Unified Phase 14/15 Pipeline + Circuit Optimizer\n");
            println!("USAGE:");
            println!("  qallow_quantum phase14 [OPTIONS]");
            println!("  qallow_quantum phase15 [OPTIONS]");
            println!("  qallow_quantum pipeline [OPTIONS]");
            println!("  qallow_quantum circuit-optimize [OPTIONS]");
            println!("  qallow_quantum qaoa-params [OPTIONS]");
            println!("  qallow_quantum vqe-params [OPTIONS]");
            println!("  qallow_quantum phase-est [OPTIONS]");
            println!("  qallow_quantum trotter [OPTIONS]\n");
            println!("EXAMPLES:");
            println!("  # Phase 14 with closed-form alpha:");
            println!("    qallow_quantum phase14 --ticks=600 --target_fidelity=0.981 \\");
            println!("      --export=/tmp/phase14.json\n");
            println!("  # Phase 14 with QAOA tuner:");
            println!("    qallow_quantum phase14 --ticks=600 --target_fidelity=0.981 \\");
            println!("      --tune_qaoa --qaoa_n=16 --qaoa_p=2 \\");
            println!("      --export=/tmp/phase14.json\n");
            println!("  # Full unified pipeline (Phase 14 → 15):");
            println!("    qallow_quantum pipeline --tune_qaoa \\");
            println!("      --export_phase14=/tmp/p14.json \\");
            println!("      --export_phase15=/tmp/p15.json \\");
            println!("      --export_pipeline=/tmp/pipeline.json\n");
            println!("  # QUANTUM CIRCUIT OPTIMIZATION:");
            println!("  # Build and optimize a 16-qubit VQE circuit:");
            println!("    qallow_quantum circuit-optimize --qubits=16 --depth=3 \\");
            println!("      --export_circuit=/tmp/circuit.json \\");
            println!("      --export_metrics=/tmp/metrics.json\n");
            println!("  # Generate hardcoded QAOA parameters:");
            println!("    qallow_quantum qaoa-params --problem_size=16 --depth=2 \\");
            println!("      --export=/tmp/qaoa_params.json\n");
            println!("  # Generate VQE initialization:");
            println!("    qallow_quantum vqe-params --problem_size=16 \\");
            println!("      --export=/tmp/vqe_params.json\n");
            println!("  # Generate phase estimation angles:");
            println!("    qallow_quantum phase-est --precision=4 \\");
            println!("      --export=/tmp/phase_est.json\n");
            println!("  # Generate Trotter decomposition:");
            println!("    qallow_quantum trotter --time_steps=10 --order=4 \\");
            println!("      --export=/tmp/trotter.json\n");
        }
    }

    Ok(())
}
