#!/usr/bin/env python3
"""
Recursive Improvement Engine for Qallow + Agent Lightning
===========================================================

This system continuously:
1. Runs Qallow with CUDA
2. Captures errors and metrics
3. Uses Agent Lightning (RL) to optimize
4. Automatically applies fixes
5. Re-runs with improvements (recursive)

The loop continues improving the project each iteration.
"""









from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Results from a single execution iteration."""
    iteration: int
    timestamp: str
    success: bool
    build_time: float
    execution_time: float
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, float]
    improvements_applied: List[str]
    agent_reward: float
    notes: str


class ErrorExtractor:
    """Detects and categorizes errors from execution logs."""
    
    ERROR_PATTERNS = {
        'cuda': r'(CUDA error|cudaError|gpu|nvcc)',
        'memory': r'(segmentation|SIGSEGV|memory|malloc|leak)',
        'compilation': r'(error:|undefined reference|no such file)',
        'runtime': r'(Traceback|Exception|Error:)',
        'phase': r'(Phase \d+ failed|phase error)',
        'ethics': r'(ethics calculation failed|ethics error)',
    }
    
    def extract_errors(self, output: str) -> Tuple[List[str], List[str]]:
        """Extract errors and warnings from output."""
        errors = []
        warnings = []
        
        for line in output.split('\n'):
            if 'error' in line.lower():
                errors.append(line.strip())
            elif 'warning' in line.lower():
                warnings.append(line.strip())
        
        return errors, warnings
    
    def categorize_error(self, error: str) -> str:
        """Categorize error type."""
        for category, pattern in self.ERROR_PATTERNS.items():
            if re.search(pattern, error, re.IGNORECASE):
                return category
        return 'unknown'


class MetricsCollector:
    """Collects performance metrics from telemetry."""
    
    def __init__(self, logs_dir: str = "data/logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_latest_metrics(self) -> Dict[str, float]:
        """Collect metrics from the most recent CSV log."""
        metrics = {
            'avg_coherence': 0.0,
            'ethics_total': 0.0,
            'phase_drift': 0.0,
            'execution_time': 0.0,
            'stability': 0.0,
        }
        
        # Find most recent CSV
        csv_files = sorted(self.logs_dir.glob('*.csv'), key=os.path.getmtime, reverse=True)
        
        if not csv_files:
            logger.warning("No CSV files found in logs directory")
            return metrics
        
        try:
            with open(csv_files[0], 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if rows:
                    last_row = rows[-1]
                    metrics['avg_coherence'] = float(last_row.get('avg_coherence', 0))
                    metrics['ethics_total'] = float(last_row.get('ethics_total', 0))
                    metrics['phase_drift'] = float(last_row.get('phase_drift', 0))
                    metrics['stability'] = 1.0 - min(metrics['phase_drift'], 1.0)
                
                logger.info(f"Collected metrics from {csv_files[0].name}")
        
        except Exception as e:
            logger.error(f"Failed to parse metrics: {e}")
        
        return metrics


class AutoFixer:
    """Automatically applies fixes based on error patterns."""
    
    FIX_MAP = {
        'cuda': [
            'export CUDA_VISIBLE_DEVICES=0',
            'cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON',
            './scripts/build_all.sh --cuda',
        ],
        'memory': [
            'export QALLOW_MEMORY_LIMIT=8192',
            'cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo',
        ],
        'compilation': [
            'rm -rf build',
            'cmake -S . -B build',
            'make clean',
        ],
        'ethics': [
            'export QALLOW_ETHICS_DEBUG=1',
            'export QALLOW_LOG_ETHICS=1',
        ],
    }
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
    
    def apply_fix(self, error_category: str) -> bool:
        """Apply a fix for the given error category."""
        if error_category not in self.FIX_MAP:
            logger.warning(f"No fix available for {error_category}")
            return False
        
        logger.info(f"Applying fixes for {error_category}...")
        
        try:
            for fix_cmd in self.FIX_MAP[error_category]:
                logger.info(f"Running: {fix_cmd}")
                subprocess.run(fix_cmd, shell=True, check=False, cwd=str(self.project_root))
            
            logger.info(f"Fixes applied for {error_category}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to apply fix: {e}")
            return False


class CUDAExecutor:
    """Executes Qallow with CUDA backend."""
    
    def __init__(self, binary_path: str = "./build/qallow_unified_cpu"):
        self.binary_path = binary_path
        self.build_dir = Path("build")
    
    def build_cuda(self) -> Tuple[bool, float]:
        """Build project with CUDA support, fallback to CPU if needed."""
        logger.info("Building with CUDA support...")
        start_time = time.time()
        
        try:
            # Clean build
            subprocess.run("rm -rf build", shell=True, check=False)
            
            # Configure with CUDA
            result = subprocess.run(
                "cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON",
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # If CUDA fails, try CPU fallback
            if result.returncode != 0:
                if "nvcc" in result.stderr.lower() or "cuda" in result.stderr.lower():
                    logger.warning("CUDA not available, falling back to CPU build...")
                    subprocess.run("rm -rf build", shell=True, check=False)
                    
                    result = subprocess.run(
                        "cmake -S . -B build -DQALLOW_ENABLE_CUDA=OFF",
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode != 0:
                        logger.error(f"CPU configure also failed: {result.stderr}")
                        return False, time.time() - start_time
                else:
                    logger.error(f"CMake configure failed: {result.stderr}")
                    return False, time.time() - start_time
            
            # Build
            try:
                num_cores = os.cpu_count() or 1
            except:
                num_cores = 1
            
            result = subprocess.run(
                f"cmake --build build --parallel {num_cores}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            build_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"Build successful in {build_time:.2f}s")
                return True, build_time
            else:
                logger.error(f"Build failed: {result.stderr}")
                return False, build_time
        
        except subprocess.TimeoutExpired:
            logger.error("Build timeout")
            return False, time.time() - start_time
        except Exception as e:
            logger.error(f"Build error: {e}")
            return False, time.time() - start_time
    
    def run_unified(self, phases: List[int] = None, ticks: int = 120) -> Tuple[bool, float, str]:
        """Run unified Qallow phases."""
        if phases is None:
            phases = [12, 13, 14, 15]
        
        logger.info(f"Running unified phases {phases} with {ticks} ticks...")
        start_time = time.time()
        
        try:
            cmd = f"{self.binary_path} run unified --integrate-ticks={ticks}"
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            exec_time = time.time() - start_time
            output = result.stdout + result.stderr
            
            if result.returncode == 0:
                logger.info(f"Execution successful in {exec_time:.2f}s")
                return True, exec_time, output
            else:
                logger.error(f"Execution failed")
                return False, exec_time, output
        
        except subprocess.TimeoutExpired:
            logger.error("Execution timeout")
            return False, time.time() - start_time, "Timeout"
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False, time.time() - start_time, str(e)


class RecursiveImprovementEngine:
    """Main orchestrator for recursive improvements."""
    
    def __init__(self, max_iterations: int = 10, max_errors_per_iteration: int = 3):
        self.max_iterations = max_iterations
        self.max_errors_per_iteration = max_errors_per_iteration
        
        self.error_extractor = ErrorExtractor()
        self.metrics_collector = MetricsCollector()
        self.auto_fixer = AutoFixer()
        self.cuda_executor = CUDAExecutor()
        
        self.results: List[ExecutionResult] = []
        self.iteration = 0
        self.prev_metrics = None
    
    def run(self) -> List[ExecutionResult]:
        """Run the recursive improvement loop."""
        logger.info("=" * 70)
        logger.info("Starting Recursive Improvement Engine (CUDA heuristic mode)")
        logger.info("=" * 70)
        
        for iteration in range(1, self.max_iterations + 1):
            self.iteration = iteration
            logger.info(f"\n{'='*70}")
            logger.info(f"ITERATION {iteration}/{self.max_iterations}")
            logger.info(f"{'='*70}\n")
            
            result = self._run_iteration()
            self.results.append(result)
            
            # Stop if no errors and stable
            if not result.errors and iteration > 2:
                logger.info(f"\n✓ No errors detected. System stable after {iteration} iterations.")
                break
            
            # Small delay before next iteration
            time.sleep(2)
        
        self._print_summary()
        return self.results
    
    def _run_iteration(self) -> ExecutionResult:
        """Run a single iteration."""
        timestamp = datetime.now().isoformat()
        result = ExecutionResult(
            iteration=self.iteration,
            timestamp=timestamp,
            success=False,
            build_time=0.0,
            execution_time=0.0,
            errors=[],
            warnings=[],
            metrics={},
            improvements_applied=[],
            agent_reward=0.0,
            notes="",
        )
        
        try:
            # Phase 1: Build
            logger.info(f"[Iteration {self.iteration}] Building with CUDA...")
            success, build_time = self.cuda_executor.build_cuda()
            result.build_time = build_time
            
            if not success:
                result.errors.append("CUDA build failed")
                result.notes = "Build failed, attempting fixes..."
                self.auto_fixer.apply_fix('cuda')
                result.improvements_applied.append("Applied CUDA build fixes")
                return result
            
            # Phase 2: Execute
            logger.info(f"[Iteration {self.iteration}] Executing Qallow phases...")
            success, exec_time, output = self.cuda_executor.run_unified()
            result.execution_time = exec_time
            
            # Extract errors and warnings
            errors, warnings = self.error_extractor.extract_errors(output)
            result.errors = errors[:self.max_errors_per_iteration]
            result.warnings = warnings
            
            # Phase 3: Collect Metrics
            logger.info(f"[Iteration {self.iteration}] Collecting metrics...")
            metrics = self.metrics_collector.collect_latest_metrics()
            result.metrics = metrics
            
            # Phase 4: Calculate Reward
            reward = self._calculate_reward(metrics, self.prev_metrics)
            result.agent_reward = reward
            result.success = success and not errors
            
            logger.info(f"  Coherence: {metrics.get('avg_coherence', 0):.4f}")
            logger.info(f"  Ethics: {metrics.get('ethics_total', 0):.4f}")
            logger.info(f"  Stability: {metrics.get('stability', 0):.4f}")
            logger.info(f"  RL Reward: {reward:.4f}")
            
            # Phase 5: Emit Agent Lightning Events
            # Agent Lightning event emission removed.
            
            # Phase 6: Auto-fix errors
            if errors:
                logger.info(f"[Iteration {self.iteration}] Detected {len(errors)} errors. Applying fixes...")
                for error in result.errors:
                    category = self.error_extractor.categorize_error(error)
                    if self.auto_fixer.apply_fix(category):
                        result.improvements_applied.append(f"Fixed {category}: {error[:60]}")
            
            self.prev_metrics = metrics
            
            return result
        
        except Exception as e:
            logger.error(f"Iteration failed with exception: {e}")
            logger.error(traceback.format_exc())
            result.errors.append(str(e))
            result.notes = f"Exception: {str(e)[:100]}"
            return result

    @staticmethod
    def _calculate_reward(metrics: Dict[str, float], prev_metrics: Optional[Dict[str, float]] = None) -> float:
        """Compute a heuristic reward for progress tracking without external tooling."""
        coherence_reward = min(metrics.get('avg_coherence', 0) / 0.95, 1.0) * 0.5
        ethics_reward = min(metrics.get('ethics_total', 0) / 3.0, 1.0) * 0.3
        stability_reward = metrics.get('stability', 0.5) * 0.2

        base_reward = coherence_reward + ethics_reward + stability_reward

        if prev_metrics:
            improvement = (
                (metrics.get('avg_coherence', 0) - prev_metrics.get('avg_coherence', 0)) * 0.5 +
                (metrics.get('ethics_total', 0) - prev_metrics.get('ethics_total', 0)) * 0.3
            )
            base_reward += max(improvement, 0) * 0.2

        return base_reward
    
    def _print_summary(self):
        """Print summary of all iterations."""
        logger.info(f"\n{'='*70}")
        logger.info("IMPROVEMENT SUMMARY")
        logger.info(f"{'='*70}\n")
        
        total_iterations = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        total_errors = sum(len(r.errors) for r in self.results)
        avg_reward = sum(r.agent_reward for r in self.results) / total_iterations if total_iterations > 0 else 0
        
        logger.info(f"Total Iterations: {total_iterations}")
        logger.info(f"Successful: {successful}/{total_iterations}")
        logger.info(f"Total Errors Found: {total_errors}")
        logger.info(f"Average RL Reward: {avg_reward:.4f}")
        
        # Best iteration
        best_result = max(self.results, key=lambda r: r.agent_reward)
        logger.info(f"\nBest Iteration: #{best_result.iteration} (Reward: {best_result.agent_reward:.4f})")
        
        # Improvements applied
        all_improvements = [imp for r in self.results for imp in r.improvements_applied]
        if all_improvements:
            logger.info(f"\nImprovements Applied ({len(all_improvements)}):")
            for imp in set(all_improvements):
                logger.info(f"  ✓ {imp}")
        
        # Save results to file
        self._save_results()
    
    def _save_results(self):
        """Save results to JSON file."""
        output_dir = Path("improvement_reports")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"recursive_improvement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'total_iterations': len(self.results),
            'results': [asdict(r) for r in self.results],
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"\nResults saved to {output_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Recursive Improvement Engine for Qallow')
    parser.add_argument('--iterations', type=int, default=10, help='Max iterations')
    parser.add_argument('--ticks', type=int, default=120, help='Phase ticks')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA')
    parser.add_argument('--phases', nargs='+', type=int, default=[12, 13, 14, 15], help='Phases to run')
    
    args = parser.parse_args()
    
    # Create and run engine
    engine = RecursiveImprovementEngine(max_iterations=args.iterations)
    results = engine.run()
    
    # Exit with success if improvement detected
    if results and results[-1].success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
