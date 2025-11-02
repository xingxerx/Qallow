#!/usr/bin/env python3
"""
Qallow + Agent Lightning Integration
======================================

This module bridges Qallow quantum-AGI phases with Microsoft's Agent Lightning
RL training framework. It enables:

1. Automated phase execution with telemetry capture
2. Agent Lightning instrumentation for RL training
3. Performance metric extraction and reward calculation
4. Multi-phase optimization via RL algorithms

Architecture:
    Qallow Phase Runner → Telemetry Parser → Agent Lightning Events → LightningStore → RL Training
"""

import os
import sys
import json
import subprocess
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Import Agent Lightning
try:
    import agentlightning as agl
except ImportError:
    print("Error: agentlightning not installed. Run: pip install agentlightning")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PhaseMetrics:
    """Captured metrics from a Qallow phase execution."""
    phase_number: int
    tick_count: int
    avg_coherence: float
    phase_drift: float
    ethics_total: float
    sustainability: float
    compassion: float
    harmony: float
    execution_time: float
    log_file: str


class QallowPhaseRunner:
    """Executes Qallow phases and captures telemetry."""
    
    def __init__(self, qallow_binary: str = "./build/qallow_unified_cpu"):
        """
        Initialize the Qallow phase runner.
        
        Args:
            qallow_binary: Path to the compiled qallow executable
        """
        self.qallow_binary = qallow_binary
        self.logs_dir = Path("data/logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        if not os.access(qallow_binary, os.X_OK):
            logger.warning(f"Qallow binary not found or not executable: {qallow_binary}")
    
    def run_phase(
        self,
        phase: int,
        ticks: int = 120,
        additional_args: Optional[List[str]] = None,
        timeout: int = 300
    ) -> Optional[PhaseMetrics]:
        """
        Execute a single Qallow phase.
        
        Args:
            phase: Phase number (12, 13, 14, 15, etc.)
            ticks: Number of execution ticks
            additional_args: Additional command-line arguments
            timeout: Execution timeout in seconds
            
        Returns:
            PhaseMetrics object if successful, None otherwise
        """
        cmd = [self.qallow_binary, "phase", str(phase), f"--ticks={ticks}"]
        
        if additional_args:
            cmd.extend(additional_args)
        
        logger.info(f"Executing: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            execution_time = time.time() - start_time
            
            if result.returncode != 0:
                logger.error(f"Phase {phase} execution failed: {result.stderr}")
                return None
            
            logger.info(f"Phase {phase} completed in {execution_time:.2f}s")
            
            # Parse telemetry output
            metrics = self._parse_telemetry(phase, execution_time)
            return metrics
            
        except subprocess.TimeoutExpired:
            logger.error(f"Phase {phase} execution timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Phase {phase} execution failed: {e}")
            return None
    
    def run_unified(
        self,
        phases: Optional[List[int]] = None,
        ticks: int = 120,
        timeout: int = 600
    ) -> Dict[int, PhaseMetrics]:
        """
        Execute unified phase workflow (default: phases 12-15).
        
        Args:
            phases: List of phases to run (default: [12, 13, 14, 15])
            ticks: Ticks per phase
            timeout: Total execution timeout
            
        Returns:
            Dictionary mapping phase numbers to metrics
        """
        if phases is None:
            phases = [12, 13, 14, 15]
        
        results = {}
        cmd = [self.qallow_binary, "run", "unified", f"--integrate-ticks={ticks}"]
        
        logger.info(f"Executing unified workflow: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            execution_time = time.time() - start_time
            
            if result.returncode != 0:
                logger.error(f"Unified execution failed: {result.stderr}")
                return results
            
            logger.info(f"Unified workflow completed in {execution_time:.2f}s")
            
            # Parse telemetry for each phase
            for phase in phases:
                metrics = self._parse_telemetry(phase, execution_time / len(phases))
                if metrics:
                    results[phase] = metrics
            
            return results
            
        except subprocess.TimeoutExpired:
            logger.error(f"Unified execution timed out after {timeout}s")
            return results
        except Exception as e:
            logger.error(f"Unified execution failed: {e}")
            return results
    
    def _parse_telemetry(self, phase: int, execution_time: float) -> Optional[PhaseMetrics]:
        """
        Parse telemetry CSV files for a phase.
        
        Returns:
            PhaseMetrics with aggregated phase data
        """
        # Determine log file
        if phase == 13:
            log_file = self.logs_dir / "phase13.csv"
        elif phase in [14, 15]:
            log_file = self.logs_dir / "lattice_integrations.csv"
        else:
            log_file = self.logs_dir / f"phase{phase}.csv"
        
        if not log_file.exists():
            logger.warning(f"Log file not found: {log_file}")
            return None
        
        try:
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if not rows:
                logger.warning(f"No data in log file: {log_file}")
                return None
            
            # Calculate aggregates
            tick_count = len(rows)
            avg_coherence = float(rows[-1].get('avg_coherence', 0))
            phase_drift = float(rows[-1].get('phase_drift', 0))
            ethics_total = float(rows[-1].get('ethics_total', 0))
            sustainability = float(rows[-1].get('sustainability', 0))
            compassion = float(rows[-1].get('compassion', 0))
            harmony = float(rows[-1].get('harmony', 0))
            
            metrics = PhaseMetrics(
                phase_number=phase,
                tick_count=tick_count,
                avg_coherence=avg_coherence,
                phase_drift=phase_drift,
                ethics_total=ethics_total,
                sustainability=sustainability,
                compassion=compassion,
                harmony=harmony,
                execution_time=execution_time,
                log_file=str(log_file)
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to parse telemetry for phase {phase}: {e}")
            return None


class QallowLightningAgent:
    """Agent Lightning instrumented Qallow quantum optimizer."""
    
    def __init__(self, runner: QallowPhaseRunner, agent_id: str = "qallow-optimizer"):
        """Initialize the Lightning-instrumented agent."""
        self.runner = runner
        self.agent_id = agent_id
        self.session_id = f"{agent_id}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Initialized QallowLightningAgent: {self.session_id}")
    
    def optimize_phase(
        self,
        phase: int,
        ticks: int = 120,
        target_coherence: float = 0.95
    ) -> Tuple[PhaseMetrics, float]:
        """
        Run a single phase with Agent Lightning instrumentation.
        
        Args:
            phase: Qallow phase number
            ticks: Execution ticks
            target_coherence: Target coherence threshold for reward calculation
            
        Returns:
            Tuple of (metrics, reward)
        """
        task_id = f"{self.agent_id}-phase-{phase}-{int(time.time())}"
        
        # Emit task start
        agl.emit_task_start(
            task_id=task_id,
            task=f"Optimize Qallow Phase {phase}",
            metadata={
                "phase": phase,
                "ticks": ticks,
                "target_coherence": target_coherence
            }
        )
        
        logger.info(f"[Lightning] Started: {task_id}")
        
        # Execute phase
        metrics = self.runner.run_phase(phase, ticks)
        
        if metrics is None:
            # Emit failure
            agl.emit_task_complete(
                task_id=task_id,
                result="FAILED",
                reward=-1.0
            )
            logger.error(f"[Lightning] Failed: {task_id}")
            return None, -1.0
        
        # Calculate reward based on coherence and ethics
        reward = self._calculate_reward(metrics, target_coherence)
        
        # Emit task completion with reward
        agl.emit_task_complete(
            task_id=task_id,
            result={
                "phase": phase,
                "coherence": metrics.avg_coherence,
                "ethics": metrics.ethics_total,
                "execution_time": metrics.execution_time
            },
            reward=reward
        )
        
        logger.info(
            f"[Lightning] Completed: {task_id} | "
            f"Coherence: {metrics.avg_coherence:.4f} | "
            f"Reward: {reward:.4f}"
        )
        
        return metrics, reward
    
    def optimize_unified(
        self,
        phases: Optional[List[int]] = None,
        ticks: int = 120,
        target_coherence: float = 0.95
    ) -> Tuple[Dict[int, PhaseMetrics], float]:
        """
        Run unified workflow with Agent Lightning instrumentation.
        
        Args:
            phases: List of phases to optimize
            ticks: Execution ticks per phase
            target_coherence: Target coherence threshold
            
        Returns:
            Tuple of (metrics_dict, cumulative_reward)
        """
        if phases is None:
            phases = [12, 13, 14, 15]
        
        task_id = f"{self.agent_id}-unified-{int(time.time())}"
        
        # Emit unified task start
        agl.emit_task_start(
            task_id=task_id,
            task="Optimize Qallow Unified Workflow",
            metadata={
                "phases": phases,
                "ticks": ticks,
                "target_coherence": target_coherence
            }
        )
        
        logger.info(f"[Lightning] Started unified: {task_id}")
        
        # Execute unified workflow
        results = self.runner.run_unified(phases, ticks)
        
        if not results:
            agl.emit_task_complete(
                task_id=task_id,
                result="FAILED",
                reward=-1.0
            )
            logger.error(f"[Lightning] Unified failed: {task_id}")
            return {}, -1.0
        
        # Calculate cumulative reward
        cumulative_reward = sum(
            self._calculate_reward(metrics, target_coherence)
            for metrics in results.values()
        ) / len(results)
        
        # Emit unified completion
        agl.emit_task_complete(
            task_id=task_id,
            result={
                "phases": list(results.keys()),
                "avg_coherence": sum(m.avg_coherence for m in results.values()) / len(results),
                "avg_ethics": sum(m.ethics_total for m in results.values()) / len(results)
            },
            reward=cumulative_reward
        )
        
        logger.info(
            f"[Lightning] Unified completed: {task_id} | "
            f"Avg Reward: {cumulative_reward:.4f}"
        )
        
        return results, cumulative_reward
    
    @staticmethod
    def _calculate_reward(
        metrics: PhaseMetrics,
        target_coherence: float
    ) -> float:
        """
        Calculate reward based on phase metrics.
        
        Reward formula:
            - Base: coherence vs target (0-1 scale)
            - Bonus: ethics alignment (sustainability + compassion + harmony / 3)
            - Penalty: phase drift
        """
        # Coherence reward: how close to target
        coherence_reward = min(
            metrics.avg_coherence / target_coherence,
            1.0
        )
        
        # Ethics reward: average ethics metrics normalized
        ethics_reward = (
            metrics.sustainability +
            metrics.compassion +
            metrics.harmony
        ) / 3.0
        
        # Drift penalty: lower is better
        drift_penalty = 1.0 - min(metrics.phase_drift, 1.0)
        
        # Combined reward: 50% coherence, 30% ethics, 20% stability
        reward = (
            0.5 * coherence_reward +
            0.3 * ethics_reward +
            0.2 * drift_penalty
        )
        
        return max(0.0, min(1.0, reward))


def main():
    """Main entry point demonstrating the integration."""
    print("\n" + "=" * 70)
    print("Qallow + Agent Lightning Integration Demo")
    print("=" * 70 + "\n")
    
    # Initialize components
    runner = QallowPhaseRunner()
    agent = QallowLightningAgent(runner, agent_id="qallow-rl-optimizer")
    
    print("[INFO] Qallow Phase Runner initialized")
    print("[INFO] Agent Lightning instrumentation enabled\n")
    
    # Demo 1: Single phase optimization
    print("-" * 70)
    print("Demo 1: Single Phase Optimization (Phase 13)")
    print("-" * 70)
    
    metrics, reward = agent.optimize_phase(phase=13, ticks=100, target_coherence=0.80)
    
    if metrics:
        print(f"\n✓ Phase 13 Results:")
        print(f"  Coherence:     {metrics.avg_coherence:.4f}")
        print(f"  Ethics Total:  {metrics.ethics_total:.4f}")
        print(f"  Phase Drift:   {metrics.phase_drift:.4f}")
        print(f"  Reward:        {reward:.4f}")
        print(f"  Exec Time:     {metrics.execution_time:.2f}s\n")
    
    # Demo 2: Unified workflow optimization
    print("-" * 70)
    print("Demo 2: Unified Workflow Optimization (Phases 12-15)")
    print("-" * 70)
    
    results, cumulative_reward = agent.optimize_unified(
        phases=[12, 13, 14, 15],
        ticks=100,
        target_coherence=0.80
    )
    
    if results:
        print(f"\n✓ Unified Workflow Results:")
        for phase, metrics in sorted(results.items()):
            print(f"  Phase {phase}:")
            print(f"    Coherence:  {metrics.avg_coherence:.4f}")
            print(f"    Ethics:     {metrics.ethics_total:.4f}")
        print(f"  Cumulative Reward: {cumulative_reward:.4f}\n")
    
    # Demo 3: Agent Lightning training setup
    print("-" * 70)
    print("Demo 3: Next Steps - Agent Lightning Training")
    print("-" * 70)
    print("""
Next, you can train the Qallow optimizer with reinforcement learning:

1. Start the LightningStore server:
   $ agl store --algorithm=ppo
   
2. The agent traces are automatically sent to LightningStore
   
3. Monitor training progress:
   $ agl store --monitor
   
4. Experiment with different RL algorithms:
   - PPO (Proximal Policy Optimization) - default, balanced
   - GRPO (Group Relative Policy Optimization) - group-aware
   - VERL (Versatile RL) - flexible, custom reward modeling

5. Fine-tune Qallow phase parameters based on training results
""")
    
    print("=" * 70)
    print("✨ Integration Demo Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
