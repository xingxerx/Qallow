#!/usr/bin/env python3
"""
Qallow AGI Integration - Complete Agent Lightning Integration
Connects all AGI self-learning components with Qallow's existing infrastructure
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import Qallow AGI modules
from agi_self_learning import QallowAGISelfLearning, create_agi_learner
from agi_telemetry_bridge import AGITelemetryBridge
from quantum_learning_system import QuantumLearningSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QallowAGIIntegration:
    """
    Complete AGI Integration for Qallow
    
    Integrates:
    - Agent Lightning RL framework
    - Quantum algorithm optimization
    - Ethics decision-making
    - Phase execution optimization
    - Telemetry and monitoring
    """
    
    def __init__(self, 
                 workspace_dir: str = '/home/xing/Qallow',
                 enable_rl: bool = True):
        """Initialize complete AGI integration"""
        
        self.workspace_dir = Path(workspace_dir)
        self.enable_rl = enable_rl
        
        # Initialize components
        logger.info("Initializing Qallow AGI Integration...")
        
        self.agi_learner = create_agi_learner(enable_rl=enable_rl)
        self.telemetry_bridge = AGITelemetryBridge(
            telemetry_dir=str(self.workspace_dir / 'telemetry')
        )
        self.quantum_learner = QuantumLearningSystem()
        
        # Integration state
        self.integration_state = {
            'enabled': True,
            'rl_enabled': enable_rl,
            'start_time': datetime.now().isoformat(),
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0
        }
        
        logger.info("✅ Qallow AGI Integration initialized successfully!")
    
    # ========================================================================
    # Quantum Algorithm Optimization
    # ========================================================================
    
    def select_optimal_quantum_algorithm(self, problem_type: str, 
                                          constraints: Dict) -> Dict:
        """
        Select optimal quantum algorithm with RL optimization
        
        Args:
            problem_type: Type of quantum problem
            constraints: Problem constraints
        
        Returns:
            Selection result with algorithm and metadata
        """
        
        logger.info(f"Selecting quantum algorithm for {problem_type}")
        
        # Use AGI learner to select algorithm
        algorithm, confidence = self.agi_learner.select_quantum_algorithm(
            problem_type, constraints
        )
        
        # Record telemetry
        self.telemetry_bridge.record_metric(
            'quantum.algorithm.selection',
            confidence,
            tags={'algorithm': algorithm, 'problem_type': problem_type}
        )
        
        result = {
            'algorithm': algorithm,
            'confidence': confidence,
            'problem_type': problem_type,
            'constraints': constraints,
            'timestamp': datetime.now().isoformat()
        }
        
        self.integration_state['total_tasks'] += 1
        self.integration_state['successful_tasks'] += 1
        
        return result
    
    # ========================================================================
    # Ethics Decision Making
    # ========================================================================
    
    def make_ethics_decision(self, scenario: Dict) -> Dict:
        """
        Make ethics decision with RL-optimized weights
        
        Args:
            scenario: Ethics scenario
        
        Returns:
            Decision with scores and reasoning
        """
        
        logger.info(f"Making ethics decision for scenario {scenario.get('id', 'unknown')}")
        
        # Use AGI learner for ethics decision
        decision = self.agi_learner.make_ethics_decision(scenario)
        
        # Record telemetry
        self.telemetry_bridge.record_metric(
            'ethics.decision.score',
            decision['total_score'],
            tags={'approved': str(decision['approved'])}
        )
        
        self.integration_state['total_tasks'] += 1
        self.integration_state['successful_tasks'] += 1
        
        return decision
    
    # ========================================================================
    # Phase Execution Optimization
    # ========================================================================
    
    def optimize_phase(self, phase_num: int, config: Dict) -> Dict:
        """
        Optimize phase execution parameters
        
        Args:
            phase_num: Phase number
            config: Current configuration
        
        Returns:
            Optimized configuration
        """
        
        logger.info(f"Optimizing phase {phase_num}")
        
        # Use AGI learner to optimize
        optimized_config = self.agi_learner.optimize_phase_execution(
            phase_num, config
        )
        
        # Record telemetry
        self.telemetry_bridge.record_metric(
            f'phase.{phase_num}.optimization',
            1.0,
            tags={'phase': str(phase_num)}
        )
        
        return optimized_config
    
    def report_phase_performance(self, phase_num: int, metrics: Dict):
        """
        Report phase execution performance
        
        Args:
            phase_num: Phase number
            metrics: Performance metrics
        """
        
        logger.info(f"Reporting phase {phase_num} performance")
        
        # Report to AGI learner
        self.agi_learner.report_phase_performance(phase_num, metrics)
        
        # Record telemetry
        for metric_name, value in metrics.items():
            self.telemetry_bridge.record_metric(
                f'phase.{phase_num}.{metric_name}',
                value,
                tags={'phase': str(phase_num)}
            )
    
    # ========================================================================
    # Quantum Learning Integration
    # ========================================================================
    
    def process_quantum_results(self, results: Dict) -> Dict:
        """
        Process quantum execution results with learning
        
        Args:
            results: Quantum execution results
        
        Returns:
            Analysis with learning signals
        """
        
        logger.info("Processing quantum results")
        
        # Use quantum learner
        analysis = self.quantum_learner.process_quantum_results(results)
        
        # Extract learning signals for RL
        if 'learning_signals' in analysis:
            for signal_name, value in analysis['learning_signals'].items():
                self.telemetry_bridge.record_metric(
                    f'quantum.{signal_name}',
                    value,
                    tags={'source': 'quantum_learner'}
                )
        
        return analysis
    
    # ========================================================================
    # Monitoring and Telemetry
    # ========================================================================
    
    def get_integration_status(self) -> Dict:
        """Get current integration status"""
        
        status = {
            'integration': self.integration_state,
            'agi_learning': self.agi_learner.get_learning_stats(),
            'telemetry': self.telemetry_bridge.generate_dashboard_data(),
            'timestamp': datetime.now().isoformat()
        }
        
        return status
    
    def export_telemetry(self):
        """Export all telemetry data"""
        
        logger.info("Exporting telemetry data")
        
        # Flush metrics
        self.telemetry_bridge.flush_metrics()
        
        # Export dashboard
        self.telemetry_bridge.export_dashboard_json()
        
        # Export AGI learning data
        self.agi_learner.export_learning_data(
            str(self.workspace_dir / 'telemetry' / 'agi_learning_data.json')
        )
        
        logger.info("✅ Telemetry exported successfully")
    
    def generate_report(self) -> str:
        """Generate integration report"""
        
        status = self.get_integration_status()
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║           Qallow AGI Integration Report                              ║
╚══════════════════════════════════════════════════════════════════════╝

Integration Status:
  • Enabled: {status['integration']['enabled']}
  • RL Enabled: {status['integration']['rl_enabled']}
  • Start Time: {status['integration']['start_time']}
  • Total Tasks: {status['integration']['total_tasks']}
  • Successful: {status['integration']['successful_tasks']}
  • Failed: {status['integration']['failed_tasks']}

AGI Learning Statistics:
  • Episodes: {status['agi_learning']['episodes']}
  • Total Reward: {status['agi_learning']['total_reward']:.3f}
  • Exploration Rate: {status['agi_learning']['exploration_rate']:.3f}
  • Active Tasks: {status['agi_learning']['active_tasks']}
  • Completed Tasks: {status['agi_learning']['completed_tasks']}

Ethics Weights:
  • Safety: {status['agi_learning']['ethics_weights']['safety']:.3f}
  • Compassion: {status['agi_learning']['ethics_weights']['compassion']:.3f}
  • Harmony: {status['agi_learning']['ethics_weights']['harmony']:.3f}

Telemetry:
  • Total Traces: {status['telemetry']['total_traces']}
  • Recent Traces: {status['telemetry']['recent_traces']}
  • Mean Reward: {status['telemetry']['reward_stats']['mean']:.3f}

Algorithm Preferences:
{self._format_algorithm_preferences(status['agi_learning']['algorithm_preferences'])}

Phase Performance:
{self._format_phase_performance(status['agi_learning']['phase_performance_summary'])}

Generated: {status['timestamp']}
╚══════════════════════════════════════════════════════════════════════╝
"""
        
        return report
    
    def _format_algorithm_preferences(self, prefs: Dict) -> str:
        """Format algorithm preferences for report"""
        if not prefs:
            return "  (No preferences learned yet)"
        
        lines = []
        for problem_type, algorithms in prefs.items():
            lines.append(f"  {problem_type}:")
            for algo, score in algorithms.items():
                lines.append(f"    - {algo}: {score:.3f}")
        
        return "\n".join(lines)
    
    def _format_phase_performance(self, perf: Dict) -> str:
        """Format phase performance for report"""
        if not perf:
            return "  (No phase performance data yet)"
        
        lines = []
        for phase, stats in perf.items():
            lines.append(f"  {phase}:")
            lines.append(f"    - Avg Reward: {stats['avg_reward']:.3f}")
            lines.append(f"    - Executions: {stats['num_executions']}")
        
        return "\n".join(lines)


# ============================================================================
# Demo and Testing
# ============================================================================

def demo_full_integration():
    """Demonstrate complete AGI integration"""
    
    print("=" * 70)
    print("Qallow AGI Integration - Complete Demo")
    print("=" * 70)
    
    # Create integration
    integration = QallowAGIIntegration(enable_rl=False)  # RL optional
    
    print("\n1. Quantum Algorithm Selection")
    print("-" * 70)
    result = integration.select_optimal_quantum_algorithm(
        problem_type='optimization',
        constraints={'max_qubits': 10, 'max_depth': 50}
    )
    print(f"   Algorithm: {result['algorithm']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    
    print("\n2. Ethics Decision")
    print("-" * 70)
    decision = integration.make_ethics_decision({
        'id': 'demo-001',
        'safety': 0.9,
        'compassion': 0.85,
        'harmony': 0.88
    })
    print(f"   Decision: {'APPROVED' if decision['approved'] else 'REJECTED'}")
    print(f"   Score: {decision['total_score']:.3f}")
    
    print("\n3. Phase Optimization")
    print("-" * 70)
    config = integration.optimize_phase(13, {'ticks': 120, 'lattice_ticks': 64})
    print(f"   Optimized: {config}")
    
    integration.report_phase_performance(13, {
        'execution_time': 2.1,
        'success_rate': 0.96,
        'error_rate': 0.04,
        'coherence': 0.91
    })
    print("   Performance reported")
    
    print("\n4. Integration Report")
    print("-" * 70)
    print(integration.generate_report())
    
    print("\n5. Export Telemetry")
    print("-" * 70)
    integration.export_telemetry()
    print("   ✅ Telemetry exported")
    
    print("\n" + "=" * 70)
    print("✨ Complete AGI Integration Demo Finished!")
    print("=" * 70)


if __name__ == "__main__":
    demo_full_integration()

