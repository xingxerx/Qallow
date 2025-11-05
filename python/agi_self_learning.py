#!/usr/bin/env python3
"""
Qallow AGI Self-Learning Module
Provides a reinforcement-style feedback loop for Qallow's AGI system without external dependencies.
"""




from datetime import datetime

# Agent Lightning integration removed; keep flag for compatibility.
AGENT_LIGHTNING_AVAILABLE = False

# Helper functions
def clip(val, min_val, max_val):
    """Clip value to range"""
    return max(min_val, min(max_val, val))

def mean(x):
    """Calculate mean"""
    return sum(x) / len(x) if x else 0

def std(x):
    """Calculate standard deviation"""
    if not x:
        return 0
    mean_val = mean(x)
    return (sum((i - mean_val) ** 2 for i in x) / len(x)) ** 0.5

# Qallow imports (optional)
try:
    from quantum_learning_system import QuantumLearningSystem
    QUANTUM_LEARNING_AVAILABLE = True
except ImportError:
    QUANTUM_LEARNING_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _emit_task_event(event_type: str, task_id: str, payload: Optional[Dict[str, Any]] = None):
    """Previously emitted Agent Lightning events; now logs for traceability."""
    details = payload or {}
    logger.debug("Task event %s for %s: %s", event_type, task_id, details)


def _emit_reward_span(task_id: str, reward_value: float):
    """Placeholder hook for reward reporting."""
    logger.debug("Reward %.4f recorded for task %s", reward_value, task_id)


class QallowAGISelfLearning:
    """
    AGI self-learning loop using internal reinforcement-style scoring.
    
    This module enables Qallow's AGI to:
    1. Learn from quantum algorithm performance
    2. Improve ethics decision-making
    3. Optimize phase execution strategies
    4. Self-improve through feedback without external services
    """
    
    def __init__(self,
                 state_file: str = 'agi_learning_state.json',
                 enable_rl: bool = True):
        """Initialize AGI self-learning system"""
        
        self.state_file = Path(state_file)
        self.enable_rl = enable_rl

        # Initialize state
        self.state = self._load_state()

        # Initialize quantum learner if available
        if QUANTUM_LEARNING_AVAILABLE:
            self.quantum_learner = QuantumLearningSystem()
        else:
            self.quantum_learner = None
        
        # Learning metrics
        self.episode_count = 0
        self.total_reward = 0.0
        self.best_performance = float('-inf')
        
        # Task tracking
        self.active_tasks = {}
        self.completed_tasks = []
        
        logger.info(f"Qallow AGI Self-Learning initialized (RL: {self.enable_rl})")
    
    def _load_state(self) -> Dict:
        """Load AGI learning state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        
        return {
            'version': '1.0.0',
            'learning_rate': 0.001,
            'discount_factor': 0.99,
            'exploration_rate': 0.1,
            'episodes': 0,
            'total_reward': 0.0,
            'best_reward': float('-inf'),
            'algorithm_preferences': {},
            'ethics_weights': {
                'safety': 1.0,
                'compassion': 1.0,
                'harmony': 1.0
            },
            'phase_performance': {},
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_state(self):
        """Save AGI learning state"""
        self.state['last_updated'] = datetime.now().isoformat()
        self.state['episodes'] = self.episode_count
        self.state['total_reward'] = self.total_reward
        self.state['best_reward'] = self.best_performance
        
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        
        logger.debug(f"State saved: {self.state_file}")
    
    # ========================================================================
    # Quantum Algorithm Selection Agent
    # ========================================================================
    
    def select_quantum_algorithm(self, 
                                  problem_type: str, 
                                  constraints: Dict) -> Tuple[str, float]:
        """
        Agent that selects optimal quantum algorithm using RL
        
        Args:
            problem_type: Type of quantum problem (e.g., 'optimization', 'simulation')
            constraints: Problem constraints (qubits, depth, etc.)
        
        Returns:
            (algorithm_name, confidence_score)
        """
        task_id = f"quantum-algo-{problem_type}-{datetime.now().timestamp()}"
        
        # Emit task start event for internal telemetry logging
        if self.enable_rl:
            _emit_task_event(
                event_type="task_start",
                task_id=task_id,
                payload={
                    'task_type': "quantum_algorithm_selection",
                    'problem_type': problem_type,
                    'constraints': constraints
                }
            )
        
        # Algorithm selection logic
        algorithm, confidence = self._select_algorithm_logic(problem_type, constraints)
        
        # Calculate reward based on expected performance
        reward = self._calculate_algorithm_reward(algorithm, problem_type, constraints)
        
        # Emit completion and reward events
        if self.enable_rl:
            _emit_task_event(
                event_type="task_complete",
                task_id=task_id,
                payload={
                    'task_type': "quantum_algorithm_selection",
                    'result': {'algorithm': algorithm, 'confidence': confidence}
                }
            )
            _emit_reward_span(task_id, reward)
        
        # Update learning state
        self._update_algorithm_preferences(problem_type, algorithm, reward)
        
        logger.info(f"Selected algorithm: {algorithm} (confidence: {confidence:.3f}, reward: {reward:.3f})")
        
        return algorithm, confidence
    
    def _select_algorithm_logic(self, problem_type: str, constraints: Dict) -> Tuple[str, float]:
        """Core algorithm selection logic with learned preferences"""
        
        # Available algorithms
        algorithms = {
            'optimization': ['QAOA', 'VQE', 'Grover'],
            'simulation': ['Trotter', 'VQE', 'QPE'],
            'search': ['Grover', 'Amplitude_Amplification'],
            'factoring': ['Shor', 'VQE'],
            'ml': ['QSVM', 'QNN', 'VQE']
        }
        
        available = algorithms.get(problem_type, ['VQE'])
        
        # Use learned preferences with exploration
        preferences = self.state['algorithm_preferences'].get(problem_type, {})

        if random.random() < self.state['exploration_rate']:
            # Explore: random selection
            algorithm = random.choice(available)
            confidence = 0.5
        else:
            # Exploit: use learned preferences
            if preferences:
                algorithm = max(preferences, key=preferences.get)
                confidence = preferences[algorithm]
            else:
                algorithm = available[0]
                confidence = 0.7
        
        return algorithm, confidence
    
    def _calculate_algorithm_reward(self, algorithm: str, problem_type: str, 
                                     constraints: Dict) -> float:
        """Calculate reward for algorithm selection"""
        
        # Base reward
        reward = 0.5
        
        # Bonus for matching problem type
        optimal_matches = {
            'optimization': ['QAOA', 'VQE'],
            'simulation': ['Trotter', 'VQE'],
            'search': ['Grover'],
            'factoring': ['Shor']
        }
        
        if algorithm in optimal_matches.get(problem_type, []):
            reward += 0.3
        
        # Penalty for constraint violations
        if 'max_qubits' in constraints:
            required_qubits = self._estimate_qubits(algorithm)
            if required_qubits > constraints['max_qubits']:
                reward -= 0.2
        
        if 'max_depth' in constraints:
            estimated_depth = self._estimate_depth(algorithm)
            if estimated_depth > constraints['max_depth']:
                reward -= 0.1
        
        return clip(reward, -1.0, 1.0)

    def _update_algorithm_preferences(self, problem_type: str, algorithm: str, reward: float):
        """Update learned algorithm preferences"""
        
        if problem_type not in self.state['algorithm_preferences']:
            self.state['algorithm_preferences'][problem_type] = {}
        
        prefs = self.state['algorithm_preferences'][problem_type]
        
        # Update with learning rate
        lr = self.state['learning_rate']
        current = prefs.get(algorithm, 0.5)
        prefs[algorithm] = current + lr * (reward - current)
        
        self._save_state()
    
    def _estimate_qubits(self, algorithm: str) -> int:
        """Estimate qubits needed for algorithm"""
        estimates = {
            'QAOA': 10, 'VQE': 8, 'Grover': 12,
            'Shor': 20, 'QPE': 15, 'Trotter': 10
        }
        return estimates.get(algorithm, 10)
    
    def _estimate_depth(self, algorithm: str) -> int:
        """Estimate circuit depth for algorithm"""
        estimates = {
            'QAOA': 50, 'VQE': 30, 'Grover': 100,
            'Shor': 200, 'QPE': 150, 'Trotter': 80
        }
        return estimates.get(algorithm, 50)
    
    # ========================================================================
    # Ethics Decision Agent
    # ========================================================================
    
    def make_ethics_decision(self, scenario: Dict) -> Dict:
        """
        Agent that makes ethics decisions using RL-optimized weights
        
        Args:
            scenario: Ethics scenario with context
        
        Returns:
            Decision with scores and reasoning
        """
        task_id = f"ethics-{scenario.get('id', datetime.now().timestamp())}"
        
        if self.enable_rl:
            _emit_task_event(
                event_type="task_start",
                task_id=task_id,
                payload={
                    'task_type': "ethics_decision",
                    'scenario': scenario
                }
            )
        
        # Calculate ethics scores using learned weights
        weights = self.state['ethics_weights']
        
        decision = {
            'safety_score': scenario.get('safety', 0.5) * weights['safety'],
            'compassion_score': scenario.get('compassion', 0.5) * weights['compassion'],
            'harmony_score': scenario.get('harmony', 0.5) * weights['harmony'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Total ethics score (E = S + C + H)
        decision['total_score'] = (
            decision['safety_score'] + 
            decision['compassion_score'] + 
            decision['harmony_score']
        )
        
        # Decision threshold
        decision['approved'] = decision['total_score'] >= 2.0
        decision['reasoning'] = self._generate_ethics_reasoning(decision)
        
        # Calculate reward based on ethics alignment
        reward = self._calculate_ethics_reward(decision, scenario)
        
        if self.enable_rl:
            _emit_task_event(
                event_type="task_complete",
                task_id=task_id,
                payload={
                    'task_type': "ethics_decision",
                    'result': decision
                }
            )
            _emit_reward_span(task_id, reward)
        
        # Update ethics weights
        self._update_ethics_weights(decision, reward)
        
        logger.info(f"Ethics decision: {decision['approved']} (score: {decision['total_score']:.3f})")
        
        return decision
    
    def _generate_ethics_reasoning(self, decision: Dict) -> str:
        """Generate human-readable reasoning for ethics decision"""
        
        reasoning = []
        
        if decision['safety_score'] > 0.8:
            reasoning.append("High safety alignment")
        if decision['compassion_score'] > 0.8:
            reasoning.append("Strong compassion factor")
        if decision['harmony_score'] > 0.8:
            reasoning.append("Excellent harmony balance")
        
        if decision['total_score'] < 2.0:
            reasoning.append("Below ethics threshold")
        
        return "; ".join(reasoning) if reasoning else "Standard evaluation"
    
    def _calculate_ethics_reward(self, decision: Dict, scenario: Dict) -> float:
        """Calculate reward for ethics decision"""
        
        # Reward for balanced scores
        scores = [
            decision['safety_score'],
            decision['compassion_score'],
            decision['harmony_score']
        ]
        
        balance_reward = 1.0 - std(scores)

        # Reward for meeting threshold
        threshold_reward = 0.5 if decision['total_score'] >= 2.0 else -0.3

        # Bonus for human feedback (if available)
        human_feedback = scenario.get('human_feedback', 0.0)

        total_reward = balance_reward * 0.4 + threshold_reward * 0.4 + human_feedback * 0.2

        return clip(total_reward, -1.0, 1.0)
    
    def _update_ethics_weights(self, decision: Dict, reward: float):
        """Update ethics weights based on reward"""
        
        lr = self.state['learning_rate']
        
        # Adjust weights based on reward
        for key in ['safety', 'compassion', 'harmony']:
            current = self.state['ethics_weights'][key]
            # Increase weight if reward is positive, decrease if negative
            adjustment = lr * reward * 0.1
            self.state['ethics_weights'][key] = clip(current + adjustment, 0.5, 2.0)
        
        self._save_state()

    # ========================================================================
    # Phase Execution Optimizer
    # ========================================================================

    def optimize_phase_execution(self, phase_num: int, config: Dict) -> Dict:
        """
        Agent that optimizes phase execution parameters using RL

        Args:
            phase_num: Phase number (12-20)
            config: Current phase configuration

        Returns:
            Optimized configuration
        """
        task_id = f"phase-{phase_num}-{datetime.now().timestamp()}"

        if self.enable_rl:
            _emit_task_event(
                event_type="task_start",
                task_id=task_id,
                payload={
                    'task_type': "phase_optimization",
                    'phase': phase_num,
                    'config': config
                }
            )

        # Get learned performance data
        phase_key = f"phase_{phase_num}"
        perf_history = self.state['phase_performance'].get(phase_key, [])

        # Optimize parameters
        optimized_config = self._optimize_phase_params(phase_num, config, perf_history)

        # Store for later reward calculation
        self.active_tasks[task_id] = {
            'phase': phase_num,
            'config': optimized_config,
            'start_time': datetime.now().isoformat()
        }

        logger.info(f"Phase {phase_num} optimized: {optimized_config}")

        return optimized_config

    def report_phase_performance(self, phase_num: int, metrics: Dict):
        """
        Report phase execution performance for RL feedback

        Args:
            phase_num: Phase number
            metrics: Performance metrics (execution_time, success_rate, etc.)
        """

        # Find corresponding task
        task_id = None
        for tid, task_data in self.active_tasks.items():
            if task_data['phase'] == phase_num:
                task_id = tid
                break

        if not task_id:
            logger.warning(f"No active task found for phase {phase_num}")
            return

        # Calculate reward from metrics
        reward = self._calculate_phase_reward(metrics)

        if self.enable_rl:
            _emit_task_event(
                event_type="task_complete",
                task_id=task_id,
                payload={
                    'task_type': "phase_optimization",
                    'phase': phase_num,
                    'metrics': metrics
                }
            )
            _emit_reward_span(task_id, reward)

        # Update phase performance history
        phase_key = f"phase_{phase_num}"
        if phase_key not in self.state['phase_performance']:
            self.state['phase_performance'][phase_key] = []

        self.state['phase_performance'][phase_key].append({
            'metrics': metrics,
            'reward': reward,
            'timestamp': datetime.now().isoformat()
        })

        # Keep only last 100 entries
        self.state['phase_performance'][phase_key] = \
            self.state['phase_performance'][phase_key][-100:]

        # Clean up active task
        del self.active_tasks[task_id]

        self._save_state()

        logger.info(f"Phase {phase_num} performance reported (reward: {reward:.3f})")

    def _optimize_phase_params(self, phase_num: int, config: Dict,
                                history: List[Dict]) -> Dict:
        """Optimize phase parameters based on history"""

        optimized = config.copy()

        if not history:
            return optimized

        # Find best performing configuration
        best_entry = max(history, key=lambda x: x.get('reward', 0))
        best_metrics = best_entry.get('metrics', {})

        # Adjust parameters based on best performance
        if 'ticks' in config and 'execution_time' in best_metrics:
            # Optimize tick count
            if best_metrics['execution_time'] < 1.0:
                optimized['ticks'] = int(config['ticks'] * 1.1)
            elif best_metrics['execution_time'] > 5.0:
                optimized['ticks'] = int(config['ticks'] * 0.9)

        if 'lattice_ticks' in config:
            # Optimize lattice ticks
            avg_reward = mean([e.get('reward', 0) for e in history[-10:]])
            if avg_reward > 0.5:
                optimized['lattice_ticks'] = int(config['lattice_ticks'] * 1.05)
            elif avg_reward < 0:
                optimized['lattice_ticks'] = int(config['lattice_ticks'] * 0.95)

        return optimized

    def _calculate_phase_reward(self, metrics: Dict) -> float:
        """Calculate reward from phase execution metrics"""

        reward = 0.0

        # Reward for fast execution
        exec_time = metrics.get('execution_time', 10.0)
        if exec_time < 2.0:
            reward += 0.3
        elif exec_time > 10.0:
            reward -= 0.2

        # Reward for high success rate
        success_rate = metrics.get('success_rate', 0.5)
        reward += success_rate * 0.4

        # Reward for low error rate
        error_rate = metrics.get('error_rate', 0.5)
        reward -= error_rate * 0.3

        # Bonus for quantum coherence
        coherence = metrics.get('coherence', 0.0)
        reward += coherence * 0.2

        return clip(reward, -1.0, 1.0)

    # ========================================================================
    # Telemetry and Monitoring
    # ========================================================================

    def get_learning_stats(self) -> Dict:
        """Get current learning statistics"""

        return {
            'episodes': self.episode_count,
            'total_reward': self.total_reward,
            'best_performance': self.best_performance,
            'exploration_rate': self.state['exploration_rate'],
            'learning_rate': self.state['learning_rate'],
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'algorithm_preferences': self.state['algorithm_preferences'],
            'ethics_weights': self.state['ethics_weights'],
            'phase_performance_summary': self._summarize_phase_performance()
        }

    def _summarize_phase_performance(self) -> Dict:
        """Summarize phase performance across all phases"""

        summary = {}

        for phase_key, history in self.state['phase_performance'].items():
            if not history:
                continue

            rewards = [e.get('reward', 0) for e in history]
            summary[phase_key] = {
                'avg_reward': mean(rewards),
                'max_reward': max(rewards),
                'min_reward': min(rewards),
                'std_reward': std(rewards),
                'num_executions': len(history)
            }

        return summary

    def export_learning_data(self, output_file: str):
        """Export learning data for analysis"""

        data = {
            'state': self.state,
            'stats': self.get_learning_stats(),
            'completed_tasks': self.completed_tasks,
            'export_time': datetime.now().isoformat()
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Learning data exported to {output_file}")


# ============================================================================
# Integration Functions
# ============================================================================

def create_agi_learner(enable_rl: bool = True) -> QallowAGISelfLearning:
    """Create and initialize AGI self-learning system"""
    return QallowAGISelfLearning(enable_rl=enable_rl)


def demo_agi_learning():
    """Demonstrate AGI self-learning capabilities"""

    print("=" * 70)
    print("Qallow AGI Self-Learning Demo")
    print("=" * 70)

    # Create learner
    learner = create_agi_learner(enable_rl=False)

    # Demo 1: Quantum Algorithm Selection
    print("\n1. Quantum Algorithm Selection Agent")
    print("-" * 70)

    algorithm, confidence = learner.select_quantum_algorithm(
        problem_type='optimization',
        constraints={'max_qubits': 10, 'max_depth': 50}
    )
    print(f"   Selected: {algorithm} (confidence: {confidence:.3f})")

    # Demo 2: Ethics Decision
    print("\n2. Ethics Decision Agent")
    print("-" * 70)

    scenario = {
        'id': 'test-001',
        'safety': 0.9,
        'compassion': 0.8,
        'harmony': 0.85,
        'human_feedback': 0.1
    }

    decision = learner.make_ethics_decision(scenario)
    print(f"   Decision: {'APPROVED' if decision['approved'] else 'REJECTED'}")
    print(f"   Total Score: {decision['total_score']:.3f}")
    print(f"   Reasoning: {decision['reasoning']}")

    # Demo 3: Phase Optimization
    print("\n3. Phase Execution Optimizer")
    print("-" * 70)

    config = learner.optimize_phase_execution(
        phase_num=13,
        config={'ticks': 120, 'lattice_ticks': 64}
    )
    print(f"   Optimized config: {config}")

    # Report performance
    learner.report_phase_performance(
        phase_num=13,
        metrics={
            'execution_time': 2.5,
            'success_rate': 0.95,
            'error_rate': 0.05,
            'coherence': 0.88
        }
    )

    # Show stats
    print("\n4. Learning Statistics")
    print("-" * 70)
    stats = learner.get_learning_stats()
    print(f"   Episodes: {stats['episodes']}")
    print(f"   Total Reward: {stats['total_reward']:.3f}")
    print(f"   Exploration Rate: {stats['exploration_rate']:.3f}")
    print(f"   Ethics Weights: {stats['ethics_weights']}")

    print("\n" + "=" * 70)
    print("✨ AGI Self-Learning Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_agi_learning()
