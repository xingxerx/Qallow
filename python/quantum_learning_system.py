#!/usr/bin/env python3
"""
Quantum Learning System - Adaptive learning from quantum workload outputs
Integrates with Qallow's adaptive learning framework
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class QuantumLearningSystem:
    """Adaptive learning system for quantum workloads"""
    
    def __init__(self, state_file: str = '/root/Qallow/adapt_state.json'):
        self.state_file = Path(state_file)
        self.state = self._load_state()
        self.history = []
        self.performance_metrics = {}
        
        logger.info("Quantum Learning System initialized")
    
    def _load_state(self) -> Dict:
        """Load learning state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        
        return {
            'learning_rate': 0.01,
            'human_score': 0.0,
            'iterations': 0,
            'best_params': None,
            'error_threshold': 0.01,
            'circuit_depth_target': 10,
            'entanglement_score': 0.0,
            'error_correction_enabled': True
        }
    
    def _save_state(self):
        """Save learning state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def process_quantum_results(self, results: Dict) -> Dict:
        """Process quantum execution results and extract learning signals"""
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'raw_results': results,
            'learning_signals': {},
            'recommendations': []
        }
        
        # Extract expectation values
        if 'expectation_values' in results:
            evs = np.array(results['expectation_values'])
            analysis['learning_signals']['mean_ev'] = float(np.mean(evs))
            analysis['learning_signals']['std_ev'] = float(np.std(evs))
            analysis['learning_signals']['max_ev'] = float(np.max(evs))
            analysis['learning_signals']['min_ev'] = float(np.min(evs))
        
        # Detect entanglement
        if 'measurements' in results:
            entanglement_score = self._detect_entanglement(results['measurements'])
            analysis['learning_signals']['entanglement_score'] = entanglement_score
            self.state['entanglement_score'] = entanglement_score
        
        # Analyze errors
        if 'errors' in results:
            errors = np.array(results['errors'])
            analysis['learning_signals']['mean_error'] = float(np.mean(errors))
            analysis['learning_signals']['max_error'] = float(np.max(errors))
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis['learning_signals'])
        
        # Update learning state
        self.state['iterations'] += 1
        self.state['human_score'] = analysis['learning_signals'].get('mean_ev', 0.0)
        self._save_state()
        
        self.history.append(analysis)
        
        return analysis
    
    def _detect_entanglement(self, measurements: Dict) -> float:
        """Detect entanglement from measurement statistics"""
        
        # Look for correlations in multi-qubit measurements
        correlation_sum = 0.0
        count = 0
        
        for label, data in measurements.items():
            if 'ZZ' in label or 'XX' in label:
                value = data.get('value', 0.0)
                correlation_sum += abs(value)
                count += 1
        
        if count == 0:
            return 0.0
        
        entanglement = correlation_sum / count
        return min(1.0, entanglement)  # Normalize to [0, 1]
    
    def _generate_recommendations(self, signals: Dict) -> List[str]:
        """Generate recommendations based on learning signals"""
        
        recommendations = []
        
        # Check entanglement
        if signals.get('entanglement_score', 0) < 0.3:
            recommendations.append("Low entanglement detected. Consider increasing circuit depth or using more CNOT gates.")
        elif signals.get('entanglement_score', 0) > 0.8:
            recommendations.append("High entanglement achieved. Circuit is well-designed for quantum advantage.")
        
        # Check errors
        if signals.get('mean_error', 0) > 0.1:
            recommendations.append("High measurement errors detected. Enable error mitigation or use error correction.")
        
        # Check expectation values
        if signals.get('std_ev', 0) > 0.5:
            recommendations.append("High variance in expectation values. Increase shot count for better statistics.")
        
        # Check mean expectation
        if signals.get('mean_ev', 0) < -0.5:
            recommendations.append("Negative mean expectation value. Verify circuit design and observable definitions.")
        
        return recommendations
    
    def adaptive_parameter_update(self, feedback_score: float) -> Dict:
        """Update parameters based on feedback"""
        
        # Adjust learning rate
        if feedback_score > self.state['human_score']:
            self.state['learning_rate'] *= 1.1  # Increase learning rate
        else:
            self.state['learning_rate'] *= 0.9  # Decrease learning rate
        
        # Clamp learning rate
        self.state['learning_rate'] = np.clip(self.state['learning_rate'], 0.001, 0.1)
        
        # Update human score
        self.state['human_score'] = feedback_score
        
        # Adjust circuit depth target
        if feedback_score > 0.7:
            self.state['circuit_depth_target'] = min(20, self.state['circuit_depth_target'] + 1)
        elif feedback_score < 0.3:
            self.state['circuit_depth_target'] = max(5, self.state['circuit_depth_target'] - 1)
        
        self._save_state()
        
        return {
            'new_learning_rate': self.state['learning_rate'],
            'new_circuit_depth': self.state['circuit_depth_target'],
            'human_score': feedback_score,
            'iteration': self.state['iterations']
        }
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        
        if not self.history:
            return {'status': 'No data available'}
        
        # Aggregate metrics
        all_evs = []
        all_errors = []
        all_entanglement = []
        
        for entry in self.history:
            signals = entry['learning_signals']
            all_evs.append(signals.get('mean_ev', 0.0))
            all_errors.append(signals.get('mean_error', 0.0))
            all_entanglement.append(signals.get('entanglement_score', 0.0))
        
        report = {
            'total_iterations': len(self.history),
            'average_expectation_value': float(np.mean(all_evs)),
            'average_error': float(np.mean(all_errors)),
            'average_entanglement': float(np.mean(all_entanglement)),
            'best_expectation_value': float(np.max(all_evs)),
            'worst_expectation_value': float(np.min(all_evs)),
            'trend': 'improving' if all_evs[-1] > all_evs[0] else 'degrading',
            'current_state': self.state
        }
        
        return report
    
    def save_learning_history(self, output_file: str = None):
        """Save learning history to file"""
        
        if output_file is None:
            output_file = f'/root/Qallow/data/quantum_learning_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'history': self.history,
                'performance_report': self.get_performance_report()
            }, f, indent=2)
        
        logger.info(f"Learning history saved to {output_path}")
        return output_path


class QuantumErrorCorrectionLearner:
    """Learn optimal error correction strategies"""
    
    def __init__(self):
        self.error_models = {}
        self.correction_strategies = {}
        self.performance_data = []
    
    def record_error_event(self, error_type: str, location: int, 
                          correction_applied: bool, success: bool):
        """Record error event for learning"""
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'location': location,
            'correction_applied': correction_applied,
            'success': success
        }
        
        self.performance_data.append(event)
    
    def analyze_error_patterns(self) -> Dict:
        """Analyze error patterns from recorded events"""
        
        if not self.performance_data:
            return {'status': 'No error data'}
        
        error_types = {}
        correction_success_rate = 0.0
        
        for event in self.performance_data:
            error_type = event['error_type']
            if error_type not in error_types:
                error_types[error_type] = {'count': 0, 'corrected': 0}
            
            error_types[error_type]['count'] += 1
            if event['correction_applied'] and event['success']:
                error_types[error_type]['corrected'] += 1
        
        if self.performance_data:
            correction_success_rate = sum(
                1 for e in self.performance_data 
                if e['correction_applied'] and e['success']
            ) / len(self.performance_data)
        
        return {
            'error_types': error_types,
            'correction_success_rate': correction_success_rate,
            'total_events': len(self.performance_data)
        }


def main():
    """Test quantum learning system"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize learning system
    learner = QuantumLearningSystem()
    
    # Simulate quantum results
    sample_results = {
        'expectation_values': [0.8, 0.7, 0.9, 0.6, 0.85],
        'errors': [0.05, 0.06, 0.04, 0.07, 0.05],
        'measurements': {
            'ZZ': {'value': 0.85, 'error': 0.05},
            'XX': {'value': 0.80, 'error': 0.06},
            'IZ': {'value': 0.1, 'error': 0.04}
        }
    }
    
    # Process results
    analysis = learner.process_quantum_results(sample_results)
    print(json.dumps(analysis, indent=2))
    
    # Get performance report
    report = learner.get_performance_report()
    print("\nPerformance Report:")
    print(json.dumps(report, indent=2))
    
    # Save history
    learner.save_learning_history()


if __name__ == "__main__":
    main()

