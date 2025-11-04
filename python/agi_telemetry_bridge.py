#!/usr/bin/env python3
"""
AGI Telemetry Bridge
Collects reinforcement-style telemetry from Qallow components and exports it for monitoring.
"""



from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AGITelemetryBridge:
    """Bridge reinforcement-learning telemetry into Qallow's monitoring stack."""
    
    def __init__(self, 
                 telemetry_dir: str = '/root/Qallow/telemetry',
                 enable_export: bool = True):
        """Initialize telemetry bridge"""
        
        self.telemetry_dir = Path(telemetry_dir)
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_export = enable_export
        self.traces = []
        self.metrics_buffer = []
        
        logger.info(f"AGI Telemetry Bridge initialized (dir: {telemetry_dir})")
    
    def capture_rl_trace(self, task_id: str, trace_data: Dict):
        """
        Capture RL training trace
        
        Args:
            task_id: Unique task identifier
            trace_data: RL trace data collected from Qallow subsystems
        """
        
        trace = {
            'task_id': task_id,
            'timestamp': datetime.now().isoformat(),
            'data': trace_data,
            'source': 'qallow_rl'
        }
        
        self.traces.append(trace)
        
        # Export to telemetry file
        if self.enable_export:
            self._export_trace(trace)
        
        logger.debug(f"Captured RL trace: {task_id}")
    
    def _export_trace(self, trace: Dict):
        """Export trace to telemetry file"""
        
        # Create daily trace file
        date_str = datetime.now().strftime('%Y%m%d')
        trace_file = self.telemetry_dir / f"rl_traces_{date_str}.jsonl"
        
        with open(trace_file, 'a') as f:
            f.write(json.dumps(trace) + '\n')
    
    def record_metric(self, metric_name: str, value: float, tags: Optional[Dict] = None):
        """
        Record a metric for telemetry
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags for categorization
        """
        
        metric = {
            'name': metric_name,
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'tags': tags or {}
        }
        
        self.metrics_buffer.append(metric)
        
        # Flush buffer if it gets too large
        if len(self.metrics_buffer) >= 100:
            self.flush_metrics()
    
    def flush_metrics(self):
        """Flush metrics buffer to file"""
        
        if not self.metrics_buffer:
            return
        
        date_str = datetime.now().strftime('%Y%m%d')
        metrics_file = self.telemetry_dir / f"rl_metrics_{date_str}.jsonl"
        
        with open(metrics_file, 'a') as f:
            for metric in self.metrics_buffer:
                f.write(json.dumps(metric) + '\n')
        
        logger.info(f"Flushed {len(self.metrics_buffer)} metrics to {metrics_file}")
        self.metrics_buffer.clear()
    
    def generate_dashboard_data(self) -> Dict:
        """
        Generate data for web dashboard
        
        Returns:
            Dashboard data structure
        """
        
        # Aggregate recent traces
        recent_traces = self.traces[-100:] if len(self.traces) > 100 else self.traces
        
        # Calculate statistics
        task_types = {}
        rewards = []
        
        for trace in recent_traces:
            task_type = trace['data'].get('task_type', 'unknown')
            task_types[task_type] = task_types.get(task_type, 0) + 1
            
            if 'reward' in trace['data']:
                rewards.append(trace['data']['reward'])
        
        dashboard = {
            'total_traces': len(self.traces),
            'recent_traces': len(recent_traces),
            'task_distribution': task_types,
            'reward_stats': {
                'mean': sum(rewards) / len(rewards) if rewards else 0,
                'max': max(rewards) if rewards else 0,
                'min': min(rewards) if rewards else 0,
                'count': len(rewards)
            },
            'last_updated': datetime.now().isoformat()
        }
        
        return dashboard
    
    def export_dashboard_json(self):
        """Export dashboard data to JSON file for web interface"""
        
        dashboard_data = self.generate_dashboard_data()
        dashboard_file = self.telemetry_dir / 'rl_dashboard.json'
        
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        logger.info(f"Dashboard data exported to {dashboard_file}")
    
    def integrate_with_qallow_telemetry(self, qallow_telemetry_file: str):
        """
        Integrate RL metrics with Qallow's existing telemetry
        
        Args:
            qallow_telemetry_file: Path to Qallow telemetry file
        """
        
        # Read existing Qallow telemetry
        qallow_data = {}
        if Path(qallow_telemetry_file).exists():
            with open(qallow_telemetry_file, 'r') as f:
                qallow_data = json.load(f)
        
        # Add RL metrics
        qallow_data['rl_learning'] = {
            'enabled': True,
            'total_traces': len(self.traces),
            'dashboard': self.generate_dashboard_data(),
            'last_sync': datetime.now().isoformat()
        }
        
        # Write back
        with open(qallow_telemetry_file, 'w') as f:
            json.dump(qallow_data, f, indent=2)
        
        logger.info(f"Integrated RL metrics with Qallow telemetry: {qallow_telemetry_file}")
    
    def get_audit_trail(self, task_id: Optional[str] = None) -> List[Dict]:
        """
        Get audit trail for RL training
        
        Args:
            task_id: Optional task ID to filter by
        
        Returns:
            List of audit trail entries
        """
        
        if task_id:
            return [t for t in self.traces if t['task_id'] == task_id]
        
        return self.traces
    
    def cleanup_old_traces(self, days_to_keep: int = 30):
        """
        Clean up old trace files
        
        Args:
            days_to_keep: Number of days to keep traces
        """
        
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for trace_file in self.telemetry_dir.glob('rl_traces_*.jsonl'):
            # Extract date from filename
            try:
                date_str = trace_file.stem.split('_')[-1]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                
                if file_date < cutoff_date:
                    trace_file.unlink()
                    logger.info(f"Deleted old trace file: {trace_file}")
            except (ValueError, IndexError):
                logger.warning(f"Could not parse date from filename: {trace_file}")


# ============================================================================
# Integration with Qallow Web Dashboard
# ============================================================================

def create_web_dashboard_endpoint(bridge: AGITelemetryBridge) -> Dict:
    """
    Create web dashboard endpoint data
    
    Args:
        bridge: Telemetry bridge instance
    
    Returns:
        Dashboard endpoint data
    """
    
    return {
        'endpoint': '/api/rl/dashboard',
        'method': 'GET',
        'data': bridge.generate_dashboard_data(),
        'refresh_interval': 5000  # 5 seconds
    }


def create_metrics_endpoint(bridge: AGITelemetryBridge) -> Dict:
    """
    Create metrics endpoint data
    
    Args:
        bridge: Telemetry bridge instance
    
    Returns:
        Metrics endpoint data
    """
    
    return {
        'endpoint': '/api/rl/metrics',
        'method': 'GET',
        'data': {
            'metrics': bridge.metrics_buffer,
            'count': len(bridge.metrics_buffer)
        }
    }


# ============================================================================
# Demo and Testing
# ============================================================================

def demo_telemetry_bridge():
    """Demonstrate telemetry bridge functionality"""
    
    print("=" * 70)
    print("AGI Telemetry Bridge Demo")
    print("=" * 70)
    
    # Create bridge
    bridge = AGITelemetryBridge(telemetry_dir='/tmp/qallow_telemetry_demo')
    
    # Simulate some RL traces
    print("\n1. Capturing RL Traces")
    print("-" * 70)
    
    for i in range(5):
        bridge.capture_rl_trace(
            task_id=f"task-{i}",
            trace_data={
                'task_type': 'quantum_algorithm_selection',
                'reward': 0.5 + i * 0.1,
                'action': f'algorithm_{i}'
            }
        )
    
    print(f"   Captured {len(bridge.traces)} traces")
    
    # Record metrics
    print("\n2. Recording Metrics")
    print("-" * 70)
    
    bridge.record_metric('rl.reward.mean', 0.75, {'agent': 'quantum_selector'})
    bridge.record_metric('rl.episodes', 100, {'agent': 'ethics_decision'})
    bridge.record_metric('rl.exploration_rate', 0.1, {'agent': 'phase_optimizer'})
    
    print(f"   Recorded {len(bridge.metrics_buffer)} metrics")
    
    # Generate dashboard
    print("\n3. Dashboard Data")
    print("-" * 70)
    
    dashboard = bridge.generate_dashboard_data()
    print(f"   Total Traces: {dashboard['total_traces']}")
    print(f"   Task Distribution: {dashboard['task_distribution']}")
    print(f"   Reward Stats: {dashboard['reward_stats']}")
    
    # Export
    print("\n4. Exporting Data")
    print("-" * 70)
    
    bridge.flush_metrics()
    bridge.export_dashboard_json()
    
    print("   ✓ Metrics flushed")
    print("   ✓ Dashboard exported")
    
    print("\n" + "=" * 70)
    print("✨ Telemetry Bridge Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_telemetry_bridge()
