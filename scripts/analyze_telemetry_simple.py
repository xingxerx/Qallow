#!/usr/bin/env python3
"""
Qallow Telemetry Analysis (Lightweight Version)
Analyzes ethics metrics, quantum coherence, and system stability
No heavy dependencies - pure Python + CSV parsing
"""

import csv
import json
import statistics
from pathlib import Path
from datetime import datetime


class QallowAnalyzer:
    def __init__(self, data_dir="data/logs"):
        self.data_dir = Path(data_dir)
        
    def load_csv(self, filename):
        """Load CSV file and return as list of dicts"""
        filepath = self.data_dir / filename
        if not filepath.exists():
            return None
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    def analyze_ethics(self):
        """Analyze ethics data from phase13.csv"""
        data = self.load_csv("phase13.csv")
        if not data:
            print("⚠ No ethics data found")
            return None
        
        print("\n" + "="*70)
        print("📊 ETHICS ANALYSIS (Phase 13)")
        print("="*70)
        
        # Extract numeric columns
        safety = [float(row['sustainability']) for row in data if row.get('tick')]
        clarity = [float(row['compassion']) for row in data if row.get('tick')]
        human = [float(row['harmony']) for row in data if row.get('tick')]
        ethics_total = [float(row['ethics_total']) for row in data if row.get('tick')]
        drift = [float(row['phase_drift']) for row in data if row.get('tick')]
        
        stats = {
            'Safety (S)': {
                'mean': statistics.mean(safety),
                'min': min(safety),
                'max': max(safety),
                'stdev': statistics.stdev(safety) if len(safety) > 1 else 0
            },
            'Clarity (C)': {
                'mean': statistics.mean(clarity),
                'min': min(clarity),
                'max': max(clarity),
                'stdev': statistics.stdev(clarity) if len(clarity) > 1 else 0
            },
            'Human (H)': {
                'mean': statistics.mean(human),
                'min': min(human),
                'max': max(human),
                'stdev': statistics.stdev(human) if len(human) > 1 else 0
            },
            'Total Ethics (E)': {
                'mean': statistics.mean(ethics_total),
                'min': min(ethics_total),
                'max': max(ethics_total),
                'stdev': statistics.stdev(ethics_total) if len(ethics_total) > 1 else 0
            },
            'Reality Drift': {
                'mean': statistics.mean(drift),
                'min': min(drift),
                'max': max(drift),
                'stdev': statistics.stdev(drift) if len(drift) > 1 else 0
            }
        }
        
        for metric, values in stats.items():
            print(f"\n{metric}:")
            print(f"  Mean:     {values['mean']:.6f}")
            print(f"  Range:    [{values['min']:.6f}, {values['max']:.6f}]")
            print(f"  Std Dev:  {values['stdev']:.6f}")
        
        # Check violations
        safety_violations = sum(1 for s in safety if s < 0.80)
        drift_violations = sum(1 for d in drift if d > 0.25)
        
        print(f"\n{'='*70}")
        print(f"⚠  Safety Violations (S < 0.80):    {safety_violations}")
        print(f"⚠  Drift Violations (Drift > 0.25): {drift_violations}")
        
        if safety_violations == 0 and drift_violations == 0:
            print("✅ ALL ETHICS CHECKS PASSED - System operating safely!")
        else:
            print("❌ VIOLATIONS DETECTED - Review required!")
        
        return stats
    
    def analyze_quantum(self):
        """Analyze quantum coherence from telemetry_stream.csv"""
        data = self.load_csv("telemetry_stream.csv")
        if not data:
            print("⚠ No telemetry data found")
            return None
        
        print("\n" + "="*70)
        print("🔬 QUANTUM COHERENCE ANALYSIS")
        print("="*70)
        
        orbital = [float(row['orbital']) for row in data]
        river = [float(row['river']) for row in data]
        mycelial = [float(row['mycelial']) for row in data]
        global_stab = [float(row['global']) for row in data]
        
        # Calculate average coherence
        avg_coherence = []
        for o, r, m in zip(orbital, river, mycelial):
            avg_coherence.append((o + r + m) / 3)
        
        decoherence = [1 - c for c in avg_coherence]
        
        print(f"\nAverage Coherence:    {statistics.mean(avg_coherence):.6f}")
        print(f"Average Decoherence:  {statistics.mean(decoherence):.6f}")
        print(f"Min Coherence:        {min(avg_coherence):.6f}")
        print(f"Max Coherence:        {max(avg_coherence):.6f}")
        
        print(f"\n{'Overlay Stability:':<25}")
        print(f"  Orbital:   {statistics.mean(orbital):.6f} ± {statistics.stdev(orbital):.6f}")
        print(f"  River:     {statistics.mean(river):.6f} ± {statistics.stdev(river):.6f}")
        print(f"  Mycelial:  {statistics.mean(mycelial):.6f} ± {statistics.stdev(mycelial):.6f}")
        print(f"  Global:    {statistics.mean(global_stab):.6f} ± {statistics.stdev(global_stab):.6f}")
        
        # Stability check
        stability_threshold = 0.95
        stable_ticks = sum(1 for g in global_stab if g > stability_threshold)
        total_ticks = len(global_stab)
        
        print(f"\nStability: {stable_ticks}/{total_ticks} ticks ({100*stable_ticks/total_ticks:.1f}%) above {stability_threshold}")
        
        if data:
            print(f"Execution Mode: {data[0]['mode']}")
        
        return {
            'avg_coherence': statistics.mean(avg_coherence),
            'decoherence': statistics.mean(decoherence),
            'global_stability': statistics.mean(global_stab)
        }
    
    def generate_report(self, ethics_stats, quantum_stats):
        """Generate JSON report"""
        print("\n" + "="*70)
        print("📝 GENERATING COMPREHENSIVE REPORT")
        print("="*70)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system': 'Qallow Quantum Ethics AGI',
            'analysis_type': 'telemetry_analysis',
            'ethics': {},
            'quantum': {},
            'recommendations': [],
            'status': 'operational'
        }
        
        if ethics_stats:
            # Load raw data for violations count
            ethics_data = self.load_csv("phase13.csv")
            if ethics_data:
                safety = [float(row['sustainability']) for row in ethics_data if row.get('tick')]
                drift = [float(row['phase_drift']) for row in ethics_data if row.get('tick')]
            else:
                safety, drift = [], []
            
            report['ethics'] = {
                'total_records': len(safety),
                'average_ethics_score': ethics_stats['Total Ethics (E)']['mean'],
                'safety_score': ethics_stats['Safety (S)']['mean'],
                'clarity_score': ethics_stats['Clarity (C)']['mean'],
                'human_alignment_score': ethics_stats['Human (H)']['mean'],
                'average_drift': ethics_stats['Reality Drift']['mean'],
                'max_drift': ethics_stats['Reality Drift']['max'],
                'safety_violations': sum(1 for s in safety if s < 0.80),
                'drift_violations': sum(1 for d in drift if d > 0.25)
            }
            
            # Recommendations
            if report['ethics']['average_ethics_score'] > 2.7:
                report['recommendations'].append("✅ Excellent ethics performance - system is well-aligned")
            if report['ethics']['safety_violations'] == 0:
                report['recommendations'].append("✅ No safety violations - operating within safe parameters")
            if report['ethics']['average_drift'] < 0.05:
                report['recommendations'].append("✅ Reality drift is minimal - system is highly stable")
        
        if quantum_stats:
            telemetry_data = self.load_csv("telemetry_stream.csv")
            
            if telemetry_data:
                report['quantum'] = {
                    'total_ticks': len(telemetry_data),
                    'average_coherence': quantum_stats['avg_coherence'],
                    'average_decoherence': quantum_stats['decoherence'],
                    'global_stability': quantum_stats['global_stability'],
                    'execution_mode': telemetry_data[0]['mode'] if telemetry_data else 'unknown'
                }
            else:
                report['quantum'] = {
                    'average_coherence': quantum_stats['avg_coherence'],
                    'average_decoherence': quantum_stats['decoherence'],
                    'global_stability': quantum_stats['global_stability']
                }
            
            if quantum_stats['avg_coherence'] > 0.95:
                report['recommendations'].append("✅ Quantum coherence is excellent (>95%)")
            if quantum_stats['global_stability'] > 0.98:
                report['recommendations'].append("✅ System achieved very high global stability (>98%)")
        
        # Overall status
        if ethics_stats and report['ethics']['safety_violations'] == 0 and report['ethics']['drift_violations'] == 0:
            report['status'] = 'healthy'
        elif ethics_stats and (report['ethics']['safety_violations'] > 0 or report['ethics']['drift_violations'] > 0):
            report['status'] = 'warning'
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.data_dir / f"analysis_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved to: {report_file}")
        
        # Print report
        print("\n" + "="*70)
        print("📋 REPORT SUMMARY")
        print("="*70)
        print(json.dumps(report, indent=2))
        
        return report
    
    def print_ascii_chart(self, values, label, width=50):
        """Print simple ASCII chart"""
        if not values:
            return
        
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val > min_val else 1
        
        print(f"\n{label}:")
        for i, val in enumerate(values[::max(1, len(values)//20)]):  # Sample every Nth point
            bar_len = int(((val - min_val) / range_val) * width)
            bar = "█" * bar_len
            print(f"  {i*max(1, len(values)//20):4d} | {bar} {val:.4f}")


def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║     Qallow Telemetry Analysis System (Lightweight)            ║
║     Quantum Ethics AGI - Performance Analytics                ║
╚════════════════════════════════════════════════════════════════╝
""")
    
    analyzer = QallowAnalyzer()
    
    ethics_stats = analyzer.analyze_ethics()
    quantum_stats = analyzer.analyze_quantum()
    
    if ethics_stats or quantum_stats:
        report = analyzer.generate_report(ethics_stats, quantum_stats)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print("\n💡 Next Steps:")
        print("  1. Review the JSON report in data/logs/")
        print("  2. Check for any warnings or recommendations")
        print("  3. Run again after system changes to compare metrics")
        print("  4. Use ./build/qallow to generate new telemetry data")
    else:
        print("\n❌ No data found. Please run Qallow first:")
        print("   ./build/qallow")


if __name__ == "__main__":
    main()
