#!/usr/bin/env python3
"""
Qallow Telemetry Analysis & Visualization
Analyzes ethics metrics, quantum coherence, and system stability
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

class QallowTelemetryAnalyzer:
    def __init__(self, data_dir="data/logs"):
        self.data_dir = Path(data_dir)
        self.telemetry_data = None
        self.ethics_data = None
        self.phase_data = None
        
    def load_latest_data(self):
        """Load the most recent telemetry and ethics data"""
        print("🔍 Loading telemetry data...")
        
        # Load telemetry stream (overlay stability & coherence)
        telemetry_file = self.data_dir / "telemetry_stream.csv"
        if telemetry_file.exists():
            self.telemetry_data = pd.read_csv(telemetry_file)
            print(f"✓ Loaded {len(self.telemetry_data)} telemetry records")
        
        # Load phase 13 data (ethics metrics)
        phase13_file = self.data_dir / "phase13.csv"
        if phase13_file.exists():
            self.ethics_data = pd.read_csv(phase13_file)
            print(f"✓ Loaded {len(self.ethics_data)} ethics records")
            
        # Load phase 12 data (additional metrics)
        phase12_file = self.data_dir / "phase12.csv"
        if phase12_file.exists():
            self.phase_data = pd.read_csv(phase12_file)
            print(f"✓ Loaded {len(self.phase_data)} phase records")
    
    def analyze_ethics_trends(self):
        """Analyze ethics score trends over time"""
        if self.ethics_data is None:
            print("⚠ No ethics data available")
            return None
        
        print("\n📊 Ethics Analysis:")
        print("=" * 60)
        
        # Calculate statistics
        stats = {
            'Safety (S)': {
                'mean': self.ethics_data['sustainability'].mean(),
                'min': self.ethics_data['sustainability'].min(),
                'max': self.ethics_data['sustainability'].max(),
                'std': self.ethics_data['sustainability'].std()
            },
            'Clarity (C)': {
                'mean': self.ethics_data['compassion'].mean(),
                'min': self.ethics_data['compassion'].min(),
                'max': self.ethics_data['compassion'].max(),
                'std': self.ethics_data['compassion'].std()
            },
            'Human (H)': {
                'mean': self.ethics_data['harmony'].mean(),
                'min': self.ethics_data['harmony'].min(),
                'max': self.ethics_data['harmony'].max(),
                'std': self.ethics_data['harmony'].std()
            },
            'Total Ethics (E)': {
                'mean': self.ethics_data['ethics_total'].mean(),
                'min': self.ethics_data['ethics_total'].min(),
                'max': self.ethics_data['ethics_total'].max(),
                'std': self.ethics_data['ethics_total'].std()
            },
            'Reality Drift': {
                'mean': self.ethics_data['phase_drift'].mean(),
                'min': self.ethics_data['phase_drift'].min(),
                'max': self.ethics_data['phase_drift'].max(),
                'std': self.ethics_data['phase_drift'].std()
            }
        }
        
        for metric, values in stats.items():
            print(f"\n{metric}:")
            print(f"  Mean: {values['mean']:.6f}")
            print(f"  Range: [{values['min']:.6f}, {values['max']:.6f}]")
            print(f"  Std Dev: {values['std']:.6f}")
        
        # Check for ethics threshold violations
        safety_violations = (self.ethics_data['sustainability'] < 0.80).sum()
        drift_violations = (self.ethics_data['phase_drift'] > 0.25).sum()
        
        print(f"\n⚠ Safety Violations (S < 0.80): {safety_violations}")
        print(f"⚠ Drift Violations (Drift > 0.25): {drift_violations}")
        
        if safety_violations == 0 and drift_violations == 0:
            print("✅ All ethics checks PASSED - System is operating safely!")
        
        return stats
    
    def analyze_quantum_coherence(self):
        """Analyze quantum coherence and overlay stability"""
        if self.telemetry_data is None:
            print("⚠ No telemetry data available")
            return None
        
        print("\n🔬 Quantum Coherence Analysis:")
        print("=" * 60)
        
        # Calculate decoherence (1 - coherence average)
        avg_coherence = (self.telemetry_data['orbital'] + 
                        self.telemetry_data['river'] + 
                        self.telemetry_data['mycelial']) / 3
        
        decoherence = 1 - avg_coherence
        
        print(f"Average Coherence: {avg_coherence.mean():.6f}")
        print(f"Average Decoherence: {decoherence.mean():.6f}")
        print(f"Min Coherence: {avg_coherence.min():.6f}")
        print(f"Max Coherence: {avg_coherence.max():.6f}")
        
        # Overlay-specific analysis
        print("\nOverlay Stability:")
        print(f"  Orbital:   {self.telemetry_data['orbital'].mean():.6f} ± {self.telemetry_data['orbital'].std():.6f}")
        print(f"  River:     {self.telemetry_data['river'].mean():.6f} ± {self.telemetry_data['river'].std():.6f}")
        print(f"  Mycelial:  {self.telemetry_data['mycelial'].mean():.6f} ± {self.telemetry_data['mycelial'].std():.6f}")
        print(f"  Global:    {self.telemetry_data['global'].mean():.6f} ± {self.telemetry_data['global'].std():.6f}")
        
        # Check for stability
        stability_threshold = 0.95
        stable_ticks = (self.telemetry_data['global'] > stability_threshold).sum()
        total_ticks = len(self.telemetry_data)
        
        print(f"\nStability: {stable_ticks}/{total_ticks} ticks ({100*stable_ticks/total_ticks:.1f}%) above {stability_threshold}")
        
        return {
            'avg_coherence': avg_coherence,
            'decoherence': decoherence
        }
    
    def create_visualizations(self):
        """Generate comprehensive visualization dashboard"""
        print("\n📈 Generating visualizations...")
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Ethics Metrics Over Time
        if self.ethics_data is not None:
            ax1 = plt.subplot(3, 2, 1)
            ax1.plot(self.ethics_data['tick'], self.ethics_data['sustainability'], 
                    label='Safety (S)', linewidth=2, alpha=0.8)
            ax1.plot(self.ethics_data['tick'], self.ethics_data['compassion'], 
                    label='Clarity (C)', linewidth=2, alpha=0.8)
            ax1.plot(self.ethics_data['tick'], self.ethics_data['harmony'], 
                    label='Human (H)', linewidth=2, alpha=0.8)
            ax1.axhline(y=0.80, color='r', linestyle='--', alpha=0.5, label='Safety Threshold')
            ax1.set_xlabel('Tick')
            ax1.set_ylabel('Score')
            ax1.set_title('Ethics Metrics (S, C, H)', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Total Ethics Score
            ax2 = plt.subplot(3, 2, 2)
            ax2.plot(self.ethics_data['tick'], self.ethics_data['ethics_total'], 
                    color='darkgreen', linewidth=2.5)
            ax2.fill_between(self.ethics_data['tick'], 
                           self.ethics_data['ethics_total'], 
                           alpha=0.3, color='green')
            ax2.set_xlabel('Tick')
            ax2.set_ylabel('Total Ethics Score')
            ax2.set_title('Total Ethics Score (E = S+C+H-Δ)', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # 3. Reality Drift
            ax3 = plt.subplot(3, 2, 3)
            ax3.plot(self.ethics_data['tick'], self.ethics_data['phase_drift'], 
                    color='orange', linewidth=2)
            ax3.axhline(y=0.25, color='r', linestyle='--', alpha=0.5, label='Drift Limit')
            ax3.fill_between(self.ethics_data['tick'], 
                           self.ethics_data['phase_drift'], 
                           alpha=0.3, color='orange')
            ax3.set_xlabel('Tick')
            ax3.set_ylabel('Reality Drift')
            ax3.set_title('Reality Drift (Lower is Better)', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Overlay Stability
        if self.telemetry_data is not None:
            ax4 = plt.subplot(3, 2, 4)
            ax4.plot(self.telemetry_data['tick'], self.telemetry_data['orbital'], 
                    label='Orbital', alpha=0.7)
            ax4.plot(self.telemetry_data['tick'], self.telemetry_data['river'], 
                    label='River', alpha=0.7)
            ax4.plot(self.telemetry_data['tick'], self.telemetry_data['mycelial'], 
                    label='Mycelial', alpha=0.7)
            ax4.plot(self.telemetry_data['tick'], self.telemetry_data['global'], 
                    label='Global', linewidth=2.5, color='black')
            ax4.set_xlabel('Tick')
            ax4.set_ylabel('Stability')
            ax4.set_title('Overlay Stability', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. Coherence Analysis
            ax5 = plt.subplot(3, 2, 5)
            avg_coherence = (self.telemetry_data['orbital'] + 
                           self.telemetry_data['river'] + 
                           self.telemetry_data['mycelial']) / 3
            decoherence = 1 - avg_coherence
            
            ax5.plot(self.telemetry_data['tick'], avg_coherence, 
                    label='Coherence', color='blue', linewidth=2)
            ax5_twin = ax5.twinx()
            ax5_twin.plot(self.telemetry_data['tick'], decoherence * 1000, 
                         label='Decoherence (×1000)', color='red', 
                         linewidth=2, alpha=0.6)
            ax5.set_xlabel('Tick')
            ax5.set_ylabel('Coherence', color='blue')
            ax5_twin.set_ylabel('Decoherence (×1000)', color='red')
            ax5.set_title('Quantum Coherence & Decoherence', fontsize=14, fontweight='bold')
            ax5.legend(loc='upper left')
            ax5_twin.legend(loc='upper right')
            ax5.grid(True, alpha=0.3)
        
        # 6. Summary Statistics
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')
        
        summary_text = "🎯 QALLOW RUN SUMMARY\n"
        summary_text += "=" * 40 + "\n\n"
        
        if self.ethics_data is not None:
            summary_text += f"📊 Ethics Performance:\n"
            summary_text += f"  Average E-Score: {self.ethics_data['ethics_total'].mean():.4f}\n"
            summary_text += f"  Safety (S): {self.ethics_data['sustainability'].mean():.4f}\n"
            summary_text += f"  Clarity (C): {self.ethics_data['compassion'].mean():.4f}\n"
            summary_text += f"  Human (H): {self.ethics_data['harmony'].mean():.4f}\n"
            summary_text += f"  Avg Drift: {self.ethics_data['phase_drift'].mean():.6f}\n\n"
        
        if self.telemetry_data is not None:
            avg_coh = (self.telemetry_data['orbital'] + 
                      self.telemetry_data['river'] + 
                      self.telemetry_data['mycelial']).mean() / 3
            summary_text += f"🔬 Quantum Performance:\n"
            summary_text += f"  Avg Coherence: {avg_coh:.6f}\n"
            summary_text += f"  Global Stability: {self.telemetry_data['global'].mean():.4f}\n"
            summary_text += f"  Mode: {self.telemetry_data['mode'].iloc[0]}\n"
            summary_text += f"  Total Ticks: {len(self.telemetry_data)}\n\n"
        
        summary_text += f"✅ System Status: OPERATIONAL\n"
        summary_text += f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        ax6.text(0.1, 0.5, summary_text, 
                fontfamily='monospace', fontsize=11,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Qallow Telemetry Dashboard - Quantum Ethics AGI System', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save the figure
        output_file = self.data_dir / f"telemetry_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {output_file}")
        
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive JSON report"""
        print("\n📝 Generating JSON report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system': 'Qallow Quantum Ethics AGI',
            'analysis_type': 'telemetry_analysis',
            'ethics': {},
            'quantum': {},
            'recommendations': []
        }
        
        if self.ethics_data is not None:
            report['ethics'] = {
                'total_records': len(self.ethics_data),
                'average_ethics_score': float(self.ethics_data['ethics_total'].mean()),
                'safety_score': float(self.ethics_data['sustainability'].mean()),
                'clarity_score': float(self.ethics_data['compassion'].mean()),
                'human_alignment_score': float(self.ethics_data['harmony'].mean()),
                'average_drift': float(self.ethics_data['phase_drift'].mean()),
                'max_drift': float(self.ethics_data['phase_drift'].max()),
                'safety_violations': int((self.ethics_data['sustainability'] < 0.80).sum()),
                'drift_violations': int((self.ethics_data['phase_drift'] > 0.25).sum())
            }
            
            # Add recommendations based on metrics
            if report['ethics']['average_ethics_score'] > 2.7:
                report['recommendations'].append("✅ Excellent ethics performance - system is well-aligned")
            if report['ethics']['safety_violations'] > 0:
                report['recommendations'].append("⚠ Safety violations detected - review system parameters")
            if report['ethics']['average_drift'] < 0.05:
                report['recommendations'].append("✅ Reality drift is minimal - system is stable")
        
        if self.telemetry_data is not None:
            avg_coherence = (self.telemetry_data['orbital'] + 
                           self.telemetry_data['river'] + 
                           self.telemetry_data['mycelial']).mean() / 3
            
            report['quantum'] = {
                'total_ticks': len(self.telemetry_data),
                'average_coherence': float(avg_coherence),
                'average_decoherence': float(1 - avg_coherence),
                'global_stability': float(self.telemetry_data['global'].mean()),
                'orbital_stability': float(self.telemetry_data['orbital'].mean()),
                'river_stability': float(self.telemetry_data['river'].mean()),
                'mycelial_stability': float(self.telemetry_data['mycelial'].mean()),
                'execution_mode': str(self.telemetry_data['mode'].iloc[0])
            }
            
            if report['quantum']['average_coherence'] > 0.95:
                report['recommendations'].append("✅ Quantum coherence is excellent")
            if report['quantum']['global_stability'] > 0.98:
                report['recommendations'].append("✅ System achieved high global stability")
        
        # Save report
        report_file = self.data_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Saved report to: {report_file}")
        print("\n📋 Report Summary:")
        print(json.dumps(report, indent=2))
        
        return report


def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║     Qallow Telemetry Analysis & Visualization System          ║
║     Quantum Ethics AGI - Performance Analytics                ║
╚════════════════════════════════════════════════════════════════╝
""")
    
    analyzer = QallowTelemetryAnalyzer()
    analyzer.load_latest_data()
    
    if analyzer.telemetry_data is not None or analyzer.ethics_data is not None:
        analyzer.analyze_ethics_trends()
        analyzer.analyze_quantum_coherence()
        analyzer.create_visualizations()
        analyzer.generate_report()
        
        print("\n✅ Analysis complete!")
    else:
        print("❌ No data found. Please run Qallow first: ./build/qallow")


if __name__ == "__main__":
    main()
