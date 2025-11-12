#!/usr/bin/env python3
"""
Qallow Real-Time Monitoring Dashboard
Monitors telemetry files and displays live updates
"""

import time
import os
import csv
from pathlib import Path
from datetime import datetime


class LiveMonitor:
    def __init__(self, data_dir="data/logs"):
        self.data_dir = Path(data_dir)
        self.telemetry_file = self.data_dir / "telemetry_stream.csv"
        self.ethics_file = self.data_dir / "phase13.csv"
        self.last_telemetry_size = 0
        self.last_ethics_size = 0
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name != 'nt' else 'cls')
    
    def get_latest_telemetry(self):
        """Get the most recent telemetry record"""
        if not self.telemetry_file.exists():
            return None
        
        with open(self.telemetry_file, 'r') as f:
            lines = list(csv.DictReader(f))
            return lines[-1] if lines else None
    
    def get_latest_ethics(self):
        """Get the most recent ethics record"""
        if not self.ethics_file.exists():
            return None
        
        with open(self.ethics_file, 'r') as f:
            lines = list(csv.DictReader(f))
            return lines[-1] if lines else None
    
    def draw_bar(self, value, width=40, threshold=None):
        """Draw ASCII progress bar"""
        value = float(value)
        filled = int(value * width)
        bar = "█" * filled + "░" * (width - filled)
        
        # Color based on threshold
        if threshold and value < threshold:
            color = "\033[91m"  # Red
        elif value > 0.95:
            color = "\033[92m"  # Green
        elif value > 0.85:
            color = "\033[93m"  # Yellow
        else:
            color = "\033[91m"  # Red
        
        reset = "\033[0m"
        return f"{color}{bar}{reset} {value:.4f}"
    
    def display_dashboard(self, telemetry, ethics):
        """Display the monitoring dashboard"""
        self.clear_screen()
        
        print("╔" + "═"*78 + "╗")
        print("║" + " "*20 + "QALLOW LIVE MONITORING DASHBOARD" + " "*26 + "║")
        print("║" + " "*22 + "Quantum Ethics AGI System" + " "*31 + "║")
        print("╚" + "═"*78 + "╝")
        print()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"🕐 Current Time: {timestamp}")
        print(f"📊 Refresh Rate: 2 seconds")
        print("━" * 80)
        
        if telemetry:
            print("\n🔬 QUANTUM COHERENCE & OVERLAY STABILITY")
            print("─" * 80)
            
            orbital = float(telemetry.get('orbital', 0))
            river = float(telemetry.get('river', 0))
            mycelial = float(telemetry.get('mycelial', 0))
            global_stab = float(telemetry.get('global', 0))
            tick = telemetry.get('tick', '?')
            mode = telemetry.get('mode', '?')
            
            print(f"Tick: {tick:<6}  Mode: {mode}")
            print()
            print(f"Orbital   │ {self.draw_bar(orbital, threshold=0.90)}")
            print(f"River     │ {self.draw_bar(river, threshold=0.90)}")
            print(f"Mycelial  │ {self.draw_bar(mycelial, threshold=0.90)}")
            print(f"Global    │ {self.draw_bar(global_stab, threshold=0.95)}")
            
            avg_coherence = (orbital + river + mycelial) / 3
            decoherence = 1 - avg_coherence
            print()
            print(f"Coherence    : {avg_coherence:.6f}")
            print(f"Decoherence  : {decoherence:.6f}")
        
        if ethics:
            print("\n📊 ETHICS MONITORING (E = S+C+H-Δ)")
            print("─" * 80)
            
            safety = float(ethics.get('sustainability', 0))
            clarity = float(ethics.get('compassion', 0))
            human = float(ethics.get('harmony', 0))
            ethics_total = float(ethics.get('ethics_total', 0))
            drift = float(ethics.get('phase_drift', 0))
            tick = ethics.get('tick', '?')
            
            print(f"Tick: {tick}")
            print()
            print(f"Safety (S)   │ {self.draw_bar(safety, threshold=0.80)}")
            print(f"Clarity (C)  │ {self.draw_bar(clarity, threshold=0.85)}")
            print(f"Human (H)    │ {self.draw_bar(human, threshold=0.90)}")
            print()
            print(f"Total Ethics Score: {ethics_total:.6f}")
            print(f"               E = {safety:.2f} + {clarity:.2f} + {human:.2f} = {ethics_total:.2f}")
            print()
            print(f"Reality Drift │ {self.draw_bar(1 - drift, threshold=0.75)}")
            print(f"Drift Value   : {drift:.6f} (limit: 0.250)")
            
            # Status indicators
            print("\n" + "─" * 80)
            status = []
            
            if safety < 0.80:
                status.append("🔴 SAFETY VIOLATION")
            else:
                status.append("✅ Safety OK")
            
            if drift > 0.25:
                status.append("🔴 DRIFT VIOLATION")
            else:
                status.append("✅ Drift OK")
            
            if ethics_total > 2.7:
                status.append("✅ Ethics Excellent")
            elif ethics_total > 2.5:
                status.append("⚠️  Ethics Acceptable")
            else:
                status.append("🔴 Ethics Low")
            
            print(" | ".join(status))
        
        if not telemetry and not ethics:
            print("\n⚠️  No data available yet. Waiting for Qallow to start...")
            print("\n💡 Run Qallow in another terminal: ./build/qallow")
        
        print("\n" + "─" * 80)
        print("Press Ctrl+C to exit")
    
    def monitor(self, refresh_interval=2):
        """Main monitoring loop"""
        print("Starting Qallow Live Monitor...")
        print(f"Monitoring: {self.data_dir}")
        time.sleep(1)
        
        try:
            while True:
                telemetry = self.get_latest_telemetry()
                ethics = self.get_latest_ethics()
                
                self.display_dashboard(telemetry, ethics)
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            self.clear_screen()
            print("\n✅ Monitoring stopped")
            print("📊 Final summary available in data/logs/")


def main():
    monitor = LiveMonitor()
    monitor.monitor(refresh_interval=2)


if __name__ == "__main__":
    main()
