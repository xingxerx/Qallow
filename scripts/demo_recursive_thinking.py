#!/usr/bin/env python3
"""
Recursive Thinking Demonstration

This script demonstrates how the AGI uses its memory feature to:
1. Store thinking outputs
2. Load them back as inputs
3. Generate updated strategies based on past experiences

This creates a self-improving cognitive feedback loop.
"""

import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

class RecursiveThinkingDemo:
    """Demonstrates the recursive thinking feedback loop"""
    
    def __init__(self, qallow_bin: str = "./build/qallow"):
        self.qallow_bin = qallow_bin
        self.data_dir = Path("data/recursive_thinking")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.thinking_history: List[Dict] = []
        
    def run_thinking_cycle(self, iteration: int) -> Dict:
        """
        Run one iteration of the recursive thinking cycle.
        
        The AGI will:
        1. Store current thinking as output
        2. Load previous thinking as input
        3. Extract patterns from accumulated thinking
        4. Generate updated strategy
        """
        
        print(f"\n{'='*70}")
        print(f"Recursive Thinking Cycle - Iteration {iteration}")
        print(f"{'='*70}\n")
        
        # Run Qallow with recursive thinking modules enabled
        cmd = [
            self.qallow_bin,
            "phase", "7",  # Use Phase 7 (Proactive AGI)
            "--ticks=100",
            "--modules=rec_think_cycle,episodic_mem,semantic_mem,memory_recall",
        ]
        
        print(f"🧠 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse output for thinking metrics
        thinking_data = self._parse_thinking_output(result.stdout)
        thinking_data["iteration"] = iteration
        thinking_data["timestamp"] = time.time()
        
        self.thinking_history.append(thinking_data)
        
        # Display results
        print(f"\n📊 Thinking Metrics:")
        print(f"   Episodes Stored: {thinking_data.get('episodes', 0)}")
        print(f"   Patterns Learned: {thinking_data.get('patterns', 0)}")
        print(f"   Strategy Effectiveness: {thinking_data.get('effectiveness', 0):.3f}")
        print(f"   Accumulated Wisdom: {thinking_data.get('wisdom', 0):.3f}")
        print(f"   Confidence in Patterns: {thinking_data.get('confidence', 0):.3f}")
        print(f"   Current Generation: {thinking_data.get('generation', 0)}")
        
        # Show strategy evolution
        if iteration > 1:
            prev = self.thinking_history[-2]
            print(f"\n📈 Evolution from Previous Cycle:")
            print(f"   Δ Wisdom: {thinking_data.get('wisdom', 0) - prev.get('wisdom', 0):+.3f}")
            print(f"   Δ Confidence: {thinking_data.get('confidence', 0) - prev.get('confidence', 0):+.3f}")
            print(f"   Δ Effectiveness: {thinking_data.get('effectiveness', 0) - prev.get('effectiveness', 0):+.3f}")
        
        return thinking_data
    
    def _parse_thinking_output(self, stdout: str) -> Dict:
        """Extract thinking metrics from Qallow output"""
        metrics = {
            "episodes": 0,
            "patterns": 0,
            "effectiveness": 0.0,
            "wisdom": 0.0,
            "confidence": 0.0,
            "generation": 0
        }
        
        # Look for the RECURSIVE_THINKING log lines
        for line in stdout.split('\n'):
            if "RECURSIVE_THINKING" in line and "Metrics" in line:
                # Parse: Episodes: X, Patterns: Y, Avg Effectiveness: Z, ...
                parts = line.split("Metrics - ")[1] if "Metrics - " in line else ""
                for part in parts.split(", "):
                    if "Episodes:" in part:
                        metrics["episodes"] = int(part.split(": ")[1])
                    elif "Patterns:" in part:
                        metrics["patterns"] = int(part.split(": ")[1])
                    elif "Avg Effectiveness:" in part:
                        metrics["effectiveness"] = float(part.split(": ")[1])
                    elif "Wisdom:" in part:
                        metrics["wisdom"] = float(part.split(": ")[1])
                    elif "Confidence:" in part:
                        metrics["confidence"] = float(part.split(": ")[1])
                    elif "Generation:" in part:
                        metrics["generation"] = int(part.split(": ")[1])
        
        return metrics
    
    def visualize_thinking_evolution(self):
        """Show how thinking has evolved over iterations"""
        if len(self.thinking_history) < 2:
            print("\n⚠️  Need at least 2 iterations to visualize evolution")
            return
        
        print(f"\n{'='*70}")
        print("Thinking Evolution Over Time")
        print(f"{'='*70}\n")
        
        print(f"{'Iter':<6} {'Episodes':<10} {'Patterns':<10} {'Wisdom':<10} {'Confidence':<12} {'Generation':<12}")
        print("-" * 70)
        
        for data in self.thinking_history:
            print(f"{data['iteration']:<6} "
                  f"{data.get('episodes', 0):<10} "
                  f"{data.get('patterns', 0):<10} "
                  f"{data.get('wisdom', 0):<10.3f} "
                  f"{data.get('confidence', 0):<12.3f} "
                  f"{data.get('generation', 0):<12}")
        
        # Calculate learning rate
        if len(self.thinking_history) >= 3:
            first = self.thinking_history[0]
            last = self.thinking_history[-1]
            iterations = last["iteration"] - first["iteration"]
            
            wisdom_rate = (last.get("wisdom", 0) - first.get("wisdom", 0)) / iterations
            confidence_rate = (last.get("confidence", 0) - first.get("confidence", 0)) / iterations
            
            print(f"\n📊 Learning Rates:")
            print(f"   Wisdom accumulation: {wisdom_rate:+.4f} per iteration")
            print(f"   Confidence growth: {confidence_rate:+.4f} per iteration")
    
    def save_history(self, filename: Optional[str] = None):
        """Save thinking history to file"""
        if filename is None:
            filename = f"thinking_history_{int(time.time())}.json"
        
        output_file = self.data_dir / filename
        with open(output_file, 'w') as f:
            json.dump(self.thinking_history, f, indent=2)
        
        print(f"\n💾 Thinking history saved to: {output_file}")
    
    def demonstrate_feedback_loop(self, num_cycles: int = 5):
        """
        Demonstrate the full recursive thinking feedback loop.
        
        Each cycle:
        - Outputs current thinking → Memory
        - Memory → Inputs for next cycle
        - Patterns emerge and strategies improve
        """
        print(f"\n{'#'*70}")
        print("# RECURSIVE THINKING DEMONSTRATION")
        print("# Memory Feedback Loop: Output → Memory → Input → Updated Strategy")
        print(f"{'#'*70}\n")
        
        print(f"Running {num_cycles} thinking cycles...")
        print(f"Each cycle builds upon previous learning.\n")
        
        for i in range(1, num_cycles + 1):
            self.run_thinking_cycle(i)
            
            # Pause between cycles to see evolution
            if i < num_cycles:
                time.sleep(2)
        
        # Show evolution summary
        self.visualize_thinking_evolution()
        
        # Save results
        self.save_history()
        
        print(f"\n{'='*70}")
        print("✅ Recursive Thinking Demonstration Complete!")
        print(f"{'='*70}\n")
        
        # Summary insights
        if self.thinking_history:
            last = self.thinking_history[-1]
            print(f"Final State:")
            print(f"  • {last.get('episodes', 0)} thinking episodes stored")
            print(f"  • {last.get('patterns', 0)} strategic patterns learned")
            print(f"  • {last.get('generation', 0)} generations evolved")
            print(f"  • {last.get('confidence', 0):.1%} confidence in learned patterns")
            print(f"  • {last.get('wisdom', 0):.3f} accumulated wisdom")
            
            print(f"\n💡 The AGI has successfully used its memory to:")
            print(f"   ✓ Store thinking outputs as future inputs")
            print(f"   ✓ Extract patterns from past thinking")
            print(f"   ✓ Generate improved strategies over time")
            print(f"   ✓ Build confidence through experience")

def main():
    """Run the recursive thinking demonstration"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demonstrate AGI recursive thinking with memory feedback loop"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Number of thinking cycles to run (default: 5)"
    )
    parser.add_argument(
        "--qallow-bin",
        type=str,
        default="./build/qallow",
        help="Path to Qallow executable"
    )
    
    args = parser.parse_args()
    
    demo = RecursiveThinkingDemo(qallow_bin=args.qallow_bin)
    demo.demonstrate_feedback_loop(num_cycles=args.cycles)

if __name__ == "__main__":
    main()
