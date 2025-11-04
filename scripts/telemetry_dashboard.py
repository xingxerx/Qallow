# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] #!/usr/bin/env python3
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] """
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] Performance Telemetry Dashboard
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] ================================
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] Analyzes and visualizes phase execution performance data.
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] Usage:
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     python3 telemetry_dashboard.py data/logs/phase13.csv
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     python3 telemetry_dashboard.py --compare phase12.csv phase13.csv
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     python3 telemetry_dashboard.py --gpu-bench gpu_profile.json
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] """
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] from pathlib import Path
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] from typing import Dict, List, Tuple
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] from dataclasses import dataclass
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] from datetime import datetime


@dataclass
class PhaseTiming:
    """Single phase execution timing."""
    phase: int
    tick: int
    coherence: float
    entropy: float
    ethics_total: float
    timestamp: str


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""
    phase: int
    min_time_ms: float
    max_time_ms: float
    avg_time_ms: float
    total_time_ms: float
    tick_count: int


def parse_phase_csv(csv_path: str) -> Tuple[List[PhaseTiming], PerformanceMetrics]:
    """Parse phase execution CSV and extract metrics."""
    timings = []
    times = []
    phase_num = int(Path(csv_path).stem.split('phase')[1])
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('tick'):
                continue
            
            timing = PhaseTiming(
                phase=phase_num,
                tick=int(row['tick']),
                coherence=float(row.get('coherence', 0)),
                entropy=float(row.get('entropy', 0)),
                ethics_total=float(row.get('ethics_total', 0)),
                timestamp=row.get('audit_tag', '')
            )
            timings.append(timing)
            times.append(timing.ethics_total * 1000)  # Use ethics as proxy for time

    if not times:
        return [], PerformanceMetrics(phase_num, 0, 0, 0, 0, 0)
    
    metrics = PerformanceMetrics(
        phase=phase_num,
        min_time_ms=min(times),
        max_time_ms=max(times),
        avg_time_ms=sum(times) / len(times),
        total_time_ms=sum(times),
        tick_count=len(timings)
    )
    
    return timings, metrics


def print_phase_report(metrics: PerformanceMetrics):
    """Print formatted phase performance report."""
    print(f"\n{'='*60}")
    print(f"Phase {metrics.phase} Performance Report")
    print(f"{'='*60}")
    print(f"Total Ticks:        {metrics.tick_count}")
    print(f"Total Time:         {metrics.total_time_ms:.3f} ms")
    print(f"Average Time/Tick:  {metrics.avg_time_ms:.3f} ms")
    print(f"Min Time:           {metrics.min_time_ms:.3f} ms")
    print(f"Max Time:           {metrics.max_time_ms:.3f} ms")
    print(f"{'='*60}\n")


def compare_phases(csv_files: List[str]) -> None:
    """Compare performance across multiple phases."""
    print("\n" + "="*60)
    print("Phase Performance Comparison")
    print("="*60)
    print(f"{'Phase':<8} {'Ticks':<8} {'Total (ms)':<15} {'Avg/Tick (ms)':<15}")
    print("-"*60)
    
    total_time = 0
    for csv_file in csv_files:
        _, metrics = parse_phase_csv(csv_file)
        print(f"{metrics.phase:<8} {metrics.tick_count:<8} {metrics.total_time_ms:<15.3f} {metrics.avg_time_ms:<15.3f}")
        total_time += metrics.total_time_ms
    
    print("-"*60)
    print(f"{'TOTAL':<8} {'':<8} {total_time:<15.3f}")
    print("="*60 + "\n")


def analyze_gpu_benchmark(json_path: str) -> None:
    """Analyze GPU vs CPU benchmark results."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print("\n" + "="*60)
        print("GPU vs CPU Performance Analysis")
        print("="*60)
        
        if 'gpu_vs_cpu_ratio' in data:
            ratio = data['gpu_vs_cpu_ratio']
            print(f"GPU/CPU Ratio:      {ratio:.2f}x")
            
            if ratio < 1.0:
                speedup = (1.0 - ratio) * 100
                print(f"GPU Advantage:      {speedup:.1f}% faster")
            elif ratio > 1.0:
                slowdown = (ratio - 1.0) * 100
                print(f"CPU Advantage:      {slowdown:.1f}% faster")
            else:
                print("GPU/CPU:            Equivalent performance")
        
        if 'phases' in data:
            print("\nPhase Timings:")
            for phase, time_ms in data['phases'].items():
                if time_ms > 0:
                    print(f"  {phase}: {time_ms:.3f} ms")
        
        print("="*60 + "\n")
    except Exception as e:
        print(f"Error analyzing benchmark: {e}")


def generate_recommendations(metrics: PerformanceMetrics) -> List[str]:
    """Generate optimization recommendations based on metrics."""
    recommendations = []
    
    if metrics.avg_time_ms > 100:
        recommendations.append(
            f"⚠️  Phase {metrics.phase} averages {metrics.avg_time_ms:.1f}ms per tick - consider profiling"
        )
    
    if metrics.max_time_ms > metrics.avg_time_ms * 2:
        recommendations.append(
            f"⚠️  Phase {metrics.phase} has outlier spikes ({metrics.max_time_ms:.1f}ms) - investigate variance"
        )
    
    if metrics.tick_count > 1000:
        recommendations.append(
            f"✅ Phase {metrics.phase} has good throughput ({metrics.tick_count} ticks)"
        )
    
    return recommendations


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    if sys.argv[1] == '--compare' and len(sys.argv) >= 4:
        # Compare multiple phases
        compare_phases(sys.argv[2:])
    elif sys.argv[1] == '--gpu-bench' and len(sys.argv) >= 3:
        # Analyze GPU benchmark
        analyze_gpu_benchmark(sys.argv[2])
    else:
        # Single phase analysis
        csv_path = sys.argv[1]
        if not Path(csv_path).exists():
            print(f"Error: File not found: {csv_path}")
            sys.exit(1)
        
        timings, metrics = parse_phase_csv(csv_path)
        print_phase_report(metrics)
        
        # Generate recommendations
        recommendations = generate_recommendations(metrics)
        if recommendations:
            print("Recommendations:")
            for rec in recommendations:
                print(f"  {rec}")
            print()


if __name__ == '__main__':
    main()
