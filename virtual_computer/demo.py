#!/usr/bin/env python3
"""
Virtual Computer Demo - Full System Integration
Shows how Lightning Agent can optimize the virtual computer system
"""


from virtual_computer import (
    VirtualComputer,
    WorkloadType,
    AgentOptimizationTasks,
)


def print_banner(text: str, width: int = 70):
    """Print a formatted banner"""
    print(f"\n{'═'*width}")
    print(f"║ {text.center(width-4)} ║")
    print(f"{'═'*width}\n")


def demo_virtual_computer():
    """Demonstrate the virtual computer system"""
    print_banner("VIRTUAL COMPUTER SYSTEM - CUDA + Neuromorphic + Photonic")
    
    # Create virtual computer
    vc = VirtualComputer()
    print("✓ Initialized Virtual Computer")
    print("  - CUDA GPU (8GB memory, 80 SMs, 128 cores/SM)")
    print("  - Neuromorphic Processor (1000 neurons, 4 layers)")
    print("  - Photonic Processor (64 waveguides, 256 gates)\n")
    
    # Create workloads
    print("Creating diversified workloads for optimization:\n")
    
    workloads = [
        (WorkloadType.GPU_COMPUTE, 5, 512, 5_000_000_000, "GPU Compute"),
        (WorkloadType.GPU_MEMORY_INTENSIVE, 4, 2048, 1_000_000_000, "GPU Memory"),
        (WorkloadType.NEURAL_INFERENCE, 6, 256, 2_000_000_000, "Neural Inference"),
        (WorkloadType.NEURAL_TRAINING, 3, 1024, 10_000_000_000, "Neural Training"),
        (WorkloadType.PHOTONIC_COMPUTE, 5, 128, 500_000_000, "Photonic Compute"),
        (WorkloadType.HYBRID_PROCESSING, 7, 768, 8_000_000_000, "Hybrid Processing"),
    ]
    
    for workload_type, priority, data_size, compute_ops, label in workloads:
        vc.create_workload(
            workload_type=workload_type,
            priority=priority,
            data_size_mb=data_size,
            compute_ops=compute_ops
        )
        print(f"  ✓ {label:20} (priority={priority}, size={data_size}MB)")
    
    print(f"\n✓ Total workloads created: {len(vc.workload_queue)}")
    
    # Execute workloads
    print_banner("EXECUTING WORKLOADS")
    print("Processing scheduled workloads...\n")
    
    start_time = time.time()
    results = vc.run_scheduled_workloads()
    elapsed = time.time() - start_time
    
    print(f"✓ Workload execution completed in {elapsed:.2f}s")
    print(f"  - Workloads executed: {results['workloads_executed']}")
    print(f"  - Total energy: {results['total_energy_consumed_j']:.3f} J")
    print(f"  - Throughput: {results['throughput_workloads_per_sec']:.2f} workloads/sec\n")
    
    # Print system status
    print_banner("SYSTEM STATUS")
    vc.print_system_status()
    
    # Show individual processor stats
    print("Individual Processor Performance:\n")
    cuda_stats = vc.cuda_gpu.get_device_properties()
    print(f"  CUDA GPU:")
    print(f"    - Compute Capability: {cuda_stats['compute_capability']}")
    print(f"    - Total Cores: {cuda_stats['total_cores']}")
    print(f"    - Memory: {cuda_stats['free_memory_mb']}/{cuda_stats['device_memory_mb']} MB free")
    print(f"    - Kernels Launched: {cuda_stats['total_launches']}")
    
    neuro_stats = vc.neuromorphic.get_stats()
    print(f"\n  Neuromorphic Processor:")
    print(f"    - Total Neurons: {neuro_stats['total_neurons']}")
    print(f"    - Total Spikes: {neuro_stats['total_spikes']}")
    print(f"    - Average Spike Rate: {neuro_stats['avg_spike_rate']:.2f} Hz")
    print(f"    - Energy: {neuro_stats['energy_consumed_uj']:.2f} µJ")
    
    photo_stats = vc.photonic.get_processor_stats()
    print(f"\n  Photonic Processor:")
    print(f"    - Photons Injected: {photo_stats['total_photons_injected']}")
    print(f"    - Photons Detected: {photo_stats['total_photons_detected']}")
    print(f"    - Detection Efficiency: {photo_stats['detection_efficiency']:.2%}")
    print(f"    - Switching Operations: {photo_stats['total_switching_operations']}\n")


def demo_optimization_tasks():
    """Demonstrate optimization tasks for the agent"""
    print_banner("AGENT OPTIMIZATION TASKS")
    
    tasks = AgentOptimizationTasks()
    
    print(f"Total optimization tasks available: {len(tasks.tasks)}\n")
    
    # Show task breakdown
    categories = {}
    for task in tasks.tasks:
        if task.category not in categories:
            categories[task.category] = 0
        categories[task.category] += 1
    
    print("Tasks by Category:")
    for category, count in sorted(categories.items()):
        print(f"  • {category:30} {count:3} tasks")
    
    print("\n" + "─"*70)
    print("High-Priority Tasks (Difficulty 5+):\n")
    
    hard_tasks = tasks.get_tasks_by_difficulty(5, 10)
    for task in hard_tasks[:5]:
        print(f"  #{task.task_id:2} {task.name:40} (difficulty: {'⭐' * task.difficulty})")
        print(f"      {task.description}")
        print(f"      Performance: {task.current_performance:.0%} → {task.optimization_goal:.0%}")
        print()
    
    print("─"*70)
    print("\nSimulating optimization on a task...\n")
    
    # Simulate optimization
    sample_task = tasks.tasks[0]
    print(f"Optimizing: {sample_task.name}")
    print(f"  Baseline: {sample_task.baseline_performance:.0%}")
    print(f"  Goal: {sample_task.optimization_goal:.0%}")
    print(f"  Current: {sample_task.current_performance:.0%}\n")
    
    # Simulate multiple optimization efforts
    for effort in [0.3, 0.6, 1.0]:
        result = tasks.simulate_optimization(sample_task.task_id, effort)
        if result["success"]:
            print(f"  Optimization effort: {effort:.0%}")
            print(f"    New performance: {result['new_performance']:.2%}")
            print(f"    Improvement: +{result['improvement_percent']:.1f}%")
            print(f"    Progress: {result['progress_to_goal']:.1f}%\n")


def demo_agent_optimization_loop():
    """Demonstrate how Lightning Agent could optimize the system"""
    print_banner("LIGHTNING AGENT OPTIMIZATION LOOP")
    
    vc = VirtualComputer()
    tasks = AgentOptimizationTasks()
    
    print("Scenario: Lightning Agent automatically optimizes virtual computer\n")
    
    # Show baseline
    print("BASELINE PERFORMANCE:")
    print(f"  Total tasks: {len(tasks.tasks)}")
    avg_baseline = sum(t.baseline_performance for t in tasks.tasks) / len(tasks.tasks)
    avg_current = sum(t.current_performance for t in tasks.tasks) / len(tasks.tasks)
    print(f"  Average performance: {avg_current:.2%}")
    print(f"  Room for improvement: {(sum(t.optimization_goal - t.current_performance for t in tasks.tasks) / len(tasks.tasks)):.2%}\n")
    
    # Simulate optimization iterations
    print("AGENT OPTIMIZATION ITERATIONS:\n")
    
    for iteration in range(1, 4):
        print(f"  Iteration {iteration}:")
        
        # Agent identifies high-impact tasks
        tasks_to_optimize = tasks.get_tasks_by_difficulty(3, 10)
        random_sample = tasks_to_optimize[:min(3, len(tasks_to_optimize))]
        
        total_improvement = 0
        for task in random_sample:
            old_perf = task.current_performance
            result = tasks.simulate_optimization(task.task_id, 0.5)
            improvement = result["improvement_percent"] if result["success"] else 0
            total_improvement += improvement
            
            print(f"    • {task.name:40} +{improvement:5.1f}%")
        
        avg_current = sum(t.current_performance for t in tasks.tasks) / len(tasks.tasks)
        print(f"    → Average performance now: {avg_current:.2%}\n")
    
    # Show final status
    print("FINAL OPTIMIZATION STATUS:")
    tasks_completed = sum(1 for t in tasks.tasks if t.current_performance >= t.optimization_goal)
    print(f"  Tasks completed: {tasks_completed}/{len(tasks.tasks)}")
    avg_final = sum(t.current_performance for t in tasks.tasks) / len(tasks.tasks)
    improvement_total = (avg_final - avg_baseline) * 100
    print(f"  Total improvement: +{improvement_total:.1f}%")
    print(f"  Final average performance: {avg_final:.2%}\n")


def main():
    """Run full demonstration"""
    print("\n" + "="*70)
    print("  QALLOW VIRTUAL COMPUTER SYSTEM")
    print("  CUDA + Neuromorphic + Photonic Processors")
    print("  Lightning Agent Optimization Platform")
    print("="*70)
    
    # Run demonstrations
    demo_virtual_computer()
    demo_optimization_tasks()
    demo_agent_optimization_loop()
    
    print_banner("DEMONSTRATION COMPLETE")
    print("✅ Virtual computer system ready for Lightning Agent optimization!")
    print("\nTo integrate with Lightning Agent:")
    print("  1. Import: from virtual_computer import VirtualComputer, AgentOptimizationTasks")
    print("  2. Create: vc = VirtualComputer()")
    print("  3. Optimize: tasks = AgentOptimizationTasks()")
    print("  4. Analyze: tasks.simulate_optimization(...)")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
