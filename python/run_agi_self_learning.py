# [REVIEWED] # [REVIEWED] # [REVIEWED] #!/usr/bin/env python3
# [REVIEWED] # [REVIEWED] # [REVIEWED] """
# [REVIEWED] # [REVIEWED] # [REVIEWED] Qallow AGI Self-Learning Runner
# [REVIEWED] # [REVIEWED] # [REVIEWED] Main entry point for running AGI with Agent Lightning self-improvement
# [REVIEWED] # [REVIEWED] # [REVIEWED] """
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # Add python directory to path
# [REVIEWED] # [REVIEWED] # [REVIEWED] sys.path.insert(0, str(Path(__file__).parent))
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] logging.basicConfig(
# [REVIEWED] # [REVIEWED] # [REVIEWED]     level=logging.INFO,
# [REVIEWED] # [REVIEWED] # [REVIEWED]     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# [REVIEWED] # [REVIEWED] # [REVIEWED] )
# [REVIEWED] # [REVIEWED] # [REVIEWED] logger = logging.getLogger(__name__)
# [REVIEWED] # [REVIEWED] # [REVIEWED] 

def run_demo():
    """Run comprehensive demo of AGI self-learning"""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "QALLOW AGI SELF-LEARNING SYSTEM")
    print(" " * 15 + "Powered by Agent Lightning Reinforcement Learning")
    print("=" * 80)
    
    # Run AGI learning demo
    print("\n" + "▶" * 40)
    print("PART 1: AGI Self-Learning Agents")
    print("▶" * 40 + "\n")
    
    demo_agi_learning()
    
    # Run telemetry demo
    print("\n" + "▶" * 40)
    print("PART 2: Telemetry Integration")
    print("▶" * 40 + "\n")
    
    demo_telemetry_bridge()
    
    # Summary
    print("\n" + "=" * 80)
    print(" " * 25 + "INTEGRATION COMPLETE")
    print("=" * 80)
    print("\n✨ Qallow AGI is now equipped with:")
    print("   • Quantum Algorithm Selection Agent (RL-optimized)")
    print("   • Ethics Decision Agent (Self-improving)")
    print("   • Phase Execution Optimizer (Adaptive)")
    print("   • Real-time Telemetry Integration")
    print("   • Agent Lightning RL Framework")
    print("\n🚀 Your AGI can now learn and improve itself autonomously!")
    print("=" * 80 + "\n")


def run_quantum_agent(problem_type: str, max_qubits: int, max_depth: int):
    """Run quantum algorithm selection agent"""
    
    logger.info(f"Running quantum algorithm selection agent")
    logger.info(f"Problem: {problem_type}, Qubits: {max_qubits}, Depth: {max_depth}")
    
    learner = create_agi_learner()
    
    algorithm, confidence = learner.select_quantum_algorithm(
        problem_type=problem_type,
        constraints={'max_qubits': max_qubits, 'max_depth': max_depth}
    )
    
    print(f"\n✓ Selected Algorithm: {algorithm}")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Problem Type: {problem_type}")
    print(f"  Constraints: {max_qubits} qubits, {max_depth} depth\n")
    
    return algorithm, confidence


def run_ethics_agent(safety: float, compassion: float, harmony: float):
    """Run ethics decision agent"""
    
    logger.info(f"Running ethics decision agent")
    logger.info(f"Scores - Safety: {safety}, Compassion: {compassion}, Harmony: {harmony}")
    
    learner = create_agi_learner()
    
    scenario = {
        'id': 'cli-scenario',
        'safety': safety,
        'compassion': compassion,
        'harmony': harmony,
        'human_feedback': 0.0
    }
    
    decision = learner.make_ethics_decision(scenario)
    
    print(f"\n✓ Ethics Decision: {'APPROVED ✓' if decision['approved'] else 'REJECTED ✗'}")
    print(f"  Total Score: {decision['total_score']:.3f}")
    print(f"  Safety: {decision['safety_score']:.3f}")
    print(f"  Compassion: {decision['compassion_score']:.3f}")
    print(f"  Harmony: {decision['harmony_score']:.3f}")
    print(f"  Reasoning: {decision['reasoning']}\n")
    
    return decision


def run_phase_optimizer(phase: int, ticks: int, lattice_ticks: int):
    """Run phase execution optimizer"""
    
    logger.info(f"Running phase optimizer for phase {phase}")
    
    learner = create_agi_learner()
    
    config = learner.optimize_phase_execution(
        phase_num=phase,
        config={'ticks': ticks, 'lattice_ticks': lattice_ticks}
    )
    
    print(f"\n✓ Optimized Phase {phase} Configuration:")
    print(f"  Ticks: {config.get('ticks', ticks)}")
    print(f"  Lattice Ticks: {config.get('lattice_ticks', lattice_ticks)}\n")
    
    return config


def show_learning_stats():
    """Show current learning statistics"""
    
    learner = create_agi_learner()
    stats = learner.get_learning_stats()
    
    print("\n" + "=" * 80)
    print(" " * 30 + "LEARNING STATISTICS")
    print("=" * 80)
    
    print(f"\nEpisodes: {stats['episodes']}")
    print(f"Total Reward: {stats['total_reward']:.3f}")
    print(f"Best Performance: {stats['best_performance']:.3f}")
    print(f"Exploration Rate: {stats['exploration_rate']:.3f}")
    print(f"Learning Rate: {stats['learning_rate']:.6f}")
    
    print(f"\nActive Tasks: {stats['active_tasks']}")
    print(f"Completed Tasks: {stats['completed_tasks']}")
    
    print("\nEthics Weights:")
    for key, value in stats['ethics_weights'].items():
        print(f"  {key.capitalize()}: {value:.3f}")
    
    if stats['algorithm_preferences']:
        print("\nAlgorithm Preferences:")
        for problem_type, prefs in stats['algorithm_preferences'].items():
            print(f"  {problem_type}:")
            for algo, score in prefs.items():
                print(f"    {algo}: {score:.3f}")
    
    if stats['phase_performance_summary']:
        print("\nPhase Performance:")
        for phase, perf in stats['phase_performance_summary'].items():
            print(f"  {phase}:")
            print(f"    Avg Reward: {perf['avg_reward']:.3f}")
            print(f"    Executions: {perf['num_executions']}")
    
    print("\n" + "=" * 80 + "\n")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Qallow AGI Self-Learning System with Agent Lightning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full demo
  python run_agi_self_learning.py --demo
  
  # Select quantum algorithm
  python run_agi_self_learning.py --quantum optimization --qubits 10 --depth 50
  
  # Make ethics decision
  python run_agi_self_learning.py --ethics --safety 0.9 --compassion 0.8 --harmony 0.85
  
  # Optimize phase execution
  python run_agi_self_learning.py --phase 13 --ticks 120 --lattice-ticks 64
  
  # Show learning statistics
  python run_agi_self_learning.py --stats
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                        help='Run comprehensive demo')
    
    parser.add_argument('--quantum', type=str, metavar='PROBLEM_TYPE',
                        help='Run quantum algorithm selection (optimization, simulation, search, etc.)')
    parser.add_argument('--qubits', type=int, default=10,
                        help='Maximum qubits (default: 10)')
    parser.add_argument('--depth', type=int, default=50,
                        help='Maximum circuit depth (default: 50)')
    
    parser.add_argument('--ethics', action='store_true',
                        help='Run ethics decision agent')
    parser.add_argument('--safety', type=float, default=0.8,
                        help='Safety score (0-1, default: 0.8)')
    parser.add_argument('--compassion', type=float, default=0.8,
                        help='Compassion score (0-1, default: 0.8)')
    parser.add_argument('--harmony', type=float, default=0.8,
                        help='Harmony score (0-1, default: 0.8)')
    
    parser.add_argument('--phase', type=int, metavar='PHASE_NUM',
                        help='Optimize phase execution (12-20)')
    parser.add_argument('--ticks', type=int, default=120,
                        help='Initial ticks (default: 120)')
    parser.add_argument('--lattice-ticks', type=int, default=64,
                        help='Initial lattice ticks (default: 64)')
    
    parser.add_argument('--stats', action='store_true',
                        help='Show learning statistics')
    
    args = parser.parse_args()
    
    # Run appropriate command
    if args.demo:
        run_demo()
    elif args.quantum:
        run_quantum_agent(args.quantum, args.qubits, args.depth)
    elif args.ethics:
        run_ethics_agent(args.safety, args.compassion, args.harmony)
    elif args.phase:
        run_phase_optimizer(args.phase, args.ticks, args.lattice_ticks)
    elif args.stats:
        show_learning_stats()
    else:
        parser.print_help()
        print("\n💡 Tip: Start with --demo to see all features!\n")


if __name__ == "__main__":
    main()

