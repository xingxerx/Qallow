#!/usr/bin/env python3
"""
Interactive Demo: How to Communicate with the Integrated System
Demonstrates all 4 communication methods
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.agents.kimi_k2_agent import create_kimi_k2_agent
import cirq
import cudaq
import numpy as np


def demo_simple_chat():
    """Demo 1: Simple Chat"""
    print("\n" + "="*70)
    print("DEMO 1: Simple Chat (Python SDK)")
    print("="*70)
    
    try:
        agent = create_kimi_k2_agent()
        
        messages = [
            "What is quantum computing?",
            "Explain QAOA optimization",
            "What is a Bell pair?",
        ]
        
        for msg in messages:
            print(f"\n👤 User: {msg}")
            response = agent.chat(msg)
            print(f"🤖 Agent: {response[:200]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure vLLM server is running: bash scripts/setup_kimi_k2_vllm.sh")


def demo_streaming_chat():
    """Demo 2: Streaming Chat"""
    print("\n" + "="*70)
    print("DEMO 2: Streaming Chat (Python SDK)")
    print("="*70)
    
    try:
        agent = create_kimi_k2_agent()
        
        print("\n👤 User: Explain quantum circuits step by step")
        print("🤖 Agent: ", end="", flush=True)
        
        for chunk in agent.chat_stream("Explain quantum circuits step by step"):
            print(chunk, end="", flush=True)
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_quantum_analysis():
    """Demo 3: Quantum Circuit Analysis"""
    print("\n" + "="*70)
    print("DEMO 3: Quantum Circuit Analysis")
    print("="*70)
    
    try:
        # Create Bell pair circuit with Cirq
        print("\n📊 Creating Bell pair circuit with Cirq...")
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key='result')
        )
        print(f"Circuit:\n{circuit}")
        
        # Simulate
        print("\n🔬 Simulating circuit...")
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        print(f"Result: {result}")
        
        # Analyze with Kimi-K2
        print("\n🤖 Analyzing with Kimi-K2...")
        agent = create_kimi_k2_agent()
        analysis = agent.chat(
            f"Analyze this quantum circuit result: {result}. "
            f"What does it represent?"
        )
        print(f"Analysis: {analysis[:300]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_qaoa_optimization():
    """Demo 4: QAOA Optimization"""
    print("\n" + "="*70)
    print("DEMO 4: QAOA Optimization with CUDA-Q")
    print("="*70)
    
    try:
        # Define QAOA kernel
        print("\n📊 Creating QAOA kernel...")
        
        @cudaq.kernel
        def qaoa():
            qubits = cudaq.qvector(3)
            h(qubits)
            for i in range(2):
                cz(qubits[i], qubits[i+1])
            for qubit in qubits:
                rx(np.pi/4, qubit)
        
        # Execute
        print("🔬 Executing QAOA kernel...")
        result = cudaq.sample(qaoa, shots_count=100)
        print(f"Result: {result}")
        
        # Analyze
        print("\n🤖 Analyzing with Kimi-K2...")
        agent = create_kimi_k2_agent()
        analysis = agent.chat(
            f"Analyze this QAOA result: {result}. "
            f"What optimization insights can you provide?"
        )
        print(f"Analysis: {analysis[:300]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_tool_calling():
    """Demo 5: Tool Calling"""
    print("\n" + "="*70)
    print("DEMO 5: Tool Calling")
    print("="*70)
    
    try:
        agent = create_kimi_k2_agent()
        
        tools = [
            {
                "name": "create_circuit",
                "description": "Create a quantum circuit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "qubits": {"type": "integer", "description": "Number of qubits"},
                        "gates": {"type": "string", "description": "Gate sequence"}
                    }
                }
            },
            {
                "name": "simulate_circuit",
                "description": "Simulate a quantum circuit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "circuit": {"type": "string", "description": "Circuit definition"}
                    }
                }
            }
        ]
        
        print("\n👤 User: Create a Bell pair circuit and simulate it")
        response = agent.chat_with_tools(
            "Create a Bell pair circuit and simulate it",
            tools=tools
        )
        print(f"🤖 Agent: {response[:300]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_interactive_loop():
    """Demo 6: Interactive Chat Loop"""
    print("\n" + "="*70)
    print("DEMO 6: Interactive Chat Loop")
    print("="*70)
    print("\n💬 Type 'exit' to quit, 'help' for commands\n")
    
    try:
        agent = create_kimi_k2_agent()
        
        while True:
            try:
                user_input = input("👤 You: ").strip()
                
                if user_input.lower() == "exit":
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == "help":
                    print("""
Commands:
  exit     - Exit the chat
  help     - Show this help
  stream   - Use streaming mode
  quantum  - Ask about quantum computing
  qaoa     - Ask about QAOA
  
Just type any message to chat!
                    """)
                    continue
                
                if not user_input:
                    continue
                
                print("🤖 Agent: ", end="", flush=True)
                response = agent.chat(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
                
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main demo menu"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              INTERACTIVE DEMO: Communication Methods                       ║
║                                                                            ║
║              How to Interact with Your Integrated System                  ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Choose a demo:
  1. Simple Chat
  2. Streaming Chat
  3. Quantum Circuit Analysis
  4. QAOA Optimization
  5. Tool Calling
  6. Interactive Chat Loop
  0. Exit

Note: Make sure vLLM server is running:
  bash scripts/setup_kimi_k2_vllm.sh
    """)
    
    demos = {
        "1": ("Simple Chat", demo_simple_chat),
        "2": ("Streaming Chat", demo_streaming_chat),
        "3": ("Quantum Circuit Analysis", demo_quantum_analysis),
        "4": ("QAOA Optimization", demo_qaoa_optimization),
        "5": ("Tool Calling", demo_tool_calling),
        "6": ("Interactive Chat Loop", demo_interactive_loop),
    }
    
    while True:
        choice = input("\n👤 Select demo (0-6): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        
        if choice in demos:
            name, demo_func = demos[choice]
            print(f"\n▶️  Running: {name}")
            demo_func()
        else:
            print("❌ Invalid choice. Please select 0-6.")


if __name__ == "__main__":
    main()

