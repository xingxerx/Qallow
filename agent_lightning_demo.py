#!/usr/bin/env python3
"""
Agent Lightning Demo - Simple Math Agent
This demonstrates how to use Microsoft's Agent Lightning framework
to train an AI agent with reinforcement learning.
"""

import os
from openai import OpenAI

# Import Agent Lightning
try:
    import agentlightning as agl
except ImportError:
    print("Error: agentlightning not installed. Run: pip install agentlightning")
    exit(1)


def simple_math_agent():
    """
    A simple agent that solves math problems.
    This demonstrates the basic Agent Lightning workflow.
    """
    
    # Initialize OpenAI client (you can use any LLM provider)
    # For demo purposes, we'll use a mock setup
    print("=" * 60)
    print("Agent Lightning Demo - Simple Math Agent")
    print("=" * 60)
    
    # Example: Create a simple task
    task = "What is 15 + 27?"
    
    print(f"\nTask: {task}")
    print("\nThis demo shows how Agent Lightning works:")
    print("1. Agent receives a task")
    print("2. Agent processes and responds")
    print("3. Agent Lightning tracks the interaction")
    print("4. You can train the agent with RL to improve performance")
    
    # In a real scenario, you would:
    # 1. Set up your LLM client
    # 2. Instrument it with Agent Lightning
    # 3. Run tasks and collect traces
    # 4. Train with RL algorithms
    
    print("\n" + "=" * 60)
    print("Key Agent Lightning Features:")
    print("=" * 60)
    print("✓ Zero code change integration (almost!)")
    print("✓ Works with ANY agent framework")
    print("✓ Supports multiple RL algorithms")
    print("✓ Selective multi-agent optimization")
    print("✓ Built-in tracing and monitoring")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Set up your LLM provider (OpenAI, Azure, etc.)")
    print("2. Create your agent logic")
    print("3. Add Agent Lightning instrumentation")
    print("4. Run training with: agl store")
    print("5. Monitor progress and iterate")
    
    print("\n" + "=" * 60)
    print("Available Commands:")
    print("=" * 60)
    print("  agl vllm      - Run vLLM with Agent Lightning")
    print("  agl store     - Run LightningStore server")
    print("  agl agentops  - Start AgentOps server")
    
    print("\n" + "=" * 60)
    print("Documentation:")
    print("=" * 60)
    print("  GitHub: https://github.com/microsoft/agent-lightning")
    print("  Docs:   https://microsoft.github.io/agent-lightning/")
    print("  Paper:  https://arxiv.org/abs/2508.03680")
    
    print("\n✨ Agent Lightning is ready to light up your AI agents! ⚡\n")


def show_architecture():
    """Display Agent Lightning architecture overview."""
    print("\n" + "=" * 60)
    print("Agent Lightning Architecture")
    print("=" * 60)
    print("""
    Your Agent (any framework)
            ↓
    agl.emit_xxx() helpers
            ↓
    LightningStore (central hub)
            ↓
    Algorithm (RL, APO, SFT, etc.)
            ↓
    Trainer (orchestrates learning)
            ↓
    Improved Agent Performance ⚡
    """)
    print("=" * 60)


def show_example_code():
    """Show example integration code."""
    print("\n" + "=" * 60)
    print("Example Integration Code")
    print("=" * 60)
    print("""
# Example: Integrate Agent Lightning with your agent

import agentlightning as agl
from openai import OpenAI

# 1. Initialize your LLM client
client = OpenAI(api_key="your-api-key")

# 2. Create your agent function
def my_agent(task):
    # Emit task start event
    agl.emit_task_start(task_id="task-1", task=task)
    
    # Run your agent logic
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": task}]
    )
    
    # Emit task completion
    agl.emit_task_complete(
        task_id="task-1",
        result=response.choices[0].message.content
    )
    
    return response

# 3. Run your agent
result = my_agent("Solve this problem: 2 + 2")

# 4. Train with RL (in separate process)
# agl store --algorithm=ppo
    """)
    print("=" * 60)


if __name__ == "__main__":
    # Run the demo
    simple_math_agent()
    show_architecture()
    show_example_code()
    
    print("\n🚀 Ready to start training your agents with RL!")
    print("📚 Check out the examples at: https://github.com/microsoft/agent-lightning/tree/main/examples\n")

