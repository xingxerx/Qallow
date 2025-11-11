#!/usr/bin/env python3
"""
Qallow AGI Full Integration Demo
Demonstrates: Roblox Game Creation + AI Research + Self-Improvement
"""

import asyncio
import sys
import subprocess

sys.path.append("..")

from qallow.master_orchestrator import QallowMasterAgent

def start_mock_server():
    """Starts the mock Roblox server intelligently."""
    print("🤖 Checking and starting mock Roblox server if needed...")
    script_path = "scripts/start_mock_server_intelligent.sh"
    try:
        # Make sure the script is executable
        subprocess.run(["chmod", "+x", script_path], check=True)
        # Run the script and capture output
        result = subprocess.run([f"./{script_path}"], check=True, capture_output=True, text=True, shell=True)
        print(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Failed to start mock server: {e}")
        print("Please ensure the script exists and you have necessary permissions.")
        sys.exit(1)

async def main():
    """
    Qallow will:
    1. Create a Roblox obby game with 10 levels
    2. Research AI NPC behavior techniques
    3. Apply the best technique to improve NPCs
    4. Test the complete game
    5. Store all learnings in memory
    """
    
    start_mock_server()
    
    print("="*60)
    print("🚀 QALLOW AGI - FULL INTEGRATION DEMO")
    print("🎮 Roblox Game Creation + 🔬 AI Self-Research")
    print("="*60)
    
    # Initialize AGI
    agi = QallowMasterAgent()
    
    # Ensure Qdrant collection is ready
    await agi.memory.initialize_collection_if_needed()
    
    # Execute complex multi-agent task
    result = await agi.execute("""
        Create a Roblox obby game with the following specifications:
        - 10 levels with increasing difficulty
        - Moving platforms with physics-based movement
        - 5 NPCs that follow the player
        - A leaderboard that tracks completion time
        - Spawn points and checkpoints
        
        After creating the game:
        1. Research "AI NPC behavior techniques" on arXiv
        2. Find and analyze the top 3 most relevant papers
        3. Extract the best technique for improving NPC behavior
        4. Apply that technique to the game's NPCs
        5. Test the updated game
        6. Store all learnings, successes, and failures
        
        Finally, generate a report of what was learned and improved.
    """)
    
    # Print results
    print("\n" + "="*60)
    print("📊 EXECUTION COMPLETE - FINAL REPORT")
    print("="*60)
    print(f"✅ Overall Success: {result.success}")
    print(f"🎮 Roblox Game ID: {result.game_id}")
    print(f"📚 Research Papers Analyzed: {result.papers_analyzed}")
    print(f"✨ Prompt Improvements Applied: {result.improvements_applied}")
    print(f"🧠 Total Experiences in Memory: {result.total_experiences}")
    
    if result.game_id:
        print(f"\n🌐 Game URL: https://www.roblox.com/games/{result.game_id}")
    
    print("\n🔄 Self-improvement cycle complete!")
    print("Qallow is now smarter and will apply these learnings to future tasks.")
    print("="*60)

if __name__ == "__main__":
    # Check dependencies
    try:
        import qdrant_client
        import arxiv
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Run demo
    asyncio.run(main())
