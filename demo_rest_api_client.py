#!/usr/bin/env python3
"""
REST API Client Demo: How to Communicate via HTTP
Shows how to interact with the FastAPI Chat Server
"""

import requests
import json
import time
from typing import Optional, Dict, Any


class QallowAPIClient:
    """Client for Qallow Chat Server REST API"""
    
    def __init__(self, base_url: str = "http://localhost:8008"):
        """Initialize API client"""
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "unhealthy"}
    
    def get_models(self) -> Dict[str, Any]:
        """Get available models"""
        try:
            response = self.session.get(f"{self.base_url}/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def chat(self, message: str, backend: str = "kimi_k2") -> str:
        """Send a chat message"""
        try:
            payload = {
                "message": message,
                "backend": backend
            }
            response = self.session.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "No response")
        except Exception as e:
            return f"Error: {e}"
    
    def chat_stream(self, message: str) -> None:
        """Stream a chat response"""
        try:
            payload = {"message": message}
            response = self.session.post(
                f"{self.base_url}/chat/stream",
                json=payload,
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            
            print("🤖 Agent: ", end="", flush=True)
            for chunk in response.iter_content(decode_unicode=True):
                if chunk:
                    print(chunk, end="", flush=True)
            print()
            
        except Exception as e:
            print(f"Error: {e}")
    
    def chat_with_tools(self, message: str, tools: list) -> str:
        """Chat with tool calling"""
        try:
            payload = {
                "message": message,
                "tools": tools
            }
            response = self.session.post(
                f"{self.base_url}/chat/tools",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"Error: {e}"


def demo_health_check():
    """Demo 1: Health Check"""
    print("\n" + "="*70)
    print("DEMO 1: Health Check")
    print("="*70)
    
    client = QallowAPIClient()
    print("\n📡 Checking server health...")
    
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    if health.get("status") == "healthy":
        print("✅ Server is healthy!")
    else:
        print("❌ Server is not responding")
        print("💡 Start the server with:")
        print("   bash scripts/setup_kimi_k2_vllm.sh &")
        print("   cd python/chat_server && uvicorn main:app --port 8008 &")


def demo_get_models():
    """Demo 2: Get Available Models"""
    print("\n" + "="*70)
    print("DEMO 2: Get Available Models")
    print("="*70)
    
    client = QallowAPIClient()
    print("\n📡 Fetching available models...")
    
    models = client.get_models()
    print(json.dumps(models, indent=2))


def demo_simple_chat():
    """Demo 3: Simple Chat"""
    print("\n" + "="*70)
    print("DEMO 3: Simple Chat")
    print("="*70)
    
    client = QallowAPIClient()
    
    messages = [
        "What is quantum computing?",
        "Explain QAOA optimization",
        "What is a Bell pair?",
    ]
    
    for msg in messages:
        print(f"\n👤 User: {msg}")
        response = client.chat(msg)
        print(f"🤖 Agent: {response[:200]}...")
        time.sleep(1)


def demo_streaming_chat():
    """Demo 4: Streaming Chat"""
    print("\n" + "="*70)
    print("DEMO 4: Streaming Chat")
    print("="*70)
    
    client = QallowAPIClient()
    
    print("\n👤 User: Explain quantum circuits step by step")
    client.chat_stream("Explain quantum circuits step by step")


def demo_tool_calling():
    """Demo 5: Tool Calling"""
    print("\n" + "="*70)
    print("DEMO 5: Tool Calling")
    print("="*70)
    
    client = QallowAPIClient()
    
    tools = [
        {
            "name": "create_circuit",
            "description": "Create a quantum circuit",
            "parameters": {
                "type": "object",
                "properties": {
                    "qubits": {"type": "integer"},
                    "gates": {"type": "string"}
                }
            }
        },
        {
            "name": "simulate_circuit",
            "description": "Simulate a quantum circuit",
            "parameters": {
                "type": "object",
                "properties": {
                    "circuit": {"type": "string"}
                }
            }
        }
    ]
    
    print("\n👤 User: Create a Bell pair circuit and simulate it")
    response = client.chat_with_tools(
        "Create a Bell pair circuit and simulate it",
        tools=tools
    )
    print(f"🤖 Agent:\n{response}")


def demo_interactive_chat():
    """Demo 6: Interactive Chat"""
    print("\n" + "="*70)
    print("DEMO 6: Interactive Chat")
    print("="*70)
    print("\n💬 Type 'exit' to quit, 'stream' for streaming mode\n")
    
    client = QallowAPIClient()
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() == "exit":
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == "stream":
                msg = input("👤 Message: ").strip()
                if msg:
                    client.chat_stream(msg)
                continue
            
            if not user_input:
                continue
            
            response = client.chat(user_input)
            print(f"🤖 Agent: {response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


def main():
    """Main demo menu"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              REST API Client Demo                                          ║
║                                                                            ║
║              How to Communicate via HTTP Requests                         ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Choose a demo:
  1. Health Check
  2. Get Available Models
  3. Simple Chat
  4. Streaming Chat
  5. Tool Calling
  6. Interactive Chat
  0. Exit

Prerequisites:
  Terminal 1: bash scripts/setup_kimi_k2_vllm.sh
  Terminal 2: cd python/chat_server && uvicorn main:app --port 8008
    """)
    
    demos = {
        "1": ("Health Check", demo_health_check),
        "2": ("Get Models", demo_get_models),
        "3": ("Simple Chat", demo_simple_chat),
        "4": ("Streaming Chat", demo_streaming_chat),
        "5": ("Tool Calling", demo_tool_calling),
        "6": ("Interactive Chat", demo_interactive_chat),
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

