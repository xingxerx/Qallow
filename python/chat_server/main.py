"""
Qallow Agent Chat Server
Enhanced with Ollama support and quantum optimization tasks
Supports: DeepSeek baseline, Ollama agent, QAOA optimization
"""
import sys
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root to path to allow importing from python/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.deepseek_baseline import DeepSeekClient, DeepSeekConfig

# Try to import Ollama agent (may not be available)
try:
    from python.agents.qallow_agent_ollama import OllamaAgent, OllamaConfig, AgentTask
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama agent not available")

# Try to import Kimi-K2 agent (may not be available)
try:
    from python.agents.kimi_k2_agent import KimiK2Agent, KimiK2Config
    KIMI_K2_AVAILABLE = True
except ImportError:
    KIMI_K2_AVAILABLE = False
    logging.warning("Kimi-K2 agent not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger("ChatServer")

# --- FastAPI App ---
app = FastAPI(
    title="Qallow Agent Chat Server",
    description="Enhanced chat server with Ollama support and quantum optimization",
    version="2.0.0",
)

# --- Agent Initialization ---
# Backend selection from environment or default to mock
BACKEND = os.getenv("QALLOW_CHAT_BACKEND", "mock")  # mock, ollama, deepseek, kimi_k2
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2:70b")
KIMI_K2_BASE_URL = os.getenv("KIMI_K2_BASE_URL", "http://localhost:8000/v1")

logger.info(f"Initializing chat server with backend: {BACKEND}")

# Initialize DeepSeek client (always available as fallback)
deepseek_config = DeepSeekConfig(backend='mock')
deepseek_agent = DeepSeekClient(deepseek_config)

# Initialize Ollama agent if available and requested
ollama_agent: Optional[OllamaAgent] = None
if BACKEND == "ollama" and OLLAMA_AVAILABLE:
    try:
        ollama_config = OllamaConfig(model=OLLAMA_MODEL)
        ollama_agent = OllamaAgent(ollama_config)
        logger.info(f"✓ Ollama agent initialized with model: {OLLAMA_MODEL}")
    except Exception as e:
        logger.error(f"Failed to initialize Ollama agent: {e}")
        logger.info("Falling back to DeepSeek mock backend")

# Initialize Kimi-K2 agent if available and requested
kimi_k2_agent: Optional[KimiK2Agent] = None
if BACKEND == "kimi_k2" and KIMI_K2_AVAILABLE:
    try:
        kimi_k2_config = KimiK2Config(base_url=KIMI_K2_BASE_URL)
        kimi_k2_agent = KimiK2Agent(kimi_k2_config)
        logger.info(f"✓ Kimi-K2 agent initialized at {KIMI_K2_BASE_URL}")
    except Exception as e:
        logger.error(f"Failed to initialize Kimi-K2 agent: {e}")
        logger.info("Falling back to DeepSeek mock backend")

# --- API Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    backend: Optional[str] = Field(None, description="Override backend: mock, ollama, deepseek")

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    backend: str
    model: Optional[str] = None

class QuantumTaskRequest(BaseModel):
    task: str = Field(..., description="Task type: qaoa_optimize, status")
    nodes: Optional[int] = Field(256, description="Number of photonic nodes")
    target_fidelity: Optional[float] = Field(0.981, description="Target fidelity")

class QuantumTaskResponse(BaseModel):
    success: bool
    task: str
    result: Dict[str, Any]
    backend: str

# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Chat with AI agent (supports multiple backends)

    Backends:
    - mock: Fast mock responses (default)
    - ollama: Local Ollama inference (requires setup)
    - deepseek: DeepSeek API (requires API key)
    - kimi_k2: Local Kimi-K2 inference (no API key needed)
    """
    logger.info(f"Chat request: '{request.message[:50]}...' (session={request.session_id})")

    # Determine backend
    backend = request.backend or BACKEND

    try:
        if backend == "kimi_k2" and kimi_k2_agent:
            # Use Kimi-K2 agent for local inference
            system_prompt = "You are Kimi, an AI assistant created by Moonshot AI. Be helpful and concise."
            reply_text = kimi_k2_agent.chat(request.message, system_prompt=system_prompt)
            model_used = "Kimi-K2-Instruct"

        elif backend == "ollama" and ollama_agent:
            # Use Ollama agent for more sophisticated responses
            # For now, use a simple prompt wrapper
            system_prompt = "You are Qallow, an AI assistant for quantum computing. Be concise and helpful."
            response = ollama_agent._query_ollama(request.message, system_prompt)
            reply_text = response["response"]
            model_used = response["model"]

        else:
            # Use DeepSeek baseline (mock or API)
            mock_state = {
                "iteration": 1,
                "current_loss": 0.5,
                "best_loss": 0.4,
                "ethics_score": 0.9,
                "backend_name": "chat_server",
                "user_input": request.message,
            }

            reasoning_result = deepseek_agent.reason_cognitive_state(
                iteration=mock_state["iteration"],
                current_loss=mock_state["current_loss"],
                best_loss=mock_state["best_loss"],
                ethics_score=mock_state["ethics_score"],
                backend_name=mock_state["backend_name"],
            )

            reply_text = reasoning_result.get("analysis", "I am functioning.")
            model_used = "deepseek-mock"

        logger.info(f"Reply generated (backend={backend})")

        return ChatResponse(
            reply=reply_text,
            session_id=request.session_id,
            backend=backend,
            model=model_used
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quantum/task", response_model=QuantumTaskResponse)
async def quantum_task(request: QuantumTaskRequest):
    """
    Execute quantum optimization tasks

    Tasks:
    - qaoa_optimize: Optimize QAOA parameters for Phase 14
    - status: Get agent status
    """
    logger.info(f"Quantum task: {request.task}")

    if not ollama_agent:
        raise HTTPException(
            status_code=503,
            detail="Ollama agent not available. Set QALLOW_CHAT_BACKEND=ollama and ensure Ollama is running."
        )

    try:
        if request.task == "qaoa_optimize":
            result = ollama_agent.optimize_qaoa(
                nodes=request.nodes,
                target_fidelity=request.target_fidelity
            )

            return QuantumTaskResponse(
                success=True,
                task=request.task,
                result=result,
                backend="ollama"
            )

        elif request.task == "status":
            result = ollama_agent.get_status()

            return QuantumTaskResponse(
                success=True,
                task=request.task,
                result=result,
                backend="ollama"
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unknown task: {request.task}")

    except Exception as e:
        logger.error(f"Quantum task error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    """Root endpoint with server info"""
    return {
        "message": "Qallow Agent Chat Server",
        "version": "2.0.0",
        "backend": BACKEND,
        "ollama_available": OLLAMA_AVAILABLE,
        "ollama_active": ollama_agent is not None,
        "endpoints": {
            "chat": "/chat",
            "quantum_task": "/quantum/task",
            "health": "/health",
            "docs": "/docs"
        }
    }


class ToolCallRequest(BaseModel):
    message: str
    tools: Optional[list] = Field(None, description="List of tool definitions")
    session_id: str = "default"

class ToolCallResponse(BaseModel):
    reply: str
    session_id: str
    backend: str
    model: Optional[str] = None
    tool_calls_made: int = 0


@app.post("/chat/tools", response_model=ToolCallResponse)
async def chat_with_tools(request: ToolCallRequest):
    """
    Chat with tool calling support (Kimi-K2 only)

    Requires:
    - Kimi-K2 backend enabled
    - Tool definitions in request
    """
    logger.info(f"Tool call request: '{request.message[:50]}...' (session={request.session_id})")

    if not kimi_k2_agent:
        raise HTTPException(
            status_code=503,
            detail="Kimi-K2 agent not available. Set QALLOW_CHAT_BACKEND=kimi_k2"
        )

    if not request.tools:
        raise HTTPException(
            status_code=400,
            detail="Tools must be provided for tool calling"
        )

    try:
        # Simple tool map for demonstration
        tool_map = {}

        system_prompt = "You are Kimi, an AI assistant created by Moonshot AI. Use tools when appropriate."
        reply = kimi_k2_agent.chat_with_tools(
            request.message,
            tools=request.tools,
            tool_map=tool_map,
            system_prompt=system_prompt
        )

        return ToolCallResponse(
            reply=reply,
            session_id=request.session_id,
            backend="kimi_k2",
            model="Kimi-K2-Instruct",
            tool_calls_made=0
        )

    except Exception as e:
        logger.error(f"Tool calling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "backend": BACKEND,
        "ollama_available": OLLAMA_AVAILABLE,
        "ollama_active": ollama_agent is not None,
        "kimi_k2_available": KIMI_K2_AVAILABLE,
        "kimi_k2_active": kimi_k2_agent is not None,
        "model": OLLAMA_MODEL if ollama_agent else ("Kimi-K2-Instruct" if kimi_k2_agent else None)
    }


# --- Main execution ---
if __name__ == "__main__":
    import uvicorn

    logger.info("╔════════════════════════════════════════════════════════════╗")
    logger.info("║  Qallow Agent Chat Server v2.1 (with Kimi-K2)            ║")
    logger.info("╚════════════════════════════════════════════════════════════╝")
    logger.info(f"Backend:         {BACKEND}")
    logger.info(f"Ollama:          {'✓ Available' if OLLAMA_AVAILABLE else '✗ Not available'}")
    logger.info(f"Ollama Active:   {'✓ Yes' if ollama_agent else '✗ No'}")
    logger.info(f"Kimi-K2:         {'✓ Available' if KIMI_K2_AVAILABLE else '✗ Not available'}")
    logger.info(f"Kimi-K2 Active:  {'✓ Yes' if kimi_k2_agent else '✗ No'}")
    if ollama_agent:
        logger.info(f"Ollama Model:    {OLLAMA_MODEL}")
    if kimi_k2_agent:
        logger.info(f"Kimi-K2 URL:     {KIMI_K2_BASE_URL}")
    logger.info("")
    logger.info("Starting server on http://127.0.0.1:8008")
    logger.info("API docs: http://127.0.0.1:8008/docs")
    logger.info("")

    uvicorn.run(app, host="127.0.0.1", port=8008)
