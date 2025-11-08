"""
Qallow Agent Chat Server
Connects the DeepSeek AI baseline to a simple web API for the native GUI.
"""
import sys
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

# Add project root to path to allow importing from python/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.deepseek_baseline import DeepSeekClient, DeepSeekConfig

# --- FastAPI App ---
app = FastAPI(
    title="Qallow Agent Chat Server",
    description="Provides an API endpoint to chat with the DeepSeek-1 agent.",
    version="1.0.0",
)

# --- Agent Initialization ---
# Use a mock backend for now for simplicity and speed.
# This can be changed to 'ollama' or 'api' if needed.
config = DeepSeekConfig(backend='mock')
agent = DeepSeekClient(config)

# --- API Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    reply: str
    session_id: str

# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Receives a message from the user, gets a reply from the agent,
    and returns the response.
    """
    print(f"Received message: '{request.message}' for session: {request.session_id}")

    # For this simple integration, we'll use the `reason_cognitive_state`
    # method as a general-purpose chat entry point.
    # We'll create a mock cognitive state.
    mock_state = {
        "iteration": 1,
        "current_loss": 0.5,
        "best_loss": 0.4,
        "ethics_score": 0.9,
        "backend_name": "chat_server",
        "user_input": request.message,
    }

    # The DeepSeekClient's reason_cognitive_state expects specific numeric inputs.
    # We'll call it with the mock data.
    reasoning_result = agent.reason_cognitive_state(
        iteration=mock_state["iteration"],
        current_loss=mock_state["current_loss"],
        best_loss=mock_state["best_loss"],
        ethics_score=mock_state["ethics_score"],
        backend_name=mock_state["backend_name"],
    )

    # For a chat-like experience, we'll format the structured output into a sentence.
    reply_text = reasoning_result.get("analysis", "I am functioning.")

    print(f"Agent reply: '{reply_text}'")

    return ChatResponse(reply=reply_text, session_id=request.session_id)

@app.get("/")
def read_root():
    return {"message": "Qallow Agent Chat Server is running."}

# --- Main execution ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Qallow Agent Chat Server...")
    uvicorn.run(app, host="127.0.0.1", port=8008)
