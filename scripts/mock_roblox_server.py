"""
A mock Roblox Studio HTTP server for testing the Qallow AGI system.

This server simulates the API endpoints that the RobloxStudioTool tries to
connect to, allowing the RobloxAgent to run without a real Roblox Studio instance.
"""
import uuid
from fastapi import FastAPI, HTTPException, Request
import uvicorn

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint to confirm the server is running."""
    return {"status": "ok"}

@app.post("/create-game")
async def create_game(request: Request):
    """Simulates creating a new base game."""
    data = await request.json()
    print(f"Mock Server: Received request to create game with data: {data}")
    game_id = f"rbx-game-{uuid.uuid4()}"
    print(f"Mock Server: Generated gameId: {game_id}")
    return {"status": "success", "gameId": game_id}

@app.post("/games/{game_id}/levels")
async def create_obby_levels(game_id: str, data: dict):
    """Simulates creating obby levels."""
    print(f"Mock Server: Received request to create levels for game {game_id} with data: {data}")
    return {"status": "success", "message": f"Created {data.get('platform_count')} platforms for level {data.get('level_number')}."}

@app.post("/games/{game_id}/npcs")
async def create_npcs(game_id: str, data: dict):
    """Simulates creating NPCs."""
    print(f"Mock Server: Received request to create NPCs for game {game_id} with data: {data}")
    return {"status": "success", "message": f"Created {data.get('count')} NPCs with behavior '{data.get('behavior')}'."}

@app.post("/games/{game_id}/publish")
async def publish_game(game_id: str):
    """Simulates publishing the game."""
    print(f"Mock Server: Received request to publish game {game_id}")
    published_url = f"https://www.roblox.com/games/{uuid.uuid4()}/qallow-agi-game"
    return {"status": "success", "published_url": published_url}

if __name__ == "__main__":
    print("Starting Mock Roblox Studio Server on http://localhost:8745...")
    uvicorn.run(app, host="127.0.0.1", port=8745)
