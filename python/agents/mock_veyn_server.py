"""
Mock VEYN telemetry server.
Serves ws://localhost:7700/stream and emits fake physiological events
so the Rust VEYN bridge (core/qallow-veyn-bridge) can be exercised end-to-end
without real biometric hardware.
"""

import asyncio
import json
import random
import time

import websockets

METRICS = ["hrv", "eeg_beta", "spo2", "presence", "sleep_stage"]


async def stream(websocket):
    print("Client connected")
    try:
        while True:
            metric = random.choice(METRICS)
            value = random.uniform(0.0, 1.0) if metric != "sleep_stage" else float(random.randint(0, 4))
            event = {"metric": metric, "value": value, "timestamp": int(time.time())}
            await websocket.send(json.dumps(event))
            print("Sent", event)
            await asyncio.sleep(1)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")


async def main():
    async with websockets.serve(stream, "localhost", 7700):
        print("Mock VEYN server listening on ws://localhost:7700/stream")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
