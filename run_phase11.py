# run_phase11_bridge_agent.py

from __future__ import annotations

from typing import Any, Optional

try:
    from agentlightning.litagent import LitAgent
except ModuleNotFoundError:  # pragma: no cover - optional dependency

    class LitAgent:  # type: ignore[too-few-public-methods]
        """Minimal stub so the bridge can run without Agent Lightning installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def send(self, payload: Any) -> None:
            # When running standalone there is no messaging layer, so ignore sends.
            pass


class QuantumBridgeAgent(LitAgent):
    def on_message(self, message: Any) -> Optional[Any]:
        """Handle Agent Lightning messages for the bridge flow."""
        if getattr(message, "content", None) != "run_bridge":
            return None

        result = self.run_quantum_bridge()
        payload = {"content": f"Bridge result: {result}", "to": getattr(message, "sender", None)}

        sender = getattr(self, "send", None)
        if callable(sender):
            sender(payload)

        return payload

    def run_quantum_bridge(self) -> Any:
        """Invoke the Phase 11 quantum bridge entrypoint."""
        from python.quantum.run_phase11_bridge import main as bridge_main

        return bridge_main()


def main() -> None:
    QuantumBridgeAgent().run_quantum_bridge()


if __name__ == "__main__":
    main()
