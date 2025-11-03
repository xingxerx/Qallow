# run_phase11_bridge_agent.py




class QuantumBridgeAgent:
    def on_message(self, message: Any) -> Optional[Any]:
        """Handle bridge messages (legacy Agent Lightning shim)."""
        if getattr(message, "content", None) != "run_bridge":
            return None

        result = self.run_quantum_bridge()
        payload = {"content": f"Bridge result: {result}", "to": getattr(message, "sender", None)}

        sender = getattr(self, "send", None)
        if callable(sender):
            sender(payload)

        return payload

    def send(self, payload: Any) -> None:  # pragma: no cover - compatibility no-op
        """Standalone shim keeps compatibility with older orchestrators."""
        # No message bus is available when Agent Lightning is removed.
        return None

    def run_quantum_bridge(self) -> Any:
        """Invoke the Phase 11 quantum bridge entrypoint."""
        from python.quantum.run_phase11_bridge import main as bridge_main

        return bridge_main()


def main() -> None:
    QuantumBridgeAgent().run_quantum_bridge()


if __name__ == "__main__":
    main()
