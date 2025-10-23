#!/usr/bin/env python3
"""Minimal HTTP API exposing entanglement simulation results.

The server exposes `/entangle?state=ghz&w=4&validate=1` style endpoints and
returns JSON describing the generated state. It leverages the same QuTiP bridge
used by the native runtime, ensuring feature parity for the web tier.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .ghz_w_sim import build_state, validate_with_qiskit, validate_with_cirq


def serialize_state(name: str, qubits: int, validate: bool) -> dict:
    ket = build_state(name, qubits)
    vector = ket.full().ravel()
    probabilities = [float(p) for p in ket.probabilities()]

    backend = "generated"
    fidelity = 1.0
    if validate:
        validation = validate_with_qiskit(vector, qubits, name)
        if validation is None:
            validation = validate_with_cirq(vector, qubits)
        if validation is not None:
            backend, fidelity = validation

    return {
        "state": name,
        "qubits": qubits,
        "backend": backend,
        "fidelity": fidelity,
        "probabilities": probabilities,
    }


class QuantumHandler(BaseHTTPRequestHandler):
    server_version = "QallowQuantumHTTP/1.0"

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        parsed = urlparse(self.path)
        if parsed.path != "/entangle":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return

        params = parse_qs(parsed.query)
        state = params.get("state", ["ghz"])[0]
        qubits = int(params.get("qubits", ["4"])[0])
        validate = params.get("validate", ["0"])[0] in {"1", "true", "True"}

        if qubits < 2 or qubits > 5:
            self.send_error(HTTPStatus.BAD_REQUEST, "qubits must be between 2 and 5")
            return

        try:
            payload = serialize_state(state, qubits, validate)
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):  # noqa: D401 - mimic BaseHTTPRequestHandler
        # Route logs through stdout with a simple prefix.
        print(f"[web-api] {self.address_string()} - {fmt % args}")


def serve(host: str = "127.0.0.1", port: int = 8713):
    server = ThreadingHTTPServer((host, port), QuantumHandler)
    print(f"[web-api] Serving entanglement API on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web-api] Shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()
