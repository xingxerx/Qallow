# Qiskit From C Example

This example shows how to embed the CPython interpreter inside a C program so you can invoke Qiskit workflows from native code.

## Prerequisites
- Python 3.10+ with development headers (`python3-dev` on Debian/Ubuntu, `python@3.x` + `brew install python` headers on macOS).
- Qiskit installed in the same Python environment you compile against:
  ```bash
  python3 -m pip install "qiskit[visualization]" qiskit-aer
  ```

## Build
```bash
cd examples/qiskit_c_embed
gcc main.c $(python3-config --cflags --ldflags --embed) -o qiskit_from_c
```

> **Note:** On Python < 3.8 remove the `--embed` flag. Ensure `python3-config` points to the interpreter with Qiskit installed.

## Run
```bash
./qiskit_from_c
```

The program spins up Python, constructs a Bell-state circuit, runs it through Aer’s statevector simulator, and prints the expectation value of `ZZ` (≈ 1 for the Bell state). Any Qiskit code can be swapped into the string literal in `main.c`.
