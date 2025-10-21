#include <Python.h>
#include <stdio.h>

/**
 * Simple embedding example that executes a Qiskit Bell-state circuit from C.
 *
 * Build (Linux/macOS):
 *   gcc main.c $(python3-config --cflags --ldflags --embed) -o qiskit_from_c
 *
 * Run:
 *   ./qiskit_from_c
 *
 * Ensure the Python environment used for compilation has Qiskit installed.
 */
int main(int argc, char *argv[]) {
  int status = 0;

  PyConfig config;
  PyConfig_InitPythonConfig(&config);
  config.parse_argv = 0;  // we are not handing args to Python

  if (PyConfig_SetBytesArgv(&config, argc, argv) < 0) {
    fprintf(stderr, "Failed to initialize Python argv\n");
    PyConfig_Clear(&config);
    return 1;
  }

  status = Py_InitializeFromConfig(&config);
  PyConfig_Clear(&config);
  if (status < 0) {
    PyErr_Print();
    fprintf(stderr, "Failed to initialize the Python interpreter\n");
    return 1;
  }

  const char *code =
      "from qiskit import QuantumCircuit\n"
      "from qiskit.quantum_info import SparsePauliOp\n"
      "qc = QuantumCircuit(2)\n"
      "qc.h(0)\n"
      "qc.cx(0, 1)\n"
      "observable = SparsePauliOp('ZZ')\n"
      "print('Bell circuit:', qc)\n"
      "from qiskit_aer import AerSimulator\n"
      "sim = AerSimulator(method='statevector')\n"
      "from qiskit.primitives import Estimator\n"
      "estimator = Estimator(backend=sim)\n"
      "job = estimator.run([(qc, [observable])])\n"
      "result = job.result()[0]\n"
      "print('Expectation ZZ =', result.data.evs[0])\n";

  status = PyRun_SimpleString(code);
  if (status != 0) {
    PyErr_Print();
    fprintf(stderr, "Embedded Qiskit execution failed\n");
    Py_Finalize();
    return 1;
  }

  if (Py_FinalizeEx() < 0) {
    fprintf(stderr, "Python finalization reported an error\n");
    return 1;
  }

  return 0;
}
