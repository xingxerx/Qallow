# Qallow Dependency Manifest
#
# This file documents all external dependencies, their versions, and sources.
# Used by bootstrap system to ensure reproducible builds.

[system-requirements]
cmake_min_version = "3.20"
gcc_min_version = "9.0"
python_min_version = "3.8"
cuda_min_version = "11.0"  # Optional, for GPU acceleration

[c-libraries]
cjson = { source = "system|git|fetch", version = "1.7.15", optional = false }
pthread = { source = "system", version = "any", optional = false }
m = { source = "system", version = "any", optional = false }  # Math library
dl = { source = "system", version = "any", optional = true }   # Dynamic linking

[cuda-libraries]
cudart = { source = "cuda-toolkit", version = "11.0+", optional = true }
cufft = { source = "cuda-toolkit", version = "11.0+", optional = true }
cublas = { source = "cuda-toolkit", version = "11.0+", optional = true }

[git-submodules]
"mcp-memory-service" = { path = "mcp-memory-service", branch = "main", optional = true }

[python-packages]
numpy = "~= 1.21.0"
scipy = "~= 1.7.0"
matplotlib = "~= 3.4.0"
pyyaml = "~= 5.4.0"
requests = "~= 2.26.0"

[python-dev-packages]
pytest = "~= 6.2.0"
pytest-cov = "~= 2.12.0"
black = "~= 21.6b0"
pylint = "~= 2.9.0"
mypy = "~= 0.910"

[python-gpu-packages]
torch = "~= 1.9.0"  # Optional, for ML acceleration
tensorflow = "~= 2.6.0"  # Optional, alternative to PyTorch
qiskit = "~= 0.27.0"  # For Phase 11 quantum bridge

[python-web-packages]
flask = "~= 2.0.0"
flask-cors = "~= 3.0.0"
werkzeug = "~= 2.0.0"

[optional-features]
cuda_support = { default = true, cmake_flag = "-DQALLOW_ENABLE_CUDA" }
quantum_bridge = { default = false, requires = ["qiskit"], note = "Phase 11 quantum simulation" }
web_dashboard = { default = false, requires = ["flask", "flask-cors"], note = "Telemetry dashboard" }
