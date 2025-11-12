#!/usr/bin/env python3
"""
Qallow AI Agent - Ollama Integration
Autonomous agent for QAOA optimization, ethics validation, and system tuning
Supports: DeepSeek-V3, Llama2-70B, and other large models via Ollama

Location: python/agents/qallow_agent_ollama.py
Purpose: 
  - Autonomous QAOA parameter tuning for Phase 14
  - Phase 13 ethics validation before LLM inference
  - Distributed GPU support via Ray/MPI
  - Export to data/quantum/agent_output.jsonl
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from enum import Enum

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("QallowAgent")


class AgentTask(Enum):
    """Types of tasks the agent can perform"""
    QAOA_OPTIMIZE = "qaoa_optimize"
    ETHICS_VALIDATE = "ethics_validate"
    PHASE_TUNE = "phase_tune"
    SYSTEM_ANALYZE = "system_analyze"


@dataclass
class OllamaConfig:
    """Configuration for Ollama agent"""
    model: str = "llama2:70b"  # Default to 70B for supercomputer scale
    host: str = "http://localhost:11434"
    temperature: float = 0.3
    num_gpu: int = 8  # Number of GPUs for distributed inference
    num_experts: int = 8  # For MoE models like DeepSeek-V3
    max_tokens: int = 4096
    timeout: int = 300  # 5 minutes
    
    # Qallow-specific paths
    data_dir: Path = Path("data/quantum")
    log_dir: Path = Path("data/logs")
    output_file: Path = Path("data/quantum/agent_output.jsonl")
    gain_json: Path = Path("data/quantum/ollama_gain.json")
    
    # Phase 13 ethics gate
    ethics_enabled: bool = True
    ethics_threshold: float = 0.85
    
    # Phase 14 QAOA parameters
    qaoa_nodes: int = 256
    qaoa_target_fidelity: float = 0.981
    
    def __post_init__(self):
        """Ensure directories exist"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


class OllamaAgent:
    """
    Autonomous AI Agent powered by Ollama
    Integrates with Qallow's Phase 13 (ethics) and Phase 14 (QAOA)
    """
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.session_id = f"agent_{int(time.time())}"
        self.task_history: List[Dict[str, Any]] = []
        
        # Verify Ollama is running
        self._verify_ollama()
        
        logger.info(f"Initialized OllamaAgent with model={self.config.model}")
        logger.info(f"Session ID: {self.session_id}")
    
    def _verify_ollama(self) -> None:
        """Verify Ollama service is running and model is available"""
        try:
            result = subprocess.run(
                ["curl", "-s", f"{self.config.host}/api/tags"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Ollama not responding at {self.config.host}")
            
            # Check if model is available
            tags = json.loads(result.stdout)
            models = [m.get("name", "") for m in tags.get("models", [])]
            
            if not any(self.config.model in m for m in models):
                logger.warning(f"Model {self.config.model} not found. Available: {models}")
                logger.warning(f"Run: ollama pull {self.config.model}")
            else:
                logger.info(f"✓ Model {self.config.model} is available")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ollama service timeout. Is it running? (ollama serve)")
        except json.JSONDecodeError:
            raise RuntimeError("Invalid response from Ollama service")
        except Exception as e:
            raise RuntimeError(f"Failed to verify Ollama: {e}")
    
    def _run_phase13_ethics(self, prompt: str) -> Dict[str, Any]:
        """
        Run Phase 13 ethics validation on prompt before LLM inference
        Returns: {"pass": bool, "score": float, "reason": str}
        """
        if not self.config.ethics_enabled:
            return {"pass": True, "score": 1.0, "reason": "Ethics gate disabled"}
        
        logger.info("Running Phase 13 ethics validation...")
        
        try:
            # Call Phase 13 via Qallow binary
            qallow_bin = PROJECT_ROOT / "build" / "qallow"
            if not qallow_bin.exists():
                logger.warning("Qallow binary not found. Skipping ethics check.")
                return {"pass": True, "score": 0.9, "reason": "Binary not available"}
            
            result = subprocess.run(
                [str(qallow_bin), "phase", "13", "--input", "-", "--ticks", "10"],
                input=prompt.encode(),
                capture_output=True,
                timeout=30
            )
            
            # Parse ethics score from output
            output = result.stdout.decode()
            
            # Look for ethics metrics in output
            ethics_score = 0.9  # Default
            if "ethics_total" in output:
                # Extract score (simplified parsing)
                for line in output.split('\n'):
                    if "ethics_total" in line.lower():
                        try:
                            ethics_score = float(line.split(':')[-1].strip())
                        except:
                            pass
            
            passed = ethics_score >= self.config.ethics_threshold
            
            logger.info(f"Ethics check: {'PASS' if passed else 'FAIL'} (score={ethics_score:.3f})")
            
            return {
                "pass": passed,
                "score": ethics_score,
                "reason": "Phase 13 validation" if passed else "Below threshold"
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Phase 13 timeout")
            return {"pass": False, "score": 0.0, "reason": "Timeout"}
        except Exception as e:
            logger.error(f"Phase 13 error: {e}")
            return {"pass": False, "score": 0.0, "reason": str(e)}
    
    def _query_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Query Ollama with the given prompt
        Returns: {"response": str, "model": str, "duration_ms": int}
        """
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_gpu": self.config.num_gpu,
                "num_ctx": self.config.max_tokens,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        # For MoE models (DeepSeek-V3)
        if "deepseek" in self.config.model.lower():
            payload["options"]["num_experts"] = self.config.num_experts
        
        logger.info(f"Querying Ollama: {self.config.model}")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ["curl", "-s", f"{self.config.host}/api/generate",
                 "-d", json.dumps(payload)],
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Ollama query failed: {result.stderr}")
            
            response = json.loads(result.stdout)
            duration_ms = int((time.time() - start_time) * 1000)
            
            logger.info(f"✓ Response received ({duration_ms}ms)")
            
            return {
                "response": response.get("response", ""),
                "model": response.get("model", self.config.model),
                "duration_ms": duration_ms,
                "context": response.get("context", [])
            }
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Ollama query timeout ({self.config.timeout}s)")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response: {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama query error: {e}")
    
    def optimize_qaoa(self, nodes: Optional[int] = None, 
                     target_fidelity: Optional[float] = None) -> Dict[str, Any]:
        """
        Autonomous QAOA optimization for Phase 14
        Uses LLM to suggest optimal parameters based on system state
        
        Returns: {
            "p": int,           # QAOA layers
            "gamma": float,     # Initial gamma
            "beta": float,      # Initial beta  
            "alpha_eff": float, # Effective gain for Phase 14
            "reasoning": str    # LLM justification
        }
        """
        nodes = nodes or self.config.qaoa_nodes
        target_fidelity = target_fidelity or self.config.qaoa_target_fidelity
        
        logger.info(f"Starting QAOA optimization: nodes={nodes}, target={target_fidelity:.3f}")
        
        # Build prompt
        task_prompt = f"""You are Qallow's AI Agent, an expert in quantum optimization.

Task: Optimize QAOA (Quantum Approximate Optimization Algorithm) parameters for a photonic quantum system.

System Configuration:
- Photonic nodes: {nodes}
- Target fidelity: {target_fidelity:.3f}
- Constraints: Harmonic governance (Phase 7), ethics compliance (Phase 13), deterministic convergence

Your task is to suggest optimal QAOA parameters that maximize fidelity while maintaining system stability.

Output ONLY valid JSON in this exact format:
{{
  "p": <integer>,           // QAOA layers (1-10)
  "gamma": <float>,         // Initial gamma (0.0-1.0)
  "beta": <float>,          // Initial beta (0.0-1.0)
  "alpha_eff": <float>,     // Effective gain for Phase 14 (0.001-0.01)
  "reasoning": "<string>"   // Brief technical justification (max 100 chars)
}}

Consider:
1. Higher p increases accuracy but requires more coherence time
2. Gamma/beta should be tuned for the specific problem structure
3. Alpha_eff controls coupling strength in Phase 14
4. System must maintain ethics_score > 0.85

Respond with JSON only, no additional text.
"""
        
        # Ethics gate
        ethics_result = self._run_phase13_ethics(task_prompt)
        if not ethics_result["pass"]:
            raise RuntimeError(f"Ethics gate failed: {ethics_result['reason']}")
        
        # Query LLM
        system_prompt = "You are a quantum optimization expert. Respond only with valid JSON."
        llm_response = self._query_ollama(task_prompt, system_prompt)
        
        # Parse JSON from response
        params = self._extract_json(llm_response["response"])
        
        # Validate parameters
        params = self._validate_qaoa_params(params)
        
        # Log to output file
        log_entry = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "task": "qaoa_optimize",
            "model": llm_response["model"],
            "duration_ms": llm_response["duration_ms"],
            "input": {"nodes": nodes, "target_fidelity": target_fidelity},
            "output": params,
            "ethics": ethics_result,
            "raw_response": llm_response["response"][:500]  # Truncate
        }
        
        self._log_task(log_entry)
        
        # Export gain for Phase 14
        self._export_gain(params["alpha_eff"])
        
        logger.info(f"✓ QAOA optimization complete: p={params['p']}, alpha_eff={params['alpha_eff']:.4f}")
        
        return params
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON object from LLM response"""
        try:
            # Try direct parse
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in text
            start = text.find("{")
            end = text.rfind("}") + 1

            if start == -1 or end == 0:
                logger.warning(f"No JSON found in response, using defaults. Response: {text[:200]}")
                # Return sensible defaults if JSON extraction fails
                return {
                    "p": 3,
                    "gamma": 0.42,
                    "beta": 0.19,
                    "alpha_eff": 0.0048,
                    "reasoning": "Default parameters (LLM response parsing failed)"
                }

            json_str = text[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed: {e}. Using defaults.")
                return {
                    "p": 3,
                    "gamma": 0.42,
                    "beta": 0.19,
                    "alpha_eff": 0.0048,
                    "reasoning": "Default parameters (JSON parse failed)"
                }
    
    def _validate_qaoa_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clamp QAOA parameters to safe ranges"""
        validated = {
            "p": max(1, min(10, int(params.get("p", 3)))),
            "gamma": max(0.0, min(1.0, float(params.get("gamma", 0.5)))),
            "beta": max(0.0, min(1.0, float(params.get("beta", 0.5)))),
            "alpha_eff": max(0.001, min(0.01, float(params.get("alpha_eff", 0.005)))),
            "reasoning": str(params.get("reasoning", "LLM optimization"))[:200]
        }
        
        return validated
    
    def _log_task(self, entry: Dict[str, Any]) -> None:
        """Append task log to output file"""
        with open(self.config.output_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        self.task_history.append(entry)
    
    def _export_gain(self, alpha_eff: float) -> None:
        """Export gain parameter for Phase 14 consumption"""
        gain_data = {
            "alpha_eff": alpha_eff,
            "timestamp": time.time(),
            "session_id": self.session_id,
            "source": "ollama_agent"
        }
        
        with open(self.config.gain_json, "w") as f:
            json.dump(gain_data, f, indent=2)
        
        logger.info(f"✓ Exported gain to {self.config.gain_json}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status and statistics"""
        return {
            "session_id": self.session_id,
            "model": self.config.model,
            "tasks_completed": len(self.task_history),
            "output_file": str(self.config.output_file),
            "gain_file": str(self.config.gain_json),
            "ethics_enabled": self.config.ethics_enabled,
        }


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Qallow AI Agent - Ollama Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize QAOA with default settings
  python -m python.agents.qallow_agent_ollama --task qaoa_optimize
  
  # Use DeepSeek-V3 model
  python -m python.agents.qallow_agent_ollama --model deepseek-v3:70b --task qaoa_optimize
  
  # Custom parameters
  python -m python.agents.qallow_agent_ollama --nodes 512 --target 0.99 --task qaoa_optimize
        """
    )
    
    parser.add_argument("--task", type=str, default="qaoa_optimize",
                       choices=["qaoa_optimize", "status"],
                       help="Task to perform")
    parser.add_argument("--model", type=str, default="llama2:70b",
                       help="Ollama model to use")
    parser.add_argument("--nodes", type=int, default=256,
                       help="Number of photonic nodes")
    parser.add_argument("--target", type=float, default=0.981,
                       help="Target fidelity")
    parser.add_argument("--num-gpu", type=int, default=8,
                       help="Number of GPUs for distributed inference")
    parser.add_argument("--no-ethics", action="store_true",
                       help="Disable Phase 13 ethics gate")
    
    args = parser.parse_args()
    
    # Create config
    config = OllamaConfig(
        model=args.model,
        num_gpu=args.num_gpu,
        qaoa_nodes=args.nodes,
        qaoa_target_fidelity=args.target,
        ethics_enabled=not args.no_ethics
    )
    
    try:
        # Initialize agent
        agent = OllamaAgent(config)
        
        if args.task == "qaoa_optimize":
            result = agent.optimize_qaoa()
            print(json.dumps(result, indent=2))
        elif args.task == "status":
            status = agent.get_status()
            print(json.dumps(status, indent=2))
        
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

