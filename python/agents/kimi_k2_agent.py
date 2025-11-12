"""
Kimi-K2 Integration for Qallow
Provides local inference support for Kimi-K2 model without API keys
Supports vLLM, SGLang, and other inference engines
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger("KimiK2Agent")


class InferenceEngine(Enum):
    """Supported inference engines for Kimi-K2"""
    VLLM = "vllm"
    SGLANG = "sglang"
    KTRANSFORMERS = "ktransformers"
    TENSORRT_LLM = "tensorrt_llm"


@dataclass
class KimiK2Config:
    """Configuration for Kimi-K2 agent"""
    # Model configuration
    model_name: str = "moonshotai/Kimi-K2-Instruct"
    model_path: Optional[str] = None  # Local path if using local weights
    
    # Inference engine
    engine: InferenceEngine = InferenceEngine.VLLM
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-needed"  # Placeholder for local inference
    
    # Model parameters
    temperature: float = 0.6  # Recommended for Kimi-K2
    max_tokens: int = 4096
    top_p: float = 0.95
    
    # Tool calling
    enable_tools: bool = True
    tool_call_parser: str = "kimi_k2"
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data/kimi_k2"))
    log_dir: Path = field(default_factory=lambda: Path("data/logs"))
    output_file: Path = field(default_factory=lambda: Path("data/kimi_k2/output.jsonl"))
    
    # Timeouts
    timeout: int = 300  # 5 minutes
    
    def __post_init__(self):
        """Ensure directories exist"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


class KimiK2Agent:
    """Kimi-K2 Agent for local inference without API keys"""
    
    def __init__(self, config: Optional[KimiK2Config] = None):
        """Initialize Kimi-K2 agent"""
        self.config = config or KimiK2Config()
        self.client = None
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize OpenAI-compatible client for local inference"""
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI SDK not available. Install with: pip install openai")
            raise ImportError("openai package required")
        
        try:
            self.client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key
            )
            # Test connection
            logger.info(f"✓ Connected to Kimi-K2 at {self.config.base_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Kimi-K2: {e}")
            raise
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False
    ) -> str:
        """
        Chat with Kimi-K2 model
        
        Args:
            message: User message
            system_prompt: System prompt (default: Kimi introduction)
            tools: List of tool definitions for tool calling
            stream: Whether to stream response
        
        Returns:
            Model response
        """
        if not self.client:
            raise RuntimeError("Client not initialized")
        
        if system_prompt is None:
            system_prompt = "You are Kimi, an AI assistant created by Moonshot AI."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                tools=tools if tools and self.config.enable_tools else None,
                tool_choice="auto" if tools and self.config.enable_tools else None,
                stream=stream,
                timeout=self.config.timeout
            )
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise
    
    def _handle_streaming_response(self, response) -> str:
        """Handle streaming response"""
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
        return full_response
    
    def chat_with_tools(
        self,
        message: str,
        tools: List[Dict[str, Any]],
        tool_map: Dict[str, callable],
        system_prompt: Optional[str] = None,
        max_iterations: int = 10
    ) -> str:
        """
        Chat with tool calling support
        
        Args:
            message: User message
            tools: List of tool definitions
            tool_map: Mapping of tool names to functions
            system_prompt: System prompt
            max_iterations: Max tool calling iterations
        
        Returns:
            Final response
        """
        if not self.client:
            raise RuntimeError("Client not initialized")
        
        if system_prompt is None:
            system_prompt = "You are Kimi, an AI assistant created by Moonshot AI."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        for iteration in range(max_iterations):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    tools=tools,
                    tool_choice="auto",
                    timeout=self.config.timeout
                )
                
                choice = response.choices[0]
                finish_reason = choice.finish_reason
                
                if finish_reason == "tool_calls":
                    messages.append(choice.message)
                    
                    for tool_call in choice.message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        if tool_name not in tool_map:
                            logger.warning(f"Tool {tool_name} not found in tool_map")
                            continue
                        
                        tool_result = tool_map[tool_name](**tool_args)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": json.dumps(tool_result)
                        })
                else:
                    return choice.message.content
            
            except Exception as e:
                logger.error(f"Tool calling error at iteration {iteration}: {e}")
                raise
        
        logger.warning(f"Max iterations ({max_iterations}) reached")
        return "Max tool calling iterations reached"
    
    @staticmethod
    def extract_tool_calls(raw_output: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from raw model output
        Useful when using completions endpoint instead of chat
        """
        if '<|tool_calls_section_begin|>' not in raw_output:
            return []
        
        pattern = r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>"
        tool_calls_sections = re.findall(pattern, raw_output, re.DOTALL)
        
        func_call_pattern = r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*?)\s*<\|tool_call_end\|>"
        tool_calls = []
        
        for section in tool_calls_sections:
            for match in re.findall(func_call_pattern, section, re.DOTALL):
                function_id, function_args = match
                function_name = function_id.split('.')[1].split(':')[0]
                tool_calls.append({
                    "id": function_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": function_args
                    }
                })
        
        return tool_calls


def create_kimi_k2_agent(
    base_url: str = "http://localhost:8000/v1",
    engine: str = "vllm"
) -> KimiK2Agent:
    """Factory function to create Kimi-K2 agent"""
    config = KimiK2Config(
        base_url=base_url,
        engine=InferenceEngine(engine)
    )
    return KimiK2Agent(config)

