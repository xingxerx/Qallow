"""
LLM Adapter for Qallow
Provides a unified interface for different LLM backends (Kimi-K2, Qwen, Llama, etc.)
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

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
logger = logging.getLogger("LLMAdapter")


@dataclass
class LLMConfig:
    """Configuration for LLM adapter"""
    model_name: str = "llm"  # Model name as served by vLLM
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-needed"
    temperature: float = 0.6
    max_tokens: int = 2048  # Reduced to leave room for input tokens
    top_p: float = 0.9


class LLMAdapter:
    """
    Unified adapter for different LLM backends
    Works with vLLM, SGLang, and other OpenAI-compatible servers
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM adapter"""
        self.config = config or LLMConfig()
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key
        )
        
        # Verify connection
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify connection to LLM server"""
        try:
            models = self.client.models.list()
            logger.info(f"✓ Connected to LLM server at {self.config.base_url}")
            logger.info(f"✓ Available models: {[m.id for m in models.data]}")
        except Exception as e:
            logger.error(f"✗ Failed to connect to LLM server: {e}")
            raise
    
    def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a chat message and get a response
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            
        Returns:
            Response text
        """
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": message})
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise
    
    def chat_stream(self, message: str, system_prompt: Optional[str] = None):
        """
        Send a chat message and stream the response
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            
        Yields:
            Response chunks
        """
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": message})
            
            stream = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            logger.error(f"Stream error: {e}")
            raise
    
    def chat_with_tools(self, message: str, tools: List[Dict[str, Any]], 
                       system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a chat message with tool calling
        
        Args:
            message: User message
            tools: List of tool definitions
            system_prompt: Optional system prompt
            
        Returns:
            Response with tool calls
        """
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": message})
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                tools=tools,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )
            
            return {
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls,
                "finish_reason": response.choices[0].finish_reason
            }
        
        except Exception as e:
            logger.error(f"Tool call error: {e}")
            raise
    
    def set_temperature(self, temperature: float):
        """Set temperature for responses"""
        self.config.temperature = temperature
        logger.info(f"Temperature set to {temperature}")
    
    def set_max_tokens(self, max_tokens: int):
        """Set maximum tokens for responses"""
        self.config.max_tokens = max_tokens
        logger.info(f"Max tokens set to {max_tokens}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            "model_name": self.config.model_name,
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p
        }


def create_llm_adapter(model_name: str = "llm", 
                       base_url: str = "http://localhost:8000/v1") -> LLMAdapter:
    """
    Create an LLM adapter with custom configuration
    
    Args:
        model_name: Model name as served by vLLM
        base_url: Base URL of the LLM server
        
    Returns:
        LLMAdapter instance
    """
    config = LLMConfig(model_name=model_name, base_url=base_url)
    return LLMAdapter(config)


# Convenience functions
def chat(message: str, model_name: str = "llm") -> str:
    """Quick chat function"""
    adapter = create_llm_adapter(model_name=model_name)
    return adapter.chat(message)


def chat_stream(message: str, model_name: str = "llm"):
    """Quick streaming chat function"""
    adapter = create_llm_adapter(model_name=model_name)
    yield from adapter.chat_stream(message)

