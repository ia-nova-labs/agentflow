"""
AgentFlow - A minimalist Python framework for building AI agents

This module provides the core Agent class for creating and running AI agents
with support for multiple LLM providers (Ollama, OpenAI, Mistral).

Example:
    >>> import asyncio
    >>> from agentflow import Agent
    >>> 
    >>> async def main():
    >>>     agent = Agent(model="llama3", debug=True)
    >>>     response = await agent.arun("Hello, who are you?")
    >>>     print(response)
    >>> 
    >>> asyncio.run(main())

Version: 0.6.0
Author: Hamadi Chaabani
License: MIT
"""

from typing import List, Dict, Optional, Any, Callable, Union
import httpx
import json
import inspect
import os
import asyncio
import logging
import re
from abc import ABC, abstractmethod
from functools import wraps


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class AgentFlowError(Exception):
    """Base exception for AgentFlow errors."""
    pass


class LLMConnectionError(AgentFlowError):
    """Raised when connection to LLM fails."""
    pass


class LLMResponseError(AgentFlowError):
    """Raised when LLM returns an invalid response."""
    pass


class ToolExecutionError(AgentFlowError):
    """Raised when a tool execution fails."""
    pass


class LoopDetectedError(AgentFlowError):
    """Raised when an infinite loop is detected."""
    pass


# --- Memory Architecture ---

class Memory(ABC):
    """Abstract base class for agent memory."""
    
    @abstractmethod
    def add(self, role: str, content: str) -> None:
        """Add a message to memory."""
        pass
    
    @abstractmethod
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in memory."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all messages from memory."""
        pass
        
    @abstractmethod
    def count(self) -> int:
        """Return the number of messages in memory."""
        pass


class InMemory(Memory):
    """Simple in-memory storage for messages."""
    
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        
    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        
    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages.copy()
        
    def clear(self) -> None:
        self.messages = []
        
    def count(self) -> int:
        return len(self.messages)


class FileMemory(Memory):
    """Persistent memory storage using a JSON file."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.messages: List[Dict[str, str]] = []
        self._load()
        
    def _load(self) -> None:
        """Load messages from file if it exists."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self.messages = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.messages = []
    
    def _save(self) -> None:
        """Save messages to file."""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.messages, f, indent=2)
            
    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._save()
        
    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages.copy()
        
    def clear(self) -> None:
        self.messages = []
        self._save()
        
    def count(self) -> int:
        return len(self.messages)


# --- Model Architecture (Async-First) ---

class Model(ABC):
    """Abstract base class for LLM providers (async-first)."""
    
    @abstractmethod
    async def agenerate(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """
        Generate a response from the model (async).
        
        This is the primary method that all models must implement.
        """
        pass
    
    def generate(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """
        Synchronous wrapper for agenerate().
        
        Provided for backward compatibility.
        """
        return asyncio.run(self.agenerate(messages, system_prompt))


class Ollama(Model):
    """Ollama LLM provider (async-first)."""
    
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        
    async def agenerate(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/api/chat"
        
        # Prepare messages
        msgs = messages.copy()
        if system_prompt:
            if msgs and msgs[0].get("role") == "system":
                pass  # Already has system prompt
            else:
                msgs.insert(0, {"role": "system", "content": system_prompt})
        
        payload = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": 0.0}
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
        except Exception as e:
            raise LLMConnectionError(f"Ollama connection failed: {str(e)}") from e
            
        try:
            data = response.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            raise LLMResponseError(f"Failed to parse Ollama response: {str(e)}") from e


class OpenAI(Model):
    """OpenAI LLM provider (async-first)."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY env var or pass api_key.")
            
    async def agenerate(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        
        msgs = messages.copy()
        if system_prompt:
            if msgs and msgs[0].get("role") == "system":
                pass
            else:
                msgs.insert(0, {"role": "system", "content": system_prompt})
                
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": msgs,
            "temperature": 0.0
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
        except Exception as e:
            raise LLMConnectionError(f"OpenAI connection failed: {str(e)}") from e
            
        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMResponseError(f"Failed to parse OpenAI response: {str(e)}") from e


class Mistral(Model):
    """Mistral AI LLM provider (async-first)."""
    
    def __init__(self, model: str = "mistral-large-latest", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key not found. Set MISTRAL_API_KEY env var or pass api_key.")
            
    async def agenerate(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        url = "https://api.mistral.ai/v1/chat/completions"
        
        msgs = messages.copy()
        if system_prompt:
            if msgs and msgs[0].get("role") == "system":
                pass
            else:
                msgs.insert(0, {"role": "system", "content": system_prompt})
                
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": msgs,
            "temperature": 0.0
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
        except Exception as e:
            raise LLMConnectionError(f"Mistral connection failed: {str(e)}") from e
            
        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMResponseError(f"Failed to parse Mistral response: {str(e)}") from e


# --- Core Agent (Async-First with Robust Loop) ---

class Agent:
    """
    A minimalist AI agent that interfaces with various LLMs (async-first with robust loop).
    """
    
    def __init__(
        self,
        model: Union[str, Model] = "llama3",
        base_url: str = "http://localhost:11434",
        memory: Optional[Memory] = None,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        tool_timeout: int = 30
    ) -> None:
        """
        Initialize an Agent instance.
        
        Args:
            model: Model name (str) or Model instance. If str, defaults to Ollama.
            base_url: Base URL for Ollama (only used if model is str).
            memory: Optional Memory instance. Defaults to InMemory().
            debug: Enable debug logging.
            logger: Custom logger instance.
            tool_timeout: Timeout for tool execution in seconds.
        """
        # Backward compatibility: if model is string, assume Ollama
        if isinstance(model, str):
            self.model = Ollama(model=model, base_url=base_url)
        else:
            self.model = model
            
        self.memory = memory if memory else InMemory()
        self._tools: Dict[str, Dict[str, Any]] = {}
        
        # Logging Setup
        self.debug = debug
        self.logger = logger or logging.getLogger("agentflow")
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        
        # Tool timeout
        self.tool_timeout = tool_timeout
        
    def tool(self, func: Callable) -> Callable:
        """Decorator to register a function as a tool."""
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or "No description provided"
        
        parameters = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation in (list, List):
                    param_type = "array"
                elif param.annotation in (dict, Dict):
                    param_type = "object"
            
            parameters[param_name] = {"type": param_type}
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        tool_schema = {
            "name": func.__name__,
            "description": doc,
            "function": func,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required
            }
        }
        
        self._tools[func.__name__] = tool_schema
        return func
        
    async def arun(self, prompt: str, max_iterations: int = 5) -> str:
        """
        Run the agent with a given prompt (async - primary method with robust loop).
        
        This is the main async entry point with enhanced error handling and loop protection.
        """
        self.memory.add("user", prompt)
        self.logger.info(f"Starting agent run: {prompt[:50]}...")
        
        # Track tool usage for loop detection
        tool_usage_history: List[str] = []
        
        for iteration in range(max_iterations):
            self.logger.info(f"Iteration {iteration + 1}/{max_iterations}")
            
            messages = self.memory.get_messages()
            system_prompt = None
            
            if self._tools:
                system_prompt = self._build_system_prompt()
            
            # Async LLM call
            self.logger.debug(f"Calling LLM with {len(messages)} messages")
            response_content = await self.model.agenerate(messages, system_prompt)
            self.logger.debug(f"LLM response: {response_content[:100]}...")
            
            # Robust tool detection with auto-repair
            tool_call = self._safe_parse_tool_call(response_content, iteration)
            
            if tool_call:
                tool_name = tool_call["tool"]
                arguments = tool_call.get("arguments", {})
                
                self.logger.info(f"Tool call detected: {tool_name} with args {arguments}")
                
                # Loop detection
                tool_signature = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
                tool_usage_history.append(tool_signature)
                
                if self._detect_loop(tool_usage_history):
                    self.logger.warning("Infinite loop detected! Breaking...")
                    error_msg = "Loop detected: same tool called repeatedly with same arguments. Please try a different approach."
                    self.memory.add("user", f"[System] {error_msg}")
                    continue
                
                # Execute tool with timeout
                try:
                    result = await self._execute_tool_with_timeout(tool_name, arguments)
                    self.logger.debug(f"Tool result: {str(result)[:100]}...")
                    self.memory.add("assistant", f"[Tool Call: {tool_name}]")
                    self.memory.add("user", f"[Tool Result: {json.dumps(result)}]")
                    continue
                except asyncio.TimeoutError:
                    error_msg = f"Tool '{tool_name}' timed out after {self.tool_timeout}s"
                    self.logger.error(error_msg)
                    self.memory.add("user", f"[Tool Error: {error_msg}]")
                    continue
                except Exception as e:
                    error_msg = f"Tool execution failed: {str(e)}"
                    self.logger.error(error_msg)
                    self.memory.add("user", f"[Tool Error: {error_msg}]")
                    continue
            else:
                self.logger.info("Final answer received")
                self.memory.add("assistant", response_content)
                return response_content
        
        final_response = "I apologize, but I've reached the maximum number of reasoning steps."
        self.logger.warning(f"Max iterations reached ({max_iterations})")
        self.memory.add("assistant", final_response)
        return final_response
    
    def run(self, prompt: str, max_iterations: int = 5) -> str:
        """
        Run the agent with a given prompt (sync wrapper).
        
        Provided for backward compatibility. Internally calls arun() via asyncio.run().
        """
        return asyncio.run(self.arun(prompt, max_iterations))
    
    async def _execute_tool_with_timeout(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool with timeout protection."""
        return await asyncio.wait_for(
            asyncio.to_thread(self._execute_tool, tool_name, arguments),
            timeout=self.tool_timeout
        )
            
    def _build_system_prompt(self) -> str:
        tools_json = []
        for name, schema in self._tools.items():
            clean_schema = schema.copy()
            del clean_schema["function"]
            tools_json.append(clean_schema)
            
        prompt = (
            "You are a helpful AI assistant with access to the following tools:\n\n"
            f"{json.dumps(tools_json, indent=2)}\n\n"
            "To use a tool, you MUST respond with a JSON object in this format:\n"
            '{"tool": "tool_name", "arguments": {"arg_name": "value"}}\n\n'
            "If you don't need to use a tool, just respond with your answer normally.\n"
            "IMPORTANT: Do not wrap the JSON in markdown code blocks. Just return the raw JSON string."
        )
        return prompt

    def _safe_parse_tool_call(self, response: str, iteration: int) -> Optional[Dict[str, Any]]:
        """
        Robustly detect and parse tool calls from LLM response with auto-repair.
        
        Handles malformed JSON gracefully with repair attempts.
        """
        response = response.strip()
        
        if not (response.startswith("{") and '"tool"' in response):
            return None
        
        # Attempt 1: Standard parsing
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "tool" in data:
                self.logger.debug("Tool call parsed successfully (standard)")
                return data
        except json.JSONDecodeError:
            self.logger.warning("JSON parsing failed, attempting auto-repair...")
        
        # Attempt 2: Auto-repair
        repaired = self._attempt_json_repair(response)
        if repaired:
            self.logger.info("JSON successfully auto-repaired")
            return repaired
        
        # Failed: Send feedback to LLM
        self.logger.warning("JSON auto-repair failed, requesting LLM to fix")
        error_msg = (
            "[System] Invalid JSON format. Please respond with valid JSON: "
            '{"tool": "tool_name", "arguments": {"arg_name": "value"}}'
        )
        self.memory.add("user", error_msg)
        return None
    
    def _attempt_json_repair(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempt to repair common JSON formatting errors."""
        # Remove markdown code blocks
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Fix trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # Try parsing
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "tool" in data:
                return data
        except:
            pass
            
        return None
    
    def _detect_loop(self, tool_history: List[str]) -> bool:
        """
        Detect if the agent is stuck in an infinite loop.
        
        Returns True if the same tool+args has been called 3+ times in a row.
        """
        if len(tool_history) < 3:
            return False
        
        # Check last 3 calls
        last_three = tool_history[-3:]
        return len(set(last_three)) == 1

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if tool_name not in self._tools:
            raise ToolExecutionError(f"Tool '{tool_name}' not found")
        tool_def = self._tools[tool_name]
        func = tool_def["function"]
        try:
            return func(**arguments)
        except Exception as e:
            raise ToolExecutionError(f"Error executing '{tool_name}': {str(e)}") from e
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.memory.clear()
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.memory.get_messages()
    
    def __repr__(self) -> str:
        model_name = self.model.model if hasattr(self.model, "model") else "Custom"
        return f"Agent(model='{model_name}', memory={self.memory.__class__.__name__}, tools={len(self._tools)})"


# Module-level convenience
__version__ = "0.6.0"
__all__ = [
    "Agent", "Model", "Ollama", "OpenAI", "Mistral",
    "Memory", "InMemory", "FileMemory",
    "AgentFlowError", "LLMConnectionError", "LLMResponseError", "ToolExecutionError", "LoopDetectedError"
]


if __name__ == "__main__":
    print(f"AgentFlow v{__version__}")
    print("This is a library module. See examples/ for usage.")
