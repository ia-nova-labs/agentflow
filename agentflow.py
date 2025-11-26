"""
AgentFlow - A minimalist Python framework for building AI agents

This module provides the core Agent class for creating and running AI agents
with support for multiple LLM providers (Ollama, OpenAI, Mistral).

Example:
    >>> from agentflow import Agent
    >>> agent = Agent(model="llama3")
    >>> response = agent.run("Hello, who are you?")
    >>> print(response)

Version: 0.4.0
Author: Hamadi Chaabani
License: MIT
"""

from typing import List, Dict, Optional, Any, Callable, Union
import httpx
import json
import inspect
import os
from abc import ABC, abstractmethod
from functools import wraps


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


# --- Model Architecture ---

class Model(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Generate a response from the model."""
        pass


class Ollama(Model):
    """Ollama LLM provider."""
    
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        
    def generate(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/api/chat"
        
        # Prepare messages
        msgs = messages.copy()
        if system_prompt:
            # Check if system prompt already exists
            if msgs and msgs[0].get("role") == "system":
                pass # Already has system prompt
            else:
                msgs.insert(0, {"role": "system", "content": system_prompt})
        
        payload = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": 0.0}
        }
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                
        except Exception as e:
            raise LLMConnectionError(f"Ollama connection failed: {str(e)}") from e
            
        try:
            data = response.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            raise LLMResponseError(f"Failed to parse Ollama response: {str(e)}") from e


class OpenAI(Model):
    """OpenAI LLM provider."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY env var or pass api_key.")
            
    def generate(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
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
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
        except Exception as e:
            raise LLMConnectionError(f"OpenAI connection failed: {str(e)}") from e
            
        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMResponseError(f"Failed to parse OpenAI response: {str(e)}") from e


class Mistral(Model):
    """Mistral AI LLM provider."""
    
    def __init__(self, model: str = "mistral-large-latest", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key not found. Set MISTRAL_API_KEY env var or pass api_key.")
            
    def generate(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
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
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
        except Exception as e:
            raise LLMConnectionError(f"Mistral connection failed: {str(e)}") from e
            
        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMResponseError(f"Failed to parse Mistral response: {str(e)}") from e


# --- Core Agent ---

class Agent:
    """
    A minimalist AI agent that interfaces with various LLMs.
    """
    
    def __init__(
        self,
        model: Union[str, Model] = "llama3",
        base_url: str = "http://localhost:11434",
        memory: Optional[Memory] = None
    ) -> None:
        """
        Initialize an Agent instance.
        
        Args:
            model: Model name (str) or Model instance. If str, defaults to Ollama.
            base_url: Base URL for Ollama (only used if model is str).
            memory: Optional Memory instance. Defaults to InMemory().
        """
        # Backward compatibility: if model is string, assume Ollama
        if isinstance(model, str):
            self.model = Ollama(model=model, base_url=base_url)
        else:
            self.model = model
            
        self.memory = memory if memory else InMemory()
        self._tools: Dict[str, Dict[str, Any]] = {}
        
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
        
    def run(self, prompt: str, max_iterations: int = 5) -> str:
        """Run the agent with a given prompt."""
        self.memory.add("user", prompt)
        
        for iteration in range(max_iterations):
            messages = self.memory.get_messages()
            system_prompt = None
            
            if self._tools:
                system_prompt = self._build_system_prompt()
            
            # Delegate generation to the model
            response_content = self.model.generate(messages, system_prompt)
            
            # Check for tool call
            tool_call = self._detect_tool_call(response_content)
            
            if tool_call:
                try:
                    result = self._execute_tool(
                        tool_call["tool"],
                        tool_call.get("arguments", {})
                    )
                    self.memory.add("assistant", f"[Tool Call: {tool_call['tool']}]")
                    self.memory.add("user", f"[Tool Result: {json.dumps(result)}]")
                    continue
                except Exception as e:
                    error_msg = f"Tool execution failed: {str(e)}"
                    self.memory.add("user", f"[Tool Error: {error_msg}]")
                    continue
            else:
                self.memory.add("assistant", response_content)
                return response_content
        
        final_response = "I apologize, but I've reached the maximum number of reasoning steps."
        self.memory.add("assistant", final_response)
        return final_response
            
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

    def _detect_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        response = response.strip()
        if response.startswith("{") and '"tool"' in response:
            try:
                if response.startswith("```json"): response = response[7:]
                if response.startswith("```"): response = response[3:]
                if response.endswith("```"): response = response[:-3]
                
                data = json.loads(response.strip())
                if isinstance(data, dict) and "tool" in data:
                    return data
            except json.JSONDecodeError:
                pass
        return None

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
        self.memory.clear()
    
    def get_history(self) -> List[Dict[str, str]]:
        return self.memory.get_messages()
    
    def __repr__(self) -> str:
        model_name = self.model.model if hasattr(self.model, "model") else "Custom"
        return f"Agent(model='{model_name}', memory={self.memory.__class__.__name__}, tools={len(self._tools)})"


# Module-level convenience
__version__ = "0.4.0"
__all__ = [
    "Agent", "Model", "Ollama", "OpenAI", "Mistral",
    "Memory", "InMemory", "FileMemory",
    "AgentFlowError", "LLMConnectionError", "LLMResponseError", "ToolExecutionError"
]


if __name__ == "__main__":
    print(f"AgentFlow v{__version__}")
    print("This is a library module. See examples/ for usage.")
