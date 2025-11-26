"""
AgentFlow - A minimalist Python framework for building AI agents

This module provides the core Agent class for creating and running AI agents
with local LLM support via Ollama.

Example:
    >>> from agentflow import Agent
    >>> agent = Agent(model="llama3")
    >>> response = agent.run("Hello, who are you?")
    >>> print(response)

Version: 0.3.0
Author: Hamadi Chaabani
License: MIT
"""

from typing import List, Dict, Optional, Any, Callable
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
                # If file is corrupted or empty, start fresh
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


class Agent:
    """
    A minimalist AI agent that interfaces with Ollama LLMs.
    
    The Agent class provides a simple interface for creating conversational
    AI agents. It manages message history and communicates with a local
    Ollama instance to generate responses.
    
    Attributes:
        model (str): The name of the Ollama model to use (e.g., "llama3").
        base_url (str): The base URL of the Ollama API endpoint.
        memory (Memory): The memory storage backend.
        
    Example:
        >>> agent = Agent(model="llama3")
        >>> response = agent.run("What is Python?")
        >>> print(response)
    """
    
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        memory: Optional[Memory] = None
    ) -> None:
        """
        Initialize an Agent instance.
        
        Args:
            model: The Ollama model name to use. Defaults to "llama3".
            base_url: The Ollama API base URL. Defaults to "http://localhost:11434".
            memory: Optional Memory instance. Defaults to InMemory().
            
        Raises:
            AgentFlowError: If initialization fails.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.memory = memory if memory else InMemory()
        self._tools: Dict[str, Dict[str, Any]] = {}  # Stores registered tools
        
    def tool(self, func: Callable) -> Callable:
        """
        Decorator to register a function as a tool for the agent.
        
        The decorator automatically extracts:
        - Tool name from function name
        - Description from docstring
        - Parameters from type hints
        
        Args:
            func: The function to register as a tool.
            
        Returns:
            The original function (unchanged).
            
        Example:
            >>> agent = Agent()
            >>> @agent.tool
            >>> def calculate(expression: str) -> float:
            >>>     \"\"\"Evaluate a mathematical expression.\"\"\"
            >>>     return eval(expression)
        """
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or "No description provided"
        
        # Extract parameters
        parameters = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            param_type = "string"  # Default type
            
            # Map Python types to JSON schema types
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
            
            # Check if parameter is required (no default value)
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        # Build tool schema
        tool_schema = {
            "name": func.__name__,
            "description": doc,
            "function": func,  # Store the actual function
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required
            }
        }
        
        self._tools[func.__name__] = tool_schema
        
        return func
        
    def run(self, prompt: str, max_iterations: int = 5) -> str:
        """
        Run the agent with a given prompt and return the response.
        
        This is the main entry point for interacting with the agent.
        The method adds the user prompt to the conversation history,
        sends it to the LLM, and returns the assistant's response.
        
        If tools are registered, the agent will automatically detect
        tool calls and execute them in a think → act loop until a
        final answer is reached or max_iterations is exceeded.
        
        Args:
            prompt: The user's input message.
            max_iterations: Maximum number of think → act iterations. Defaults to 5.
            
        Returns:
            The agent's response as a string.
            
        Raises:
            LLMConnectionError: If unable to connect to Ollama.
            LLMResponseError: If the LLM returns an invalid response.
            
        Example:
            >>> agent = Agent(model="llama3")
            >>> response = agent.run("Hello!")
            >>> print(response)
        """
        # Add user message to history
        self.memory.add("user", prompt)
        
        # Think → Act loop
        for iteration in range(max_iterations):
            # Prepare messages with system prompt if tools are available
            messages_to_send = self.memory.get_messages()
            
            if self._tools:
                # Inject system prompt with tool information
                system_prompt = self._build_system_prompt()
                # Check if system prompt already exists at the beginning
                if messages_to_send and messages_to_send[0].get("role") == "system":
                    # Update existing system prompt (simplified for now)
                    pass 
                else:
                    messages_to_send.insert(0, {
                        "role": "system",
                        "content": system_prompt
                    })
            
            # Get response from LLM
            response_content = self._call_llm(messages_to_send)
            
            # Check for tool call
            tool_call = self._detect_tool_call(response_content)
            
            if tool_call:
                # Execute the tool
                try:
                    result = self._execute_tool(
                        tool_call["tool"],
                        tool_call.get("arguments", {})
                    )
                    
                    # Add tool execution to history
                    self.memory.add("assistant", f"[Tool Call: {tool_call['tool']}]")
                    self.memory.add("user", f"[Tool Result: {json.dumps(result)}]")
                    
                    # Continue the loop to get next response
                    continue
                    
                except Exception as e:
                    # Report error to LLM
                    error_msg = f"Tool execution failed: {str(e)}"
                    self.memory.add("user", f"[Tool Error: {error_msg}]")
                    continue
            else:
                # No tool call - this is the final answer
                self.memory.add("assistant", response_content)
                return response_content
        
        # Max iterations reached
        final_response = "I apologize, but I've reached the maximum number of reasoning steps. Please try simplifying your request."
        self.memory.add("assistant", final_response)
        return final_response
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Make an API call to Ollama and return the response.
        
        This is an internal method that handles the HTTP communication
        with the Ollama API. It sends the conversation history and
        receives the generated response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            
        Returns:
            The generated text response from the LLM.
            
        Raises:
            LLMConnectionError: If the HTTP request fails.
            LLMResponseError: If the response format is invalid.
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,  # v0.1 doesn't support streaming yet
            "options": {
                "temperature": 0.0  # Deterministic for tool calling
            }
        }
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                
        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Error: {str(e)}"
            ) from e
        except httpx.TimeoutException as e:
            raise LLMConnectionError(
                f"Request to Ollama timed out. Error: {str(e)}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise LLMConnectionError(
                f"Ollama returned error status {e.response.status_code}: {str(e)}"
            ) from e
        except Exception as e:
            raise LLMConnectionError(
                f"Unexpected error communicating with Ollama: {str(e)}"
            ) from e
        
        # Parse response
        try:
            data = response.json()
            message = data.get("message", {})
            content = message.get("content", "")
            
            if not content:
                raise LLMResponseError(
                    "Ollama returned empty response content"
                )
            
            return content
            
        except json.JSONDecodeError as e:
            raise LLMResponseError(
                f"Failed to parse Ollama response as JSON: {str(e)}"
            ) from e
        except Exception as e:
            raise LLMResponseError(
                f"Unexpected error parsing Ollama response: {str(e)}"
            ) from e
            
    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool definitions."""
        tools_json = []
        for name, schema in self._tools.items():
            # Create a clean schema copy without the function object
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
        """
        Detect if the response contains a tool call.
        
        Args:
            response: The LLM's text response.
            
        Returns:
            Dictionary with tool name and arguments if found, else None.
        """
        response = response.strip()
        
        # Simple heuristic: check if it looks like JSON and has "tool" key
        if response.startswith("{") and '"tool"' in response:
            try:
                # Clean up potential markdown code blocks
                if response.startswith("```json"):
                    response = response[7:]
                if response.startswith("```"):
                    response = response[3:]
                if response.endswith("```"):
                    response = response[:-3]
                
                data = json.loads(response.strip())
                
                if isinstance(data, dict) and "tool" in data:
                    return data
            except json.JSONDecodeError:
                pass
                
        return None

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a registered tool.
        
        Args:
            tool_name: Name of the tool to execute.
            arguments: Dictionary of arguments to pass.
            
        Returns:
            The result of the tool execution.
            
        Raises:
            ToolExecutionError: If tool not found or execution fails.
        """
        if tool_name not in self._tools:
            raise ToolExecutionError(f"Tool '{tool_name}' not found")
            
        tool_def = self._tools[tool_name]
        func = tool_def["function"]
        
        try:
            return func(**arguments)
        except Exception as e:
            raise ToolExecutionError(f"Error executing '{tool_name}': {str(e)}") from e
    
    def clear_history(self) -> None:
        """
        Clear the conversation history.
        
        This removes all messages from the agent's memory, allowing
        you to start a fresh conversation.
        
        Example:
            >>> agent = Agent()
            >>> agent.run("Hello")
            >>> agent.clear_history()
            >>> # Now the agent has no memory of previous messages
        """
        self.memory.clear()
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        
        Returns:
            A list of message dictionaries containing the full conversation.
            
        Example:
            >>> agent = Agent()
            >>> agent.run("Hello")
            >>> history = agent.get_history()
            >>> print(len(history))  # 2 (user + assistant)
        """
        return self.memory.get_messages()
    
    def __repr__(self) -> str:
        """Return a string representation of the Agent."""
        return f"Agent(model='{self.model}', memory={self.memory.__class__.__name__}, messages={self.memory.count()}, tools={len(self._tools)})"


# Module-level convenience
__version__ = "0.3.0"
__all__ = ["Agent", "Memory", "InMemory", "FileMemory", "AgentFlowError", "LLMConnectionError", "LLMResponseError", "ToolExecutionError"]


if __name__ == "__main__":
    # Simple test when running module directly
    print(f"AgentFlow v{__version__}")
    print("This is a library module. See examples/ for usage.")
