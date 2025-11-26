"""
AgentFlow - A minimalist Python framework for building AI agents

This module provides the core Agent class for creating and running AI agents
with local LLM support via Ollama.

Example:
    >>> from agentflow import Agent
    >>> agent = Agent(model="llama3")
    >>> response = agent.run("Hello, who are you?")
    >>> print(response)

Version: 0.1.0
Author: Hamadi Chaabani
License: MIT
"""

from typing import List, Dict, Optional, Any
import httpx
import json


class AgentFlowError(Exception):
    """Base exception for AgentFlow errors."""
    pass


class LLMConnectionError(AgentFlowError):
    """Raised when connection to LLM fails."""
    pass


class LLMResponseError(AgentFlowError):
    """Raised when LLM returns an invalid response."""
    pass


class Agent:
    """
    A minimalist AI agent that interfaces with Ollama LLMs.
    
    The Agent class provides a simple interface for creating conversational
    AI agents. It manages message history and communicates with a local
    Ollama instance to generate responses.
    
    Attributes:
        model (str): The name of the Ollama model to use (e.g., "llama3").
        base_url (str): The base URL of the Ollama API endpoint.
        messages (List[Dict[str, str]]): Conversation history.
        
    Example:
        >>> agent = Agent(model="llama3")
        >>> response = agent.run("What is Python?")
        >>> print(response)
        >>> # Continue the conversation
        >>> response = agent.run("Tell me more about it")
        >>> print(response)
    """
    
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434"
    ) -> None:
        """
        Initialize an Agent instance.
        
        Args:
            model: The Ollama model name to use. Defaults to "llama3".
            base_url: The Ollama API base URL. Defaults to "http://localhost:11434".
            
        Raises:
            AgentFlowError: If initialization fails.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.messages: List[Dict[str, str]] = []
        
    def run(self, prompt: str) -> str:
        """
        Run the agent with a given prompt and return the response.
        
        This is the main entry point for interacting with the agent.
        The method adds the user prompt to the conversation history,
        sends it to the LLM, and returns the assistant's response.
        
        Args:
            prompt: The user's input message.
            
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
        self.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Get response from LLM
        response_content = self._call_llm(self.messages)
        
        # Add assistant response to history
        self.messages.append({
            "role": "assistant",
            "content": response_content
        })
        
        return response_content
    
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
            "stream": False  # v0.1 doesn't support streaming yet
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
        self.messages = []
    
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
        return self.messages.copy()
    
    def __repr__(self) -> str:
        """Return a string representation of the Agent."""
        return f"Agent(model='{self.model}', messages={len(self.messages)})"


# Module-level convenience
__version__ = "0.1.0"
__all__ = ["Agent", "AgentFlowError", "LLMConnectionError", "LLMResponseError"]


if __name__ == "__main__":
    # Simple test when running module directly
    print(f"AgentFlow v{__version__}")
    print("This is a library module. See examples/ for usage.")
