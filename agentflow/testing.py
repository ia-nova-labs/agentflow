"""
AgentFlow Testing Utilities

This module provides testing utilities for AgentFlow, including:
- MockModel: Mock LLM for testing without real API calls
- AgentTestClient: FastAPI-inspired test client for agents

Example:
    >>> from agentflow import Agent
    >>> from agentflow.testing import MockModel, AgentTestClient
    >>> 
    >>> model = MockModel(responses=["Hello!"])
    >>> agent = Agent(model=model)
    >>> client = AgentTestClient(agent)
    >>> 
    >>> response = client.run("Hello")
    >>> assert "Hello" in response

Author: Hamadi Chaabani
License: MIT
"""

from typing import List, Dict, Any, Optional
import asyncio
from . import Agent, Model


class MockModel(Model):
    """
    Mock model for testing without real LLM calls.
    
    Returns predefined responses in sequence.
    """
    
    def __init__(self, responses: List[str]):
        """
        Initialize MockModel.
        
        Args:
            responses: List of responses to return in order.
        """
        self.responses = responses
        self.call_count = 0
        self.calls: List[Dict[str, Any]] = []
        
    async def agenerate(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: Optional[str] = None
    ) -> str:
        """Return next response from the list."""
        # Record the call
        self.calls.append({
            "messages": messages.copy(),
            "system_prompt": system_prompt
        })
        
        if self.call_count >= len(self.responses):
            raise IndexError(
                f"MockModel ran out of responses. "
                f"Provided {len(self.responses)}, needed {self.call_count + 1}"
            )
        
        response = self.responses[self.call_count]
        self.call_count += 1
        return response
    
    def reset(self):
        """Reset call count and history."""
        self.call_count = 0
        self.calls = []


class AgentTestClient:
    """
    Test client for AgentFlow agents (inspired by FastAPI TestClient).
    
    Provides easy testing and assertion helpers.
    
    Example:
        >>> client = AgentTestClient(agent)
        >>> response = client.run("Hello")
        >>> client.assert_tool_called("my_tool")
    """
    
    def __init__(self, agent: Agent):
        """
        Initialize TestClient.
        
        Args:
            agent: The agent to test.
        """
        self.agent = agent
        self.interaction_history: List[Dict[str, str]] = []
        
    def run(self, prompt: str, **kwargs) -> str:
        """
        Run agent synchronously and record interaction.
        
        Args:
            prompt: User prompt.
            **kwargs: Additional arguments for agent.run().
            
        Returns:
            Agent response.
        """
        response = self.agent.run(prompt, **kwargs)
        self.interaction_history.append({
            "prompt": prompt,
            "response": response
        })
        return response
    
    async def arun(self, prompt: str, **kwargs) -> str:
        """
        Run agent asynchronously and record interaction.
        
        Args:
            prompt: User prompt.
            **kwargs: Additional arguments for agent.arun().
            
        Returns:
            Agent response.
        """
        response = await self.agent.arun(prompt, **kwargs)
        self.interaction_history.append({
            "prompt": prompt,
            "response": response
        })
        return response
    
    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """
        Extract all tool calls from agent history.
        
        Returns:
            List of tool call information.
        """
        tool_calls = []
        messages = self.agent.get_history()
        
        for msg in messages:
            content = msg.get("content", "")
            if content.startswith("[Tool Call:"):
                # Extract tool name
                tool_name = content.replace("[Tool Call:", "").replace("]", "").strip()
                tool_calls.append({"tool": tool_name})
        
        return tool_calls
    
    def assert_tool_called(self, tool_name: str, times: Optional[int] = None):
        """
        Assert that a tool was called.
        
        Args:
            tool_name: Name of the tool.
            times: Optional number of times it should have been called.
            
        Raises:
            AssertionError: If assertion fails.
        """
        tool_calls = self.get_tool_calls()
        matching_calls = [tc for tc in tool_calls if tc["tool"] == tool_name]
        
        if len(matching_calls) == 0:
            raise AssertionError(f"Tool '{tool_name}' was not called")
        
        if times is not None and len(matching_calls) != times:
            raise AssertionError(
                f"Tool '{tool_name}' was called {len(matching_calls)} times, "
                f"expected {times}"
            )
    
    def assert_tool_not_called(self, tool_name: str):
        """
        Assert that a tool was NOT called.
        
        Args:
            tool_name: Name of the tool.
            
        Raises:
            AssertionError: If assertion fails.
        """
        tool_calls = self.get_tool_calls()
        matching_calls = [tc for tc in tool_calls if tc["tool"] == tool_name]
        
        if len(matching_calls) > 0:
            raise AssertionError(
                f"Tool '{tool_name}' was called {len(matching_calls)} times, "
                f"expected 0"
            )
    
    def assert_response_contains(self, text: str):
        """
        Assert that the last response contains text.
        
        Args:
            text: Text to search for.
            
        Raises:
            AssertionError: If assertion fails.
        """
        if not self.interaction_history:
            raise AssertionError("No interactions recorded")
        
        last_response = self.interaction_history[-1]["response"]
        if text not in last_response:
            raise AssertionError(
                f"Response does not contain '{text}'. "
                f"Response: {last_response[:100]}..."
            )
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get agent conversation history."""
        return self.agent.get_history()
    
    def clear_history(self):
        """Clear agent history and interaction history."""
        self.agent.clear_history()
        self.interaction_history = []
    
    def __repr__(self) -> str:
        return f"AgentTestClient(agent={self.agent}, interactions={len(self.interaction_history)})"


__all__ = ["MockModel", "AgentTestClient"]
