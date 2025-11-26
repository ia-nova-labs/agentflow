"""
Unit tests for AgentFlow

These tests use MockModel and AgentTestClient to test agent behavior
without making real LLM API calls.

Run with:
    python -m pytest tests/test_agent.py
    
Or simply:
    python tests/test_agent.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow import Agent, InMemory
from agentflow.testing import MockModel, AgentTestClient


def test_basic_interaction():
    """Test basic agent interaction."""
    model = MockModel(responses=["Hello! I'm an AI assistant."])
    agent = Agent(model=model)
    client = AgentTestClient(agent)
    
    response = client.run("Hello")
    
    assert "assistant" in response.lower() or "AI" in response
    assert model.call_count == 1


def test_multi_turn_conversation():
    """Test multi-turn conversation with memory."""
    model = MockModel(responses=[
        "Nice to meet you, Alice!",
        "Your name is Alice."
    ])
    agent = Agent(model=model)
    client = AgentTestClient(agent)
    
    client.run("My name is Alice")
    response = client.run("What's my name?")
    
    assert "Alice" in response
    assert len(client.get_history()) == 4  # 2 user + 2 assistant


def test_tool_calling():
    """Test that agent can call tools."""
    model = MockModel(responses=[
        '{"tool": "calculate", "arguments": {"expression": "2+2"}}',
        "The answer is 4"
    ])
    agent = Agent(model=model)
    
    @agent.tool
    def calculate(expression: str) -> float:
        """Calculate a mathematical expression."""
        return eval(expression)
    
    client = AgentTestClient(agent)
    response = client.run("What is 2+2?")
    
    client.assert_tool_called("calculate")
    assert "4" in response


def test_tool_not_called():
    """Test assertion for tool not being called."""
    model = MockModel(responses=["I don't need tools for this."])
    agent = Agent(model=model)
    
    @agent.tool
    def unused_tool(x: int) -> int:
        return x * 2
    
    client = AgentTestClient(agent)
    client.run("Hello")
    
    client.assert_tool_not_called("unused_tool")


def test_response_contains():
    """Test response content assertion."""
    model = MockModel(responses=["Python is a programming language."])
    agent = Agent(model=model)
    client = AgentTestClient(agent)
    
    client.run("What is Python?")
    
    client.assert_response_contains("Python")
    client.assert_response_contains("programming")


def test_tool_called_multiple_times():
    """Test tool called specific number of times."""
    model = MockModel(responses=[
        '{"tool": "count", "arguments": {"n": 1}}',
        '{"tool": "count", "arguments": {"n": 2}}',
        "Done counting to 2"
    ])
    agent = Agent(model=model)
    
    @agent.tool
    def count(n: int) -> int:
        return n
    
    client = AgentTestClient(agent)
    client.run("Count to 2", max_iterations=10)
    
    client.assert_tool_called("count", times=2)


def test_memory_persistence():
    """Test that memory persists across interactions."""
    model = MockModel(responses=["Got it", "You said: test message"])
    agent = Agent(model=model, memory=InMemory())
    client = AgentTestClient(agent)
    
    client.run("Remember: test message")
    response = client.run("What did I say?")
    
    assert "test message" in response


def test_clear_history():
    """Test clearing agent history."""
    model = MockModel(responses=["Hello", "Hi again"])
    agent = Agent(model=model)
    client = AgentTestClient(agent)
    
    client.run("First message")
    assert len(client.get_history()) == 2  # user + assistant
    
    client.clear_history()
    assert len(client.get_history()) == 0
    
    client.run("Second message")
    assert len(client.get_history()) == 2  # Only new interaction


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("Basic Interaction", test_basic_interaction),
        ("Multi-turn Conversation", test_multi_turn_conversation),
        ("Tool Calling", test_tool_calling),
        ("Tool Not Called", test_tool_not_called),
        ("Response Contains", test_response_contains),
        ("Tool Called Multiple Times", test_tool_called_multiple_times),
        ("Memory Persistence", test_memory_persistence),
        ("Clear History", test_clear_history),
    ]
    
    print("\n" + "=" * 60)
    print("Running AgentFlow Unit Tests")
    print("=" * 60 + "\n")
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"✅ {name}")
            passed += 1
        except AssertionError as e:
            print(f"❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {name}: Unexpected error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
