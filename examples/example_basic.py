"""
Basic AgentFlow Example

This example demonstrates the simplest way to create and use an AI agent
with AgentFlow. It shows:
- Creating an agent
- Single interaction
- Multi-turn conversation
- Error handling

Requirements:
- Ollama must be running locally (ollama serve)
- A model must be installed (ollama pull llama3)
"""

import sys
import os

# Add parent directory to path so we can import agentflow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow import Agent, LLMConnectionError, LLMResponseError


def example_single_interaction():
    """Example 1: Simple single interaction."""
    print("=" * 60)
    print("Example 1: Single Interaction")
    print("=" * 60)
    
    # Create an agent
    agent = Agent(model="llama3")
    
    # Run a single prompt
    prompt = "Hello! In one sentence, who are you?"
    print(f"\nUser: {prompt}")
    
    response = agent.run(prompt)
    print(f"Agent: {response}")
    print()


def example_multi_turn_conversation():
    """Example 2: Multi-turn conversation with history."""
    print("=" * 60)
    print("Example 2: Multi-turn Conversation")
    print("=" * 60)
    
    # Create an agent
    agent = Agent(model="llama3")
    
    # First interaction
    prompt1 = "My name is Alice. What's a good programming language to learn?"
    print(f"\nUser: {prompt1}")
    response1 = agent.run(prompt1)
    print(f"Agent: {response1}")
    
    # Second interaction - agent should remember the context
    prompt2 = "Do you remember my name?"
    print(f"\nUser: {prompt2}")
    response2 = agent.run(prompt2)
    print(f"Agent: {response2}")
    
    # Show conversation history
    print(f"\nConversation history: {len(agent.get_history())} messages")
    print()


def example_error_handling():
    """Example 3: Error handling when Ollama is not available."""
    print("=" * 60)
    print("Example 3: Error Handling")
    print("=" * 60)
    
    # Try to connect to a non-existent server
    agent = Agent(model="llama3", base_url="http://localhost:99999")
    
    try:
        response = agent.run("Hello")
        print(f"Agent: {response}")
    except LLMConnectionError as e:
        print(f"\n❌ Connection Error (expected): {e}")
    except LLMResponseError as e:
        print(f"\n❌ Response Error: {e}")
    
    print()


def example_clear_history():
    """Example 4: Clear conversation history."""
    print("=" * 60)
    print("Example 4: Clear History")
    print("=" * 60)
    
    agent = Agent(model="llama3")
    
    # First conversation
    print("\nFirst conversation:")
    agent.run("My favorite color is blue.")
    print(f"Messages in history: {len(agent.get_history())}")
    
    # Clear history
    agent.clear_history()
    print("\nAfter clearing history:")
    print(f"Messages in history: {len(agent.get_history())}")
    
    # New conversation - agent won't remember previous context
    print("\nNew conversation (agent won't remember blue):")
    response = agent.run("What's my favorite color?")
    print(f"Agent: {response}")
    print()


def main():
    """Run all examples."""
    print("\n" + "🤖 AgentFlow Basic Examples".center(60))
    print()
    
    try:
        # Example 1: Single interaction
        example_single_interaction()
        
        # Example 2: Multi-turn conversation
        example_multi_turn_conversation()
        
        # Example 4: Clear history
        example_clear_history()
        
        # Example 3: Error handling (run last as it's expected to fail)
        example_error_handling()
        
        print("=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
        
    except LLMConnectionError as e:
        print(f"\n❌ Could not connect to Ollama: {e}")
        print("\nMake sure:")
        print("  1. Ollama is installed (https://ollama.ai)")
        print("  2. Ollama is running (run: ollama serve)")
        print("  3. llama3 model is installed (run: ollama pull llama3)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
