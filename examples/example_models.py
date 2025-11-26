"""
AgentFlow Multi-Model Example

This example demonstrates how to use different LLM providers with AgentFlow.
It shows:
1. Ollama (Local) - Default
2. OpenAI (Cloud) - Requires OPENAI_API_KEY
3. Mistral (Cloud) - Requires MISTRAL_API_KEY

Requirements:
- Ollama running locally (for Example 1)
- Environment variables set for Cloud APIs (for Examples 2 & 3)
"""

import sys
import os
import time

# Add parent directory to path so we can import agentflow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow import Agent, Ollama, OpenAI, Mistral


def example_ollama():
    """Example 1: Local Ollama Model (Default)."""
    print("=" * 60)
    print("Example 1: Local Ollama (llama3)")
    print("=" * 60)
    
    # Method A: String shortcut (defaults to Ollama)
    agent = Agent(model="llama3")
    
    # Method B: Explicit class (same result)
    # model = Ollama(model="llama3")
    # agent = Agent(model=model)
    
    print(f"Using model: {agent.model.model}")
    try:
        response = agent.run("What is the capital of France?")
        print(f"Agent: {response}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running!")
    print()


def example_openai():
    """Example 2: OpenAI GPT-4o."""
    print("=" * 60)
    print("Example 2: OpenAI (gpt-4o)")
    print("=" * 60)
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Skipping: OPENAI_API_KEY environment variable not set.")
        return

    try:
        # Initialize OpenAI model
        model = OpenAI(model="gpt-4o", api_key=api_key)
        agent = Agent(model=model)
        
        print(f"Using model: {agent.model.model}")
        response = agent.run("Explain quantum computing in one sentence.")
        print(f"Agent: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
    print()


def example_mistral():
    """Example 3: Mistral AI."""
    print("=" * 60)
    print("Example 3: Mistral (mistral-large-latest)")
    print("=" * 60)
    
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Skipping: MISTRAL_API_KEY environment variable not set.")
        return

    try:
        # Initialize Mistral model
        model = Mistral(model="mistral-large-latest", api_key=api_key)
        agent = Agent(model=model)
        
        print(f"Using model: {agent.model.model}")
        response = agent.run("What are the three laws of robotics?")
        print(f"Agent: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
    print()


def main():
    """Run all examples."""
    print("\n" + "🤖 AgentFlow Multi-Model Examples".center(60))
    print()
    
    example_ollama()
    example_openai()
    example_mistral()
        
    print("=" * 60)
    print("✅ Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
