"""
AgentFlow Memory Example

This example demonstrates how to use different memory backends in AgentFlow.
It shows:
1. Default InMemory storage (ephemeral)
2. FileMemory storage (persistent across runs)
3. Clearing memory

Requirements:
- Ollama must be running locally (ollama serve)
- llama3 model recommended (ollama pull llama3)
"""

import sys
import os
import time

# Add parent directory to path so we can import agentflow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow import Agent, FileMemory, InMemory


def example_in_memory():
    """Example 1: Default In-Memory storage (lost when script ends)."""
    print("=" * 60)
    print("Example 1: In-Memory Storage (Default)")
    print("=" * 60)
    
    # By default, Agent uses InMemory
    agent = Agent(model="llama3")
    print(f"Memory type: {type(agent.memory).__name__}")
    
    agent.run("My favorite fruit is apple.")
    print("Agent knows my favorite fruit is apple.")
    
    response = agent.run("What is my favorite fruit?")
    print(f"Agent: {response}")
    
    print("Note: This memory will be lost when the script finishes.")
    print()


def example_file_memory():
    """Example 2: File-based storage (persists across runs)."""
    print("=" * 60)
    print("Example 2: File Memory (Persistent)")
    print("=" * 60)
    
    memory_file = "conversation_history.json"
    
    # Create agent with FileMemory
    memory = FileMemory(memory_file)
    agent = Agent(model="llama3", memory=memory)
    print(f"Memory type: {type(agent.memory).__name__}")
    print(f"Current message count: {agent.memory.count()}")
    
    if agent.memory.count() == 0:
        print("\n[First Run] Teaching agent something new...")
        agent.run("My secret code is 12345.")
        print("Agent learned the secret code.")
    else:
        print("\n[Subsequent Run] Asking agent to recall...")
        response = agent.run("What is my secret code?")
        print(f"Agent: {response}")
        
        # Clean up for next run
        print("\nClearing memory for next demonstration...")
        agent.clear_history()
        
    print(f"\nConversation saved to: {os.path.abspath(memory_file)}")
    print()


def main():
    """Run all examples."""
    print("\n" + "🧠 AgentFlow Memory Examples".center(60))
    print()
    
    try:
        example_in_memory()
        example_file_memory()
        
        # Run file memory example again to show persistence
        print("Running FileMemory example again to demonstrate persistence...")
        time.sleep(1)
        example_file_memory()
        
        print("=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
        
        # Clean up the file
        if os.path.exists("conversation_history.json"):
            os.remove("conversation_history.json")
            print("Cleaned up temporary memory file.")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
