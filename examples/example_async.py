"""
AgentFlow Async Example

This example demonstrates the new async-first API in AgentFlow v0.5.

It shows:
1. Async usage with await agent.arun() (recommended)
2. Sync usage with agent.run() (backward compatible)
3. Performance comparison (optional)

Requirements:
- Ollama must be running locally (ollama serve)
- llama3 model recommended (ollama pull llama3)
"""

import sys
import os
import time
import asyncio

# Add parent directory to path so we can import agentflow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow import Agent


async def example_async_usage():
    """Example 1: Async usage (recommended)."""
    print("=" * 60)
    print("Example 1: Async Usage (await agent.arun())")
    print("=" * 60)
    
    agent = Agent(model="llama3")
    
    print("\nAsking: What is the capital of France?")
    start = time.time()
    
    # This is the async way - recommended
    response = await agent.arun("What is the capital of France?")
    
    elapsed = time.time() - start
    print(f"Agent: {response}")
    print(f"Time: {elapsed:.2f}s")
    print()


def example_sync_usage():
    """Example 2: Sync usage (backward compatible)."""
    print("=" * 60)
    print("Example 2: Sync Usage (agent.run()) - Backward Compatible")
    print("=" * 60)
    
    agent = Agent(model="llama3")
    
    print("\nAsking: What is 2 + 2?")
    start = time.time()
    
    # This is the sync way - still works for backward compatibility
    response = agent.run("What is 2 + 2?")
    
    elapsed = time.time() - start
    print(f"Agent: {response}")
    print(f"Time: {elapsed:.2f}s")
    print()


async def example_async_with_tools():
    """Example 3: Async with tools."""
    print("=" * 60)
    print("Example 3: Async with Tools")
    print("=" * 60)
    
    agent = Agent(model="llama3")
    
    # Register a tool
    @agent.tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        # Mock implementation
        return f"Sunny, 25°C in {city}"
    
    print("\nAsking: What's the weather in Paris?")
    response = await agent.arun("What's the weather in Paris?")
    print(f"Agent: {response}")
    print()


async def example_concurrent_agents():
    """Example 4: Run multiple agents concurrently (async advantage)."""
    print("=" * 60)
    print("Example 4: Concurrent Agents (Async Advantage)")
    print("=" * 60)
    
    print("\nRunning 3 agents concurrently...")
    start = time.time()
    
    # Create 3 agents with different questions
    agent1 = Agent(model="llama3")
    agent2 = Agent(model="llama3")
    agent3 = Agent(model="llama3")
    
    # Run them all concurrently
    results = await asyncio.gather(
        agent1.arun("What is Python?"),
        agent2.arun("What is JavaScript?"),
        agent3.arun("What is Rust?")
    )
    
    elapsed = time.time() - start
    
    print(f"Agent 1: {results[0][:80]}...")
    print(f"Agent 2: {results[1][:80]}...")
    print(f"Agent 3: {results[2][:80]}...")
    print(f"\nTotal time for 3 concurrent requests: {elapsed:.2f}s")
    print("(Much faster than running sequentially!)")
    print()


async def main():
    """Run all async examples."""
    print("\n" + "⚡ AgentFlow Async Examples (v0.5)".center(60))
    print()
    
    try:
        await example_async_usage()
        example_sync_usage()  # Sync example doesn't need await
        await example_async_with_tools()
        await example_concurrent_agents()
        
        print("=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
        print("\n💡 Tip: Use 'await agent.arun()' for best performance")
        print("       Use 'agent.run()' for backward compatibility")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
