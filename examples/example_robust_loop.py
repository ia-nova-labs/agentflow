"""
AgentFlow Robust Loop Example

This example demonstrates the robust loop features in AgentFlow v0.6.

It shows:
1. JSON auto-repair handling malformed LLM responses
2. Loop detection preventing infinite tool calls
3. Tool timeout protection
4. Debug mode with structured logging
5. Intelligent retry with LLM feedback

Requirements:
- Ollama must be running locally (ollama serve)
- llama3 model recommended (ollama pull llama3)
"""

import sys
import os
import asyncio
import time

# Add parent directory to path so we can import agentflow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow import Agent


async def example_debug_mode():
    """Example 1: Debug mode with structured logging."""
    print("=" * 60)
    print("Example 1: Debug Mode + Structured Logging")
    print("=" * 60)
    
    # Enable debug mode to see detailed logs
    agent = Agent(model="llama3", debug=True)
    
    @agent.tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Sunny, 25°C in {city}"
    
    print("\nAsking: What's the weather in Paris?")
    print("(Watch the detailed logs below)\n")
    
    response = await agent.arun("What's the weather in Paris?")
    print(f"\nFinal Answer: {response}")
    print()


async def example_tool_timeout():
    """Example 2: Tool timeout protection."""
    print("=" * 60)
    print("Example 2: Tool Timeout Protection")
    print("=" * 60)
    
    # Set a short timeout for demonstration
    agent = Agent(model="llama3", tool_timeout=3, debug=True)
    
    @agent.tool
    def slow_operation(seconds: int) -> str:
        """Simulate a slow operation that takes some seconds."""
        print(f"  [Tool] Sleeping for {seconds} seconds...")
        time.sleep(seconds)
        return f"Completed after {seconds} seconds"
    
    print("\nAsking: Run a slow operation for 5 seconds")
    print("(This will timeout after 3 seconds)\n")
    
    try:
        response = await agent.arun("Run a slow operation for 5 seconds")
        print(f"\nResponse: {response}")
    except Exception as e:
        print(f"\nCaught error: {e}")
    print()


async def example_loop_detection():
    """Example 3: Infinite loop detection."""
    print("=" * 60)
    print("Example 3: Infinite Loop Detection")
    print("=" * 60)
    
    agent = Agent(model="llama3", debug=True)
    
    @agent.tool
    def check_status() -> str:
        """Check the system status."""
        return "Status: Pending"
    
    print("\nSimulating a scenario where the LLM might keep calling the same tool...")
    print("(The loop detector will prevent infinite calls)\n")
    
    # This might trigger loop detection if LLM keeps calling check_status
    response = await agent.arun(
        "Keep checking the status until it's 'Ready'. "
        "Use the check_status tool to verify."
    )
    print(f"\nResponse: {response}")
    print()


async def example_json_auto_repair():
    """Example 4: Demonstrate JSON auto-repair (conceptual)."""
    print("=" * 60)
    print("Example 4: JSON Auto-Repair (Conceptual)")
    print("=" * 60)
    
    print("\nThe agent now handles malformed JSON responses gracefully:")
    print("- Removes markdown code blocks")
    print("- Fixes trailing commas")
    print("- Sends feedback to LLM if repair fails")
    print("\nThis happens automatically in the background!")
    
    agent = Agent(model="llama3", debug=False)
    
    @agent.tool
    def calculate(expression: str) -> float:
        """Calculate a mathematical expression."""
        try:
            return eval(expression)
        except:
            return 0.0
    
    response = await agent.arun("What is 25 * 17?")
    print(f"\nResponse: {response}")
    print()


async def example_max_iterations():
    """Example 5: Max iterations protection."""
    print("=" * 60)
    print("Example 5: Max Iterations Protection")
    print("=" * 60)
    
    agent = Agent(model="llama3", debug=True)
    
    @agent.tool
    def count(n: int) -> int:
        """Return the number."""
        return n
    
    print("\nAsking a question that might require many steps...")
    print("(Agent will stop at max_iterations)\n")
    
    response = await agent.arun(
        "Count from 1 to 10 using the count tool for each number",
        max_iterations=3  # Reduced to demonstrate
    )
    print(f"\nResponse: {response}")
    print()


async def main():
    """Run all examples."""
    print("\n" + "🛡️ AgentFlow Robust Loop Examples (v0.6)".center(60))
    print()
    
    try:
        await example_debug_mode()
        print("\n" + "-" * 60 + "\n")
        
        await example_tool_timeout()
        print("\n" + "-" * 60 + "\n")
        
        await example_loop_detection()
        print("\n" + "-" * 60 + "\n")
        
        await example_json_auto_repair()
        print("\n" + "-" * 60 + "\n")
        
        await example_max_iterations()
        
        print("=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
        print("\n💡 Key Features:")
        print("   • Structured logging with debug mode")
        print("   • Automatic JSON repair")
        print("   • Loop detection")
        print("   • Tool timeout protection")
        print("   • Intelligent retry with LLM feedback")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
