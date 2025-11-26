"""
AgentFlow Tools Example

This example demonstrates how to use the @agent.tool decorator to give
your agent capabilities like calculation and data retrieval.

Requirements:
- Ollama must be running locally (ollama serve)
- llama3 model recommended (ollama pull llama3)
"""

import sys
import os
import math

# Add parent directory to path so we can import agentflow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow import Agent


def example_calculator_tool():
    """Example 1: Simple calculator tool."""
    print("=" * 60)
    print("Example 1: Calculator Tool")
    print("=" * 60)
    
    agent = Agent(model="llama3")
    
    # Register a tool
    @agent.tool
    def calculate(expression: str) -> float:
        """
        Evaluate a mathematical expression.
        Useful for calculations that are hard for LLMs.
        """
        print(f"  [Tool] Calculating: {expression}")
        try:
            # Safe evaluation of math expressions
            allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            return eval(expression, {"__builtins__": {}}, allowed_names)
        except Exception as e:
            return f"Error: {str(e)}"
            
    # Run a prompt that requires calculation
    prompt = "What is the square root of 12345 multiplied by pi?"
    print(f"\nUser: {prompt}")
    
    response = agent.run(prompt)
    print(f"\nAgent: {response}")
    print()


def example_weather_tool():
    """Example 2: Mock weather tool with structured data."""
    print("=" * 60)
    print("Example 2: Weather Tool")
    print("=" * 60)
    
    agent = Agent(model="llama3")
    
    @agent.tool
    def get_weather(city: str) -> dict:
        """Get the current weather for a specific city."""
        print(f"  [Tool] Fetching weather for: {city}")
        
        # Mock data
        weather_data = {
            "paris": {"temp": 22, "condition": "sunny", "humidity": 60},
            "london": {"temp": 15, "condition": "rainy", "humidity": 85},
            "new york": {"temp": 25, "condition": "cloudy", "humidity": 55},
            "tokyo": {"temp": 28, "condition": "clear", "humidity": 40}
        }
        
        city_lower = city.lower()
        if city_lower in weather_data:
            return weather_data[city_lower]
        else:
            return {"error": "City not found"}
            
    prompt = "What's the weather like in Paris and London right now?"
    print(f"\nUser: {prompt}")
    
    response = agent.run(prompt)
    print(f"\nAgent: {response}")
    print()


def example_multi_step_reasoning():
    """Example 3: Multi-step reasoning with multiple tools."""
    print("=" * 60)
    print("Example 3: Multi-step Reasoning")
    print("=" * 60)
    
    agent = Agent(model="llama3")
    
    @agent.tool
    def get_product_price(product_name: str) -> float:
        """Get the price of a product."""
        print(f"  [Tool] Checking price for: {product_name}")
        prices = {
            "laptop": 1200.00,
            "phone": 800.00,
            "headphones": 150.00
        }
        return prices.get(product_name.lower(), 0.0)
        
    @agent.tool
    def calculate_tax(price: float, tax_rate: float) -> float:
        """Calculate tax amount."""
        print(f"  [Tool] Calculating tax: {price} * {tax_rate}")
        return price * tax_rate
        
    prompt = "How much would 3 laptops cost including 20% tax?"
    print(f"\nUser: {prompt}")
    
    response = agent.run(prompt)
    print(f"\nAgent: {response}")
    print()


def main():
    """Run all examples."""
    print("\n" + "🛠️ AgentFlow Tool Examples".center(60))
    print()
    
    try:
        example_calculator_tool()
        example_weather_tool()
        example_multi_step_reasoning()
        
        print("=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
