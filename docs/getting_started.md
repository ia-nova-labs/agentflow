# Getting Started with AgentFlow

Welcome to AgentFlow! This guide will help you create your first AI agent in less than 5 minutes.

## Prerequisites

Before you start, make sure you have:

1. **Python 3.9 or higher** installed
   ```bash
   python --version  # Should be 3.9+
   ```

2. **Ollama** installed and running
   - Download from [https://ollama.ai](https://ollama.ai)
   - After installation, start Ollama:
     ```bash
     ollama serve
     ```
   - Pull a model (we'll use llama3):
     ```bash
     ollama pull llama3
     ```

## Installation

1. Clone or download AgentFlow:
   ```bash
   git clone <repository-url>
   cd agentflow
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

That's it! You're ready to create your first agent.

## Your First Agent

Create a new file called `my_first_agent.py`:

```python
from agentflow import Agent

# Create an agent
agent = Agent(model="llama3")

# Have a conversation
response = agent.run("Hello! What can you help me with?")
print(response)
```

Run it:
```bash
python my_first_agent.py
```

🎉 Congratulations! You just created your first AI agent.

## Understanding the Basics

### Creating an Agent

```python
from agentflow import Agent

# Use default settings (llama3 model, localhost:11434)
agent = Agent()

# Or customize the model
agent = Agent(model="mistral")

# Or use a different Ollama server
agent = Agent(model="llama3", base_url="http://192.168.1.100:11434")
```

### Running Prompts

The `run()` method is your main interface:

```python
response = agent.run("What is Python?")
print(response)
```

### Multi-Turn Conversations

Agents automatically remember conversation history:

```python
agent = Agent()

# First message
agent.run("My name is Alice")

# Second message - agent remembers context
response = agent.run("What's my name?")
print(response)  # Will answer "Alice"
```

### Managing History

```python
# Get conversation history
history = agent.get_history()
print(f"Total messages: {len(history)}")

# Clear history to start fresh
agent.clear_history()
```

## Working with Tools (New in v0.2)

AgentFlow v0.2 introduces the `@agent.tool` decorator, allowing you to give your agent capabilities like calculation, web search, or API access.

### Creating a Tool

Simply decorate a Python function with `@agent.tool`. The agent will automatically understand how to use it based on the function name, docstring, and type hints.

```python
@agent.tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)
```

### Using Tools

Once registered, the agent will decide when to use the tool:

```python
# The agent will use the calculate tool for this
response = agent.run("What is 123 * 456?")
print(response)
```

### Multi-Step Reasoning

The agent can use tools multiple times to solve complex problems:

```python
@agent.tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    # ... implementation ...
    return "Sunny, 25°C"

response = agent.run("Is it hotter in Paris or London right now?")
# Agent will:
# 1. Call get_weather("Paris")
# 2. Call get_weather("London")
# 3. Compare and answer
```

### Best Practices for Tools

1. **Type Hints**: Always use Python type hints (`str`, `int`, `float`, `bool`). The agent uses these to know what arguments to pass.
2. **Docstrings**: Write clear docstrings describing *what* the tool does and *when* to use it.
3. **Return Values**: Return simple data types (strings, numbers, dictionaries) that are easy for the LLM to understand.

## Error Handling

Always handle potential errors:

```python
from agentflow import Agent, LLMConnectionError, LLMResponseError

agent = Agent()

try:
    response = agent.run("Hello!")
    print(response)
except LLMConnectionError as e:
    print(f"Connection error: {e}")
    print("Is Ollama running?")
except LLMResponseError as e:
    print(f"Response error: {e}")
```

## Common Issues

### "Failed to connect to Ollama"

**Problem**: AgentFlow can't reach your Ollama server.

**Solutions**:
1. Make sure Ollama is running: `ollama serve`
2. Check the port (default is 11434)
3. If using a custom URL, verify it's correct

### "Ollama returned empty response"

**Problem**: The model returned no content.

**Solutions**:
1. Make sure the model is properly installed: `ollama pull llama3`
2. Try a different model
3. Check Ollama logs for errors

### "Model not found"

**Problem**: The specified model isn't available.

**Solutions**:
1. List available models: `ollama list`
2. Pull the model: `ollama pull <model-name>`

### "Tool execution failed"

**Problem**: A tool raised an exception during execution.

**Solutions**:
1. Check your tool implementation for errors
2. Add try/except blocks within your tool function
3. Return a descriptive error string instead of raising an exception

## Managing Memory (New in v0.3)

AgentFlow v0.3 introduces flexible memory management. By default, agents use `InMemory` storage which is lost when the script ends. You can use `FileMemory` to persist conversations.

### Using Persistent Memory

```python
from agentflow import Agent, FileMemory

# Create persistent memory
memory = FileMemory("history.json")

# Create agent with this memory
agent = Agent(memory=memory)

# This conversation will be saved to history.json
agent.run("My name is Alice")
```

If you restart the script, the agent will remember everything!

### Custom Memory

You can create your own memory backend (e.g., for a database) by inheriting from the `Memory` class:

```python
from agentflow import Memory

class RedisMemory(Memory):
    def add(self, role, content): ...
    def get_messages(self): ...
    def clear(self): ...
    def count(self): ...
```

## Using Different Models (New in v0.4)

AgentFlow v0.4 supports multiple LLM providers. You can easily switch between local and cloud models.

### Local Models (Ollama)

This is the default. You can specify the model name directly:

```python
# Uses Ollama by default
agent = Agent(model="llama3")
```

### Cloud Models (OpenAI, Mistral)

To use cloud models, you need to set your API keys in environment variables (`OPENAI_API_KEY`, `MISTRAL_API_KEY`) or pass them explicitly.

```python
from agentflow import Agent, OpenAI, Mistral

# Use OpenAI
agent = Agent(model=OpenAI(model="gpt-4o"))

# Use Mistral
agent = Agent(model=Mistral(model="mistral-large-latest"))
```

## Examples

Check out the `examples/` directory for more:

- `example_basic.py` - Comprehensive basic usage demonstrations
- `example_tools.py` - Demonstrates calculator and weather tools
- `example_memory.py` - Shows how to use persistent memory
- `example_models.py` - Demonstrates switching between Ollama, OpenAI, and Mistral

To run an example:
```bash
cd examples
python example_models.py
```

## What's Next?

Now that you have the basics down, you can:

1. **Experiment** with different models
2. **Explore** multi-turn conversations
3. **Try Tools** to give your agent superpowers
4. **Use Memory** to save conversations
5. **Switch Models** to use the best AI for the job
6. **Wait for v0.5** which will add:
   - Real Think → Act loop (Observation, Reasoning, Action)
   - Improved agent autonomy

## API Reference

### Agent Class

```python
Agent(model: str = "llama3", base_url: str = "http://localhost:11434")
```

**Methods**:
- `run(prompt: str, max_iterations: int = 5) -> str` - Send a prompt and get a response
- `tool(func: Callable) -> Callable` - Decorator to register a tool
- `get_history() -> List[Dict[str, str]]` - Get conversation history
- `clear_history() -> None` - Clear conversation history

**Attributes**:
- `model: Model` - The LLM provider instance
- `memory: Memory` - The memory backend
- `_tools: Dict` - Registered tools

### Model Classes

- `Ollama` - Local LLM provider (default)
- `OpenAI` - OpenAI API provider
- `Mistral` - Mistral AI API provider

### Memory Classes

- `InMemory` - Ephemeral list-based storage (default)
- `FileMemory` - JSON file-based persistent storage

### Exceptions

- `AgentFlowError` - Base exception for all AgentFlow errors
- `LLMConnectionError` - Connection to Ollama failed
- `LLMResponseError` - Invalid response from Ollama
- `ToolExecutionError` - Tool execution failed

## Philosophy

AgentFlow follows these principles:

- **Explicit > Magic**: Clear APIs, no hidden behavior
- **Minimal by default**: Start simple, add complexity when needed
- **Framework, not library**: You're in control
- **Documentation first**: If it's not documented, it doesn't exist

## Get Help

- Check the [examples](../examples/)
- Read the [README](../README.md)
- Review the [CHANGELOG](../CHANGELOG.md) for latest features

## Next Steps

Ready to dive deeper? Here's what's coming in future versions:

- **v0.5**: Advanced reasoning loops
- **v0.6**: MCP integration
- **v0.7**: Multi-agent workflows
- **v1.0**: Production-ready release

Happy building! 🚀
