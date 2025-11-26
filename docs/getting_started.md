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

## Examples

Check out the `examples/` directory for more:

- `example_basic.py` - Comprehensive basic usage demonstrations

To run an example:
```bash
cd examples
python example_basic.py
```

## What's Next?

Now that you have the basics down, you can:

1. **Experiment** with different models
2. **Explore** multi-turn conversations
3. **Wait for v0.2** which will add:
   - Tool support (@agent.tool decorator)
   - Function calling capabilities
   - Think → Act loop

## API Reference

### Agent Class

```python
Agent(model: str = "llama3", base_url: str = "http://localhost:11434")
```

**Methods**:
- `run(prompt: str) -> str` - Send a prompt and get a response
- `get_history() -> List[Dict[str, str]]` - Get conversation history
- `clear_history() -> None` - Clear conversation history

**Attributes**:
- `model: str` - The model name
- `base_url: str` - The Ollama API URL
- `messages: List[Dict[str, str]]` - Message history

### Exceptions

- `AgentFlowError` - Base exception for all AgentFlow errors
- `LLMConnectionError` - Connection to Ollama failed
- `LLMResponseError` - Invalid response from Ollama

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

- **v0.2**: Tools and function calling
- **v0.3**: Memory persistence
- **v0.4**: Multi-model support (OpenAI, Mistral, Ollama)
- **v0.5**: Advanced reasoning loops
- **v0.6**: MCP integration
- **v0.7**: Multi-agent workflows
- **v1.0**: Production-ready release

Happy building! 🚀
