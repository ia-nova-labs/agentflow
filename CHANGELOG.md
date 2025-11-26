# Changelog

All notable changes to AgentFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2025-11-26

### Added
- 🚀 **Async-First Architecture** - Framework transformed to async-first like FastAPI
- `Model.agenerate()` - Primary async method for all LLM providers
- `Agent.arun()` - Primary async method for running agents
- Sync wrappers (`generate()`, `run()`) for backward compatibility
- `httpx.AsyncClient` for all HTTP communications
- `example_async.py` demonstrating async usage and concurrent execution
- Robust tool call parsing with `_safe_parse_tool_call()`

### Changed
- **BREAKING (Internal)**: All model implementations now use `async`/`await`
- `Ollama`, `OpenAI`, `Mistral` now implement `agenerate()` as primary method
- `Agent.run()` is now a sync wrapper around `arun()`
- Version bumped to 0.5.0 to reflect major architectural shift

### Migration Guide

```python
# v0.4 (still works in v0.5)
agent = Agent()
response = agent.run("Hello")

# v0.5 (recommended for best performance)
import asyncio

async def main():
    agent = Agent()
   response = await agent.arun("Hello")
    
asyncio.run(main())
```

## [0.4.0] - 2025-11-26

### Added
- Unified `Model` abstract base class for LLM providers
- `Ollama` class for local model support
- `OpenAI` class for OpenAI API support
- `Mistral` class for Mistral AI API support
- `example_models.py` demonstrating multi-model usage
- Documentation for "Using Different Models"

### Changed
- Refactored `Agent` to use `Model` interface
- `Agent` now accepts `model` object or string (for backward compatibility)
- Moved LLM communication logic from `Agent` to `Model` classes

## [0.3.0] - 2025-11-26

### Added
- `Memory` abstract base class for flexible memory management
- `InMemory` class for default ephemeral storage
- `FileMemory` class for JSON-based persistent storage
- `Agent` now accepts a `memory` parameter
- `example_memory.py` demonstrating persistence usage
- Documentation for "Managing Memory"

### Changed
- Refactored `Agent` to use `Memory` interface instead of internal list
- `clear_history()` and `get_history()` now delegate to memory backend
- Updated `Agent.__repr__` to show memory type

## [0.2.0] - 2025-11-26

### Added
- `@agent.tool` decorator for easy tool registration
- Automatic tool schema generation from docstrings and type hints
- Think → Act loop for tool execution (max 5 iterations)
- `ToolExecutionError` for handling tool failures
- `example_tools.py` demonstrating calculator and weather tools
- Comprehensive documentation for "Working with Tools"

### Changed
- `Agent.run()` now supports `max_iterations` parameter
- Updated `Agent` class to store and manage tools
- Enhanced system prompt to include tool definitions

## [0.1.0] - 2025-11-26

### Added
- Initial release of AgentFlow
- Core `Agent` class with basic functionality
- `run(prompt: str)` method for simple agent interaction
- Ollama integration for local LLM support
- Basic message handling and conversation management
- Type hints and comprehensive docstrings
- MIT License
- Initial documentation and examples
- Getting started guide

### Technical Details
- Single-file architecture (<300 lines)
- Synchronous API (async coming in future releases)
- HTTP-based Ollama API integration via httpx
- Simple prompt → LLM → response loop

### Known Limitations
- Only supports Ollama models
- No tool support (coming in v0.2)
- No memory persistence (coming in v0.3)
- No streaming responses yet
- Synchronous only

[0.1.0]: https://github.com/yourusername/agentflow/releases/tag/v0.1.0
