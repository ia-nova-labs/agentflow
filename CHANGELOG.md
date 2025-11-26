# Changelog

All notable changes to AgentFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2025-11-26

### Added
- 🛡️ **Robust Loop Enhancement** - Production-ready Think → Act loop
- JSON auto-repair for malformed LLM responses (removes markdown, fixes trailing commas)
- Infinite loop detection (detects same tool+args called 3+ times)
- Tool timeout protection with `asyncio.wait_for()` (default 30s, configurable)
- Structured logging system with debug mode (`debug=True`)
- Intelligent retry logic - sends feedback to LLM when JSON parsing fails
- `LoopDetectedError` exception for loop detection
- `tool_timeout` parameter in Agent.__init__()
- `logger` parameter for custom logging
- `example_robust_loop.py` demonstrating all robustness features

### Changed
- `_safe_parse_tool_call()` now attempts JSON auto-repair before failing
- `arun()` now includes comprehensive logging at INFO and DEBUG levels
- Tool execution wrapped in timeout protection
- Enhanced error messages sent back to LLM for self-correction

### Fixed
- Malformed JSON responses from LLMs no longer crash the loop
- Infinite loops are detected and broken automatically
- Long-running tools don't block indefinitely

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

## [0.4.0] - 2025-11-26

### Added
- Unified `Model` abstract base class for LLM providers
- `Ollama` class for local model support
- `OpenAI` class for OpenAI API support
- `Mistral` class for Mistral AI API support
- `example_models.py` demonstrating multi-model usage

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

### Changed
- Refactored `Agent` to use `Memory` interface instead of internal list
- `clear_history()` and `get_history()` now delegate to memory backend

## [0.2.0] - 2025-11-26

### Added
- `@agent.tool` decorator for easy tool registration
- Automatic tool schema generation from docstrings and type hints
- Think → Act loop for tool execution (max 5 iterations)
- `ToolExecutionError` for handling tool failures
- `example_tools.py` demonstrating calculator and weather tools

### Changed
- `Agent.run()` now supports `max_iterations` parameter
- Enhanced system prompt to include tool definitions

## [0.1.0] - 2025-11-26

### Added
- Initial release of AgentFlow
- Core `Agent` class with basic functionality
- Ollama integration for local LLM support
- Basic message handling and conversation management
- Type hints and comprehensive docstrings
- MIT License
- Initial documentation and examples

[0.1.0]: https://github.com/yourusername/agentflow/releases/tag/v0.1.0
