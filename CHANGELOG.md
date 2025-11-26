# Changelog

All notable changes to AgentFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2025-11-26

### Added
- 🧪 **TestClient (Killer Feature)** - FastAPI-inspired testing for agents
- `testing.py` module with `MockModel` and `AgentTestClient`
- `MockModel` - Test agents without real LLM API calls
- `AgentTestClient` - Wrapper for easy testing with assertion helpers
- Assertion helpers:
  - `assert_tool_called(tool_name, times=None)` - Verify tool was called
  - `assert_tool_not_called(tool_name)` - Verify tool was NOT called
  - `assert_response_contains(text)` - Verify response content
- `example_testing.py` demonstrating all testing patterns
- `tests/test_agent.py` with 8 comprehensive unit tests
- Testing runs completely offline (no LLM required)

### Why This Matters
This is THE differentiator from competitors:
- **LangChain**: No easy testing ❌
- **CrewAI**: Cannot mock LLMs ❌
- **AutoGen**: Testing nightmare ❌
- **AgentFlow**: TestClient makes it trivial ✅

## [0.6.0] - 2025-11-26

### Added
- 🛡️ **Robust Loop Enhancement** - Production-ready Think → Act loop
- JSON auto-repair for malformed LLM responses
- Infinite loop detection (same tool+args 3x)
- Tool timeout protection (default 30s, configurable)
- Structured logging system with debug mode
- Intelligent retry logic with LLM feedback
- `LoopDetectedError` exception
- `example_robust_loop.py` demonstrating robustness features

### Changed
- `_safe_parse_tool_call()` attempts JSON auto-repair
- `arun()` includes comprehensive logging
- Tool execution wrapped in timeout protection

## [0.5.0] - 2025-11-26

### Added
- 🚀 **Async-First Architecture** - Framework transformed like FastAPI
- `Model.agenerate()` - Primary async method for all LLM providers
- `Agent.arun()` - Primary async method for running agents
- Sync wrappers (`generate()`, `run()`) for backward compatibility
- `httpx.AsyncClient` for all HTTP communications
- `example_async.py` demonstrating async usage and concurrent execution

### Changed
- **BREAKING (Internal)**: All model implementations now use `async`/`await`
- `Agent.run()` is now a sync wrapper around `arun()`

## [0.4.0] - 2025-11-26

### Added
- Unified `Model` abstract base class for LLM providers
- `Ollama`, `OpenAI`, `Mistral` classes
- `example_models.py` demonstrating multi-model usage

### Changed
- Refactored `Agent` to use `Model` interface

## [0.3.0] - 2025-11-26

### Added
- `Memory` abstract base class for flexible memory management
- `InMemory` and `FileMemory` classes
- `example_memory.py` demonstrating persistence

### Changed
- Refactored `Agent` to use `Memory` interface

## [0.2.0] - 2025-11-26

### Added
- `@agent.tool` decorator for easy tool registration
- Automatic tool schema generation
- Think → Act loop for tool execution
- `example_tools.py` demonstrating tools

## [0.1.0] - 2025-11-26

### Added
- Initial release of AgentFlow
- Core `Agent` class with basic functionality
- Ollama integration for local LLM support
- Basic message handling and conversation management

[0.1.0]: https://github.com/yourusername/agentflow/releases/tag/v0.1.0
