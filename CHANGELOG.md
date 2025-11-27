# Changelog

All notable changes to AgentFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2025-11-27

### Added
- 🔌 **MCP Client Integration** - First Python framework with native MCP support
- `mcp.py` module with `MCPClient` class
- stdio transport for local MCP servers
- JSON-RPC 2.0 protocol implementation
- Automatic tool discovery from MCP servers (`tools/list`)
- Automatic tool calling via MCP (`tools/call`)
- `Agent.add_mcp_tools()` method for seamless integration
- Async context manager support for MCPClient
- `example_mcp.py` demonstrating filesystem and Git servers
- Support for mixing local Python tools with MCP tools

### Why This Matters
MCP (Model Context Protocol) is Anthropic's vision for the future of AI tooling.
AgentFlow is THE FIRST Python agent framework with native MCP support, positioning
it as future-forward and ecosystem-compatible.

## [0.7.0] - 2025-11-26

### Added
- 🧪 **TestClient (Killer Feature)** - FastAPI-inspired testing
- `testing.py` module with `MockModel` and `AgentTestClient`
- Assertion helpers: `assert_tool_called`, `assert_tool_not_called`, `assert_response_contains`
- `example_testing.py` with comprehensive demos
- `tests/test_agent.py` with 8 unit tests
- Completely offline testing (no LLM required)

## [0.6.0] - 2025-11-26

### Added
- 🛡️ **Robust Loop Enhancement** - Production-ready Think → Act loop
- JSON auto-repair for malformed responses
- Infinite loop detection
- Tool timeout protection (default 30s)
- Structured logging with debug mode
- `example_robust_loop.py`

## [0.5.0] - 2025-11-26

### Added
- 🚀 **Async-First Architecture** - FastAPI-inspired
- `Model.agenerate()` and `Agent.arun()` as primary methods
- Sync wrappers for backward compatibility
- `httpx.AsyncClient` for all HTTP
- `example_async.py`

## [0.4.0] - 2025-11-26

### Added
- Unified `Model` ABC
- `Ollama`, `OpenAI`, `Mistral` classes
- `example_models.py`

## [0.3.0] - 2025-11-26

### Added
- `Memory` ABC with `InMemory` and `FileMemory`
- `example_memory.py`

## [0.2.0] - 2025-11-26

### Added
- `@agent.tool` decorator
- Tool schema generation
- Think → Act loop
- `example_tools.py`

## [0.1.0] - 2025-11-26

### Added
- Initial release
- Core `Agent` class
- Ollama integration
- Basic conversation management
