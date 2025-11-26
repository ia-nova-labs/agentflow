# Changelog

All notable changes to AgentFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
