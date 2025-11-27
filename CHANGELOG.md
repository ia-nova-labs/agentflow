# Changelog

All notable changes to AgentFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-27

### 🎉 Production Release

AgentFlow v1.0.0 is production-ready! This release represents the culmination of
8 sprints of development, delivering a minimalist yet powerful framework for building AI agents.

### Added
- **PyPI Distribution** - `pip install agentflow`
- `setup.py` and `pyproject.toml` for proper packaging
- GitHub Actions CI/CD for automated testing
- Professional README with badges and comparison table
- `pytest.ini` configuration

### Changed
- Updated version to 1.0.0 across all files
- Enhanced documentation and examples

### Framework Highlights (v0.1-v1.0)

**v0.1** - Foundation
- Core Agent class with Ollama integration

**v0.2** - Tools & Decorator
- `@agent.tool` decorator with automatic schema generation

**v0.3** - Memory Management
- Memory ABC with InMemory and FileMemory

**v0.4** - Multi-Model Support
- Ollama, OpenAI, and Mistral providers

**v0.5** - Async-First Architecture
- Primary `agenerate()` and `arun()` methods
- Sync wrappers for backward compatibility

**v0.6** - Robust Loop Enhancement
- JSON auto-repair
- Infinite loop detection
- Tool timeout protection
- Structured logging

**v0.7** - TestClient (Killer Feature)
- MockModel for offline testing
- AgentTestClient with assertion helpers
- THE differentiator from competitors

**v0.8** - MCP Client Integration
- First Python framework with native MCP support
- stdio transport for local MCP servers
- Seamless tool discovery and integration

**v1.0** - Stabilization & Distribution
- Production-ready packaging
- CI/CD pipeline
- PyPI publishing

## Previous Versions

See [CHANGELOG history](CHANGELOG.md) for detailed version history v0.1-v0.8.

---

**AgentFlow v1.0.0** - Production-ready. Fully tested. Simply powerful.
