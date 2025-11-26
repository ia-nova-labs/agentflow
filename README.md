# AgentFlow

**A minimalist Python framework for building AI agents**

AgentFlow is designed with one goal: make creating AI agents as simple as creating a Flask app. Inspired by Flask v0.1's philosophy of "readable in 15 minutes," AgentFlow provides a clean, explicit API for building intelligent agents.

## 🎯 Philosophy

- **Minimalist**: Start with a single file, grow as you need
- **Explicit > Magic**: Clear APIs over hidden complexity
- **Testable**: Easy to test, easy to debug
- **Flexible**: Adapt to your needs, not ours

## 🚀 Quick Start

```python
from agentflow import Agent

# Create an agent
agent = Agent(model="llama3")

# Run it
response = agent.run("Hello, who are you?")
print(response)
```

That's it. No boilerplate, no configuration files, no magic.

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- [Ollama](https://ollama.ai/) running locally

### Install

```bash
pip install -r requirements.txt
```

## 🎓 Learn More

Check out the [Getting Started Guide](docs/getting_started.md) for a complete walkthrough.

## 🗺️ Roadmap

AgentFlow is under active development. Here's what's coming:

- **v0.1** (Current): Minimal agent with Ollama support ✅
- **v0.2**: Tool decorator and function calling
- **v0.3**: Memory management (in-memory & file-based)
- **v0.4**: Multi-model support (OpenAI, Mistral, Ollama)
- **v0.5**: Enhanced think → act loop
- **v0.6**: MCP client integration
- **v0.7**: Multi-agent workflows
- **v1.0**: Production-ready with full documentation

## 📖 Documentation

- [Getting Started](docs/getting_started.md)
- [API Reference](docs/api_reference.md) *(coming soon)*
- [Examples](examples/)

## 🤝 Contributing

AgentFlow is open-source and contributions are welcome! This is an early-stage project, so things may change rapidly.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with inspiration from:
- Flask's minimalist design philosophy
- The simplicity of early web frameworks
- The belief that AI tooling should be accessible to everyone

---

**Version**: 0.1.0  
**Author**: Hamadi Chaabani  
**Status**: Alpha - Active Development
