"""
AgentFlow - A minimalist Python framework for building AI agents

Setup configuration for PyPI distribution.
"""

from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agentflow-ai",
    version="1.0.1",
    author="Hamadi Chaabani",
    author_email="chaabani.hammadi@gmail.com",
    description="Minimalist Python framework for building AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ia-nova-labs/agentflow",
    py_modules=["agentflow", "mcp", "testing"],
    install_requires=[
        "httpx>=0.27.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="ai agents llm ollama openai mistral mcp testing async",
    project_urls={
        "Documentation": "https://github.com/ia-nova-labs/agentflow-docs",
        "Examples": "https://github.com/ia-nova-labs/agentflow-examples",
        "Source": "https://github.com/ia-nova-labs/agentflow",
        "Bug Reports": "https://github.com/ia-nova-labs/agentflow/issues",
    },
)
