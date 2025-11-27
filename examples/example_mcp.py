"""
AgentFlow MCP (Model Context Protocol) Example

This example demonstrates how to use MCP servers with AgentFlow agents.

MCP allows agents to use tools from external servers, expanding capabilities
beyond local Python functions.

Requirements:
- Node.js and npm installed
- MCP server packages (installed automatically via npx)
- Ollama running locally (ollama serve)

Example MCP Servers:
- @modelcontextprotocol/server-filesystem - File operations
- @modelcontextprotocol/server-git - Git operations
- @modelcontextprotocol/server-sqlite - SQLite database
"""

import sys
import os
import asyncio

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow import Agent
from agentflow.mcp import MCPClient


async def example_filesystem_server():
    """Example 1: Using filesystem MCP server."""
    print("=" * 60)
    print("Example 1: Filesystem MCP Server")
    print("=" * 60)
    
    print("\n📁 Starting filesystem MCP server...")
    print("   Server can read/write files in /tmp directory\n")
    
    # Create agent
    agent = Agent(model="llama3", debug=False)
    
    # Connect to filesystem MCP server
    # This server provides file operations for /tmp directory
    mcp_client = MCPClient(
        transport="stdio",
        command=[
            "npx",
            "@modelcontextprotocol/server-filesystem",
            "/tmp"
        ]
    )
    
    try:
        await mcp_client.connect()
        print("✅ Connected to filesystem server")
        
        # Discover and register MCP tools
        await agent.add_mcp_tools(mcp_client)
        print(f"✅ Registered {len(agent._tools)} tools from MCP server\n")
        
        # List available tools
        print("Available tools:")
        for tool_name in agent._tools.keys():
            print(f"  • {tool_name}")
        print()
        
        # Use agent with MCP tools
        print("🤖 Agent using MCP filesystem tools...\n")
        
        # Create a test file
        response = await agent.arun(
            "Create a file called /tmp/hello.txt with the content 'Hello from AgentFlow MCP!'"
        )
        print(f"Response: {response}\n")
        
        # Read the file back
        response = await agent.arun(
            "Read the content of /tmp/hello.txt"
        )
        print(f"Response: {response}\n")
        
    finally:
        await mcp_client.disconnect()
        print("✅ Disconnected from MCP server")
    
    print()


async def example_git_server():
    """Example 2: Using Git MCP server."""
    print("=" * 60)
    print("Example 2: Git MCP Server")
    print("=" * 60)
    
    print("\n🔧 Starting Git MCP server...")
    print("   Server provides Git operations\n")
    
    agent = Agent(model="llama3", debug=False)
    
    # Connect to Git MCP server
    mcp_client = MCPClient(
        transport="stdio",
        command=[
            "npx",
            "@modelcontextprotocol/server-git",
            "/tmp"
        ]
    )
    
    try:
        await mcp_client.connect()
        print("✅ Connected to Git server")
        
        await agent.add_mcp_tools(mcp_client)
        print(f"✅ Registered {len(agent._tools)} Git tools\n")
        
        # List tools
        print("Available Git tools:")
        for tool_name in agent._tools.keys():
            print(f"  • {tool_name}")
        print()
        
        # Use Git operations
        print("🤖 Agent using MCP Git tools...\n")
        
        response = await agent.arun(
            "Show me the Git status of the repository in /tmp"
        )
        print(f"Response: {response}\n")
        
    except Exception as e:
        print(f"⚠️  Note: {str(e)}")
        print("   (Git server requires a valid Git repository)")
    finally:
        await mcp_client.disconnect()
        print("✅ Disconnected from MCP server")
    
    print()


async def example_mixed_tools():
    """Example 3: Combining local tools with MCP tools."""
    print("=" * 60)
    print("Example 3: Mixed Local + MCP Tools")
    print("=" * 60)
    
    print("\n🔀 Combining local Python tools with MCP tools...\n")
    
    agent = Agent(model="llama3", debug=False)
    
    # Add local Python tool
    @agent.tool
    def calculate(expression: str) -> float:
        """Calculate a mathematical expression."""
        try:
            return eval(expression)
        except:
            return 0.0
    
    print("✅ Registered local tool: calculate")
    
    # Add MCP filesystem tools
    mcp_client = MCPClient(
        transport="stdio",
        command=[
            "npx",
            "@modelcontextprotocol/server-filesystem",
            "/tmp"
        ]
    )
    
    try:
        await mcp_client.connect()
        await agent.add_mcp_tools(mcp_client)
        print(f"✅ Registered MCP tools from filesystem server")
        
        total_tools = len(agent._tools)
        print(f"\n📊 Total tools available: {total_tools}")
        print("   • 1 local Python tool")
        print(f"   • {total_tools - 1} MCP tools\n")
        
        # Use both types of tools
        print("🤖 Agent using BOTH local and MCP tools...\n")
        
        response = await agent.arun(
            "Calculate 25 * 17, then save the result to /tmp/calculation.txt"
        )
        print(f"Response: {response}\n")
        
    finally:
        await mcp_client.disconnect()
        print("✅ Disconnected from MCP server")
    
    print()


async def example_mcp_context_manager():
    """Example 4: Using MCP client as async context manager."""
    print("=" * 60)
    print("Example 4: MCP Client Context Manager")
    print("=" * 60)
    
    print("\n🔄 Using async context manager for automatic cleanup...\n")
    
    agent = Agent(model="llama3", debug=False)
    
    # Use context manager for automatic connect/disconnect
    async with MCPClient(
        transport="stdio",
        command=["npx", "@modelcontextprotocol/server-filesystem", "/tmp"]
    ) as mcp_client:
        
        print("✅ Connected (automatic)")
        
        await agent.add_mcp_tools(mcp_client)
        print(f"✅ Registered {len(agent._tools)} tools\n")
        
        print("🤖 Agent working with MCP tools...\n")
        
        response = await agent.arun(
            "List all files in /tmp directory"
        )
        print(f"Response: {response[:200]}...\n")
    
    print("✅ Disconnected (automatic)")
    print()


async def main():
    """Run all MCP examples."""
    print("\n" + "🔌 AgentFlow MCP Examples (v0.8)".center(60))
    print()
    
    print("ℹ️  These examples use official MCP servers from Anthropic.")
    print("   Servers are downloaded automatically via npx.\n")
    
    try:
        await example_filesystem_server()
        print("-" * 60 + "\n")
        
        await example_git_server()
        print("-" * 60 + "\n")
        
        await example_mixed_tools()
        print("-" * 60 + "\n")
        
        await example_mcp_context_manager()
        
        print("=" * 60)
        print("✅ All MCP examples completed!")
        print("=" * 60)
        print("\n💡 Key Features:")
        print("   • Connect to MCP servers (stdio transport)")
        print("   • Automatic tool discovery")
        print("   • Seamless integration with agents")
        print("   • Mix local + MCP tools")
        print("   • Async context manager support")
        print("\n🚀 AgentFlow = First Python framework with native MCP!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
