"""
AgentFlow MCP (Model Context Protocol) Client

This module provides MCP client functionality for connecting to MCP servers
and using their tools seamlessly within AgentFlow agents.

MCP Specification: https://modelcontextprotocol.io

Example:
    >>> from agentflow.mcp import MCPClient
    >>> 
    >>> # Connect to filesystem server
    >>> client = MCPClient(
    ...     transport="stdio",
    ...     command=["npx", "@modelcontextprotocol/server-filesystem", "/tmp"]
    ... )
    >>> await client.connect()
    >>> tools = await client.list_tools()
    >>> result = await client.call_tool("read_file", {"path": "/tmp/file.txt"})

Author: Hamadi Chaabani
License: MIT
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
import subprocess
import uuid


class MCPError(Exception):
    """Base exception for MCP errors."""
    pass


class MCPConnectionError(MCPError):
    """Raised when connection to MCP server fails."""
    pass


class MCPToolError(MCPError):
    """Raised when a tool call fails."""
    pass


class MCPClient:
    """
    Client for connecting to MCP (Model Context Protocol) servers.
    
    Currently supports stdio transport for local MCP servers.
    """
    
    def __init__(self, transport: str = "stdio", **kwargs):
        """
        Initialize MCP client.
        
        Args:
            transport: Transport type ("stdio" only for now).
            **kwargs: Transport-specific configuration.
                For stdio: command (List[str]) - Command to start server.
        """
        if transport != "stdio":
            raise ValueError(f"Unsupported transport: {transport}. Only 'stdio' is supported.")
        
        self.transport = transport
        self.config = kwargs
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self._tools_cache: Optional[List[Dict]] = None
        
    async def connect(self):
        """Connect to MCP server."""
        if self.transport == "stdio":
            await self._connect_stdio()
        else:
            raise MCPConnectionError(f"Unsupported transport: {self.transport}")
    
    async def _connect_stdio(self):
        """Connect via stdio transport."""
        command = self.config.get("command")
        if not command:
            raise MCPConnectionError("'command' is required for stdio transport")
        
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Send initialize request
            await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "agentflow",
                    "version": "0.8.0"
                }
            })
            
            # Send initialized notification
            await self._send_notification("notifications/initialized")
            
        except Exception as e:
            raise MCPConnectionError(f"Failed to start MCP server: {str(e)}") from e
    
    async def _send_request(self, method: str, params: Optional[Dict] = None) -> Dict:
        """Send JSON-RPC request and get response."""
        if not self.process:
            raise MCPConnectionError("Not connected to MCP server")
        
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        
        try:
            # Send request
            request_line = json.dumps(request) + "\n"
            self.process.stdin.write(request_line)
            self.process.stdin.flush()
            
            # Read response
            response_line = self.process.stdout.readline()
            if not response_line:
                raise MCPConnectionError("Server closed connection")
            
            response = json.loads(response_line)
            
            if "error" in response:
                raise MCPToolError(f"MCP error: {response['error']}")
            
            return response.get("result", {})
            
        except Exception as e:
            raise MCPConnectionError(f"Request failed: {str(e)}") from e
    
    async def _send_notification(self, method: str, params: Optional[Dict] = None):
        """Send JSON-RPC notification (no response expected)."""
        if not self.process:
            raise MCPConnectionError("Not connected to MCP server")
        
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }
        
        try:
            notification_line = json.dumps(notification) + "\n"
            self.process.stdin.write(notification_line)
            self.process.stdin.flush()
        except Exception as e:
            raise MCPConnectionError(f"Notification failed: {str(e)}") from e
    
    async def list_tools(self) -> List[Dict]:
        """
        List available tools from the MCP server.
        
        Returns:
            List of tool schemas.
        """
        if self._tools_cache is not None:
            return self._tools_cache
        
        result = await self._send_request("tools/list")
        tools = result.get("tools", [])
        self._tools_cache = tools
        return tools
    
    async def call_tool(self, name: str, arguments: Optional[Dict] = None) -> Any:
        """
        Call a tool on the MCP server.
        
        Args:
            name: Tool name.
            arguments: Tool arguments.
            
        Returns:
            Tool result.
        """
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments or {}
        })
        
        # Extract content from result
        content = result.get("content", [])
        if content and len(content) > 0:
            return content[0].get("text", "")
        
        return result
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            self.process = None
    
    def __repr__(self) -> str:
        status = "connected" if self.process else "disconnected"
        return f"MCPClient(transport='{self.transport}', status='{status}')"
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


__all__ = ["MCPClient", "MCPError", "MCPConnectionError", "MCPToolError"]
