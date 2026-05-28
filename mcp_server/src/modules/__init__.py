"""
MCP Server Tool Modules.

This package contains modular tool implementations that can be plugged into
the base MCP server. Each module is self-contained and provides its own tools,
handlers, and configuration.
"""

from .base import ToolModule

__all__ = ["ToolModule"]
