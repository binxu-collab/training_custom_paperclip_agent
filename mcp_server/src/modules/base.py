"""
Base module interface for MCP server tool modules.

This module defines the abstract base class that all tool modules must implement
to be compatible with the modular MCP server architecture.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any

from mcp.types import Tool


class ToolCategory(str, Enum):
    """Categories for tool classification in the UI.
    
    Used to group tools by their primary function for display
    in execution traces (e.g., "Performed 3 searches, 2 queries").
    """
    SEARCH = "search"   # Full-text search, pattern matching, BLAST
    QUERY = "query"     # Direct data retrieval, lookups, metadata
    EXECUTE = "execute" # Code execution, compilation
    FILES = "files"     # File operations (upload, read, create, delete)
    AGENTS = "agents"   # Sub-agent invocations

logger = logging.getLogger(__name__)


class ToolModule(ABC):
    """Abstract base class for MCP tool modules.

    All tool modules must inherit from this class and implement the required methods.
    This ensures a consistent interface for registering and using tool modules.
    """

    def __init__(self):
        """Initialize the tool module."""
        self.config = None
        self.session_manager = None
        self._initialized = False
        self._ready = False

    @abstractmethod
    def get_name(self) -> str:
        """Return the unique name of this module.

        Returns:
            str: A unique identifier for this module (e.g., "database", "search").
        """
        pass

    @abstractmethod
    def get_tools(self) -> list[Tool]:
        """Return list of MCP tools provided by this module.

        Returns:
            List[Tool]: List of MCP Tool objects that this module provides.
        """
        pass

    @abstractmethod
    def get_handlers(self) -> dict[str, Callable]:
        """Return dictionary mapping tool names to their handler functions.

        Returns:
            Dict[str, Callable]: Dictionary where keys are tool names and values
                                are async functions that handle tool calls.
        """
        pass

    def initialize(self, config: Any, session_manager: Any) -> None:
        """Initialize the module with configuration and session manager.

        Args:
            config: Server configuration object
            session_manager: Session manager instance for handling user sessions
        """
        self.config = config
        self.session_manager = session_manager
        self._initialized = True
        logger.info(f"Initialized module: {self.get_name()}")

    def is_initialized(self) -> bool:
        """Check if the module has been initialized."""
        return self._initialized

    def is_ready(self) -> bool:
        """Check if the module is ready to handle requests."""
        return self._ready

    async def ensure_ready(self) -> None:
        """Ensure the module is ready for use.

        This method is called before the first tool request and can be used
        for async initialization tasks like loading models, connecting to APIs, etc.
        """
        if not self._initialized:
            raise RuntimeError(f"Module {self.get_name()} must be initialized before ensuring ready")

        if not self._ready:
            await self._async_initialize()
            self._ready = True
            logger.info(f"Module {self.get_name()} is ready")

    async def _async_initialize(self) -> None:
        """Override this method for async initialization tasks.

        This is called by ensure_ready() and should contain any async setup
        that needs to happen before the module can handle requests.

        Note: This is not an abstract method because it's optional to override.
        """
        # Default implementation does nothing - subclasses can override if needed

    def cleanup(self) -> None:
        """Clean up module resources.

        This method is called when the module is being unregistered or
        the server is shutting down. Override to clean up resources.
        """
        logger.info(f"Cleaning up module: {self.get_name()}")

    def get_description(self) -> str:
        """Return a human-readable description of this module.

        Returns:
            str: A description of what this module provides.
        """
        return f"Tool module: {self.get_name()}"

    def get_version(self) -> str:
        """Return the version of this module.

        Returns:
            str: Version string for this module.
        """
        return "1.0.0"

    def get_dependencies(self) -> list[str]:
        """Return list of module names this module depends on.

        Returns:
            List[str]: List of module names that must be loaded before this one.
        """
        return []

    def supports_tool(self, tool_name: str) -> bool:
        """Check if this module supports a specific tool.

        Args:
            tool_name: Name of the tool to check

        Returns:
            bool: True if this module provides the specified tool
        """
        return tool_name in self.get_handlers()

    def validate_config(self) -> list[str]:
        """Validate the module's configuration.

        Returns:
            List[str]: List of validation error messages, empty if valid.
        """
        if not self._initialized:
            return ["Module not initialized"]
        return []


class SimpleToolModule(ToolModule):
    """Simple implementation of ToolModule for basic use cases.

    This class provides a concrete implementation that can be used directly
    for simple modules that just need to register tools and handlers.
    """

    def __init__(self, name: str, tools: list[Tool], handlers: dict[str, Callable]):
        """Initialize a simple tool module.

        Args:
            name: Unique name for this module
            tools: List of MCP Tool objects
            handlers: Dictionary mapping tool names to handler functions
        """
        super().__init__()
        self._name = name
        self._tools = tools
        self._handlers = handlers

    def get_name(self) -> str:
        return self._name

    def get_tools(self) -> list[Tool]:
        return self._tools

    def get_handlers(self) -> dict[str, Callable]:
        return self._handlers
