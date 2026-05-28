"""
Core MCP server framework.

This module provides the base MCP server functionality including authentication,
configuration, session management, and response handling.

Uses lazy imports to avoid loading heavy dependencies (like gxl_filesystem)
when only lightweight utilities (like environment.py) are needed.
"""

__all__ = [
    "MCPServer",
    "validate_api_key",
    "is_authentication_required",
    "get_api_validator",
    "get_config",
    "validate_config",
    "SessionManager",
    "SandboxSession",
    "create_response_manager",
]


def __getattr__(name: str):
    """Lazy import to avoid loading heavy dependencies unnecessarily."""
    if name in ("validate_api_key", "is_authentication_required", "get_api_validator"):
        from .auth import get_api_validator, is_authentication_required, validate_api_key
        return {"validate_api_key": validate_api_key,
                "is_authentication_required": is_authentication_required,
                "get_api_validator": get_api_validator}[name]
    elif name == "MCPServer":
        from .base_server import MCPServer
        return MCPServer
    elif name in ("get_config", "validate_config"):
        from .config import get_config, validate_config
        return {"get_config": get_config, "validate_config": validate_config}[name]
    elif name == "create_response_manager":
        from .response_manager import create_response_manager
        return create_response_manager
    elif name in ("SessionManager", "SandboxSession"):
        from .session_manager import SandboxSession, SessionManager
        return {"SessionManager": SessionManager, "SandboxSession": SandboxSession}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
