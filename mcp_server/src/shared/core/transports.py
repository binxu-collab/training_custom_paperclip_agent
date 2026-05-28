"""
Transport implementations for MCP servers.

This module provides transport layer implementations for serving MCP servers
via different protocols (stdio, HTTP).
"""

import json
import logging
import os
import sys
import traceback
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any

from .auth import get_rate_limit_info, is_api_key_auth_enabled, is_authentication_required, validate_api_key
from .base_server import MCPServer
from .port_utils import kill_process_on_port

logger = logging.getLogger(__name__)

# URL of the OAuth authorization server (unified backend with Firebase login).
# When set, MCP OAuth discovery points here instead of using auto-approve OAuth.
_OAUTH_SERVER_URL = os.environ.get("OAUTH_SERVER_URL", "")

# Firebase project ID for verifying Bearer tokens as Firebase ID tokens.
_FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "")

# Cache for the google.auth HTTP transport (created lazily)
_google_request = None


def _verify_firebase_token(token: str) -> bool:
    """Verify a Bearer token as a Firebase ID token.

    Returns True if the token is a valid Firebase ID token.
    Falls back to False (not a Firebase token) so other auth
    methods can be tried.
    """
    if not _FIREBASE_PROJECT_ID:
        return False
    try:
        global _google_request
        if _google_request is None:
            from google.auth.transport import requests as google_requests
            _google_request = google_requests.Request()
        from google.oauth2 import id_token
        id_token.verify_firebase_token(
            token, _google_request, audience=_FIREBASE_PROJECT_ID
        )
        return True
    except Exception:
        return False


# ============================================================================
# Transport Abstraction
# ============================================================================


class ServerTransport(ABC):
    """Abstract base class for server transport implementations."""

    def __init__(self, server: MCPServer):
        self.server = server

    @abstractmethod
    async def run(self):
        """Run the server with this transport."""
        pass


# ============================================================================
# Stdio Transport
# ============================================================================


class StdioTransport(ServerTransport):
    """stdio transport for local MCP clients (Claude Desktop, etc.)."""

    def __init__(self, server: MCPServer):
        super().__init__(server)
        logger.info("Initializing stdio transport")

    async def run(self):
        """Run the server using stdio transport."""
        # Check authentication for stdio MCP server
        if is_authentication_required():
            api_key = os.getenv("MCP_API_KEY")
            if not api_key:
                logger.error(
                    "Authentication required: MCP_API_KEY environment variable not set"
                )
                logger.error(
                    "Please set MCP_API_KEY environment variable with a valid API key"
                )
                sys.exit(1)

            if not validate_api_key(api_key, "stdio-mcp"):
                logger.error("Authentication failed: Invalid MCP_API_KEY")
                sys.exit(1)

            logger.info("Authentication successful for stdio MCP server")
        else:
            logger.info("Authentication disabled for stdio MCP server")

        # Import MCP stdio (only needed for stdio transport)
        import mcp.server.stdio

        try:
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await self.server.server.run(
                    read_stream,
                    write_stream,
                    self.server.server.create_initialization_options(),
                )
        finally:
            # Clean up modules
            self.server.cleanup()


# ============================================================================
# HTTP Transport
# ============================================================================


class HTTPTransport(ServerTransport):
    """FastAPI-based HTTP transport for Cloud Run deployment."""

    def __init__(
        self,
        server: MCPServer,
        host: str = "0.0.0.0",
        port: int = 8080,
        server_name: str = "MCP Server",
        server_version: str = "1.0.0",
        custom_routes_builder: Any = None,
        path_prefix: str = "",
    ):
        super().__init__(server)
        self.host = host
        self.port = port
        self.server_name = server_name
        self.server_version = server_version
        self.custom_routes_builder = custom_routes_builder
        self.path_prefix = path_prefix.rstrip("/") if path_prefix else ""
        self.app = None
        self._oauth_codes: dict[str, dict] = {}
        self._oauth_tokens: set[str] = set()
        logger.info(f"Initializing HTTP transport on {host}:{port}")
        if self.path_prefix:
            logger.info(f"Path prefix: {self.path_prefix}")
        if custom_routes_builder:
            logger.info("Custom routes builder registered")

    def _create_app(self):
        """Create and configure the FastAPI application."""
        from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel

        # Force unbuffered output for Cloud Run
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Manage application lifespan - startup and shutdown."""
            # Startup
            logger.info("✓ HTTP server initialized with existing MCP server")
            if hasattr(self.server, "on_startup"):
                try:
                    await self.server.on_startup()
                except Exception as e:
                    logger.warning(f"Server on_startup hook failed: {e}")
            yield
            # Shutdown
            try:
                self.server.cleanup()
                logger.info("MCP server cleanup completed")
            except Exception as e:
                logger.error(f"Error during server cleanup: {e}")

        # Initialize FastAPI app with lifespan
        app = FastAPI(
            title=self.server_name,
            description=f"HTTP wrapper for the {self.server_name}",
            version=self.server_version,
            lifespan=lifespan,
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
        )

        # Add CORS middleware — MCP servers are called server-to-server (no browser).
        # ALLOWED_ORIGINS restricts which origins may make credentialed requests.
        # An empty list disables cross-origin browser access entirely.
        _raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
        _allowed_origins: list[str] = [
            o.strip() for o in _raw_origins.split(",") if o.strip()
        ]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_allowed_origins,
            allow_credentials=bool(_allowed_origins),
            allow_methods=["GET", "POST", "OPTIONS", "HEAD", "DELETE"],
            allow_headers=[
                "Authorization", "Content-Type", "X-API-Key",
                "Mcp-Session-Id", "X-Session-ID", "X-Connection-Id",
            ],
            expose_headers=["Mcp-Session-Id"],
        )

        # Pydantic models for request/response
        from pydantic import field_validator

        class ToolRequest(BaseModel):
            name: str
            arguments: dict[str, Any] = {}

            @field_validator("arguments", mode="before")
            @classmethod
            def convert_empty_string_to_dict(cls, v):
                """Convert empty string arguments to empty dict."""
                if v == "" or v is None:
                    return {}
                return v

        class MCPRequest(BaseModel):
            jsonrpc: str = "2.0"
            id: Any
            method: str
            params: dict[str, Any] = {}

        # Helper functions
        def serialize_tool_with_meta(tool):
            """Serialize a Tool object, dropping null fields for spec compliance."""
            tool_dict = tool.model_dump(exclude_none=True)
            if hasattr(tool, "_meta") and tool._meta is not None:
                tool_dict["_meta"] = tool._meta
            return tool_dict

        _api_key_auth_ok = is_api_key_auth_enabled()

        def authenticate_request(
            x_api_key: str = Header(None, alias="X-API-Key"),
            authorization: str = Header(None),
        ) -> str:
            """FastAPI dependency to authenticate requests."""
            if not is_authentication_required():
                return x_api_key or "no-auth-required"

            # Try X-API-Key header (dev/staging only)
            if x_api_key and _api_key_auth_ok:
                if validate_api_key(x_api_key, "http-request"):
                    return x_api_key
                raise HTTPException(status_code=403, detail="Invalid API key")

            # Try Authorization: Bearer header (OAuth / Firebase token)
            if authorization and authorization.lower().startswith("bearer "):
                token = authorization[7:].strip()
                if token in self._oauth_tokens:
                    return token
                if _verify_firebase_token(token):
                    return token
                if _api_key_auth_ok and validate_api_key(token, "http-bearer"):
                    return token
                raise HTTPException(status_code=401, detail="Invalid token")

            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        async def handle_mcp_request(
            jsonrpc: str,
            id: Any,
            method: str,
            params: dict[str, Any],
            connection_id: str | None = None,
            agent_id: str | None = None,
            agent_config_path: str | None = None,
        ):
            """Handle MCP JSON-RPC requests."""
            try:
                if method == "initialize":
                    # Validate required params
                    if not params:
                        return {
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {
                                "code": -32602,
                                "message": "initialize params are required",
                            },
                        }

                    # Get requested protocol version from client
                    client_protocol_version = params.get(
                        "protocolVersion", "2024-11-05"
                    )

                    # Server supported versions (latest first)
                    supported_versions = ["2025-03-26", "2024-11-05"]

                    # Negotiate version: if client version is supported, use it; otherwise use latest
                    if client_protocol_version in supported_versions:
                        negotiated_version = client_protocol_version
                    else:
                        negotiated_version = supported_versions[
                            0
                        ]  # Return latest supported
                        logger.info(
                            f"Client requested unsupported version {client_protocol_version}, "
                            f"negotiating to {negotiated_version}"
                        )

                    # SessionId is optional - generate one if not provided
                    requested_session_id = params.get("sessionId")

                    # Create or retrieve session with the specified ID (or generate new one)
                    session_id = self.server.session_manager.get_or_create_session(
                        session_id=requested_session_id
                    )
                    logger.info(f"Initialized session {session_id}")

                    result = {
                        "protocolVersion": negotiated_version,
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": self.server_name,
                            "version": self.server_version,
                        },
                        "sessionId": session_id,
                    }
                    instructions = self.server.get_instructions()
                    if instructions:
                        result["instructions"] = instructions
                    return {
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": result,
                    }

                elif method == "notifications/initialized":
                    # Per JSON-RPC 2.0 spec: notifications do not require or expect a response
                    logger.info(
                        "Client sent initialized notification - ready for operations"
                    )
                    return None  # No response for notifications

                elif method == "ping":
                    return {"jsonrpc": "2.0", "id": id, "result": {}}

                elif method == "tools/list":
                    tools = await self.server.get_tools_list()
                    serialized_tools = [
                        serialize_tool_with_meta(tool) for tool in tools
                    ]
                    return {
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {"tools": serialized_tools},
                    }

                elif method == "tools/call":
                    import time

                    call_start = time.perf_counter()

                    tool_name = params.get("name")
                    arguments = params.get("arguments", {})

                    if not tool_name:
                        return {
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {
                                "code": -32602,
                                "message": "Tool name is required.",
                            },
                        }

                    if not connection_id:
                        return {
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {
                                "code": -32602,
                                "message": "Session ID is required.",
                            },
                        }

                    # Ensure session is registered for timing tracking
                    if hasattr(self.server, "session_manager"):
                        self.server.session_manager.get_or_create_session(
                            session_id=connection_id
                        )

                    try:
                        result = await self.server.call_tool_by_name(
                            tool_name,
                            arguments,
                            connection_id,
                            agent_id=agent_id,
                            agent_config_path=agent_config_path,
                        )
                    except Exception as e:
                        elapsed_ms = int((time.perf_counter() - call_start) * 1000)
                        return {
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {
                                "code": -32603,
                                "message": f"[⏱️ {elapsed_ms}ms] {str(e)}",
                            },
                        }

                    # Calculate elapsed time for this call
                    elapsed_ms = int((time.perf_counter() - call_start) * 1000)
                    if elapsed_ms < 1000:
                        call_time_str = f"{elapsed_ms}ms"
                    else:
                        call_time_str = f"{elapsed_ms/1000:.1f}s"

                    # Get session elapsed time (time since session created)
                    session_elapsed_str = ""
                    if hasattr(self.server, "session_manager"):
                        session = self.server.session_manager.get_session(connection_id)
                        if session:
                            session_elapsed_s = time.time() - session.created_at
                            if session_elapsed_s < 60:
                                session_elapsed_str = f"{int(session_elapsed_s)}s"
                            else:
                                mins = int(session_elapsed_s // 60)
                                secs = int(session_elapsed_s % 60)
                                session_elapsed_str = f"{mins}m{secs}s"

                    # Handle different response types for MCP format
                    content = "No result returned"
                    if result:
                        if hasattr(result, "content") and result.content:
                            content = (
                                result.content[0].text
                                if result.content
                                else "No result returned"
                            )
                        elif (
                            isinstance(result, list)
                            and len(result) > 0
                            and hasattr(result[0], "text")
                        ):
                            content = result[0].text
                        else:
                            content = str(result)

                    # Estimate tokens in response (chars/4)
                    est_tokens = len(content) // 4

                    # Update cumulative tokens and format string
                    cumulative_tokens = est_tokens
                    if hasattr(self.server, "session_manager") and session:
                        session.total_tokens += est_tokens
                        cumulative_tokens = session.total_tokens

                    # Format: "turn_tokens / cumulative_tokens tok"
                    if cumulative_tokens >= 1000:
                        tokens_str = f"{est_tokens} / {cumulative_tokens//1000}K tok"
                    else:
                        tokens_str = f"{est_tokens} / {cumulative_tokens} tok"

                    # Format timing prefix
                    if session_elapsed_str:
                        timing_prefix = f"[⏱️ {call_time_str} | {tokens_str} | session: {session_elapsed_str}]"
                    else:
                        timing_prefix = f"[⏱️ {call_time_str} | {tokens_str}]"

                    # Banner-first: prepend timing for text content; for JSON
                    # content (e.g. meta.json), inject as _timing key so jq
                    # piping still works. Mirrors the /tools/call endpoint
                    # below and matches the format SFT data was generated with.
                    try:
                        import json as json_mod

                        content_json = json_mod.loads(content)
                        if isinstance(content_json, dict):
                            result_with_timing = {
                                "_timing": timing_prefix,
                                **content_json,
                            }
                            final_content = json_mod.dumps(result_with_timing)
                        else:
                            final_content = timing_prefix + " " + content
                    except (json_mod.JSONDecodeError, TypeError):
                        final_content = timing_prefix + " " + content

                    return {
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "content": [{"type": "text", "text": final_content}]
                        },
                    }

                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {
                            "code": -32601,
                            "message": f"Method '{method}' not found",
                        },
                    }

            except Exception as e:
                logger.error(f"Error handling MCP request: {e}")
                return {
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32603, "message": str(e)},
                }

        # ============================================================================
        # Routes
        # ============================================================================

        @app.get("/")
        async def root():
            """Root endpoint for basic connectivity check."""
            return {
                "message": f"{self.server_name} is running",
                "status": "healthy",
                "version": self.server_version,
            }

        @app.get("/health")
        async def health_check():
            """Health check endpoint for Cloud Run."""
            return {
                "status": "healthy",
                "server_initialized": self.server is not None,
                "authentication_required": is_authentication_required(),
            }

        @app.get("/auth/status")
        async def auth_status(api_key: str = Depends(authenticate_request)):
            """Check authentication status and rate limits."""
            rate_info = (
                get_rate_limit_info(api_key) if api_key != "no-auth-required" else None
            )
            return {
                "authenticated": True,
                "api_key_required": is_authentication_required(),
                "rate_limit": rate_info,
            }

        # MCP initialize endpoints
        @app.get("/initialize")
        async def get_initialize():
            """GET version of MCP initialize."""
            return await handle_mcp_request("2.0", 1, "initialize", {})

        @app.post("/initialize")
        async def post_initialize():
            """POST version of MCP initialize."""
            return await handle_mcp_request("2.0", 1, "initialize", {})

        # MCP ping endpoints
        @app.get("/ping")
        async def get_ping():
            """GET version of MCP ping."""
            return await handle_mcp_request("2.0", 1, "ping", {})

        @app.post("/ping")
        async def post_ping():
            """POST version of MCP ping."""
            return await handle_mcp_request("2.0", 1, "ping", {})

        # Tools list endpoints
        @app.get("/tools/list")
        async def get_tools_list(api_key: str = Depends(authenticate_request)):
            """GET version of tools/list."""
            return await handle_mcp_request("2.0", 1, "tools/list", {})

        @app.post("/tools/list")
        async def post_tools_list(api_key: str = Depends(authenticate_request)):
            """POST version of tools/list."""
            return await handle_mcp_request("2.0", 1, "tools/list", {})

        @app.get("/tools")
        async def list_tools(api_key: str = Depends(authenticate_request)):
            """List all available tools."""
            try:
                tools = await self.server.get_tools_list()
                serialized_tools = [serialize_tool_with_meta(tool) for tool in tools]
                return {"tools": serialized_tools}
            except Exception as e:
                logger.error(f"Error listing tools: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/tools")
        async def post_list_tools(api_key: str = Depends(authenticate_request)):
            """POST version of list all available tools."""
            try:
                tools = await self.server.get_tools_list()
                serialized_tools = [serialize_tool_with_meta(tool) for tool in tools]
                return {"tools": serialized_tools}
            except Exception as e:
                logger.error(f"Error listing tools: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Tools call endpoints
        @app.get("/tools/call")
        async def get_call_tool(
            name: str = Query(..., description="Tool name to call"),
            api_key: str = Depends(authenticate_request),
            x_connection_id: str = Header(None, alias="X-Connection-Id"),
            x_session_id: str = Header(None, alias="X-Session-ID"),
            x_agent_id: str = Header(None, alias="X-Agent-ID"),
            x_agent_config_path: str | None = Header(None, alias="X-Agent-Config-Path"),
            **kwargs,
        ):
            """GET version of tools/call with query parameters."""
            # Build arguments from query parameters
            arguments = {
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "name",
                    "api_key",
                    "x_connection_id",
                    "x_session_id",
                    "x_agent_id",
                    "x_agent_config_path",
                ]
            }

            # Prefer X-Session-ID (from ToolDispatcher) over X-Connection-Id
            connection_id = x_session_id or x_connection_id
            if not connection_id:
                raise HTTPException(
                    status_code=400,
                    detail="Session ID is required. Please provide X-Session-ID or X-Connection-Id header.",
                )

            logger.info(
                f"🔧 GET Tool call '{name}' | Connection: {connection_id} | X-Session-ID: {x_session_id} | X-Agent-ID: {x_agent_id}"
            )

            return await handle_mcp_request(
                "2.0",
                1,
                "tools/call",
                {"name": name, "arguments": arguments},
                connection_id,
                x_agent_id,
                x_agent_config_path,
            )

        @app.post("/tools/call")
        async def call_tool(
            request: ToolRequest,
            api_key: str = Depends(authenticate_request),
            x_connection_id: str = Header(None, alias="X-Connection-Id"),
            x_session_id: str = Header(None, alias="X-Session-ID"),
            x_agent_id: str = Header(None, alias="X-Agent-ID"),
            x_agent_config_path: str | None = Header(None, alias="X-Agent-Config-Path"),
            x_paper_uuid: str = Header(None, alias="X-Paper-UUID"),
        ):
            """Call a specific tool with arguments."""
            import time

            start_time = time.perf_counter()

            try:
                # Prefer X-Session-ID (from ToolDispatcher) over X-Connection-Id
                connection_id = x_session_id or x_connection_id
                if not connection_id:
                    raise HTTPException(
                        status_code=400,
                        detail="Session ID is required. Please provide in header.",
                    )

                # Resolve paper_uuid: use header if sent, otherwise fall back to stored session value
                paper_uuid = x_paper_uuid
                if hasattr(self.server, "session_manager"):
                    self.server.session_manager.get_or_create_session(
                        session_id=connection_id,
                        paper_uuid=paper_uuid,
                    )
                    if not paper_uuid:
                        session = self.server.session_manager.get_session(connection_id)
                        if session and session.paper_uuid:
                            paper_uuid = session.paper_uuid

                logger.info(
                    f"🔧 Tool call '{request.name}' | Connection: {connection_id} | X-Agent-ID: {x_agent_id} | X-Paper-UUID: {paper_uuid}"
                )

                result = await self.server.call_tool_by_name(
                    request.name,
                    request.arguments,
                    connection_id,
                    api_key=api_key,
                    agent_id=x_agent_id,
                    agent_config_path=x_agent_config_path,
                    paper_uuid=paper_uuid,
                )

                # Calculate elapsed time for this call
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)

                # Format call timing
                if elapsed_ms < 1000:
                    call_time_str = f"{elapsed_ms}ms"
                else:
                    call_time_str = f"{elapsed_ms/1000:.1f}s"

                # Get session elapsed time (time since session created)
                session_elapsed_str = ""
                if hasattr(self.server, "session_manager"):
                    session = self.server.session_manager.get_session(connection_id)
                    if session:
                        session_elapsed_s = time.time() - session.created_at
                        if session_elapsed_s < 60:
                            session_elapsed_str = f"{int(session_elapsed_s)}s"
                        else:
                            mins = int(session_elapsed_s // 60)
                            secs = int(session_elapsed_s % 60)
                            session_elapsed_str = f"{mins}m{secs}s"

                # Handle different response types and get content
                if result:
                    if hasattr(result, "content") and result.content:
                        content = (
                            result.content[0].text
                            if result.content
                            else "No result returned"
                        )
                    elif (
                        isinstance(result, list)
                        and len(result) > 0
                        and hasattr(result[0], "text")
                    ):
                        content = result[0].text
                    else:
                        content = str(result)
                else:
                    content = "No result returned"

                # Estimate tokens in response (chars/4)
                est_tokens = len(content) // 4

                # Update cumulative tokens and format string
                cumulative_tokens = est_tokens
                if hasattr(self.server, "session_manager") and session:
                    session.total_tokens += est_tokens
                    cumulative_tokens = session.total_tokens

                # Format: "turn_tokens / cumulative_tokens tok"
                if cumulative_tokens >= 1000:
                    tokens_str = f"{est_tokens} / {cumulative_tokens//1000}K tok"
                else:
                    tokens_str = f"{est_tokens} / {cumulative_tokens} tok"

                # Format timing prefix: [call_time | tokens | session_elapsed]
                if session_elapsed_str:
                    timing_prefix = f"[⏱️ {call_time_str} | {tokens_str} | session: {session_elapsed_str}]"
                else:
                    timing_prefix = f"[⏱️ {call_time_str} | {tokens_str}]"

                # If content is JSON, inject timing inside it; otherwise prepend
                try:
                    import json as json_mod

                    content_json = json_mod.loads(content)
                    if isinstance(content_json, dict):
                        # Inject timing as first key for visibility
                        result_with_timing = {"_timing": timing_prefix, **content_json}
                        return {"result": json_mod.dumps(result_with_timing)}
                except (json_mod.JSONDecodeError, TypeError):
                    pass

                # Fallback: prepend timing for non-JSON content
                return {"result": timing_prefix + " " + content}

            except Exception as e:
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                logger.error(f"Error calling tool {request.name}: {e}")
                raise HTTPException(
                    status_code=500, detail=f"[⏱️ {elapsed_ms}ms] {str(e)}"
                )

        # MCP v1 endpoints
        @app.post("/mcp/v1/initialize")
        async def mcp_v1_initialize(
            request: Request,
            x_connection_id: str = Header(None, alias="X-Connection-Id"),
        ):
            """Handle MCP v1 initialize endpoint."""
            try:
                body = await request.json()
                return await handle_mcp_request(
                    body.get("jsonrpc", "2.0"),
                    body.get("id", 1),
                    "initialize",
                    body.get("params", {}),
                    x_connection_id,
                )
            except Exception as e:
                logger.error(f"Error in MCP v1 initialize: {e}")
                return {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": -32700, "message": "Parse error"},
                }

        @app.post("/mcp/v1/tools/list")
        async def mcp_v1_tools_list(
            request: Request, api_key: str = Depends(authenticate_request)
        ):
            """Handle MCP v1 tools/list endpoint."""
            try:
                body = await request.json()
                return await handle_mcp_request(
                    body.get("jsonrpc", "2.0"),
                    body.get("id", 1),
                    "tools/list",
                    body.get("params", {}),
                )
            except Exception as e:
                logger.error(f"Error in MCP v1 tools/list: {e}")
                return {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": -32700, "message": "Parse error"},
                }

        @app.post("/mcp/v1/tools/call")
        async def mcp_v1_tools_call(
            request: Request, api_key: str = Depends(authenticate_request)
        ):
            """Handle MCP v1 tools/call endpoint."""
            try:
                body = await request.json()
                return await handle_mcp_request(
                    body.get("jsonrpc", "2.0"),
                    body.get("id", 1),
                    "tools/call",
                    body.get("params", {}),
                )
            except Exception as e:
                logger.error(f"Error in MCP v1 tools/call: {e}")
                return {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": -32700, "message": "Parse error"},
                }

        def _extract_session_and_auth(request: Request) -> tuple[str | None, str | None]:
            """Extract session ID and API key from MCP / standard headers."""
            session_id = (
                request.headers.get("mcp-session-id")
                or request.headers.get("x-session-id")
                or request.headers.get("x-connection-id")
            )
            if not session_id:
                import uuid
                session_id = f"mcp-{uuid.uuid4().hex[:12]}"

            api_key = request.headers.get("x-api-key")
            if not api_key:
                auth = request.headers.get("authorization", "")
                if auth.lower().startswith("bearer "):
                    api_key = auth[7:].strip()
            if api_key and is_authentication_required():
                if api_key in self._oauth_tokens:
                    return session_id, api_key
                if _verify_firebase_token(api_key):
                    return session_id, api_key
                if _api_key_auth_ok:
                    if not validate_api_key(api_key, "mcp-jsonrpc"):
                        return session_id, None
                else:
                    return session_id, None
            return session_id, api_key

        def _require_auth(request: Request):
            """Return a 401 Response if auth is required but missing/invalid, else None."""
            from fastapi.responses import Response as FastAPIResponse

            if not is_authentication_required():
                return None
            _, api_key = _extract_session_and_auth(request)
            if api_key is not None:
                return None
            return FastAPIResponse(
                status_code=401,
                content=json.dumps({"error": "Authentication required"}),
                media_type="application/json",
                headers={"WWW-Authenticate": "Bearer"},
            )

        @app.post("/mcp")
        async def mcp_jsonrpc_handler(request: Request):
            """Handle MCP JSON-RPC requests with structured input."""
            from fastapi.responses import Response as FastAPIResponse

            try:
                body = await request.json()
            except Exception:
                return {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}

            auth_resp = _require_auth(request)
            if auth_resp is not None:
                return auth_resp

            session_id, api_key = _extract_session_and_auth(request)

            result = await handle_mcp_request(
                body.get("jsonrpc", "2.0"),
                body.get("id"),
                body.get("method", ""),
                body.get("params", {}),
                connection_id=session_id,
            )

            if result is None:
                return FastAPIResponse(status_code=204)

            resp = FastAPIResponse(
                content=json.dumps(result, default=str),
                media_type="application/json",
            )
            if session_id:
                resp.headers["Mcp-Session-Id"] = session_id
            return resp

        @app.post("/")
        async def mcp_raw_handler(request: Request):
            """Handle raw JSON-RPC requests for better client compatibility."""
            from fastapi.responses import Response as FastAPIResponse

            auth_resp = _require_auth(request)
            if auth_resp is not None:
                return auth_resp

            try:
                body = await request.json()
                session_id, api_key = _extract_session_and_auth(request)

                result = await handle_mcp_request(
                    body.get("jsonrpc", "2.0"),
                    body.get("id"),
                    body.get("method"),
                    body.get("params", {}),
                    connection_id=session_id,
                )

                if result is None:
                    return FastAPIResponse(status_code=204)

                resp = FastAPIResponse(
                    content=json.dumps(result, default=str),
                    media_type="application/json",
                )
                if session_id:
                    resp.headers["Mcp-Session-Id"] = session_id
                return resp
            except Exception as e:
                logger.error(f"Error parsing raw MCP request: {e}")
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }

        # Session cleanup endpoints
        @app.delete("/sessions/{session_id}")
        async def delete_session(session_id: str):
            """Delete a session by session ID."""
            try:
                # Destroy the session
                success = await self.server.session_manager.destroy_session(session_id)
                if success:
                    logger.info(f"🧹 Deleted session {session_id}")
                    return {"status": "deleted", "session_id": session_id}
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Session {session_id} not found or could not be deleted",
                    )
            except Exception as e:
                logger.error(f"Error deleting session: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/sessions/cleanup")
        async def cleanup_session(request: Request):
            """Cleanup session by session ID (sent in body)."""
            try:
                body = await request.json()
                session_id = body.get("session_id")
                if not session_id:
                    raise HTTPException(status_code=400, detail="session_id required")

                # Check if session exists
                if session_id in self.server.session_manager.sessions:
                    await self.server.session_manager.destroy_session(session_id)
                    logger.info(f"🧹 Cleaned up session {session_id}")
                    return {"status": "cleaned_up", "session_id": session_id}
                else:
                    return {"status": "no_session", "session_id": session_id}
            except Exception as e:
                logger.error(f"Error cleaning up session: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # ============================================================================
        # SSE Transport Endpoints (for Claude Code and MCP clients)
        # ============================================================================
        # Use raw ASGI routing for proper MCP SSE transport compatibility

        from mcp.server.sse import SseServerTransport
        from starlette.routing import Mount, Route

        # Create SSE transport - messages endpoint path as seen by clients
        sse_transport = SseServerTransport("/sse/messages/")
        mcp_server = self.server

        # Raw ASGI endpoint for SSE connection (GET /sse/)
        async def sse_endpoint(scope, receive, send):
            """Raw ASGI handler for SSE connection."""
            logger.info("🔗 New SSE connection from MCP client (e.g., Claude Code)")
            try:
                async with sse_transport.connect_sse(scope, receive, send) as (
                    read_stream,
                    write_stream,
                ):
                    logger.info("🚀 Starting MCP server run loop for SSE connection")
                    await mcp_server.server.run(
                        read_stream,
                        write_stream,
                        mcp_server.server.create_initialization_options(),
                    )
            except Exception as e:
                logger.error(f"Error in SSE connection: {e}")

        # Raw ASGI endpoint for SSE messages (POST /sse/messages/)
        async def sse_messages_endpoint(scope, receive, send):
            """Raw ASGI handler for SSE POST messages."""
            logger.debug("📨 Received SSE POST message")
            await sse_transport.handle_post_message(scope, receive, send)

        # Create routes using Route with ASGI apps directly
        # The key is to NOT use the endpoint parameter (which expects request->response)
        # but instead mount ASGI apps directly
        sse_route = Route("/sse/", endpoint=None)
        sse_route.app = sse_endpoint

        sse_messages_route = Route("/sse/messages/", endpoint=None, methods=["POST"])
        sse_messages_route.app = sse_messages_endpoint

        # Also handle /sse without trailing slash
        sse_route_no_slash = Route("/sse", endpoint=None)
        sse_route_no_slash.app = sse_endpoint

        # Insert routes at the beginning so they take precedence
        app.router.routes.insert(0, sse_messages_route)
        app.router.routes.insert(0, sse_route)
        app.router.routes.insert(0, sse_route_no_slash)

        logger.info("✓ SSE endpoints added at /sse and /sse/messages/")

        # Add custom routes if provided
        if self.custom_routes_builder:
            logger.info("Adding custom routes...")
            self.custom_routes_builder(app, self.server)

        return app

    async def run(self):
        """Run the server using HTTP transport."""
        import uvicorn

        # Check for and kill any existing process on the port
        logger.info(f"🔍 Checking for existing process on port {self.port}...")
        killed = kill_process_on_port(self.port)
        if killed:
            logger.info(f"✓ Cleared port {self.port}, ready to start server")
        else:
            logger.info(f"✓ Port {self.port} is available")

        # Create the FastAPI app
        self.app = self._create_app()

        app_to_serve = self.app
        if self.path_prefix:
            from starlette.applications import Starlette
            from starlette.responses import JSONResponse, RedirectResponse
            from starlette.routing import Mount, Route

            transport = self

            def _get_base_url(request) -> str:
                proto = request.headers.get("x-forwarded-proto", "https")
                host = request.headers.get("host", request.base_url.hostname)
                return f"{proto}://{host}"

            async def root_info(request):
                return JSONResponse({
                    "endpoints": {self.path_prefix: self.server_name},
                    "health": f"{self.path_prefix}/health",
                })

            # ── OAuth 2.1 discovery ──────────────────────────────────
            # When OAUTH_SERVER_URL is set (e.g. https://sy.gxl.ai), discovery
            # points to the unified backend's Firebase OAuth endpoints so MCP
            # clients authenticate with the same flow as the Paperclip CLI.

            async def oauth_protected_resource(request):
                base = _get_base_url(request)
                auth_server = _OAUTH_SERVER_URL or base
                # Use origin-only resource so it matches any service path on
                # the shared gateway (FDA at /fda/, Papers at /paperclip/).
                return JSONResponse({
                    "resource": base,
                    "authorization_servers": [auth_server],
                    "bearer_methods_supported": ["header"],
                })

            async def oauth_authorization_server(request):
                base = _get_base_url(request)
                if _OAUTH_SERVER_URL:
                    return JSONResponse({
                        "issuer": _OAUTH_SERVER_URL,
                        "authorization_endpoint": f"{_OAUTH_SERVER_URL}/api/oauth/authorize",
                        "token_endpoint": f"{_OAUTH_SERVER_URL}/api/oauth/token",
                        "registration_endpoint": f"{base}/oauth/register",
                        "response_types_supported": ["code"],
                        "grant_types_supported": ["authorization_code", "refresh_token"],
                        "token_endpoint_auth_methods_supported": ["none"],
                        "code_challenge_methods_supported": ["S256"],
                    })
                return JSONResponse({
                    "issuer": base,
                    "authorization_endpoint": f"{base}/oauth/authorize",
                    "token_endpoint": f"{base}/oauth/token",
                    "registration_endpoint": f"{base}/oauth/register",
                    "response_types_supported": ["code"],
                    "grant_types_supported": ["authorization_code"],
                    "token_endpoint_auth_methods_supported": ["none"],
                    "code_challenge_methods_supported": ["S256"],
                })

            # ── OAuth 2.1 endpoints ──────────────────────────────────

            async def oauth_register(request):
                """RFC 7591 dynamic client registration – auto-approve."""
                import uuid as _uuid
                body = await request.json()
                return JSONResponse({
                    "client_id": f"mcp-{_uuid.uuid4().hex[:12]}",
                    "client_name": body.get("client_name", "MCP Client"),
                    "redirect_uris": body.get("redirect_uris", []),
                    "grant_types": ["authorization_code"],
                    "response_types": ["code"],
                    "token_endpoint_auth_method": "none",
                }, status_code=201)

            async def oauth_authorize(request):
                """Authorization endpoint – auto-approve, redirect with code."""
                import uuid as _uuid
                redirect_uri = request.query_params.get("redirect_uri", "")
                state = request.query_params.get("state", "")
                code_challenge = request.query_params.get("code_challenge", "")

                code = _uuid.uuid4().hex
                transport._oauth_codes[code] = {
                    "challenge": code_challenge,
                    "redirect_uri": redirect_uri,
                }

                sep = "&" if "?" in redirect_uri else "?"
                target = f"{redirect_uri}{sep}code={code}"
                if state:
                    target += f"&state={state}"
                return RedirectResponse(target, status_code=302)

            async def oauth_token(request):
                """Token endpoint – exchange code for bearer token."""
                import base64
                import hashlib
                import uuid as _uuid

                body = await request.form()
                code = body.get("code", "")
                code_verifier = body.get("code_verifier", "")

                stored = transport._oauth_codes.pop(code, None)
                if not stored:
                    return JSONResponse({"error": "invalid_grant"}, status_code=400)

                if stored["challenge"] and code_verifier:
                    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
                    computed = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
                    if computed != stored["challenge"]:
                        return JSONResponse({"error": "invalid_grant"}, status_code=400)

                token = f"mcp_{_uuid.uuid4().hex}"
                transport._oauth_tokens.add(token)
                return JSONResponse({
                    "access_token": token,
                    "token_type": "Bearer",
                    "expires_in": 86400,
                })

            wrapper = Starlette(routes=[
                Route("/.well-known/oauth-protected-resource", oauth_protected_resource),
                Route("/.well-known/oauth-authorization-server", oauth_authorization_server),
                Route("/oauth/register", oauth_register, methods=["POST"]),
                Route("/oauth/authorize", oauth_authorize),
                Route("/oauth/token", oauth_token, methods=["POST"]),
                Route("/", root_info),
                Mount(self.path_prefix, app=self.app),
            ])
            app_to_serve = wrapper
            logger.info(f"Mounted app at {self.path_prefix}")

        logger.info(f"🌐 Starting HTTP server on {self.host}:{self.port}")

        try:
            config = uvicorn.Config(
                app=app_to_serve,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=True,
            )
            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            logger.error(f"❌ FATAL: Failed to start uvicorn: {e}")
            traceback.print_exc()
            raise
