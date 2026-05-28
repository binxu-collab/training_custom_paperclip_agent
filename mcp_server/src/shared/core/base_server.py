"""
Base MCP Server class with modular architecture.

This module provides the core MCP server functionality that can be extended
with pluggable tool modules.
"""

import asyncio
import inspect
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from mcp.server import Server
from mcp.types import TextContent, Tool

from .config import get_config
from .response_manager import create_response_manager
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class ToolModule(ABC):
    """Abstract base class for tool modules."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the module name."""
        pass

    @abstractmethod
    def get_tools(self) -> list[Tool]:
        """Return list of tools provided by this module."""
        pass

    @abstractmethod
    def get_handlers(self) -> dict[str, callable]:
        """Return dictionary of tool name -> handler function mappings."""
        pass

    @abstractmethod
    def initialize(self, config: Any, session_manager: SessionManager) -> None:
        """Initialize the module with configuration and session manager."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up module resources."""
        pass

    @abstractmethod
    async def ensure_ready(self) -> None:
        """Ensure module is ready for use (async initialization)."""
        pass


class MCPServer:
    """Base MCP server with modular tool architecture."""

    def __init__(self, server_name: str = "mcp-server"):
        """Initialize the base MCP server."""
        self.server = Server(server_name)
        self.modules: dict[str, ToolModule] = {}
        self.tool_handlers: dict[str, callable] = {}
        self.excluded_tools: list[str] = []
        self.included_tools: list[str] = []

        # Initialize core components
        self.session_manager = SessionManager()
        config = get_config()
        self.response_manager = create_response_manager(
            self.session_manager, max_tokens=config.max_response_tokens
        )

        logger.info(f"Initialized base MCP server: {server_name}")

    def register_module(self, module: ToolModule) -> None:
        """Register a tool module with the server."""
        module_name = module.get_name()

        if module_name in self.modules:
            logger.warning(f"Module '{module_name}' is already registered, replacing")

        self.modules[module_name] = module

        # Initialize the module
        module.initialize(get_config(), self.session_manager)

        # Register module's handlers
        handlers = module.get_handlers()
        for tool_name, handler in handlers.items():
            if tool_name in self.tool_handlers:
                logger.warning(f"Tool handler '{tool_name}' already exists, replacing")
            self.tool_handlers[tool_name] = handler

        logger.info(f"Registered module '{module_name}' with {len(handlers)} tools")

    def unregister_module(self, module_name: str) -> None:
        """Unregister a tool module from the server."""
        if module_name not in self.modules:
            logger.warning(f"Module '{module_name}' is not registered")
            return

        module = self.modules[module_name]

        # Remove module's handlers
        handlers = module.get_handlers()
        for tool_name in handlers:
            if tool_name in self.tool_handlers:
                del self.tool_handlers[tool_name]

        # Clean up module
        module.cleanup()

        # Remove module
        del self.modules[module_name]

        logger.info(f"Unregistered module '{module_name}'")

    def set_excluded_tools(self, excluded_tools: list[str]) -> None:
        """Set list of tools to exclude from registration."""
        self.excluded_tools = [tool.lower() for tool in excluded_tools]
        logger.info(f"Set excluded tools: {self.excluded_tools}")

    def set_included_tools(self, included_tools: list[str]) -> None:
        """Set list of tools to include (overrides excluded_tools when set)."""
        # import pdb; pdb.set_trace()
        self.included_tools = [tool.lower() for tool in included_tools]
        logger.info(f"Set included tools: {self.included_tools}")

    async def ensure_modules_ready(self) -> None:
        """Ensure all registered modules are ready for use."""
        for module_name, module in self.modules.items():
            try:
                await module.ensure_ready()

                # Re-register handlers in case they were populated during ensure_ready()
                handlers = module.get_handlers()
                for tool_name, handler in handlers.items():
                    if tool_name in self.tool_handlers:
                        logger.debug(f"Updating handler for tool '{tool_name}'")
                    self.tool_handlers[tool_name] = handler

                logger.debug(
                    f"Module '{module_name}' is ready with {len(handlers)} handlers"
                )
            except Exception as e:
                logger.error(f"Error ensuring module '{module_name}' is ready: {e}")

    @staticmethod
    def _inject_description_param(tool: Tool) -> Tool:
        """Add an optional ``description`` property to a tool's inputSchema.

        The LLM is instructed (via system prompt) to populate this field with a
        short, user-facing sentence describing what the call is doing.  The
        frontend reads ``tool.input.description`` and displays it in the
        execution trace instead of the raw tool name + args.
        """
        schema = tool.inputSchema
        props = schema.get("properties")
        if props is None:
            props = {}
            schema["properties"] = props
        if "description" not in props:
            props["description"] = {
                "type": "string",
                "description": (
                    "Short plain-language sentence (< 80 chars) describing what "
                    "this call does, shown to the user in the execution trace. "
                    "Describe the intent, not the mechanics."
                ),
            }
        return tool

    def get_instructions(self) -> str | None:
        """Return server instructions for MCP clients. Override in subclasses."""
        return None

    def get_all_tools(self) -> list[Tool]:
        """Get all tools from all registered modules."""
        all_tools = []

        for module_name, module in self.modules.items():
            try:
                module_tools = module.get_tools()
                if self.included_tools:
                    filtered_tools = [
                        tool
                        for tool in module_tools
                        if tool.name.lower() in self.included_tools
                    ]
                    included_count = len(filtered_tools)
                    if included_count > 0:
                        logger.info(
                            f"Module '{module_name}': included {included_count} tools"
                        )
                else:
                    filtered_tools = [
                        tool
                        for tool in module_tools
                        if tool.name.lower() not in self.excluded_tools
                    ]
                    excluded_count = len(module_tools) - len(filtered_tools)
                    if excluded_count > 0:
                        logger.info(
                            f"Module '{module_name}': excluded {excluded_count} tools"
                        )

                all_tools.extend(filtered_tools)

            except Exception as e:
                logger.error(f"Error getting tools from module '{module_name}': {e}")

        # Inject optional `description` parameter into every tool so the LLM
        # can provide a user-facing label for the execution trace.
        all_tools = [self._inject_description_param(t) for t in all_tools]

        logger.info(f"Total tools available: {len(all_tools)}")
        return all_tools

    async def call_tool_handler(
        self,
        name: str,
        arguments: dict[str, Any],
        connection_id: str = "default",
        agent_id: str | None = None,
        api_key: str | None = None,
        agent_config_path: str | None = None,
        paper_uuid: str | None = None,
    ) -> list[TextContent]:
        """Call a tool handler by name."""
        logger.info(
            f"🔧 TOOL CALL - {name} from connection: {connection_id}, agent: {agent_id}, paper_uuid: {paper_uuid}"
        )

        # Ensure modules are ready (async initialization)
        await self.ensure_modules_ready()

        # Strip version suffixes (e.g., netsolp_1_0 -> netsolp)
        base_name = name
        if "_" in name and name.split("_")[-1].replace(".", "").isdigit():
            # Remove version suffix pattern like _1_0, _2_1, etc.
            parts = name.split("_")
            if len(parts) >= 2 and parts[-1].replace(".", "").isdigit():
                # Check if second-to-last part is also a digit (for _1_0 pattern)
                if len(parts) >= 3 and parts[-2].replace(".", "").isdigit():
                    base_name = "_".join(parts[:-2])
                else:
                    base_name = "_".join(parts[:-1])
                logger.info(f"Stripped version suffix: {name} -> {base_name}")

        # # Check tool filtering rules
        # if self.included_tools:
        #     # If include_tools is set, only allow tools in that list
        #     if base_name.lower() not in self.included_tools:
        #         logger.warning(f"Tool '{base_name}' is not in included tools and cannot be executed")
        #         return [TextContent(type="text", text=f"Error: Tool '{base_name}' is not in included tools")]
        # else:
        #     # Otherwise, check if tool is excluded
        #     if base_name.lower() in self.excluded_tools:
        #         logger.warning(f"Tool '{base_name}' is excluded and cannot be executed")
        #         return [TextContent(type="text", text=f"Error: Tool '{base_name}' is excluded")]

        # Find and call the handler - try base name first, then original name
        tool_name_to_use = None
        if base_name in self.tool_handlers:
            tool_name_to_use = base_name
        elif name in self.tool_handlers:
            tool_name_to_use = name

        if tool_name_to_use:
            try:
                handler = self.tool_handlers[tool_name_to_use]

                # Strip the injected `description` field before dispatching.
                # _inject_description_param adds it to every tool schema so the
                # LLM populates it for the UI trace, but handler functions don't
                # accept it as a keyword argument.
                arguments.pop("description", None)
                skip_truncation = arguments.pop("skip_truncation", None) is True

                # Check if handler accepts parameters
                sig = inspect.signature(handler)

                # Build kwargs based on what handler accepts
                call_kwargs = {}
                if "session_id" in sig.parameters or "connection_id" in sig.parameters:
                    call_kwargs[
                        (
                            "connection_id"
                            if "connection_id" in sig.parameters
                            else "session_id"
                        )
                    ] = connection_id
                if "api_key" in sig.parameters:
                    call_kwargs["api_key"] = api_key
                if "agent_id" in sig.parameters:
                    call_kwargs["agent_id"] = agent_id
                if "agent_config_path" in sig.parameters:
                    call_kwargs["agent_config_path"] = agent_config_path
                if "paper_uuid" in sig.parameters:
                    call_kwargs["paper_uuid"] = paper_uuid

                raw_response = await handler(arguments, **call_kwargs)

                # Save tool call result to JSON file using GXLFileSystem
                try:
                    # Use Pacific time for timestamps
                    pacific_time = datetime.now(ZoneInfo("America/Los_Angeles"))
                    timestamp = pacific_time.strftime("%H%M%S")

                    # Use full tool name without module prefix
                    tool_base_name = (
                        tool_name_to_use.split("__")[-1]
                        if "__" in tool_name_to_use
                        else tool_name_to_use
                    )

                    # Construct filename in session .gxl/tool_calls/
                    result_filename = (
                        f".gxl/tool_calls/{tool_base_name}_{timestamp}_1.json"
                    )

                    # Parse raw_response to extract all output data.
                    # Prefer _raw_result (structured dict stashed by compact
                    # formatters) so disk files stay machine-readable even
                    # when the LLM-facing text is a compact string.
                    output_data = []
                    if raw_response and isinstance(raw_response, list):
                        for item in raw_response:
                            raw = getattr(item, "_raw_result", None)
                            if raw is not None:
                                output_data.append(raw)
                            elif hasattr(item, "text"):
                                # Try to parse as JSON, fallback to text
                                try:
                                    parsed = json.loads(item.text)
                                    output_data.append(parsed)
                                except json.JSONDecodeError:
                                    output_data.append(item.text)
                            else:
                                output_data.append(str(item))
                    else:
                        output_data = str(raw_response) if raw_response else None

                    # If only one output item, unwrap the list
                    if isinstance(output_data, list) and len(output_data) == 1:
                        output_data = output_data[0]

                    # Prepare content
                    content = json.dumps(
                        {
                            "tool": tool_name_to_use,
                            "timestamp": pacific_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                            "session_id": connection_id,
                            "agent_id": agent_id,
                            "input": {
                                k: v
                                for k, v in arguments.items()
                                if k not in ["api_key", "session_id", "session_manager"]
                            },
                            "output": output_data,
                        },
                        indent=2,
                    )

                    # Write in background so the agent gets its response immediately
                    async def _bg_write(
                        session_id=connection_id, path=result_filename, data=content
                    ):
                        try:
                            from gxl_filesystem import GXLFileSystem

                            fs = GXLFileSystem(
                                session_id=session_id,
                                base_dir=os.getenv(
                                    "LOCAL_SESSION_STORAGE_ROOT",
                                    "/workspaces/gxl/sessions",
                                ),
                            )
                            await fs.write_file(path, data)
                            logger.info(f"Saved tool call (input/output) to {path}")
                        except Exception as exc:
                            logger.warning(f"Background save failed for {path}: {exc}")

                    asyncio.create_task(_bg_write())
                except Exception as e:
                    logger.warning(f"Failed to save tool result to file: {e}")

                # Process response through response manager.
                # Skip re-saving when the agent reads from /session_files/ —
                # those are already-saved large responses and re-processing
                # creates an unreadable cascade of saved files.
                # Also skip when the caller explicitly opts out (CLI callers
                # want full output, not the truncated preview).
                skip_save = (
                    skip_truncation
                    or (
                        tool_name_to_use == "bash"
                        and isinstance(arguments.get("command"), str)
                        and "/session_files/" in arguments["command"]
                    )
                )
                if raw_response:
                    return await self.response_manager.process_response(
                        tool_name_to_use, raw_response, connection_id=connection_id,
                        skip_save=skip_save,
                    )
                else:
                    return [TextContent(type="text", text="No response from tool")]

            except Exception as e:
                logger.error(f"Error in tool {tool_name_to_use}: {str(e)}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
        else:
            logger.warning(f"Unknown tool: {name} (also tried: {base_name})")
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    def setup_handlers(self) -> None:
        """Setup MCP server handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            # Ensure modules are ready
            await self.ensure_modules_ready()
            return self.get_all_tools()

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict[str, Any]
        ) -> list[TextContent]:
            return await self.call_tool_handler(name, arguments)

    async def call_tool_by_name(
        self,
        name: str,
        arguments: dict[str, Any],
        connection_id: str = "default",
        api_key: str | None = None,
        agent_id: str | None = None,
        agent_config_path: str | None = None,
        paper_uuid: str | None = None,
    ) -> list[TextContent]:
        """External interface for calling tools by name."""
        return await self.call_tool_handler(
            name,
            arguments,
            connection_id,
            api_key=api_key,
            agent_id=agent_id,
            agent_config_path=agent_config_path,
            paper_uuid=paper_uuid,
        )

    async def get_tools_list(self) -> list[Tool]:
        """External interface for getting all tools."""
        await self.ensure_modules_ready()
        return self.get_all_tools()

    def cleanup(self) -> None:
        """Clean up server resources."""
        for module_name, module in self.modules.items():
            try:
                module.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up module '{module_name}': {e}")

        logger.info("MCP server cleanup completed")
