#!/usr/bin/env python3
"""Paperclip MCP Server — Discrete tools for biomedical paper search and reading.

Benchmark counterpart to the paperclip CLI. Exposes the same capabilities as
structured MCP tools instead of a monolithic bash/CLI interface.

Requires a running biomedrxiv MCP backend (PAPERCLIP_MCP_URL, default localhost:8085).
"""

import argparse
import logging
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from modules.paperclip.tools import PaperclipModule
from shared.core.base_server import MCPServer
from shared.core.config import ServerConfig
from shared.core.session_manager import SessionManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("paperclip-mcp")


class PaperclipServer(MCPServer):
    """Paperclip MCP Server with discrete paper tools."""

    def __init__(self):
        super().__init__("paperclip-mcp")
        self.config = ServerConfig()
        self.session_manager = SessionManager()
        self._register_modules()
        self.setup_handlers()
        logger.info("Paperclip MCP Server initialized")

    def _register_modules(self):
        module = PaperclipModule()
        module.initialize(self.config, self.session_manager)
        self.register_module(module)
        logger.info(f"Registered paperclip module with {len(module.get_tools())} tools")


def run_http(host: str = "0.0.0.0", port: int = 8090):
    import asyncio

    from shared.core.transports import HTTPTransport

    server = PaperclipServer()
    transport = HTTPTransport(
        server,
        host=host,
        port=port,
        server_name="Paperclip MCP Benchmark",
        server_version="1.0.0",
    )
    asyncio.run(transport.run())


def main():
    parser = argparse.ArgumentParser(description="Paperclip MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--local", action="store_true", default=False,
                        help="Run in local mode (accepted for consistency with other servers)")
    args = parser.parse_args()

    from shared.core.environment import load_environment

    load_environment()

    if args.transport == "http":
        run_http(host=args.host, port=args.port)
    else:
        from shared.core.transports import StdioTransport

        server = PaperclipServer()
        transport = StdioTransport(server)
        import asyncio

        asyncio.run(transport.run())


if __name__ == "__main__":
    main()
