#!/bin/bash
# Start the papers MCP server (HTTP mode) for use during RL training.
#
# The server exposes the paperclip tool over HTTP, which the RL training
# environment calls via the GXL inference engine.
#
# Requirements:
#   - gxl-tools package installed (see pyproject.toml)
#   - Cloud SQL proxy running (for biomedrxiv DB access)
#   - ALLOWED_API_KEYS env var set
#
# Usage:
#   ALLOWED_API_KEYS="your-secret-key" MCP_PORT=8093 bash start_papers_mcp.sh

set -e

PORT=${MCP_PORT:-8093}
HOST=${MCP_HOST:-0.0.0.0}

if [ -z "${ALLOWED_API_KEYS:-}" ]; then
    echo "ERROR: ALLOWED_API_KEYS env var must be set"
    echo "  e.g. ALLOWED_API_KEYS=mysecretkey bash start_papers_mcp.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

echo "Starting papers MCP server on $HOST:$PORT"
uv run --project "$SCRIPT_DIR" python -m mcps.papers.servers.papers_server \
    --host "$HOST" \
    --port "$PORT"
