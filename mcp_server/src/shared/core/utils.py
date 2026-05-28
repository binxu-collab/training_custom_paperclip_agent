"""
Shared utility functions for result management and file operations.

This module provides functions for saving results to sandbox with consistent
formatting and error handling across all tool modules.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


async def save_result_to_sandbox(
    function_name: str,
    result: Any,
    arguments: dict[str, Any],
    module_name: str = "general",
    is_json: bool = True,
    session_manager=None,
    connection_id: str | None = None,
    api_key: str | None = None,
    agent_id: str | None = None,
):
    """
    Save function result to sandbox and return the sandbox file path.

    Args:
        function_name: Name of the function that generated the result
        result: The result data to save
        arguments: Arguments passed to the function
        module_name: Name of the module (used for sandbox directory naming)
        is_json: Whether the result can be serialized as JSON
        session_manager: Session manager instance to use active session sandbox
        connection_id: Connection ID for session context
        api_key: Optional API key for authentication
        agent_id: Optional agent ID - if subagent, saves to subagents/{agent_id}/

    Returns:
        Sandbox file path with sandbox:// prefix, or None if session_manager not available

    Raises:
        Exception: If saving fails (logged but not raised)
    """
    try:
        # Return None if no session manager
        if not session_manager:
            logger.warning("No session manager available, skipping sandbox save")
            return None

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        file_extension = "json" if is_json else "txt"
        filename = f"{function_name}_{timestamp}_{unique_id}.{file_extension}"

        # Prepare content to save - save raw result directly (no metadata wrapper)
        # Metadata (function name, arguments, timestamp) is preserved in filename
        if is_json:
            content = json.dumps(result, indent=2, default=str)
        else:
            content = str(result)

        # Try to use active session sandbox if session_manager is available
        if hasattr(session_manager, "upload_file_to_session_sandbox"):
            try:
                # Use the SAME connection ID that the sandbox module uses by default
                # This ensures all search/database results go to the same sandbox as sandbox commands
                if not connection_id:
                    connection_id = "default"

                # Auto-initialize sandbox for the session if needed
                session_id = await session_manager.auto_initialize_sandbox_for_session(
                    connection_id, api_key=api_key
                )

                # Upload file to the active session sandbox
                upload_result = await session_manager.upload_file_to_session_sandbox(
                    session_id,
                    filename,
                    content,
                    api_key=api_key,
                    agent_id=agent_id,
                )
                logger.info(f"Upload result: {upload_result}")

                # Use the actual file_path from upload result (includes subagents/ prefix for subagents)
                actual_file_path = upload_result.get("file_path", f"{filename}")
                sandbox_file_path = f"sandbox://{actual_file_path}"

                logger.info(f"Saved {module_name} result to session sandbox: {sandbox_file_path}")
                return sandbox_file_path

            except Exception as e:
                logger.warning(f"Failed to save to session sandbox: {e}")
                return None

        return None

    except Exception as e:
        logger.error(f"Failed to save result to sandbox: {e}")
        return None


def try_json_serialize(result: Any) -> tuple[str, bool]:
    """
    Try to serialize result as JSON, fallback to string representation.

    Args:
        result: The result to serialize

    Returns:
        Tuple of (serialized_string, is_json_success)
    """
    try:
        result_text = json.dumps(result, indent=2, default=str)
        return result_text, True
    except (TypeError, ValueError):
        return str(result), False


def truncate_with_sandbox_info(
    result_text: str,
    sandbox_file_path: str,
    max_length: int = 5000,
    custom_message: str | None = None,
    truncated_display: str | None = None,
) -> str:
    """
    Truncate result text and append sandbox file information.

    Args:
        result_text: The result text to potentially truncate
        sandbox_file_path: Path to the sandbox file containing full results
        max_length: Maximum length before truncation (default: 5000)
        custom_message: Optional custom message template. Use {sandbox_file_path} as placeholder.
        truncated_display: Custom display of result
    Returns:
        Truncated text with sandbox file information
    """
    message = f"Full results saved to {sandbox_file_path}"
    if custom_message:
        message = custom_message.format(sandbox_file_path=sandbox_file_path)
    if truncated_display:
        return truncated_display + f"\n\n{message}"
    else:
        if len(result_text) > max_length:
            return result_text[:max_length] + f"\n... (truncated)\n\n{message}"
        else:
            return result_text + f"\n\n{message}"


async def process_function_result(
    function_name: str,
    result: Any,
    arguments: dict[str, Any],
    module_name: str,
    session_manager=None,
    connection_id: str | None = None,
    api_key: str | None = None,
    agent_id: str | None = None,
    custom_message: str | None = None,
    truncated_display: str | None = None,
) -> str:
    """
    Complete workflow for processing function results: serialize, save, and truncate.

    Args:
        function_name: Name of the function that generated the result
        result: The result data
        arguments: Arguments passed to the function
        module_name: Name of the module (for sandbox directory naming)
        session_manager: Session manager instance to use active session sandbox
        connection_id: Connection ID for session context
        api_key: Optional API key for authentication
        agent_id: Optional agent ID - if subagent, saves to subagents/{agent_id}/
        custom_message: Optional custom message template with {sandbox_file_path} placeholder
        truncated_display: Optional display message for truncation
    Returns:
        Processed result text (truncated with sandbox info)
    """
    # Try JSON serialization first
    result_text, is_json = try_json_serialize(result)

    # Save full result to sandbox
    sandbox_file_path = await save_result_to_sandbox(
        function_name,
        result,
        arguments,
        module_name,
        is_json,
        session_manager,
        connection_id,
        api_key,
        agent_id,
    )

    # Return truncated result with sandbox info
    if sandbox_file_path:
        return truncate_with_sandbox_info(
            result_text,
            sandbox_file_path,
            custom_message=custom_message,
            truncated_display=truncated_display if truncated_display else None,
        )
    else:
        return result_text
