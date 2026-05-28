"""
Response management utilities for handling large tool responses.
Automatically saves responses that exceed size limits to sandbox files.
"""

import logging
from datetime import datetime

from mcp.types import TextContent

logger = logging.getLogger(__name__)


class ResponseManager:
    """Manages tool responses and handles large response file saving."""

    def __init__(
        self, session_manager, max_response_size: int = 50_000
    ):  # ~12500 tokens (4 chars/token)
        """
        Initialize ResponseManager.

        Args:
            session_manager: The SessionManager instance for file operations
            max_response_size: Maximum response size in characters before saving to file (~12500 tokens)
        """
        self.session_manager = session_manager
        self.max_response_size = max_response_size
        self.preview_size = (
            5000  # First 5000 characters to show in response (~1250 tokens)
        )

    def calculate_response_size(self, response: list[TextContent]) -> int:
        """Calculate the total size of a response in bytes."""
        total_size = 0
        for content in response:
            if hasattr(content, "text") and content.text:
                total_size += len(content.text.encode("utf-8"))
        return total_size

    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    async def process_response(
        self,
        tool_name: str,
        response: list[TextContent],
        connection_id: str = "default_connection",
        *,
        skip_save: bool = False,
    ) -> list[TextContent]:
        """
        Process tool response and save to file if it's too large.

        Args:
            tool_name: Name of the tool that generated the response
            response: The tool response content
            connection_id: Connection ID for sandbox association
            skip_save: If True, return response as-is without saving to file

        Returns:
            Processed response (either original or truncated with file info)
        """
        try:
            if skip_save:
                return response

            response_size = self.calculate_response_size(response)

            # If response is small enough, return as-is
            if response_size < self.max_response_size:
                return response

            logger.info(
                f"📁 Large response detected ({self.format_file_size(response_size)}) for tool '{tool_name}', saving to sandbox..."
            )

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{tool_name}_response_{timestamp}.txt"

            # Combine all response text
            combined_text = self._combine_response_text(response, tool_name)

            # Save to sandbox
            await self._save_response_to_sandbox(connection_id, filename, combined_text)

            # Create truncated response with file info
            truncated_response = self._create_truncated_response(
                tool_name, filename, combined_text, response_size
            )

            return truncated_response

        except Exception as e:
            logger.error(f"Error processing large response for tool '{tool_name}': {e}")
            # Return original response if processing fails
            return response

    def _combine_response_text(
        self, response: list[TextContent], tool_name: str
    ) -> str:
        """Combine all response content into a single text with metadata."""
        lines = []
        lines.append(f"Tool Response: {tool_name}")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("=" * 80)
        lines.append("")

        for i, content in enumerate(response):
            if hasattr(content, "text") and content.text:
                if len(response) > 1:
                    lines.append(f"--- Response Part {i + 1} ---")
                lines.append(content.text)
                lines.append("")

        return "\n".join(lines)

    async def _save_response_to_sandbox(
        self, connection_id: str, filename: str, content: str
    ) -> None:
        """Save response content so it is reachable via /session_files/."""
        try:
            session_id = await self.session_manager.auto_initialize_sandbox_for_session(
                connection_id
            )

            await self.session_manager.upload_file_to_session_sandbox(
                session_id, f"files/{filename}", content
            )

            logger.info(f"Saved large response to session_files: {filename}")

        except Exception as e:
            logger.error(f"Failed to save response to sandbox: {e}")
            raise

    def _create_truncated_response(
        self, tool_name: str, filename: str, full_content: str, original_size: int
    ) -> list[TextContent]:
        """Create a truncated response with file information and preview."""

        # Extract preview content (first 5000 characters)
        preview_content = full_content[: self.preview_size]
        if len(full_content) > self.preview_size:
            preview_content += "\n\n[... content truncated ...]"

        response_text = f"""🔄 Large Response Saved

Tool: {tool_name}
Original Size: {self.format_file_size(original_size)}
Saved to File: {filename}

To read the full output, use: cat /session_files/{filename}

Preview (first {self.preview_size} characters):
{"-" * 60}
{preview_content}
{"-" * 60}
"""

        return [TextContent(type="text", text=response_text)]


def create_response_manager(session_manager, max_tokens: int = 12500) -> ResponseManager:
    """
    Create a ResponseManager instance.

    Args:
        session_manager: The SessionManager instance
        max_tokens: Approximate max tokens before saving to file (uses ~4 chars/token)

    Returns:
        Configured ResponseManager instance
    """
    max_size_chars = max_tokens * 4  # ~4 characters per token
    return ResponseManager(session_manager, max_size_chars)
