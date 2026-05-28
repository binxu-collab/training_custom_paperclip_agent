"""
Session and E2B sandbox management for the MCP server.
Provides stateful connections with isolated E2B sandbox environments per session.
"""

import asyncio
import builtins
import contextlib
import hashlib
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from gxl_filesystem.filesystem import GXLFileSystem
from mcps.protein_design.utils.file_analyzer import FileAnalyzer

from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class SandboxSession:
    """Represents a client session with an associated E2B sandbox."""

    session_id: str
    sandbox_id: str | None = None
    sandbox_url: str | None = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    files: dict[str, Any] = field(default_factory=dict)
    status: str = "created"  # created, initializing, active, error, destroyed
    api_key_hash: str | None = None  # Hash of the API key associated with this session
    total_tokens: int = 0  # Cumulative token count for this session
    paper_uuid: str | None = None  # For paper2agent: scopes filesystem to single paper


def get_required_api_key() -> str:
    """Get the required API key from environment."""
    api_key = os.getenv("MCP_API_KEY")
    if not api_key:
        raise ValueError("MCP_API_KEY environment variable not set")
    return api_key


def validate_api_key(provided_key: str | None) -> bool:
    """Validate that provided API key matches required key."""
    require_api_key = os.getenv("REQUIRE_API_KEY", "true").lower() == "true"
    if not require_api_key:
        return True  # No authentication required

    required_key = os.getenv("MCP_API_KEY")
    if not required_key:
        return True  # No authentication required if key not set

    # Authentication is required - validate the provided key
    if not provided_key:
        return False

    is_valid = provided_key == required_key
    if not is_valid:
        logger.warning("validate_api_key: Provided API key does not match MCP_API_KEY")
    else:
        logger.debug("validate_api_key: API key validated successfully")
    return is_valid


class SessionManager:
    """Manages client sessions and their associated CodeSandbox instances."""

    def __init__(self, session_timeout: int | None = None):
        config = get_config()
        self.sessions: dict[str, SandboxSession] = {}
        self.session_timeout = session_timeout or config.session.timeout_seconds
        self.cleanup_interval = config.session.cleanup_interval_seconds
        self.max_sessions = config.session.max_sessions_per_instance
        self.file_analyzer = FileAnalyzer()
        self._cleanup_task = None
        # Track which agent_ids are subagents
        self._subagent_ids: set[str] = set()

    def _get_filesystem(
        self, session_id: str, agent_id: str | None = None
    ) -> GXLFileSystem:
        """Get a GXLFileSystem instance for the session."""
        # Always use auto-detection (Local or E2B).
        # Remote provider logic has been removed as per zero-trust local architecture.
        return GXLFileSystem(
            session_id=session_id,
            agent_id=agent_id,
            provider="auto",
            base_dir=os.getenv("LOCAL_SESSION_STORAGE_ROOT"),
        )

    async def start_cleanup_task(self):
        """Start the background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())

    async def stop_cleanup_task(self):
        """Stop the background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

    def _hash_api_key(self, api_key: str) -> str:
        """Create a hash of the API key for session association."""
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]

    def associate_session_with_api_key(self, session_id: str, api_key: str) -> bool:
        """Associate a session with an API key."""
        if session_id in self.sessions:
            self.sessions[session_id].api_key_hash = self._hash_api_key(api_key)
            logger.info(f"Associated session {session_id} with API key")
            return True
        return False

    def get_sessions_for_api_key(self, api_key: str) -> list[SandboxSession]:
        """Get all sessions associated with an API key."""
        api_key_hash = self._hash_api_key(api_key)
        return [
            session
            for session in self.sessions.values()
            if session.api_key_hash == api_key_hash
        ]

    def validate_session_access(self, session_id: str, api_key: str | None) -> bool:
        """Validate that the API key has access to this session.

        Args:
            session_id: Session ID to access
            api_key: API key making the request

        Returns:
            True if access allowed, False otherwise
        """
        if not validate_api_key(api_key):
            return False

        # Get session
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"Session {session_id[:8]}... not found")
            return False

        # For now, all valid API keys can access all sessions
        # Future: check session.api_key_hash matches provided key
        return True

    def get_or_create_session(
        self,
        session_id: str | None = None,
        api_key: str | None = None,
        paper_uuid: str | None = None,
    ) -> str:
        """Get existing session or create new one.

        Args:
            session_id: Optional session ID. If provided and exists, returns it. If not provided, creates new UUID.
            api_key: Optional API key for session association
            paper_uuid: Optional paper UUID for paper2agent mode (scopes filesystem to single paper)

        Returns:
            The session ID
        """
        logger.info(
            f"📍 get_or_create_session called with session_id: {session_id[:8] if session_id else 'None'}..."
        )

        # If session_id provided and exists, update and return it
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            session.last_accessed = time.time()
            # Update paper_uuid if provided (for paper2agent sessions)
            if paper_uuid:
                session.paper_uuid = paper_uuid
            logger.info(f"🔄 REUSING existing session {session_id[:8]}...")
            return session_id

        # Create new session
        if len(self.sessions) >= self.max_sessions:
            raise ValueError(
                f"Maximum number of sessions ({self.max_sessions}) reached"
            )

        # Use provided session_id or generate new UUID
        new_session_id = session_id or str(uuid.uuid4())
        session = SandboxSession(session_id=new_session_id, paper_uuid=paper_uuid)

        # Associate with API key if provided
        if api_key and api_key != "no-auth-required":
            session.api_key_hash = self._hash_api_key(api_key)

        self.sessions[new_session_id] = session

        logger.info(f"✨ CREATED NEW session {new_session_id[:8]}... paper_uuid={paper_uuid}")
        return new_session_id

    def create_session(self) -> str:
        """Create a new session and return the session ID. (Legacy method)"""
        # Check if we've reached the maximum number of sessions
        if len(self.sessions) >= self.max_sessions:
            raise ValueError(
                f"Maximum number of sessions ({self.max_sessions}) reached"
            )

        session_id = str(uuid.uuid4())
        session = SandboxSession(session_id=session_id)
        self.sessions[session_id] = session
        logger.info(f"Created new session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> SandboxSession | None:
        """Get a session by ID and update last accessed time."""
        session = self.sessions.get(session_id)
        if session:
            session.last_accessed = time.time()
        return session

    async def initialize_sandbox_for_session(
        self, session_id: str, template: str = None, files: dict[str, str] = None
    ) -> dict[str, Any]:
        """Initialize a sandbox for a session (local or E2B based on configuration)."""
        if not session_id:
            raise ValueError("Session ID is required")

        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.sandbox_id:
            logger.info(
                f"Session {session_id} already has sandbox at {session.sandbox_url}"
            )
            return {
                "id": session.sandbox_id,
                "url": session.sandbox_url,
                "template": template or "local",
                "status": "active",
            }

        try:
            session.status = "initializing"

            fs = self._get_filesystem(session_id)
            info = await fs.initialize(template=template, files=files)

            # Update session with sandbox info
            session.sandbox_id = getattr(
                info, "id", session_id
            )  # SandboxInfo usually has id/session_id
            session.sandbox_url = getattr(info, "path", "")
            session.status = "active"

            logger.info(
                f"✅ Initialized {fs.provider_name} sandbox for session {session_id}"
            )

            return {
                "id": session.sandbox_id,
                "url": session.sandbox_url,
                "template": template or "local",
                "status": "active",
                "path": session.sandbox_url,
            }

        except Exception as e:
            session.status = "error"
            logger.error(f"Failed to initialize sandbox for session {session_id}: {e}")
            raise

    async def upload_file_to_session_sandbox(
        self,
        session_id: str,
        file_path: str,
        content: str,
        api_key: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Upload a file to a session's sandbox.

        Args:
            session_id: Session ID for the sandbox
            file_path: Path to save file at (relative to sandbox root)
            content: File content
            api_key: Optional API key for authentication
            agent_id: Optional agent ID - if subagent, routes to subagents/{agent_id}/
        """
        # Validate inputs
        if not session_id:
            raise ValueError("Session ID is required")
        if not file_path or not file_path.strip():
            raise ValueError("File path is required and cannot be empty")
        if not isinstance(content, str):
            raise ValueError("File content must be a string")

        # Authorization check
        if not self.validate_session_access(session_id, api_key):
            raise PermissionError(f"Access denied to session {session_id[:8]}...")

        # Check file size
        config = get_config()
        e2b_config = getattr(config, "e2b", None)
        max_file_size_mb = e2b_config.max_file_size_mb if e2b_config else 10
        max_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        if len(content.encode("utf-8")) > max_size:
            raise ValueError(f"File exceeds maximum size of {max_file_size_mb}MB")

        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if not session.sandbox_id:
            raise ValueError(f"Session {session_id} has no active sandbox")

        if session.status != "active":
            raise ValueError(
                f"Session {session_id} is not active (status: {session.status})"
            )

        # Determine sandbox type and upload accordingly
        try:
            fs = self._get_filesystem(session_id, agent_id)
            result = await fs.write_file(file_path, content)

            # Track uploaded files in session metadata
            session.files[file_path] = {
                "size": len(content),
                "uploaded_at": time.time(),
                "agent_id": agent_id,
                "provider": fs.provider_name,
            }

            return result

        except Exception as e:
            logger.error(f"Error uploading file to session {session_id}: {e}")
            raise

    async def execute_in_session_sandbox(
        self, session_id: str, command: str, api_key: str | None = None
    ) -> dict[str, Any]:
        """Execute a command in a session's sandbox."""
        # Validate inputs
        if not session_id:
            raise ValueError("Session ID is required")
        if not command or not command.strip():
            raise ValueError("Command is required and cannot be empty")

        # Authorization check
        if not self.validate_session_access(session_id, api_key):
            raise PermissionError(f"Access denied to session {session_id[:8]}...")

        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if not session.sandbox_id:
            raise ValueError(f"Session {session_id} has no active sandbox")

        if session.status != "active":
            raise ValueError(
                f"Session {session_id} is not active (status: {session.status})"
            )

        # Validate command (basic security check)
        dangerous_commands = ["rm -rf", "del /", "rm /*"]
        command_lower = command.lower()
        for dangerous in dangerous_commands:
            if dangerous in command_lower:
                raise ValueError(
                    f"Command contains potentially dangerous operation: {dangerous}"
                )

        # Determine sandbox type and execute accordingly
        try:
            fs = self._get_filesystem(session_id)
            result = await fs.execute_command(command)

            # Map ExecutionResult (object) to dict if needed, or return dict directly if FS returns dict?
            # GXLFileSystem returns ExecutionResult object, we need to serialize it for API
            return {
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": result.command,
            }

        except Exception as e:
            logger.error(f"Error executing command in session {session_id}: {e}")
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "command": command,
            }

    async def get_session_sandbox_info(self, session_id: str) -> dict[str, Any]:
        """Get sandbox information for a session."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        try:
            fs = self._get_filesystem(session_id)
            return await fs.get_info()
        except Exception as e:
            logger.warning(f"Could not get filesystem info for {session_id}: {e}")
            # Fallback to cached session info
            return {
                "id": session.sandbox_id,
                "url": session.sandbox_url,
                "status": session.status,
                "error": str(e),
            }

    async def destroy_session(self, session_id: str) -> bool:
        """Destroy a session and its associated sandbox."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        success = True
        try:
            fs = self._get_filesystem(session_id)
            success = await fs.destroy()
        except Exception as e:
            logger.error(f"Error destroying session {session_id}: {e}")
            success = False

        # Remove session regardless of sandbox deletion success
        del self.sessions[session_id]

        session.status = "destroyed"

        logger.info(f"Destroyed session: {session_id}")
        return success

    def list_sessions(self) -> dict[str, dict[str, Any]]:
        """List all active sessions with their information."""
        return {
            session_id: {
                "session_id": session.session_id,
                "sandbox_id": session.sandbox_id,
                "sandbox_url": session.sandbox_url,
                "created_at": session.created_at,
                "last_accessed": session.last_accessed,
                "status": session.status,
                "file_count": len(session.files),
            }
            for session_id, session in self.sessions.items()
        }

    async def _cleanup_expired_sessions(self):
        """Background task to cleanup expired sessions."""
        while True:
            try:
                current_time = time.time()
                expired_sessions = [
                    session_id
                    for session_id, session in self.sessions.items()
                    if current_time - session.last_accessed > self.session_timeout
                ]

                for session_id in expired_sessions:
                    logger.info(f"Cleaning up expired session: {session_id}")
                    await self.destroy_session(session_id)

                # Run cleanup at configured interval
                await asyncio.sleep(self.cleanup_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup task: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def auto_initialize_sandbox_for_session(
        self, session_id: str, api_key: str | None = None
    ) -> str:
        """Auto-initialize a sandbox for a session if not already exists.

        Args:
            session_id: The session ID
            api_key: Optional API key for validation
        """
        # Get or create session
        session_id = self.get_or_create_session(session_id=session_id, api_key=api_key)

        # Validate access
        if not self.validate_session_access(session_id, api_key):
            raise PermissionError(f"Access denied to session {session_id[:8]}...")

        session = self.get_session(session_id)

        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Check if sandbox already exists
        if session.sandbox_id:
            logger.info(f"📦 Sandbox already exists for session {session_id}")
            return session_id

        # Auto-initialize sandbox with empty template for file processing
        logger.info(f"📦 Auto-initializing empty sandbox for session {session_id}")
        await self.initialize_sandbox_for_session(session_id, template="empty")

        return session_id

    async def process_downloaded_file_to_sandbox(
        self,
        session_id: str,
        file_path: str,
        content: bytes,
        extract_archives: bool = True,
        analyze_files: bool = True,
    ) -> dict[str, Any]:
        """Process a downloaded file by uploading to sandbox and extracting if needed."""
        # Ensure session and sandbox exist
        session_id = await self.auto_initialize_sandbox_for_session(
            session_id, api_key=None
        )

        # Determine file name from path
        import urllib.parse

        parsed_url = urllib.parse.urlparse(file_path)
        filename = os.path.basename(parsed_url.path) or "downloaded_file"

        # Write content to temp file for analysis
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f"_{filename}"
        ) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Analyze the downloaded file
            file_info = self.file_analyzer.analyze_file(
                temp_path, include_preview=analyze_files
            )

            # Upload file to sandbox
            content_str = (
                content.decode("utf-8", errors="ignore")
                if file_info.is_text
                else str(content)
            )
            await self.upload_file_to_session_sandbox(session_id, filename, content_str)

            result = {
                "session_id": session_id,
                "uploaded_file": filename,
                "file_info": {
                    "name": file_info.name,
                    "size": file_info.size_human,
                    "type": file_info.file_type,
                    "is_archive": file_info.is_archive,
                },
                "extracted_files": [],
            }

            # Extract if it's an archive and extraction is enabled
            if extract_archives and file_info.is_archive:
                logger.info(f"📂 Extracting archive {filename} in sandbox")

                # Execute extraction command in sandbox
                if filename.endswith(".zip"):
                    extract_cmd = f"python -c \"import zipfile; zipfile.ZipFile('{filename}').extractall('.')\""
                elif filename.endswith((".tar.gz", ".tgz")):
                    extract_cmd = f"tar -xzf {filename}"
                elif filename.endswith(".tar"):
                    extract_cmd = f"tar -xf {filename}"
                else:
                    extract_cmd = f"echo 'Unsupported archive format: {filename}'"

                extract_result = await self.execute_in_session_sandbox(
                    session_id, extract_cmd
                )

                # List files after extraction
                list_result = await self.execute_in_session_sandbox(
                    session_id, "ls -la"
                )

                result["extraction_result"] = extract_result
                result["files_after_extraction"] = list_result

                # Try to analyze extracted files
                if analyze_files:
                    list_files_result = await self.execute_in_session_sandbox(
                        session_id, "find . -type f -name '*' | head -20"
                    )
                    result["extracted_file_list"] = list_files_result

            return result

        finally:
            # Clean up temp file
            with contextlib.suppress(builtins.BaseException):
                os.unlink(temp_path)

    async def download_file_from_session_sandbox(
        self, session_id: str, file_path: str, api_key: str | None = None
    ) -> bytes:
        """Download a file from a session's sandbox."""
        # Validate inputs
        if not session_id:
            raise ValueError("Session ID is required")
        if not file_path or not file_path.strip():
            raise ValueError("File path is required and cannot be empty")

        # Authorization check
        if not self.validate_session_access(session_id, api_key):
            raise PermissionError(f"Access denied to session {session_id[:8]}...")

        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        try:
            fs = self._get_filesystem(session_id)
            content = await fs.read_file(file_path)

            # Handle binary files (base64 encoded by GXLFileSystem)
            if content.startswith("[Binary file - Base64 encoded]\n"):
                import base64

                return base64.b64decode(content.split("\n", 1)[1])

            return content.encode("utf-8")

        except FileNotFoundError:
            # Re-raise with better message or let it propagate
            raise FileNotFoundError(f"File not found in session: {file_path}")
        except Exception as e:
            logger.error(f"Failed to download file from session {session_id}: {e}")
            raise

    async def list_files_in_session_sandbox(
        self,
        session_id: str,
        path: str | None = None,
        pattern: str | None = None,
        api_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """List files in a session's sandbox.

        Args:
            session_id: Session ID
            path: Optional folder path within sandbox (e.g., ".gxl/tool_calls")
            pattern: Optional glob pattern to filter files (e.g., "*.json")
            api_key: Optional API key for authorization

        Returns:
            List of file info dicts with keys: path, name, folder, size, mime_type, created_at, modified_at
        """
        # Validate inputs
        if not session_id:
            raise ValueError("Session ID is required")

        # Authorization check
        if not self.validate_session_access(session_id, api_key):
            raise PermissionError(f"Access denied to session {session_id[:8]}...")

        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        try:
            fs = self._get_filesystem(session_id)
            return await fs.list_files_structured(folder=path, pattern=pattern)
        except Exception as e:
            logger.error(f"Failed to list files in session {session_id}: {e}")
            raise

    async def execute_in_connection_sandbox(
        self, connection_id: str, command: str, api_key: str | None = None
    ) -> dict[str, Any]:
        """Execute a command in the sandbox for a connection/session.
        Note: connection_id is now treated as session_id directly.
        DEPRECATED: Use execute_in_session_sandbox with auto_initialize_sandbox_for_session instead.
        """
        # Treat connection_id as session_id and auto-initialize if needed
        session_id = await self.auto_initialize_sandbox_for_session(
            connection_id, api_key=api_key
        )

        return await self.execute_in_session_sandbox(
            session_id, command, api_key=api_key
        )

    async def download_file_from_connection_sandbox(
        self, connection_id: str, file_path: str, api_key: str | None = None
    ) -> bytes:
        """
        Download a file from the sandbox associated with a connection/session.
        Note: connection_id is now treated as session_id directly.
        DEPRECATED: Use download_file_from_session_sandbox with auto_initialize_sandbox_for_session instead.
        """
        # Treat connection_id as session_id and auto-initialize if needed
        session_id = await self.auto_initialize_sandbox_for_session(
            connection_id, api_key=api_key
        )

        return await self.download_file_from_session_sandbox(
            session_id, file_path, api_key=api_key
        )

    async def upload_file_to_connection_sandbox(
        self,
        connection_id: str,
        file_path: str,
        content: str,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Upload a file to the sandbox associated with a connection/session.
        Note: connection_id is now treated as session_id directly.
        DEPRECATED: Use upload_file_to_session_sandbox with auto_initialize_sandbox_for_session instead.
        """
        # Treat connection_id as session_id and auto-initialize if needed
        session_id = await self.auto_initialize_sandbox_for_session(
            connection_id, api_key=api_key
        )

        # Now use the regular upload method with the session_id
        return await self.upload_file_to_session_sandbox(
            session_id, file_path, content, api_key=api_key
        )
