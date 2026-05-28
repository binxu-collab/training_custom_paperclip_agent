"""
Virtual Terminal - A sandboxed shell interface for virtual filesystems.

Provides a familiar Unix-like terminal experience while ensuring safety:
- Allowlist-based command execution (no rm, chmod, sudo, etc.)
- Path validation (stays within virtual filesystem)
- Sandboxed Python execution
- Stateful sessions (cwd, history, aliases)

Usage:
    terminal = VirtualTerminal(filesystem)
    result = await terminal.execute("ls -la /papers/")
    result = await terminal.execute("cat Methods.lines | grep 'learning rate'")
"""

import asyncio
import json
import logging
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

_SOURCE_DISPLAY = {
    "biorxiv": "bioRxiv",
    "medrxiv": "medRxiv",
    "biomedrxiv": "bioRxiv",
    "pmc": "PubMed Central",
    "arxiv": "arXiv",
    "openalex": "Abstracts",
    "abstract_only": "Abstracts",
    "abstracts": "Abstracts",
    "ctgov_registry": "ClinicalTrials.gov",
    "ctgov": "ClinicalTrials.gov",
    "clinicaltrials.gov": "ClinicalTrials.gov",
}


# =============================================================================
# Command Definitions
# =============================================================================

ALLOWED_COMMANDS = {
    # Navigation
    "cd",
    "pwd",
    "ls",
    "tree",
    # Reading
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "wc",
    # Searching
    "grep",
    "egrep",
    "fgrep",
    "search",
    "searches",
    "lookup",
    "lookup-citation",
    "scan",
    "sql",
    "export",
    # Listing/browsing
    # Text processing
    "awk",
    "sed",
    "sort",
    "uniq",
    "cut",
    "tr",
    # JSON
    "jq",
    # Analysis
    "ask_image",
    "filter",
    "map",
    "reduce",
    # Utility
    "echo",
    "printf",
    "env",
    "history",
    "mode",
    "help",
    "man",
    "skill",
    "tee",
    # HTTP download
    "curl",
    # Paper links (database accession references)
    "links",
    "links-search",
    "links-browse",
    "links-stats",
}

# Bash keywords that indicate unsupported shell syntax
BASH_KEYWORDS = {
    # Loop constructs
    "for",
    "do",
    "done",
    "while",
    "until",
    # Conditionals
    "if",
    "then",
    "else",
    "elif",
    "fi",
    "case",
    "esac",
    # Functions
    "function",
    # Other
    "select",
    "in",
    "time",
    "coproc",
}

BLOCKED_COMMANDS = {
    # Destructive
    "rm",
    "rmdir",
    "unlink",
    "shred",
    # Move/copy outside sandbox
    "mv",
    "cp",
    # Permissions
    "chmod",
    "chown",
    "chgrp",
    # Process management
    "kill",
    "pkill",
    "killall",
    # Privilege escalation
    "sudo",
    "su",
    "doas",
    # Network (raw) - curl is allowed (handled by _cmd_curl in FDA/BioMedRxiv terminals)
    "wget",
    "nc",
    "netcat",
    # Remote access
    "ssh",
    "scp",
    "rsync",
    "ftp",
    "sftp",
    # Disk operations
    "dd",
    "mkfs",
    "mount",
    "umount",
    # Code injection vectors
    "eval",
    "exec",
    "source",
    ".",
}


@dataclass
class ParsedCommand:
    """A parsed shell command."""

    program: str
    args: list[str] = field(default_factory=list)
    stdin_redirect: str | None = None
    stdout_redirect: str | None = None
    append_redirect: bool = False


@dataclass
class TerminalResult:
    """Result of a terminal command execution."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    cwd: str = "/papers/"
    metadata: dict = None  # structured data (e.g. results_id) for chaining

    def to_dict(self) -> dict:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "cwd": self.cwd,
        }


# =============================================================================
# Citation helpers
# =============================================================================


def _extract_citation_key(ref_content: str, original_query: str) -> str:
    """Return a compact search term for locating a reference in body text.

    Handles:
      - Numbered refs: "[1] Smith..." or "1. Smith..." → "[1]"
      - Paren-numbered: "1) Smith..."                  → "(1)"
      - Author-year: "Smith, J. (2020)..."             → "Smith"
    Falls back to the original query if no pattern matches.
    """
    content = ref_content.strip()

    # [N] at start
    m = re.match(r"^\[(\d+)\]", content)
    if m:
        return f"[{m.group(1)}]"

    # N. at start
    m = re.match(r"^(\d+)\.\s", content)
    if m:
        return f"[{m.group(1)}]"

    # N) at start
    m = re.match(r"^(\d+)\)\s", content)
    if m:
        return f"({m.group(1)})"

    # Author-year: extract first author last name (enough for ILIKE search)
    last_m = re.match(r"^([A-ZÀ-Ö][a-zà-ö]+(?:[- ][A-ZÀ-Ö][a-zà-ö]+)?)", content)
    if last_m:
        return last_m.group(1)

    # Fallback: first alphabetic word of the reference
    words = content.split()
    if words and words[0][0].isalpha():
        return words[0].rstrip(".,;")

    return original_query


# =============================================================================
# Virtual Terminal
# =============================================================================


class VirtualTerminal:
    """A sandboxed terminal for virtual filesystem interaction."""

    def __init__(self, filesystem_module=None):
        """
        Initialize the virtual terminal.

        Args:
            filesystem_module: The VirtualFilesystemModule to operate on.
                             If None, terminal runs in standalone mode.
        """
        self.fs = filesystem_module

        # Determine root path from filesystem (e.g., /papers/ for BioMedRxiv, /documents/ for FDA)
        root_path = "/papers/"  # Default
        if filesystem_module and hasattr(filesystem_module, "path_parser"):
            root_name = filesystem_module.path_parser.root_name
            root_path = f"/{root_name}/"

        self.cwd = root_path
        self.env = {
            "USER": "agent",
            "HOME": root_path,
            "SHELL": "/bin/vsh",
            "PATH": "/bin:/usr/bin",
            "TERM": "xterm-256color",
            "PWD": root_path,
        }
        self.history: list[str] = []
        self._last_search_results_id: str | None = None
        self.aliases: dict[str, str] = {
            "ll": "ls -la",
            "la": "ls -a",
            "l": "ls -CF",
            "..": "cd ..",
            "...": "cd ../..",
        }
        self.variables: dict[str, str] = {}

        # Sandbox provider per session (lazy-loaded)
        # In staging/prod: uses E2B remote sandbox
        # In dev: uses local filesystem at sessions/{session_id}/files/
        self._e2b_providers: dict[str, Any] = {}
        self._use_e2b: bool | None = None  # Lazily determined

        # Prompt identity — subclasses override these
        self.hostname = "vsh"
        self.home_dir = root_path  # e.g. "/papers/" or "/fda/"

        # Tag-based search accumulator: searches with the same --tag value
        # accumulate into one deduplicated result set. Maps tag -> results_id.
        self._search_accumulator_ids: dict[str, str] = {}
        self._ACCUMULATOR_CAP = 1000

        # Command handlers
        self._handlers: dict[str, Callable] = {
            "cd": self._cmd_cd,
            "pwd": self._cmd_pwd,
            "ls": self._cmd_ls,
            "tree": self._cmd_tree,
            "cat": self._cmd_cat,
            "head": self._cmd_head,
            "tail": self._cmd_tail,
            "wc": self._cmd_wc,
            "grep": self._cmd_grep,
            "egrep": self._cmd_grep,
            "fgrep": self._cmd_grep,
            "search": self._cmd_search,
            "searches": self._cmd_searches,
            "lookup": self._cmd_lookup,
            "scan": self._cmd_scan,
            "sql": self._cmd_sql,
            "export": self._cmd_export,
            # Analysis commands (delegate to filesystem module)
            "ask_image": self._cmd_ask_image,
            "ask-image": self._cmd_ask_image,
            "filter": self._cmd_filter,
            "map": self._cmd_map,
            "reduce": self._cmd_reduce,
            "sort": self._cmd_sort,
            "uniq": self._cmd_uniq,
            "awk": self._cmd_awk,
            "sed": self._cmd_sed,
            "cut": self._cmd_cut,
            "tr": self._cmd_tr,
            "echo": self._cmd_echo,
            "printf": self._cmd_echo,
            "env": self._cmd_env,
            "history": self._cmd_history,
            "mode": self._cmd_mode,
            "help": self._cmd_help,
            "man": self._cmd_help,
            "skill": self._cmd_skill,
            "jq": self._cmd_jq,
            "tee": self._cmd_tee,
            "less": self._cmd_cat,  # Alias to cat
            "more": self._cmd_cat,  # Alias to cat
        }

    @property
    def root_path(self) -> str:
        """Get the root path (e.g., /papers/ or /fda/) based on filesystem."""
        if self.fs and hasattr(self.fs, "path_parser"):
            return f"/{self.fs.path_parser.root_name}/"
        return "/papers/"  # Default

    # =========================================================================
    # Public API
    # =========================================================================

    async def execute(
        self, command: str, session_id: str = "default"
    ) -> TerminalResult:
        """
        Execute a shell command.

        Args:
            command: The command string to execute (supports pipes, &&, ;)
            session_id: Session ID for stateful operations

        Returns:
            TerminalResult with stdout, stderr, exit_code
        """
        start_time = time.perf_counter()

        # Record in history
        command = command.strip()
        if command and not command.startswith("#"):
            self.history.append(command)

        # Empty command
        if not command or command.startswith("#"):
            return TerminalResult(cwd=self.cwd)

        # Handle heredoc: `cat > file << 'EOF'\ncontent\nEOF`
        # Converts to a redirect-based write before normal processing.
        command = self._expand_heredoc(command, session_id)

        # Split on command separators (&&, ||, ;) while respecting quotes
        commands = self._split_command_chain(command)

        # Execute each command in sequence
        result = TerminalResult(cwd=self.cwd)
        all_stdout = []

        for cmd_str, operator in commands:
            cmd_str = cmd_str.strip()
            if not cmd_str:
                continue

            # Check previous result for && and ||
            if operator == "&&" and result.exit_code != 0:
                continue  # Skip if previous failed
            elif operator == "||" and result.exit_code == 0:
                continue  # Skip if previous succeeded

            # Execute single command/pipeline
            result = await self._execute_single(cmd_str, session_id)
            if result.stdout:
                all_stdout.append(result.stdout.rstrip("\n"))

        # Combine stdout from all commands
        if all_stdout:
            result = TerminalResult(
                stdout="\n".join(all_stdout) + "\n",
                stderr=result.stderr,
                exit_code=result.exit_code,
                cwd=self.cwd,
            )

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Terminal: '{command}' completed in {elapsed:.0f}ms")
        return result

    def _split_command_chain(self, command: str) -> list[tuple[str, str]]:
        """Split command by &&, ||, ; while respecting quotes.

        Returns list of (command, preceding_operator) tuples.
        First command has empty operator.
        """
        result = []
        current = []
        in_quotes = False
        quote_char = None
        i = 0
        prev_operator = ""

        while i < len(command):
            char = command[i]

            # Handle quotes
            if char in ('"', "'") and (i == 0 or command[i - 1] != "\\"):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                current.append(char)
                i += 1
                continue

            # Check for operators outside quotes
            if not in_quotes:
                # Check for &&
                if command[i : i + 2] == "&&":
                    result.append(("".join(current), prev_operator))
                    current = []
                    prev_operator = "&&"
                    i += 2
                    continue
                # Check for ||
                elif command[i : i + 2] == "||":
                    result.append(("".join(current), prev_operator))
                    current = []
                    prev_operator = "||"
                    i += 2
                    continue
                # Check for ;
                elif char == ";":
                    result.append(("".join(current), prev_operator))
                    current = []
                    prev_operator = ";"
                    i += 1
                    continue

            current.append(char)
            i += 1

        # Add final command
        if current:
            result.append(("".join(current), prev_operator))

        return result

    async def _execute_single(self, command: str, session_id: str) -> TerminalResult:
        """Execute a single command or pipeline (no &&, ||, ;)."""
        # Expand aliases
        command = self._expand_aliases(command)

        # Shell expansions (in order: brace, variable, command substitution)
        try:
            command = self._expand_braces(command)
            command = self._expand_variables(command)
            command = await self._expand_command_substitution(command, session_id)
        except Exception as e:
            return TerminalResult(
                stderr=f"vsh: expansion error: {e}",
                exit_code=1,
                cwd=self.cwd,
            )

        # Parse pipeline
        try:
            pipeline = self._parse_pipeline(command)
        except ValueError as e:
            return TerminalResult(
                stderr=f"vsh: {e}",
                exit_code=1,
                cwd=self.cwd,
            )

        # Validate safety
        for cmd in pipeline:
            is_safe, error_msg = self._is_safe(cmd)
            if not is_safe:
                return TerminalResult(
                    stderr=f"vsh: {cmd.program}: {error_msg}",
                    exit_code=126,
                    cwd=self.cwd,
                )

        # Execute pipeline
        try:
            return await self._execute_pipeline(pipeline, session_id)
        except Exception as e:
            logger.error(f"Terminal error: {e}", exc_info=True)
            return TerminalResult(
                stderr=f"vsh: {e}",
                exit_code=1,
                cwd=self.cwd,
            )

    def get_prompt(self) -> str:
        """Get the shell prompt string."""
        user = self.env.get("USER", "agent")
        home = self.home_dir.rstrip("/")
        cwd_display = self.cwd.replace(self.home_dir, "~/")
        if cwd_display == home:
            cwd_display = "~"
        return f"{user}@{self.hostname}:{cwd_display}$ "

    def get_completions(self, partial: str) -> list[str]:
        """Get tab completions for partial input."""
        if " " not in partial:
            # Command completion
            matches = [c for c in ALLOWED_COMMANDS if c.startswith(partial)]
            matches.extend([a for a in self.aliases if a.startswith(partial)])
            return sorted(set(matches))

        # Path completion would go here (requires async)
        return []

    # =========================================================================
    # Parsing
    # =========================================================================

    def _expand_aliases(self, command: str) -> str:
        """Expand aliases in command."""
        parts = command.split()
        if parts and parts[0] in self.aliases:
            parts[0] = self.aliases[parts[0]]
            return " ".join(parts)
        return command

    def _expand_variables(self, command: str) -> str:
        """Expand shell variables like $VAR and ${VAR}."""

        # Pattern for ${VAR} and $VAR
        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            return self.env.get(var_name, "")

        # ${VAR} form
        command = re.sub(r"\$\{([^}]+)\}", replace_var, command)
        # $VAR form (word characters only)
        command = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", replace_var, command)

        return command

    def _expand_heredoc(self, command: str, session_id: str) -> str:
        """Extract heredoc and convert to a write + remaining command.

        Handles patterns like:
            cat > /.gxl/data.txt << 'EOF'
            hello world
            EOF
            head /.gxl/data.txt

        Converts to:
            echo 'hello world' > /.gxl/data.txt && head /.gxl/data.txt
        """
        import re as _re

        # Match: command << ['"]?DELIM['"]? \n content \n DELIM
        m = _re.search(
            r'<<\s*[\'"]?(\w+)[\'"]?\s*\n',
            command,
        )
        if not m:
            return command

        delimiter = m.group(1)
        heredoc_start = m.start()  # position of <<
        content_start = m.end()    # position after delimiter\n

        # Find the closing delimiter (must be on its own line)
        pattern = _re.compile(r'^' + _re.escape(delimiter) + r'\s*$', _re.MULTILINE)
        end_match = pattern.search(command, content_start)
        if not end_match:
            return command

        heredoc_content = command[content_start:end_match.start()].rstrip("\n")
        after_heredoc = command[end_match.end():].strip()

        # Extract the command before << (e.g., "cat > /.gxl/script.py")
        before_heredoc = command[:heredoc_start].strip()

        # Find redirect target in the before part
        redirect_match = _re.search(r'>\s*(\S+)', before_heredoc)
        if redirect_match:
            target_file = redirect_match.group(1)
            # Build: write the content to the file, then run remaining commands
            # Use the _session_files_write path by converting to echo > redirect
            write_cmd = f'echo {json.dumps(heredoc_content)} > {target_file}'
            if after_heredoc:
                return f"{write_cmd} && {after_heredoc}"
            return write_cmd
        else:
            # No redirect — the heredoc content becomes stdin for the command
            # Convert to echo | command
            cmd_program = before_heredoc.strip()
            piped = f'echo {json.dumps(heredoc_content)} | {cmd_program}'
            if after_heredoc:
                return f"{piped} && {after_heredoc}"
            return piped

    def _expand_braces(self, command: str) -> str:
        """Expand brace expressions like {a,b,c} and {1..5}.

        Properly skips braces inside single or double quoted strings.
        """
        result = command

        max_iterations = 10
        for _ in range(max_iterations):
            # Find brace expressions that are NOT inside quoted strings
            match = self._find_unquoted_brace(result)
            if not match:
                break

            start, end = match
            brace_expr = result[start:end]
            expanded = self._expand_single_brace(brace_expr)

            if expanded == brace_expr:
                break

            result = result[:start] + expanded + result[end:]

        return result

    def _find_unquoted_brace(self, text: str) -> tuple[int, int] | None:
        """Find the first {..} brace expression that is not inside quotes."""
        in_single = False
        in_double = False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif ch == "{" and not in_single and not in_double:
                depth = 1
                j = i + 1
                while j < len(text) and depth > 0:
                    if text[j] == "{":
                        depth += 1
                    elif text[j] == "}":
                        depth -= 1
                    j += 1
                if depth == 0:
                    return (i, j)
            i += 1
        return None

    def _expand_single_brace(self, expr: str) -> str:
        """Expand a single brace expression."""
        inner = expr[1:-1]  # Remove { }

        # Range expansion: {1..5} or {a..z}
        range_match = re.match(r"^(\d+)\.\.(\d+)$", inner)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            step = 1 if start <= end else -1
            return " ".join(str(i) for i in range(start, end + step, step))

        # Character range: {a..z}
        char_range_match = re.match(r"^([a-zA-Z])\.\.([a-zA-Z])$", inner)
        if char_range_match:
            start, end = ord(char_range_match.group(1)), ord(char_range_match.group(2))
            step = 1 if start <= end else -1
            return " ".join(chr(i) for i in range(start, end + step, step))

        # List expansion: {a,b,c}
        if "," in inner:
            items = inner.split(",")
            return " ".join(items)

        return expr  # No expansion

    async def _expand_command_substitution(self, command: str, session_id: str) -> str:
        """Expand command substitutions like $(cmd) and `cmd`."""
        result = command

        # Safety: limit recursion depth
        max_iterations = 5

        for _ in range(max_iterations):
            # Find $(...) - need to handle nested parens
            match = re.search(r"\$\(([^)]+)\)", result)
            if match:
                inner_cmd = match.group(1)
                # Execute the inner command
                inner_result = await self._execute_single(inner_cmd, session_id)
                # Replace with output (strip trailing newline)
                output = (
                    inner_result.stdout.rstrip("\n")
                    if inner_result.exit_code == 0
                    else ""
                )
                result = result[: match.start()] + output + result[match.end() :]
                continue

            # Find `...` (backticks)
            match = re.search(r"`([^`]+)`", result)
            if match:
                inner_cmd = match.group(1)
                inner_result = await self._execute_single(inner_cmd, session_id)
                output = (
                    inner_result.stdout.rstrip("\n")
                    if inner_result.exit_code == 0
                    else ""
                )
                result = result[: match.start()] + output + result[match.end() :]
                continue

            break  # No more substitutions

        return result

    async def _expand_glob_async(
        self, pattern: str, session_id: str = "default"
    ) -> list[str]:
        """Expand glob patterns like *.txt, file?.log using virtual filesystem.

        Returns list of matching paths, or [pattern] if no matches.
        Supports both virtual filesystem paths and /session_files/ paths.
        """
        import fnmatch

        if not any(c in pattern for c in "*?["):
            return [pattern]  # Not a glob pattern

        # Determine the directory to list
        if "/" in pattern:
            dir_part = pattern.rsplit("/", 1)[0]
            file_pattern = pattern.rsplit("/", 1)[1]
        else:
            dir_part = self.cwd
            file_pattern = pattern

        # Handle /session_files/ paths via session_files_ls
        if self._is_session_files_path(dir_part):
            try:
                ls_result = await self._session_files_ls(dir_part, session_id)
                if ls_result.exit_code != 0:
                    return [pattern]
                matches = []
                for line in ls_result.stdout.splitlines():
                    line = line.strip()
                    if not line or line == "(empty)":
                        continue
                    # Format: "-rw-r--r--    1234  filename" or "drwxr-xr-x  dirname/"
                    parts = line.split(None, 2)
                    if len(parts) >= 2:
                        name = parts[-1].rstrip("/")
                    else:
                        continue
                    if fnmatch.fnmatch(name, file_pattern):
                        matches.append(f"{dir_part}/{name}")
                return matches if matches else [pattern]
            except Exception:
                return [pattern]

        if not self.fs:
            return [pattern]  # No filesystem to query

        # Validate and normalize path
        full_dir = self._validate_path(dir_part)
        if full_dir is None:
            return [pattern]

        try:
            # List directory contents
            result = await self.fs._ls(
                path=full_dir if full_dir.endswith("/") else full_dir + "/",
                session_id=session_id,
            )
            if "error" in result:
                return [pattern]

            contents = result.get("contents", [])
            matches = []

            for item in contents:
                if isinstance(item, dict):
                    name = item.get("name", "")
                elif isinstance(item, str):
                    name = item
                else:
                    continue

                if fnmatch.fnmatch(name, file_pattern):
                    if "/" in pattern:
                        matches.append(f"{dir_part}/{name}")
                    else:
                        matches.append(name)

            return matches if matches else [pattern]
        except Exception:
            return [pattern]

    async def _expand_args_globs(
        self, args: list[str], session_id: str = "default"
    ) -> list[str]:
        """Expand any glob patterns in argument list."""
        expanded = []
        for arg in args:
            if any(c in arg for c in "*?[") and not arg.startswith("-"):
                glob_expanded = await self._expand_glob_async(arg, session_id)
                expanded.extend(glob_expanded)
            else:
                expanded.append(arg)
        return expanded

    def _parse_pipeline(self, command: str) -> list[ParsedCommand]:
        """Parse a command string into a pipeline of commands."""
        # Split by pipe
        pipe_parts = self._split_pipes(command)

        commands = []
        for part in pipe_parts:
            cmd = self._parse_single_command(part.strip())
            commands.append(cmd)

        return commands

    def _split_pipes(self, command: str) -> list[str]:
        """Split command by pipes, respecting quotes."""
        parts = []
        current = []
        in_quotes = False
        quote_char = None

        for char in command:
            if char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
                current.append(char)
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current.append(char)
            elif char == "|" and not in_quotes:
                parts.append("".join(current))
                current = []
            else:
                current.append(char)

        if current:
            parts.append("".join(current))

        return [p for p in parts if p.strip()]

    @staticmethod
    def _find_redirect_outside_quotes(command: str, token: str) -> int:
        """Find the position of a redirect token (>, >>, <) outside of quotes.

        Returns the index of the token in the command string, or -1 if not found
        outside of quotes. This prevents splitting on > or < inside SQL strings
        like: sql "SELECT * FROM t WHERE x >= 5"
        """
        in_single = False
        in_double = False
        i = 0
        while i < len(command):
            ch = command[i]
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif not in_single and not in_double:
                if command[i : i + len(token)] == token:
                    return i
            i += 1
        return -1

    def _parse_single_command(self, command: str) -> ParsedCommand:
        """Parse a single command (no pipes)."""
        # Strip 2>/dev/null and 2>&1 (stderr redirects vsh doesn't support, silently discard)
        import re
        command = re.sub(r'\s+2>/dev/null\b', '', command)
        command = re.sub(r'\s+2>&1\b', '', command)

        # Handle redirects (quote-aware to avoid splitting on > or < inside strings)
        stdout_redirect = None
        stdin_redirect = None
        append_mode = False

        # Check for >> first (must come before > check)
        pos = self._find_redirect_outside_quotes(command, ">>")
        if pos >= 0:
            stdout_redirect = command[pos + 2 :].strip()
            command = command[:pos]
            append_mode = True
        else:
            # Check for single > (but not >=, which is SQL comparison)
            pos = self._find_redirect_outside_quotes(command, ">")
            if pos >= 0:
                # Skip if it's part of >= (SQL operator)
                if pos + 1 < len(command) and command[pos + 1] == "=":
                    pass  # Not a redirect, it's >=
                else:
                    stdout_redirect = command[pos + 1 :].strip()
                    command = command[:pos]

        pos = self._find_redirect_outside_quotes(command, "<")
        if pos >= 0:
            # Skip if it's part of <= (SQL operator)
            if pos + 1 < len(command) and command[pos + 1] == "=":
                pass  # Not a redirect, it's <=
            else:
                stdin_redirect = command[pos + 1 :].strip()
                command = command[:pos]

        # Parse command and args
        try:
            tokens = shlex.split(command.strip())
        except ValueError as e:
            raise ValueError(f"parse error: {e}")

        if not tokens:
            raise ValueError("empty command")

        return ParsedCommand(
            program=tokens[0],
            args=tokens[1:],
            stdin_redirect=stdin_redirect,
            stdout_redirect=stdout_redirect,
            append_redirect=append_mode,
        )

    # =========================================================================
    # Safety
    # =========================================================================

    def _is_safe(self, cmd: ParsedCommand) -> tuple[bool, str | None]:
        """Check if a command is safe to execute.

        Returns:
            (is_safe, error_message) - error_message is None if safe
        """
        # Check for bash keywords (unsupported shell syntax)
        if cmd.program in BASH_KEYWORDS:
            return (
                False,
                f"bash syntax not supported ('{cmd.program}' is a shell keyword). vsh only supports basic commands and pipes.",
            )

        # Check against blocklist first
        if cmd.program in BLOCKED_COMMANDS:
            return False, "permission denied"

        # Check against allowlist
        if cmd.program not in ALLOWED_COMMANDS:
            # Provide helpful suggestions for common but unsupported commands
            suggestions = {
                # Text processing alternatives
                "perl": "Use grep for pattern matching. E.g., 'grep -P \"pattern\" file'.",
                # File operations
                "find": "Use 'ls' with globs. E.g., 'ls /papers/*/sections/*.lines' or 'ls -R /papers/'.",
                "locate": "Use 'search' to find papers by content, or 'ls' with globs for files.",
                "xargs": "Use shell pipelines directly. E.g., 'cat file | grep pattern'.",
                "tee": "Output is returned directly - no need for tee.",
                # Editors (read-only filesystem)
                "nano": "Filesystem is read-only. Use 'cat', 'head', or 'less' to view content.",
                "vim": "Filesystem is read-only. Use 'cat', 'head', or 'less' to view content.",
                "vi": "Filesystem is read-only. Use 'cat', 'head', or 'less' to view content.",
                "emacs": "Filesystem is read-only. Use 'cat', 'head', or 'less' to view content.",
                # Network (sandboxed)
                "wget": "Network access not available. Use 'search' to find papers in the database.",
                "ssh": "Network access not available. This is a sandboxed virtual filesystem.",
                # Write operations (/papers/ is read-only)
                "rm": "/papers/ is read-only. Use /.gxl/ for files you create.",
                "mv": "/papers/ is read-only. Use /.gxl/ for files you create.",
                "cp": "/papers/ is read-only. To copy paper content: cat content.lines > /.gxl/paper.txt",
                "mkdir": "/papers/ is read-only. Use /.gxl/ for writable storage.",
                "touch": "/papers/ is read-only. Write files to /.gxl/ instead: echo 'data' > /.gxl/file.txt",
                "chmod": "Filesystem is read-only. Cannot modify permissions.",
                # Misc
                "sudo": "No elevated privileges in virtual filesystem.",
                "apt": "Package management not available in virtual filesystem.",
                "brew": "Package management not available in virtual filesystem.",
                "pip": "Package management not available. Python has pandas, numpy, scipy, scikit-learn, seaborn, statsmodels available.",
            }
            hint = suggestions.get(cmd.program)
            if hint:
                return False, f"command not found. Suggestion: {hint}"
            return (
                False,
                f"command not found. Available: {', '.join(sorted(ALLOWED_COMMANDS)[:10])}...",
            )

        return True, None

    def _validate_path(self, path: str) -> str | None:
        """
        Validate and normalize a path within the virtual filesystem.
        Returns None if path escapes the sandbox.
        """
        # Handle relative paths
        if not path.startswith("/"):
            path = os.path.join(self.cwd, path)

        # Normalize
        path = os.path.normpath(path)

        # Determine allowed roots from filesystem's path parser
        root_name = "papers"  # Default for BioMedRxiv
        if self.fs and hasattr(self.fs, "path_parser"):
            root_name = self.fs.path_parser.root_name

        # Must stay within allowed roots (plus /tmp for cached results, /youtube for transcripts, /.gxl for scratch)
        allowed_roots = [
            f"/{root_name}/",
            f"/{root_name}",
            "/tmp/",
            "/tmp",
            "/youtube/",
            "/youtube",
            "/.gxl/",
            "/.gxl",
        ]
        if not any(
            path.startswith(root) or path == root.rstrip("/") for root in allowed_roots
        ):
            return None

        return path

    # =========================================================================
    # E2B Sandbox Integration (for /.gxl/)
    # =========================================================================

    def _is_session_files_path(self, path: str) -> bool:
        """Check if a path is under /.gxl/."""
        normalized = path.rstrip("/")
        return normalized == "/.gxl" or normalized.startswith("/.gxl/")

    def _session_files_to_sandbox_path(self, path: str) -> str:
        """Convert /.gxl/foo.csv to the relative path foo.csv for the sandbox."""
        stripped = path.lstrip("/")
        if stripped.startswith(".gxl/"):
            return stripped[len(".gxl/") :]
        if stripped == ".gxl":
            return ""
        return stripped

    def _should_use_e2b(self) -> bool:
        """Determine whether to use E2B or local filesystem for /.gxl/.

        Respects SANDBOX_PROVIDER to stay consistent with GXLFileSystem's provider
        selection so writes and reads always use the same backend.
        """
        if self._use_e2b is not None:
            return self._use_e2b

        provider = os.getenv("SANDBOX_PROVIDER", "local").lower()
        self._use_e2b = provider in ("e2b", "gcr")

        logger.info(f"Session files backend: {'E2B' if self._use_e2b else 'local'}")
        return self._use_e2b

    def _get_local_session_dir(self, session_id: str) -> str:
        """Get the local /.gxl/ scratch directory path.

        Maps to CWD/.gxl/ so files land in the user's project directory.
        """
        base = os.getenv("LOCAL_SESSION_STORAGE_ROOT", os.path.join(os.getcwd(), ".gxl"))
        return base

    def _ensure_local_session_dir(self, session_id: str) -> str:
        """Ensure the local session files directory exists and return its path."""
        d = self._get_local_session_dir(session_id)
        os.makedirs(d, exist_ok=True)
        return d

    async def _get_e2b_provider(self, session_id: str):
        """Get or create an E2B sandbox provider for a session. Returns None if E2B is not available."""
        if session_id in self._e2b_providers:
            return self._e2b_providers[session_id]

        try:
            from gxl_filesystem.e2b_provider import E2BProvider
        except ImportError:
            logger.debug("E2B provider not available (gxl_filesystem not installed)")
            return None

        api_key = os.getenv("E2B_API_KEY")
        if not api_key:
            logger.debug("E2B_API_KEY not set, session_files not available")
            return None

        provider = E2BProvider(session_id=session_id, api_key=api_key)
        self._e2b_providers[session_id] = provider
        return provider

    # ---- /.gxl/ operations (dispatch to E2B or local) ----

    async def _session_files_ls(self, path: str, session_id: str) -> TerminalResult:
        """List files at /.gxl/."""
        if self._should_use_e2b():
            result = await self._session_files_ls_e2b(path, session_id)
            # Merge with local files (e.g. Datalab output written directly to disk)
            local_result = self._session_files_ls_local(path, session_id)
            if local_result.exit_code == 0 and local_result.stdout.strip():
                if result.exit_code != 0 or not result.stdout.strip():
                    return local_result
                # Merge: combine unique entries from both
                e2b_lines = set(result.stdout.strip().splitlines())
                local_lines = set(local_result.stdout.strip().splitlines())
                merged = sorted(e2b_lines | local_lines)
                return TerminalResult(stdout="\n".join(merged) + "\n", cwd=self.cwd)
            return result
        return self._session_files_ls_local(path, session_id)

    async def _session_files_cat(self, path: str, session_id: str) -> TerminalResult:
        """Read a file from /.gxl/."""
        if self._should_use_e2b():
            return await self._session_files_cat_e2b(path, session_id)
        return self._session_files_cat_local(path, session_id)

    async def _session_files_write(
        self, path: str, content: str, session_id: str, append: bool = False
    ) -> TerminalResult:
        """Write content to a file at /.gxl/."""
        if self._should_use_e2b():
            return await self._session_files_write_e2b(
                path, content, session_id, append=append
            )
        return self._session_files_write_local(path, content, session_id, append=append)

    async def _session_files_grep(
        self, pattern: str, path: str, flags: str, session_id: str
    ) -> TerminalResult:
        """Grep a file at /.gxl/."""
        # For grep, always read the file content and return it for the caller to grep
        cat_result = await self._session_files_cat(path, session_id)
        return cat_result  # Caller will grep the content

    async def _session_files_head_tail(
        self, path: str, n: int, is_tail: bool, session_id: str
    ) -> TerminalResult:
        """Head or tail a file at /.gxl/."""
        cat_result = await self._session_files_cat(path, session_id)
        if cat_result.exit_code != 0:
            return cat_result
        lines = cat_result.stdout.split("\n")
        if is_tail:
            selected = lines[-n:]
        else:
            selected = lines[:n]
        return TerminalResult(stdout="\n".join(selected) + "\n", cwd=self.cwd)

    async def _session_files_wc(self, path: str, session_id: str) -> TerminalResult:
        """Word count on a file at /.gxl/."""
        cat_result = await self._session_files_cat(path, session_id)
        if cat_result.exit_code != 0:
            return cat_result
        text = cat_result.stdout
        l = text.count("\n")
        w = len(text.split())
        c = len(text)
        rel = self._session_files_to_sandbox_path(path) or "."
        return TerminalResult(stdout=f"{l:8}{w:8}{c:8} {rel}\n", cwd=self.cwd)

    # ---- Local filesystem implementations (dev) ----

    def _session_files_ls_local(self, path: str, session_id: str) -> TerminalResult:
        """List files in the local session directory."""
        local_dir = self._ensure_local_session_dir(session_id)
        rel_path = self._session_files_to_sandbox_path(path)
        target = os.path.join(local_dir, rel_path) if rel_path else local_dir

        if not os.path.exists(target):
            return TerminalResult(stdout="(empty)\n", cwd=self.cwd)

        if os.path.isfile(target):
            size = os.path.getsize(target)
            return TerminalResult(
                stdout=f"-rw-r--r-- {size:>8} {os.path.basename(target)}\n",
                cwd=self.cwd,
            )

        try:
            entries = sorted(os.listdir(target))
            if not entries:
                return TerminalResult(stdout="(empty)\n", cwd=self.cwd)
            lines = []
            for entry in entries:
                if entry.startswith("."):
                    continue
                full = os.path.join(target, entry)
                if os.path.isdir(full):
                    lines.append(f"drwxr-xr-x  {entry}/")
                else:
                    size = os.path.getsize(full)
                    lines.append(f"-rw-r--r-- {size:>8}  {entry}")
            return TerminalResult(
                stdout="\n".join(lines) + "\n" if lines else "(empty)\n", cwd=self.cwd
            )
        except Exception as e:
            return TerminalResult(
                stderr=f"vsh: ls: {path}: {e}", exit_code=1, cwd=self.cwd
            )

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

    def _session_files_cat_local(self, path: str, session_id: str) -> TerminalResult:
        """Read a file from the local session directory."""
        local_dir = self._ensure_local_session_dir(session_id)
        rel_path = self._session_files_to_sandbox_path(path)
        if not rel_path:
            return TerminalResult(
                stderr="vsh: cat: /.gxl/: Is a directory",
                exit_code=1,
                cwd=self.cwd,
            )

        target = os.path.join(local_dir, rel_path)

        # Prevent path traversal
        if not os.path.realpath(target).startswith(os.path.realpath(local_dir)):
            return TerminalResult(
                stderr=f"vsh: cat: {path}: Permission denied", exit_code=1, cwd=self.cwd
            )

        if not os.path.exists(target):
            return TerminalResult(
                stderr=f"vsh: cat: {path}: No such file", exit_code=1, cwd=self.cwd
            )

        # Image files: return a citation marker instead of binary content
        ext = os.path.splitext(rel_path)[1].lower()
        if ext in self._IMAGE_EXTENSIONS:
            size_bytes = os.path.getsize(target)
            if size_bytes < 1024:
                size_str = f"{size_bytes}B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f}KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.1f}MB"
            return TerminalResult(
                stdout=(
                    f"Image: {rel_path} ({size_str})\n"
                    f'{{{{"type": "image", "path": "{rel_path}"}}}}\n'
                ),
                cwd=self.cwd,
            )

        try:
            with open(target) as f:
                content = f.read()
            return TerminalResult(stdout=content, cwd=self.cwd)
        except UnicodeDecodeError:
            size_bytes = os.path.getsize(target)
            return TerminalResult(
                stdout=f"[Binary file: {rel_path} ({size_bytes} bytes)]\n",
                cwd=self.cwd,
            )
        except Exception as e:
            return TerminalResult(
                stderr=f"vsh: cat: {path}: {e}", exit_code=1, cwd=self.cwd
            )

    def _session_files_write_local(
        self, path: str, content: str, session_id: str, append: bool = False
    ) -> TerminalResult:
        """Write content to a file in the local session directory."""
        local_dir = self._ensure_local_session_dir(session_id)
        rel_path = self._session_files_to_sandbox_path(path)
        if not rel_path:
            return TerminalResult(
                stderr="vsh: write: /.gxl/: Is a directory",
                exit_code=1,
                cwd=self.cwd,
            )

        target = os.path.join(local_dir, rel_path)

        # Prevent path traversal
        if not os.path.realpath(
            os.path.dirname(target) if os.path.dirname(target) else local_dir
        ).startswith(os.path.realpath(local_dir)):
            return TerminalResult(
                stderr=f"vsh: write: {path}: Permission denied",
                exit_code=1,
                cwd=self.cwd,
            )

        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            mode = "a" if append else "w"
            with open(target, mode) as f:
                f.write(content)
            return TerminalResult(cwd=self.cwd)
        except Exception as e:
            return TerminalResult(
                stderr=f"vsh: write: {path}: {e}", exit_code=1, cwd=self.cwd
            )

    # ---- E2B sandbox implementations (staging/prod) ----

    async def _session_files_ls_e2b(self, path: str, session_id: str) -> TerminalResult:
        """List files in the E2B sandbox at /.gxl/."""
        provider = await self._get_e2b_provider(session_id)
        if not provider:
            return TerminalResult(
                stderr="vsh: ls: /.gxl/: E2B sandbox not available (E2B_API_KEY not set)",
                exit_code=1,
                cwd=self.cwd,
            )

        try:
            listing = await provider.list_files()
            return TerminalResult(stdout=listing + "\n", cwd=self.cwd)
        except Exception as e:
            return TerminalResult(
                stderr=f"vsh: ls: {path}: {e}", exit_code=1, cwd=self.cwd
            )

    async def _session_files_cat_e2b(
        self, path: str, session_id: str
    ) -> TerminalResult:
        """Read a file from the E2B sandbox at /.gxl/."""
        provider = await self._get_e2b_provider(session_id)
        if not provider:
            return TerminalResult(
                stderr="vsh: cat: E2B sandbox not available (E2B_API_KEY not set)",
                exit_code=1,
                cwd=self.cwd,
            )

        try:
            rel_path = self._session_files_to_sandbox_path(path)
            if not rel_path:
                return TerminalResult(
                    stderr="vsh: cat: /.gxl/: Is a directory",
                    exit_code=1,
                    cwd=self.cwd,
                )

            # Image files: return a citation marker instead of binary
            ext = os.path.splitext(rel_path)[1].lower()
            if ext in self._IMAGE_EXTENSIONS:
                return TerminalResult(
                    stdout=(
                        f"Image: {rel_path}\n"
                        f'{{{{"type": "image", "path": "{rel_path}"}}}}\n'
                    ),
                    cwd=self.cwd,
                )

            content = await provider.read_file(rel_path)
            return TerminalResult(stdout=content, cwd=self.cwd)
        except Exception as e:
            # ResponseManager writes via GXLFileSystem with a files/ prefix,
            # but the terminal path resolution doesn't include it for E2B.
            # Try the files/-prefixed path before falling back to local.
            if rel_path:
                try:
                    content = await provider.read_file(f"files/{rel_path}")
                    return TerminalResult(stdout=content, cwd=self.cwd)
                except Exception:
                    pass
            if not isinstance(e, FileNotFoundError):
                logger.warning(f"E2B read failed for {path}: {e}, trying local fallback")
            # Fall back to local filesystem (e.g. files written by Datalab)
            local_result = self._session_files_cat_local(path, session_id)
            if local_result.exit_code == 0:
                return local_result
            return TerminalResult(
                stderr=f"vsh: cat: {path}: No such file",
                exit_code=1,
                cwd=self.cwd,
            )

    async def _session_files_write_e2b(
        self, path: str, content: str, session_id: str, append: bool = False
    ) -> TerminalResult:
        """Write content to a file in the E2B sandbox at /.gxl/."""
        provider = await self._get_e2b_provider(session_id)
        if not provider:
            return TerminalResult(
                stderr="vsh: write: E2B sandbox not available (E2B_API_KEY not set)",
                exit_code=1,
                cwd=self.cwd,
            )

        try:
            rel_path = self._session_files_to_sandbox_path(path)
            if not rel_path:
                return TerminalResult(
                    stderr="vsh: write: /.gxl/: Is a directory",
                    exit_code=1,
                    cwd=self.cwd,
                )

            if append:
                try:
                    existing = await provider.read_file(rel_path)
                    content = existing + content
                except FileNotFoundError:
                    pass

            await provider.write_file(rel_path, content)
            return TerminalResult(cwd=self.cwd)
        except Exception as e:
            return TerminalResult(
                stderr=f"vsh: write: {path}: {e}", exit_code=1, cwd=self.cwd
            )

    # =========================================================================
    # Pipeline Execution
    # =========================================================================

    async def _execute_pipeline(
        self,
        pipeline: list[ParsedCommand],
        session_id: str,
    ) -> TerminalResult:
        """Execute a pipeline of commands.

        Supports auto-chaining: if a command produces a results_id (e.g. search),
        the next grep in the pipeline automatically receives --from <results_id>.
        This enables: search "query" | grep "pattern"
        """
        stdin = ""
        prev_results_id = None

        for i, cmd in enumerate(pipeline):
            # Auto-chain: if previous command produced a results_id and this
            # is a grep without --from, inject --from automatically
            if (prev_results_id
                    and cmd.program == "grep"
                    and "--from" not in cmd.args):
                cmd.args = ["--from", prev_results_id] + cmd.args

            # Get handler
            handler = self._handlers.get(cmd.program)
            if not handler:
                # Provide helpful suggestions for common unsupported commands
                suggestions = {
                    "sed": "Use head/tail with pipes. E.g., 'head -N | tail -M' for line ranges, or 'grep' with -v for filtering.",
                    "awk": "Use cut or grep for text processing. E.g., 'cut -d: -f1' for field extraction.",
                    "perl": "Use grep for pattern matching. E.g., 'grep -P \"pattern\" file'.",
                    "find": "Use 'ls' with globs. E.g., 'ls /papers/*/sections/*.lines' to find files.",
                    "xargs": "Use shell pipelines directly. E.g., 'cat file | grep pattern' instead of xargs.",
                    "tee": "Output is returned directly - no need for tee in virtual filesystem.",
                    "nano": "Files are read-only in the virtual filesystem. Use 'cat' to view content.",
                    "vim": "Files are read-only in the virtual filesystem. Use 'cat' to view content.",
                    "wget": "External network access not available. Use 'search' to find content in the database.",
                }
                hint = suggestions.get(cmd.program, "")
                hint_msg = (
                    f"\nHint: {hint}"
                    if hint
                    else "\nUse 'help' to see available commands."
                )
                return TerminalResult(
                    stderr=f"vsh: {cmd.program}: command not found{hint_msg}",
                    exit_code=127,
                    cwd=self.cwd,
                )

            # Auto-chain: if previous command produced a results_id and this
            # command is grep (without --from), inject --from automatically
            if (prev_results_id
                    and cmd.program in ("grep", "egrep", "fgrep")
                    and "--from" not in cmd.args):
                cmd.args = ["--from", prev_results_id] + cmd.args

            # Intercept --help / -h for any command
            if "--help" in cmd.args or (cmd.args == ["-h"]):
                help_text = COMMAND_HELP.get(cmd.program, f"No help available for '{cmd.program}'")
                return TerminalResult(stdout=help_text + "\n", cwd=self.cwd)

            # Execute
            result = await handler(cmd.args, stdin=stdin, session_id=session_id)

            if result.exit_code != 0:
                return result

            # Track results_id for chaining
            prev_results_id = (
                result.metadata.get("results_id")
                if result.metadata else None
            )

            stdin = result.stdout

        # Handle final redirect if present
        final_cmd = pipeline[-1]
        if final_cmd.stdout_redirect:
            redirect_path = self._validate_path(final_cmd.stdout_redirect)
            if redirect_path and self._is_session_files_path(redirect_path):
                write_result = await self._session_files_write(
                    redirect_path,
                    result.stdout,
                    session_id,
                    append=final_cmd.append_redirect,
                )
                if write_result.exit_code != 0:
                    return write_result
                rel = self._session_files_to_sandbox_path(redirect_path)
                result = TerminalResult(
                    stdout=f"[Output written to /.gxl/{rel}]\n",
                    cwd=self.cwd,
                )
            else:
                result.stdout = f"[Output redirect to {final_cmd.stdout_redirect} - only /.gxl/ is writable]\n"

        return result

    # =========================================================================
    # Command Implementations
    # =========================================================================

    async def _cmd_cd(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Change directory."""
        target = args[0] if args else self.env.get("HOME", "/papers/")

        # Handle ~ expansion
        if target.startswith("~"):
            target = target.replace("~", "/papers", 1)

        # Validate path
        new_path = self._validate_path(target)
        if new_path is None:
            return TerminalResult(
                stderr=f"vsh: cd: {target}: Permission denied",
                exit_code=1,
                cwd=self.cwd,
            )

        # Check if it's a valid directory (would need filesystem check)
        # For now, accept any valid path
        old_cwd = self.cwd
        self.cwd = new_path if new_path.endswith("/") else new_path + "/"
        self.env["PWD"] = self.cwd

        # Show context when entering /.gxl/
        stdout = ""
        if self.cwd != old_cwd and self._is_session_files_path(self.cwd):
            if self._should_use_e2b():
                stdout = "Session files (E2B sandbox). Files here persist to GCS.\n"
            else:
                stdout = "Session files (local). Files here persist for the session.\n"

        # Auto-show paper summary when entering a paper directory for the first time
        if (
            self.fs
            and self.cwd != old_cwd
            and not self._is_session_files_path(self.cwd)
        ):
            # Check if we just entered a paper-level directory: /papers/UUID/ or /documents/UUID/
            parts = self.cwd.strip("/").split("/")
            if len(parts) == 2 and parts[0] in ("papers", "documents"):
                try:
                    doc_id = parts[1]
                    stat_result = await self.fs._stat(
                        path=self.cwd, session_id=session_id
                    )
                    if isinstance(stat_result, dict) and "error" not in stat_result:
                        title = stat_result.get("title", "")
                        total_chars = stat_result.get("total_chars", 0)
                        lines_count = stat_result.get("total_lines", "")
                        sections = stat_result.get("section_count", "")
                        info_parts = []
                        if title:
                            info_parts.append(title[:80])
                        meta = []
                        if total_chars:
                            meta.append(f"~{total_chars // 4} tokens")
                        elif lines_count:
                            meta.append(f"~{lines_count * 40 // 4} tokens")
                        if sections:
                            meta.append(f"{sections} sections")
                        if meta:
                            info_parts.append("  " + ", ".join(meta))
                        if info_parts:
                            stdout = "\n".join(info_parts) + "\n"
                except Exception:
                    pass  # Don't fail cd if stat fails

        return TerminalResult(stdout=stdout, cwd=self.cwd)

    async def _cmd_pwd(
        self, args: list[str], stdin: str = "", **kwargs
    ) -> TerminalResult:
        """Print working directory."""
        return TerminalResult(stdout=self.cwd + "\n", cwd=self.cwd)

    async def _cmd_ls(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """List directory contents."""
        # Parse args
        show_all = "-a" in args or "-la" in args or "-al" in args
        long_format = "-l" in args or "-la" in args or "-al" in args

        # Get path(s) - support globs
        path_args = [a for a in args if not a.startswith("-")]
        if path_args:
            # Expand globs
            path_args = await self._expand_args_globs(path_args, session_id)
        path = path_args[0] if path_args else self.cwd

        # Validate path
        full_path = self._validate_path(path)
        if full_path is None:
            return TerminalResult(
                stderr=f"vsh: ls: {path}: Permission denied",
                exit_code=1,
                cwd=self.cwd,
            )

        # Handle /tmp/searches/ directory (legacy alias for cached/durable search results)
        if full_path.rstrip("/") == "/tmp/searches" or full_path.startswith(
            "/tmp/searches/"
        ):
            if hasattr(self, "_search_results_cache") and self._search_results_cache:
                files = list(self._search_results_cache.keys())
                output_lines = [f.split("/")[-1] for f in files]
                if not output_lines:
                    return TerminalResult(stdout="(empty)\n", cwd=self.cwd)
                return TerminalResult(
                    stdout="\n".join(output_lines) + "\n", cwd=self.cwd
                )
            else:
                session_results = await self._session_files_ls(
                    "/session_files/searches", session_id
                )
                if session_results.exit_code == 0:
                    return session_results
                return TerminalResult(
                    stdout="(empty - run 'searches' command first)\n", cwd=self.cwd
                )

        # Handle /tmp/ directory
        if full_path.rstrip("/") == "/tmp":
            output_lines = ["searches/"]
            return TerminalResult(stdout="\n".join(output_lines) + "\n", cwd=self.cwd)

        # Handle /.gxl/ via sandbox (E2B in staging/prod, local in dev)
        if self._is_session_files_path(full_path):
            return await self._session_files_ls(full_path, session_id)

        # Use filesystem module if available
        if self.fs:
            try:
                # Ensure path ends with / for directories
                ls_path = full_path if full_path.endswith("/") else full_path + "/"
                result = await self.fs._ls(path=ls_path, session_id=session_id)
                if "error" in result:
                    error_msg = result["error"]
                    hint = result.get("hint", "")

                    # Only show special message for listing /papers/ root with no specific UUID
                    if (
                        "Cannot list path: /papers/" in error_msg
                        and ls_path.rstrip("/") == "/papers"
                    ):
                        error_msg = (
                            f"Cannot list all papers directly. Use 'papers_find' to search or specify a paper UUID.\n"
                            f"Example: ls /papers/<uuid>/\n"
                            f"        papers_find --query 'CRISPR'"
                        )
                    elif hint:
                        error_msg = f"{error_msg}\n{hint}"

                    return TerminalResult(
                        stderr=f"vsh: ls: {error_msg}",
                        exit_code=1,
                        cwd=self.cwd,
                    )

                # Handle single-item results (like figure or image info)
                if result.get("type") == "figure":
                    output_lines = [
                        f"Figure: {result.get('figure_id', 'unknown')}",
                        f"  Graphic: {result.get('graphic', 'N/A')}",
                        f"  Line: {result.get('line_number', 'N/A')}",
                        f"  Caption: {result.get('caption', 'N/A')}",
                    ]
                    hint = result.get("hint")
                    if hint:
                        output_lines.append(f"\n💡 {hint}")
                    return TerminalResult(
                        stdout="\n".join(output_lines) + "\n", cwd=self.cwd
                    )

                if result.get("type") == "image":
                    output_lines = [
                        f"Image: {result.get('filename', 'unknown')}",
                    ]
                    hint = result.get("hint")
                    if hint:
                        output_lines.append(f"\n💡 {hint}")
                    return TerminalResult(
                        stdout="\n".join(output_lines) + "\n", cwd=self.cwd
                    )

                # Format output
                contents = result.get("contents", [])
                output_lines = []

                # Character limit (~10k tokens ≈ 40k chars)
                MAX_CHARS = 40000
                total_items = len(contents) if isinstance(contents, list) else 0
                total_papers = result.get("total_papers")
                char_count = 0
                items_shown = 0

                # /papers/ is read-only; /.gxl/ is read-write
                is_readonly = full_path.startswith(self.root_path)
                dir_perm = "dr-xr-xr-x" if is_readonly else "drwxr-xr-x"
                file_perm = "-r--r--r--" if is_readonly else "-rw-r--r--"

                if isinstance(contents, list):
                    if long_format:
                        for item in contents:
                            if isinstance(item, dict):
                                name = (
                                    item.get("name")
                                    or item.get("path", "").split("/")[-2]
                                    or str(item)
                                )
                                title = item.get("title", "")
                                lines_count = item.get("lines", "")
                                if title:
                                    line1 = f"{dir_perm}  {name}"
                                    line2 = f"           └─ {title[:60]}{'...' if len(title) > 60 else ''}"
                                    new_chars = len(line1) + len(line2) + 2
                                    if char_count + new_chars > MAX_CHARS:
                                        break
                                    output_lines.append(line1)
                                    output_lines.append(line2)
                                    char_count += new_chars
                                elif lines_count:
                                    approx_tokens = lines_count * 40 // 4
                                    line = f"{file_perm}  ~{approx_tokens:>5} tokens  {name}"
                                    if char_count + len(line) > MAX_CHARS:
                                        break
                                    output_lines.append(line)
                                    char_count += len(line) + 1
                                else:
                                    line = f"{file_perm}  {name}"
                                    if char_count + len(line) > MAX_CHARS:
                                        break
                                    output_lines.append(line)
                                    char_count += len(line) + 1
                            else:
                                line = f"{file_perm}  {item}"
                                if char_count + len(line) > MAX_CHARS:
                                    break
                                output_lines.append(line)
                                char_count += len(line) + 1
                            items_shown += 1
                    else:
                        names = []
                        for item in contents:
                            if isinstance(item, dict):
                                name = (
                                    item.get("name")
                                    or item.get("path", "").split("/")[-2]
                                    or str(item)
                                )
                            else:
                                name = str(item)

                            if char_count + len(name) + 2 > MAX_CHARS:
                                break
                            names.append(name)
                            char_count += len(name) + 2
                            items_shown += 1

                        output_lines.append("  ".join(names))
                        if is_readonly:
                            output_lines.append(f"  (read-only — use /.gxl/ for writable storage)")

                    # Show remaining count
                    remaining = total_items - items_shown
                    if remaining > 0:
                        output_lines.append(f"  ... [{remaining} more]")

                    # For papers, show total available
                    if total_papers and total_papers > total_items:
                        output_lines.append(
                            f"  ({total_papers:,} total papers - use papers_find to search)"
                        )
                else:
                    output_lines.append(str(contents))

                # Show hint from _ls result (e.g. how to use ask_image for figures)
                ls_hint = result.get("hint")
                if ls_hint:
                    output_lines.append(f"\n💡 {ls_hint}")

                output = "\n".join(output_lines) + "\n" if output_lines else ""
                return TerminalResult(stdout=output, cwd=self.cwd)
            except Exception as e:
                return TerminalResult(
                    stderr=f"vsh: ls: {e}",
                    exit_code=1,
                    cwd=self.cwd,
                )

        # Fallback without filesystem
        return TerminalResult(
            stdout=f"(contents of {full_path})\n",
            cwd=self.cwd,
        )

    async def _cmd_tree(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Display directory tree."""
        # Parse args
        max_depth = 3
        for i, arg in enumerate(args):
            if arg in ("-L", "--depth") and i + 1 < len(args):
                try:
                    max_depth = int(args[i + 1])
                except ValueError:
                    pass

        path_args = [a for a in args if not a.startswith("-") and not a.isdigit()]
        path = path_args[0] if path_args else self.cwd

        full_path = self._validate_path(path)
        if full_path is None:
            return TerminalResult(
                stderr=f"vsh: tree: {path}: Permission denied",
                exit_code=1,
                cwd=self.cwd,
            )

        # Build tree recursively
        lines = [full_path]
        await self._build_tree(full_path, lines, "", max_depth, 0, session_id)

        return TerminalResult(stdout="\n".join(lines) + "\n", cwd=self.cwd)

    async def _build_tree(
        self,
        path: str,
        lines: list[str],
        prefix: str,
        max_depth: int,
        current_depth: int,
        session_id: str,
    ):
        """Recursively build tree output."""
        if current_depth >= max_depth:
            return

        if not self.fs:
            return

        try:
            result = await self.fs._ls(path=path, session_id=session_id)
            contents = result.get("contents", [])

            for i, item in enumerate(contents):
                is_last = i == len(contents) - 1
                connector = "└── " if is_last else "├── "

                if isinstance(item, dict):
                    name = item.get("name", str(item))
                    item_type = item.get("type", "file")
                else:
                    name = str(item)
                    item_type = "dir" if name.endswith("/") else "file"

                lines.append(f"{prefix}{connector}{name}")

                if item_type == "dir":
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    subpath = os.path.join(path, name.rstrip("/"))
                    await self._build_tree(
                        subpath,
                        lines,
                        new_prefix,
                        max_depth,
                        current_depth + 1,
                        session_id,
                    )
        except Exception:
            pass

    async def _cmd_cat(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Concatenate and display file contents.

        cat FILE                 # show file (default: first ~1000 chars for large files)
        cat -n FILE              # with line numbers
        cat --lines N FILE       # show first N lines
        cat --lines N-M FILE     # show line range N to M
        cat --full FILE          # show entire file (no truncation)
        """
        show_line_numbers = "-n" in args
        show_full = "--full" in args

        # Parse --lines N or --lines N-M
        cat_limit = None   # None = use default character-based truncation
        if show_full:
            cat_limit = 999_999  # effectively no limit
        cat_start = None
        cat_end = None
        _args_filtered = []
        skip_next = False
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg == "--full":
                continue
            if arg == "--lines" and i + 1 < len(args):
                val = args[i + 1]
                skip_next = True
                if "-" in val and not val.startswith("-"):
                    parts = val.split("-", 1)
                    try:
                        cat_start = int(parts[0])
                        cat_end = int(parts[1])
                        cat_limit = cat_end - cat_start + 1
                    except ValueError:
                        pass
                else:
                    try:
                        cat_limit = int(val)
                    except ValueError:
                        pass
                continue
            _args_filtered.append(arg)
        args = _args_filtered

        files = [a for a in args if not a.startswith("-")]

        # Expand globs
        if files:
            files = await self._expand_args_globs(files, session_id)

        if not files and not stdin:
            return TerminalResult(
                stderr="vsh: cat: missing file operand",
                exit_code=1,
                cwd=self.cwd,
            )

        # If stdin provided (from pipe), use that
        if stdin and not files:
            if show_line_numbers:
                lines = stdin.split("\n")
                numbered = [f"{i+1:6}\t{line}" for i, line in enumerate(lines)]
                return TerminalResult(stdout="\n".join(numbered), cwd=self.cwd)
            return TerminalResult(stdout=stdin, cwd=self.cwd)

        # Read files
        output_parts = []
        for file_path in files:
            # Check for cached search results first (from `searches` command)
            resolved_path = (
                file_path
                if file_path.startswith("/")
                else f"{self.cwd.rstrip('/')}/{file_path}"
            )

            # Special case: references.md is in the session results dir
            if file_path == "references.md" or resolved_path.endswith("/references.md"):
                try:
                    if self.fs and hasattr(self.fs, "results_registry"):
                        session_dir = self.fs.results_registry._get_session_dir(
                            session_id
                        )
                        refs_path = session_dir / "references.md"
                        if refs_path.exists():
                            output_parts.append(refs_path.read_text())
                            continue
                    return TerminalResult(
                        stderr="vsh: cat: references.md: No references registered yet. Use: cite LINE [LINE...]",
                        exit_code=1,
                        cwd=self.cwd,
                    )
                except Exception as e:
                    return TerminalResult(
                        stderr=f"vsh: cat: references.md: {e}",
                        exit_code=1,
                        cwd=self.cwd,
                    )

            if resolved_path.startswith("/tmp/searches/"):
                if (
                    hasattr(self, "_search_results_cache")
                    and resolved_path in self._search_results_cache
                ):
                    output_parts.append(self._search_results_cache[resolved_path])
                    continue
                else:
                    session_file_path = (
                        f"/session_files/searches/{file_path.split('/')[-1]}"
                    )
                    session_result = await self._session_files_cat(
                        session_file_path, session_id
                    )
                    if session_result.exit_code == 0:
                        output_parts.append(session_result.stdout)
                        continue
                    return TerminalResult(
                        stderr=f"vsh: cat: {file_path}: No such file (search results may have expired)",
                        exit_code=1,
                        cwd=self.cwd,
                    )

            # Handle /.gxl/ via sandbox
            if self._is_session_files_path(resolved_path):
                result = await self._session_files_cat(resolved_path, session_id)
                if result.exit_code != 0:
                    return result
                content = result.stdout
                if show_line_numbers:
                    content_lines = content.split("\n")
                    content = "\n".join(
                        f"{i+1:6}\t{line}" for i, line in enumerate(content_lines)
                    )
                output_parts.append(content)
                continue

            full_path = self._validate_path(file_path)
            if full_path is None:
                return TerminalResult(
                    stderr=f"vsh: cat: {file_path}: Permission denied",
                    exit_code=1,
                    cwd=self.cwd,
                )

            if self.fs:
                try:
                    result = await self.fs._cat(
                        path=full_path,
                        session_id=session_id,
                        truncate=False,
                    )
                    if "error" in result:
                        error_msg = result["error"]
                        hint = result.get("hint", "")
                        if hint:
                            error_msg = f"{error_msg}\n\n💡 {hint}"
                        return TerminalResult(
                            stderr=f"vsh: cat: {error_msg}",
                            exit_code=1,
                            cwd=self.cwd,
                        )

                    # Binary file (image/figure) — return download metadata
                    if result.get("type") == "binary":
                        url = result.get("download_url", "")
                        fname = result.get("filename", file_path.rsplit("/", 1)[-1])
                        caption = result.get("caption", "")
                        time_ms = result.get("time_ms", "")
                        if url:
                            content = json.dumps({
                                "type": "binary_download",
                                "download_url": url,
                                "filename": fname,
                                "caption": caption,
                                "hint": f"Redirect to download: cat {file_path} > {fname}",
                            }, indent=2)
                            output_parts.append(content)
                            continue
                        hint = result.get("hint", "")
                        return TerminalResult(
                            stderr=f"vsh: cat: Cannot download {fname}" + (f"\n💡 {hint}" if hint else ""),
                            exit_code=1,
                            cwd=self.cwd,
                        )

                    # Format output from _cat result
                    if result.get("type") == "json":
                        # JSON content (like meta.json)
                        content = json.dumps(result.get("content", {}), indent=2)
                        if show_line_numbers:
                            content_lines = content.split("\n")
                            content = "\n".join(
                                f"{i+1:6}\t{line}"
                                for i, line in enumerate(content_lines)
                            )
                    else:
                        # Line-based content (.lines files)
                        lines = result.get("lines", [])
                        total_lines = result.get(
                            "total_lines", len(lines) if isinstance(lines, list) else 0
                        )
                        if isinstance(lines, list):
                            CAT_CHAR_LIMIT = 1000
                            use_char_limit = cat_limit is None
                            CAT_LINE_LIMIT = cat_limit if cat_limit is not None else 999_999

                            if cat_start is not None:
                                lines = lines[cat_start - 1 : cat_end]

                            total_chars = sum(
                                len(str(l.get("content", ""))) if isinstance(l, dict) else len(str(l))
                                for l in lines
                            )

                            needs_truncation = (
                                (use_char_limit and total_chars > CAT_CHAR_LIMIT)
                                or (not use_char_limit and len(lines) > CAT_LINE_LIMIT)
                            )

                            if needs_truncation:
                                sections_found: list[tuple[int, str]] = []
                                for line in lines:
                                    if isinstance(line, dict):
                                        ln = line.get("line", "?")
                                        c = str(line.get("content", ""))
                                        stripped = c.strip()
                                        if (
                                            stripped.startswith("#")
                                            or (
                                                stripped.startswith("**")
                                                and stripped.endswith("**")
                                                and len(stripped) > 10
                                            )
                                            or re.match(
                                                r"^(Figure|Fig|Table|Supplementary|Section|Appendix)\b",
                                                stripped,
                                                re.IGNORECASE,
                                            )
                                        ):
                                            sections_found.append((ln, stripped[:80]))

                                header_parts = [
                                    f"[~{total_chars // 4} tokens total, showing first ~{CAT_CHAR_LIMIT} chars]"
                                ]
                                if sections_found:
                                    header_parts.append("")
                                    header_parts.append(
                                        f"Sections detected ({len(sections_found)}):"
                                    )
                                    for ln, title in sections_found[:20]:
                                        header_parts.append(f"  L{ln}: {title}")
                                    if len(sections_found) > 20:
                                        header_parts.append(
                                            f"  ... and {len(sections_found) - 20} more"
                                        )
                                header_parts.append("")

                                if use_char_limit:
                                    show_lines = []
                                    char_count = 0
                                    for line in lines:
                                        line_text = str(line.get("content", "")) if isinstance(line, dict) else str(line)
                                        if char_count + len(line_text) > CAT_CHAR_LIMIT and show_lines:
                                            break
                                        show_lines.append(line)
                                        char_count += len(line_text)
                                else:
                                    show_lines = lines[:CAT_LINE_LIMIT]

                                formatted_lines = list(header_parts)
                            else:
                                show_lines = lines
                                formatted_lines = []

                            for line in show_lines:
                                if isinstance(line, dict):
                                    line_num = line.get("line", "?")
                                    line_content = line.get("content", "")
                                    formatted_lines.append(
                                        f"L{line_num}: {line_content}"
                                    )
                                else:
                                    formatted_lines.append(str(line))
                            content = "\n".join(formatted_lines)
                            remaining = len(lines) - len(show_lines)
                            if remaining <= 0:
                                remaining = total_lines - len(lines)
                            if remaining > 0:
                                content += (
                                    f"\n\n[{remaining} more lines not shown]"
                                    f"\n  cat --full {file_path}          # read entire file (⚠️  may be large for LLM context)"
                                    f'\n  cat {file_path} > output.txt    # save full text to local file'
                                    f'\n  grep "TERM" {file_path}         # search within file'
                                )
                        else:
                            content = str(lines)

                    output_parts.append(content)
                except Exception as e:
                    return TerminalResult(
                        stderr=f"vsh: cat: {file_path}: {e}",
                        exit_code=1,
                        cwd=self.cwd,
                    )
            else:
                output_parts.append(f"(contents of {full_path})")

        return TerminalResult(stdout="\n".join(output_parts) + "\n", cwd=self.cwd)

    async def _cmd_head(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Display first lines of file."""
        n_lines = 10
        for i, arg in enumerate(args):
            if arg == "-n" and i + 1 < len(args):
                try:
                    n_lines = int(args[i + 1])
                except ValueError:
                    pass
            elif arg.startswith("-") and arg[1:].isdigit():
                n_lines = int(arg[1:])

        files = [a for a in args if not a.startswith("-") and not a.isdigit()]

        # Expand globs
        if files:
            files = await self._expand_args_globs(files, session_id)

        if stdin:
            all_lines = stdin.split("\n")
            lines = all_lines[:n_lines]
            output = "\n".join(lines)
            # Show "[and N more]" if there are more lines
            remaining = len(all_lines) - len(lines)
            if remaining > 0:
                output += f"\n[and {remaining} more]"
            return TerminalResult(stdout=output + "\n", cwd=self.cwd)

        if not files:
            return TerminalResult(
                stderr="vsh: head: missing file operand",
                exit_code=1,
                cwd=self.cwd,
            )

        file_path = files[0]
        full_path = self._validate_path(file_path)
        if full_path is None:
            return TerminalResult(
                stderr=f"vsh: head: {file_path}: Permission denied",
                exit_code=1,
                cwd=self.cwd,
            )

        # Handle /.gxl/ via sandbox
        if self._is_session_files_path(full_path):
            return await self._session_files_head_tail(
                full_path, n_lines, is_tail=False, session_id=session_id
            )

        if self.fs:
            try:
                result = await self.fs._head(
                    path=full_path, n=n_lines, session_id=session_id
                )
                if "error" in result:
                    return TerminalResult(
                        stderr=f"vsh: head: {result['error']}",
                        exit_code=1,
                        cwd=self.cwd,
                    )

                lines = result.get("lines", [])
                total_lines = result.get("total_lines", len(lines))
                formatted = []
                for line in lines:
                    if isinstance(line, dict):
                        line_num = line.get("line", "?")
                        line_content = line.get("content", "")
                        formatted.append(f"L{line_num}: {line_content}")
                    else:
                        formatted.append(str(line))
                output = "\n".join(formatted)
                remaining = total_lines - len(lines)
                if remaining > 0:
                    output += f"\n[and {remaining} more]"
                return TerminalResult(stdout=output + "\n", cwd=self.cwd)
            except Exception as e:
                return TerminalResult(
                    stderr=f"vsh: head: {e}",
                    exit_code=1,
                    cwd=self.cwd,
                )

        return TerminalResult(
            stdout=f"(first {n_lines} lines of {full_path})\n", cwd=self.cwd
        )

    async def _cmd_tail(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Display last lines of file."""
        n_lines = 10
        for i, arg in enumerate(args):
            if arg == "-n" and i + 1 < len(args):
                try:
                    n_lines = int(args[i + 1])
                except ValueError:
                    pass
            elif arg.startswith("-") and arg[1:].isdigit():
                n_lines = int(arg[1:])

        files = [a for a in args if not a.startswith("-") and not a.isdigit()]

        # Expand globs
        if files:
            files = await self._expand_args_globs(files, session_id)

        if stdin:
            all_lines = stdin.split("\n")
            lines = all_lines[-n_lines:]
            output = ""
            # Show "[N earlier lines]" if there are more lines before
            skipped = len(all_lines) - len(lines)
            if skipped > 0:
                output = f"[{skipped} earlier lines]\n"
            output += "\n".join(lines)
            return TerminalResult(stdout=output + "\n", cwd=self.cwd)

        if not files:
            return TerminalResult(
                stderr="vsh: tail: missing file operand",
                exit_code=1,
                cwd=self.cwd,
            )

        file_path = files[0]
        full_path = self._validate_path(file_path)
        if full_path is None:
            return TerminalResult(
                stderr=f"vsh: tail: {file_path}: Permission denied",
                exit_code=1,
                cwd=self.cwd,
            )

        # Handle /.gxl/ via sandbox
        if self._is_session_files_path(full_path):
            return await self._session_files_head_tail(
                full_path, n_lines, is_tail=True, session_id=session_id
            )

        if self.fs:
            try:
                result = await self.fs._tail(
                    path=full_path, n=n_lines, session_id=session_id
                )
                if "error" in result:
                    return TerminalResult(
                        stderr=f"vsh: tail: {result['error']}",
                        exit_code=1,
                        cwd=self.cwd,
                    )

                lines = result.get("lines", [])
                total_lines = result.get("total_lines", len(lines))
                formatted = []
                for line in lines:
                    if isinstance(line, dict):
                        line_num = line.get("line", "?")
                        line_content = line.get("content", "")
                        formatted.append(f"L{line_num}: {line_content}")
                    else:
                        formatted.append(str(line))
                output = "\n".join(formatted)
                skipped = total_lines - len(lines)
                if skipped > 0:
                    output = f"[{skipped} lines above]\n" + output
                return TerminalResult(stdout=output + "\n", cwd=self.cwd)
            except Exception as e:
                return TerminalResult(
                    stderr=f"vsh: tail: {e}",
                    exit_code=1,
                    cwd=self.cwd,
                )

        return TerminalResult(
            stdout=f"(last {n_lines} lines of {full_path})\n", cwd=self.cwd
        )

    async def _cmd_wc(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Word, line, character count."""
        count_lines = "-l" in args or not any(a.startswith("-") for a in args)
        count_words = "-w" in args or not any(a.startswith("-") for a in args)
        count_chars = (
            "-c" in args or "-m" in args or not any(a.startswith("-") for a in args)
        )

        files = [a for a in args if not a.startswith("-")]

        # Expand globs
        if files:
            files = await self._expand_args_globs(files, session_id)

        def count(text: str) -> tuple[int, int, int]:
            lines = text.count("\n")
            words = len(text.split())
            chars = len(text)
            return lines, words, chars

        if stdin:
            l, w, c = count(stdin)
            parts = []
            if count_lines:
                parts.append(f"{l:8}")
            if count_words:
                parts.append(f"{w:8}")
            if count_chars:
                parts.append(f"{c:8}")
            return TerminalResult(stdout="".join(parts) + "\n", cwd=self.cwd)

        if not files:
            return TerminalResult(
                stderr="vsh: wc: missing file operand",
                exit_code=1,
                cwd=self.cwd,
            )

        # Read file and count
        file_path = files[0]
        full_path = self._validate_path(file_path)
        if full_path is None:
            return TerminalResult(
                stderr=f"vsh: wc: {file_path}: Permission denied",
                exit_code=1,
                cwd=self.cwd,
            )

        # Handle /.gxl/ via sandbox
        if self._is_session_files_path(full_path):
            cat_result = await self._session_files_cat(full_path, session_id)
            if cat_result.exit_code != 0:
                return cat_result
            l, w, c = count(cat_result.stdout)
            parts = []
            if count_lines:
                parts.append(f"{l:8}")
            if count_words:
                parts.append(f"{w:8}")
            if count_chars:
                parts.append(f"{c:8}")
            parts.append(f" {file_path}")
            return TerminalResult(stdout="".join(parts) + "\n", cwd=self.cwd)

        if self.fs:
            try:
                result = await self.fs._cat(path=full_path, session_id=session_id)
                lines = result.get("lines", [])
                content = "\n".join(
                    str(line.get("content", line) if isinstance(line, dict) else line)
                    for line in lines
                )
                l, w, c = count(content)
                parts = []
                if count_lines:
                    parts.append(f"{l:8}")
                if count_words:
                    parts.append(f"{w:8}")
                if count_chars:
                    parts.append(f"{c:8}")
                parts.append(f" {file_path}")
                return TerminalResult(stdout="".join(parts) + "\n", cwd=self.cwd)
            except Exception as e:
                return TerminalResult(
                    stderr=f"vsh: wc: {e}",
                    exit_code=1,
                    cwd=self.cwd,
                )

        return TerminalResult(
            stdout=f"       0       0       0 {file_path}\n", cwd=self.cwd
        )

    def _grep_structured_lines(
        self,
        lines: list[dict],
        regex,
        filename: str = "",
        invert_match: bool = False,
        only_matching: bool = False,
        count_only: bool = False,
        list_files: bool = False,
        suppress_filename: bool = False,
        before_context: int = 0,
        after_context: int = 0,
        max_count: int = 0,
    ) -> tuple[str, bool]:
        """Grep structured lines preserving actual line numbers.

        For .lines files that return {"line": N, "content": "..."}, this method
        preserves the actual database line numbers instead of using sequential indices.
        This is critical for the cite command to work correctly.
        """
        output_lines = []
        has_any_match = False
        match_count = 0

        line_data = []
        for line in lines:
            if isinstance(line, dict):
                line_data.append(
                    {
                        "num": line.get("line", len(line_data) + 1),
                        "content": str(line.get("content", "")),
                    }
                )
            else:
                line_data.append(
                    {
                        "num": len(line_data) + 1,
                        "content": str(line),
                    }
                )

        # Find matching indices (respecting max_count limit)
        matching_indices = []
        for i, data in enumerate(line_data):
            match = regex.search(data["content"])
            if (match and not invert_match) or (not match and invert_match):
                matching_indices.append(i)
                has_any_match = True
                match_count += 1
                # Stop early if we've hit max_count
                if max_count > 0 and match_count >= max_count:
                    break

        matching_indices_set = set(matching_indices)

        if list_files and has_any_match:
            return (filename, True)

        if count_only:
            return (str(match_count), has_any_match)

        # Determine which lines to include (matches + context)
        lines_to_show = set()
        for idx in matching_indices:
            # Add before context
            for b in range(max(0, idx - before_context), idx):
                lines_to_show.add(b)
            # Add the match itself
            lines_to_show.add(idx)
            # Add after context
            for a in range(idx + 1, min(len(line_data), idx + 1 + after_context)):
                lines_to_show.add(a)

        prev_idx = -2
        for idx in sorted(lines_to_show):
            data = line_data[idx]
            actual_line_num = data["num"]
            line_content = data["content"]
            is_match = idx in matching_indices_set

            line_label = f"L{actual_line_num}"

            # Add separator if there's a gap between groups
            if before_context > 0 or after_context > 0:
                if prev_idx >= 0 and idx > prev_idx + 1:
                    output_lines.append("--")
            prev_idx = idx

            if only_matching and is_match:
                # -o: Print only the matching part
                for m in regex.finditer(line_content):
                    prefix = (
                        f"{filename}:" if filename and not suppress_filename else ""
                    )
                    output_lines.append(f"{prefix}{line_label}:{m.group()}")
            else:
                prefix = f"{filename}:" if filename and not suppress_filename else ""
                # Use : for matches, - for context lines
                sep = ":" if is_match else "-"
                output_lines.append(f"{prefix}{line_label}{sep}{line_content}")

        return ("\n".join(output_lines), has_any_match)

    async def _cmd_grep(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Search for patterns.

        Supported flags:
            -i: Case insensitive matching
            -n: Show line numbers
            -c: Count matches only
            -v: Invert match (show non-matching lines)
            -o: Print only the matching part of lines
            -e PATTERN: Explicit pattern (can repeat for multi-pattern OR)
            -E: Extended regex (default, for compatibility)
            -F: Fixed strings (literal match, no regex)
            -w: Match whole words only
            -l: List only filenames with matches
            -h: Suppress filename prefix
            -m NUM: Stop after NUM matches
            -A NUM: Show NUM lines after match
            -B NUM: Show NUM lines before match
            -C NUM: Show NUM lines before and after match
        """
        ignore_case = "-i" in args
        show_line_numbers = "-n" in args
        count_only = "-c" in args
        invert_match = "-v" in args
        only_matching = "-o" in args
        fixed_strings = "-F" in args
        whole_word = "-w" in args
        list_files = "-l" in args
        suppress_filename = "-h" in args
        # -E is already default behavior (extended regex)

        # Parse flags that take numeric arguments (-A, -B, -C, -m)
        after_context = 0
        before_context = 0
        max_count = 0  # 0 means no limit
        skip_indices = set()  # Track indices to skip when extracting pattern/files

        # Parse -e patterns (multiple patterns OR'd together)
        explicit_patterns: list[str] = []

        for i, arg in enumerate(args):
            if arg in ("-A", "-B", "-C", "-m") and i + 1 < len(args):
                try:
                    num = int(args[i + 1])
                    skip_indices.add(i)
                    skip_indices.add(i + 1)
                    if arg == "-A":
                        after_context = num
                    elif arg == "-B":
                        before_context = num
                    elif arg == "-C":
                        after_context = before_context = num
                    elif arg == "-m":
                        max_count = num
                except ValueError:
                    pass
            elif arg == "-e" and i + 1 < len(args):
                explicit_patterns.append(args[i + 1])
                skip_indices.add(i)
                skip_indices.add(i + 1)
            elif arg == "--from" and i + 1 < len(args):
                skip_indices.add(i)
                skip_indices.add(i + 1)

        # Parse --from (grep within a search result set)
        from_results = None
        for i, arg in enumerate(args):
            if arg == "--from" and i + 1 < len(args):
                from_results = args[i + 1]
                skip_indices.add(i)
                skip_indices.add(i + 1)

        # Get pattern and files (excluding flags and their arguments)
        pattern_args = [
            a
            for i, a in enumerate(args)
            if not a.startswith("-") and i not in skip_indices
        ]

        if explicit_patterns:
            # -e: OR multiple patterns together
            if fixed_strings:
                pattern = "|".join(re.escape(p) for p in explicit_patterns)
            else:
                pattern = "|".join(f"(?:{p})" for p in explicit_patterns)
            files = [
                a
                for i, a in enumerate(args)
                if not a.startswith("-") and i not in skip_indices
            ]
        elif pattern_args:
            pattern = pattern_args[0]
            files = pattern_args[1:] if len(pattern_args) > 1 else []
        else:
            return TerminalResult(
                stderr="vsh: grep: missing pattern",
                exit_code=1,
                cwd=self.cwd,
            )

        # --from: grep content blocks of papers from a search result set
        if from_results and self.fs and hasattr(self.fs, 'grep_content'):
            try:
                saved = self.fs.results_registry.load(from_results, session_id)
                if not saved:
                    return TerminalResult(
                        stderr=f"grep: results not found: {from_results}",
                        exit_code=1, cwd=self.cwd,
                    )
                papers = saved.get("papers", [])
                doc_ids = [p.get("document_id", "") for p in papers if p.get("document_id")]
                if not doc_ids:
                    return TerminalResult(
                        stderr=f"grep: no papers in {from_results}",
                        exit_code=1, cwd=self.cwd,
                    )
                matches = await self.fs.grep_content(
                    regex=pattern, document_ids=doc_ids,
                    section_filter=None, limit=100,
                )
                if not matches:
                    return TerminalResult(
                        stdout=f"No matches for /{pattern}/ in {len(doc_ids)} papers from {from_results}\n",
                        cwd=self.cwd,
                    )
                lines = []
                for m in matches:
                    did = m.get("document_id", "")
                    ln = m.get("line_number", "")
                    content = m.get("content", "")[:200]
                    sec = m.get("section", "")
                    lines.append(f"{did}:L{ln} ({sec}) {content}")
                header = f"Matched /{pattern}/ in {len(matches)} blocks across {len(set(m.get('document_id') for m in matches))} papers"
                return TerminalResult(
                    stdout=header + "\n" + "\n".join(lines) + "\n",
                    cwd=self.cwd,
                )
            except Exception as e:
                return TerminalResult(
                    stderr=f"grep --from: {e}", exit_code=1, cwd=self.cwd,
                )

        # Expand globs in file arguments
        if files:
            files = await self._expand_args_globs(files, session_id)

        # Build the regex pattern
        if fixed_strings:
            # -F: Escape all regex special characters for literal matching
            python_pattern = re.escape(pattern)
        else:
            # Convert shell-style escaping to Python regex:
            # - Shell uses \| for alternation, Python uses just |
            # - Shell uses \( \) for groups, Python uses ( )
            # - Shell uses \+ for one-or-more, Python uses +
            python_pattern = (
                pattern.replace(r"\|", "|")
                .replace(r"\(", "(")
                .replace(r"\)", ")")
                .replace(r"\+", "+")
            )

        # -w: Wrap pattern in word boundaries
        if whole_word:
            python_pattern = r"\b" + python_pattern + r"\b"

        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(python_pattern, flags)
        except re.error as e:
            return TerminalResult(
                stderr=f"vsh: grep: invalid regex: {e}",
                exit_code=1,
                cwd=self.cwd,
            )

        def grep_text(text: str, filename: str = "") -> tuple[str, bool]:
            """Returns (output, has_matches)"""
            lines = text.split("\n")
            output_lines = []
            has_any_match = False
            match_count = 0

            # Find matching line indices (respecting max_count limit)
            matching_indices = []
            for i, line in enumerate(lines):
                match = regex.search(line)
                if (match and not invert_match) or (not match and invert_match):
                    matching_indices.append(i)
                    has_any_match = True
                    match_count += 1
                    # Stop early if we've hit max_count
                    if max_count > 0 and match_count >= max_count:
                        break

            matching_indices_set = set(matching_indices)

            if list_files and has_any_match:
                return (filename, True)

            if count_only:
                return (str(match_count), has_any_match)

            # Determine which lines to include (matches + context)
            lines_to_show = set()
            for idx in matching_indices:
                # Add before context
                for b in range(max(0, idx - before_context), idx):
                    lines_to_show.add(b)
                # Add the match itself
                lines_to_show.add(idx)
                # Add after context
                for a in range(idx + 1, min(len(lines), idx + 1 + after_context)):
                    lines_to_show.add(a)

            # Build output
            prev_idx = -2  # Track for separator between groups
            for idx in sorted(lines_to_show):
                line = lines[idx]
                line_num = idx + 1  # 1-indexed for display
                is_match = idx in matching_indices_set

                # Add separator if there's a gap between groups
                if before_context > 0 or after_context > 0:
                    if prev_idx >= 0 and idx > prev_idx + 1:
                        output_lines.append("--")
                prev_idx = idx

                if only_matching and is_match:
                    # -o: Print only the matching part
                    for m in regex.finditer(line):
                        prefix = (
                            f"{filename}:" if filename and not suppress_filename else ""
                        )
                        if show_line_numbers:
                            output_lines.append(f"{prefix}{line_num}:{m.group()}")
                        else:
                            output_lines.append(
                                f"{prefix}{m.group()}" if prefix else m.group()
                            )
                else:
                    prefix = (
                        f"{filename}:" if filename and not suppress_filename else ""
                    )
                    # Use : for matches, - for context lines (like real grep)
                    sep = ":" if is_match else "-"
                    if show_line_numbers:
                        output_lines.append(f"{prefix}{line_num}{sep}{line}")
                    else:
                        output_lines.append(
                            f"{prefix}{line}" if not prefix else f"{prefix}{line}"
                        )

            return ("\n".join(output_lines), has_any_match)

        def _display_match_content(m: dict, max_chars: int = 250) -> str:
            """Keep the matched term visible when rendering long grep lines."""
            content = m.get("content") or ""
            if len(content) <= max_chars:
                return content

            needle = str(m.get("match") or "")
            pos = content.lower().find(needle.lower()) if needle else -1
            if pos < 0:
                return content[:max_chars]

            window_start = max(0, pos - max_chars // 3)
            window_end = window_start + max_chars
            snippet = content[window_start:window_end]
            if window_start > 0:
                snippet = "…" + snippet[1:]
            if window_end < len(content):
                snippet = snippet[:-1] + "…"
            return snippet

        # Parse --from (grep within a search result set via SQL)
        grep_from = None
        for i, arg in enumerate(args):
            if arg == "--from" and i + 1 < len(args):
                grep_from = args[i + 1]

        # Grep stdin if provided
        if stdin and not files and not grep_from:
            output, has_matches = grep_text(stdin)
            return TerminalResult(
                stdout=output + "\n" if output else "",
                exit_code=0 if has_matches else 1,
                cwd=self.cwd,
            )

        # --from: grep content blocks of papers from a search result set
        if grep_from and self.fs:
            result = await self.fs._grep(
                regex=python_pattern,
                from_results=grep_from,
                limit=50,
                session_id=session_id,
            )
            if "error" in result:
                return TerminalResult(
                    stderr=f"grep: {result['error']}", exit_code=1, cwd=self.cwd)

            papers = result.get("papers", [])
            if not papers:
                return TerminalResult(
                    stdout=f"No matches for /{python_pattern}/ in {grep_from}\n",
                    exit_code=1, cwd=self.cwd)

            out_lines = []
            total_matches = 0
            for p in papers:
                doc_id = p.get("document_id", "?")
                display_id = doc_id[:8] if len(doc_id) > 20 else doc_id
                matches = p.get("matches", [])
                total_matches += len(matches)
                out_lines.append(f"  {display_id}/ ({len(matches)} matches)")
                for m in matches[:3]:
                    section = m.get("section", "")
                    content = _display_match_content(m)
                    sec_tag = f"[{section}] " if section else ""
                    out_lines.append(f"    {sec_tag}{content}")
                if len(matches) > 3:
                    out_lines.append(f"    ... +{len(matches) - 3} more")
                out_lines.append("")

            results_id = result.get("results_id", "")
            header = (f"Matched {total_matches} paragraphs across {len(papers)} papers "
                      f"[results_id: {results_id}]")
            _meta = {"results_id": results_id} if results_id else None
            return TerminalResult(
                stdout=header + "\n\n" + "\n".join(out_lines) + "\n",
                exit_code=0, cwd=self.cwd, metadata=_meta)

        # Grep files
        if not files:
            return TerminalResult(
                stderr="vsh: grep: missing file operand",
                exit_code=1,
                cwd=self.cwd,
            )

        all_output = []
        any_match_found = False
        files_with_matches = []

        recursive = "-r" in args or "-rl" in args or "-rn" in args or "-ri" in args

        # Corpus-wide grep: route `grep "pattern" /papers/` to the slab-grep
        # service for sub-second regex search across the entire corpus.
        corpus_roots = {"/documents", "/papers", "/documents/", "/papers/"}
        resolved_files = [self._validate_path(f) for f in files]
        is_corpus_grep = (
            self.fs
            and hasattr(self.fs, "_grep")
            and len(files) == 1
            and resolved_files[0]
            and resolved_files[0].rstrip("/") in {"/documents", "/papers"}
        )
        if is_corpus_grep:
            try:
                result = await self.fs._grep(
                    regex=python_pattern,
                    path="/papers/",
                    limit=max_count if (max_count > 0) else 500,
                    session_id=session_id,
                )
                if "error" in result:
                    return TerminalResult(
                        stderr=f"grep: {result['error']}",
                        exit_code=1,
                        cwd=self.cwd,
                    )
                papers = result.get("papers", [])
                if not papers:
                    return TerminalResult(
                        stdout=f"No matches for /{python_pattern}/ across corpus\n",
                        exit_code=1,
                        cwd=self.cwd,
                    )
                if count_only:
                    total = sum(len(p.get("matches", [])) for p in papers)
                    return TerminalResult(
                        stdout=f"{total}\n", cwd=self.cwd,
                    )
                out_lines = []
                total_matches = 0
                for p in papers:
                    doc_id = p.get("document_id", "?")
                    matches = p.get("matches", [])
                    total_matches += len(matches)
                    out_lines.append(f"  {doc_id}/ ({len(matches)} matches)")
                    for m in matches[:3]:
                        section = m.get("section", "")
                        content = _display_match_content(m)
                        sec_tag = f"[{section}] " if section else ""
                        out_lines.append(f"    {sec_tag}{content}")
                    if len(matches) > 3:
                        out_lines.append(f"    ... +{len(matches) - 3} more")
                    out_lines.append("")

                results_id = result.get("results_id", "")
                header = (
                    f"Matched {total_matches} paragraphs across {len(papers)} papers"
                    f" [results_id: {results_id}]"
                )
                _meta = {"results_id": results_id} if results_id else None
                return TerminalResult(
                    stdout=header + "\n\n" + "\n".join(out_lines) + "\n",
                    exit_code=0,
                    cwd=self.cwd,
                    metadata=_meta,
                )
            except Exception as e:
                return TerminalResult(
                    stderr=f"grep: corpus search error: {e}",
                    exit_code=2,
                    cwd=self.cwd,
                )

        for file_path in files:
            full_path = self._validate_path(file_path)
            if full_path is None:
                return TerminalResult(
                    stderr=f"vsh: grep: {file_path}: Permission denied",
                    exit_code=1,
                    cwd=self.cwd,
                )

            if recursive and full_path.rstrip("/") in ("/documents", "/papers"):
                return TerminalResult(
                    stderr=(
                        f"vsh: grep -r is not supported on {file_path} (virtual directory with thousands of documents).\n"
                        f"Use `search \"{pattern}\"` to find documents matching your query, then grep individual files:\n"
                        f"  search \"{pattern}\"\n"
                        f"  grep -i \"{pattern}\" /documents/<document_id>/content.lines"
                    ),
                    exit_code=1,
                    cwd=self.cwd,
                )

            # Handle /.gxl/ scratch files via sandbox (see _is_session_files_path)
            if self._is_session_files_path(full_path):
                cat_result = await self._session_files_cat(full_path, session_id)
                if cat_result.exit_code != 0:
                    return cat_result
                filename = file_path if len(files) > 1 and not suppress_filename else ""
                output, has_matches = grep_text(cat_result.stdout, filename)
                if has_matches:
                    any_match_found = True
                    files_with_matches.append(file_path)
                if output:
                    all_output.append(output)
                continue

            if self.fs:
                try:
                    result = await self.fs._cat(path=full_path, session_id=session_id)

                    # Check for errors first - distinguish from "no matches"
                    if "error" in result:
                        error_msg = result["error"]
                        # Database/connection errors should be clearly reported
                        if any(
                            x in str(error_msg).lower()
                            for x in ["ssl", "connection", "timeout", "eof"]
                        ):
                            return TerminalResult(
                                stderr=f"vsh: grep: database error: {error_msg}\nHint: This is a connection issue - retry the command",
                                exit_code=2,  # Exit code 2 = error (vs 1 = no match)
                                cwd=self.cwd,
                            )
                        return TerminalResult(
                            stderr=f"vsh: grep: {file_path}: {error_msg}",
                            exit_code=2,
                            cwd=self.cwd,
                        )

                    lines = result.get("lines", [])

                    # Check if lines have structured data with actual line numbers
                    # (e.g., from .lines files which return {"line": N, "content": "..."})
                    has_structured_lines = (
                        lines and isinstance(lines[0], dict) and "line" in lines[0]
                    )

                    if has_structured_lines:
                        # Always use structured grep for .lines files so that
                        # line numbers are preserved in the output — the
                        # model uses line number + document_id for citations.
                        filename = (
                            file_path
                            if len(files) > 1 and not suppress_filename
                            else ""
                        )
                        output, has_matches = self._grep_structured_lines(
                            lines,
                            regex,
                            filename,
                            invert_match=invert_match,
                            only_matching=only_matching,
                            count_only=count_only,
                            list_files=list_files,
                            suppress_filename=suppress_filename,
                            before_context=before_context,
                            after_context=after_context,
                            max_count=max_count,
                        )
                    else:
                        # Fall back to plain text grep (sequential line numbers)
                        content = "\n".join(
                            str(
                                line.get("content", line)
                                if isinstance(line, dict)
                                else line
                            )
                            for line in lines
                        )
                        filename = (
                            file_path
                            if len(files) > 1 and not suppress_filename
                            else ""
                        )
                        output, has_matches = grep_text(content, filename)
                    if has_matches:
                        any_match_found = True
                        if list_files:
                            files_with_matches.append(file_path)
                        elif output:
                            all_output.append(output)
                except Exception as e:
                    error_str = str(e)
                    # Database errors should be clearly flagged for retry
                    if any(
                        x in error_str.lower()
                        for x in ["ssl", "connection", "timeout", "eof", "operational"]
                    ):
                        return TerminalResult(
                            stderr=f"vsh: grep: database error: {error_str}\nHint: This is a connection issue - retry the command",
                            exit_code=2,
                            cwd=self.cwd,
                        )
                    return TerminalResult(
                        stderr=f"vsh: grep: {file_path}: {e}",
                        exit_code=2,
                        cwd=self.cwd,
                    )

        # Handle -l flag: output only filenames with matches
        if list_files:
            if files_with_matches:
                return TerminalResult(
                    stdout="\n".join(files_with_matches) + "\n",
                    exit_code=0,
                    cwd=self.cwd,
                )
            return TerminalResult(
                stdout="(no matches found)\n",
                cwd=self.cwd,
            )

        return TerminalResult(
            stdout=(
                "\n".join(all_output) + "\n" if all_output else "(no matches found)\n"
            ),
            cwd=self.cwd,
        )

    # Session-scoped references registry: session_id -> list of {ref_num, doc_id, line_number, ...}
    _references: dict[str, list[dict]] = {}

    def _get_references(self, session_id: str) -> list[dict]:
        """Get or create the references list for this session."""
        if session_id not in self._references:
            self._references[session_id] = []
        return self._references[session_id]

    def _find_existing_ref(
        self, refs: list[dict], doc_id: str, line_number: int
    ) -> dict | None:
        """Find an existing reference by doc_id + line_number."""
        for r in refs:
            if r.get("doc_id") == doc_id and r.get("line_number") == line_number:
                return r
        return None

    def _write_references_md(self, refs: list[dict], session_id: str) -> None:
        """Write references.md to the session results directory."""
        if not self.fs or not hasattr(self.fs, "results_registry"):
            return
        try:
            session_dir = self.fs.results_registry._get_session_dir(session_id)
            refs_path = session_dir / "references.md"

            lines = ["# References", ""]

            # Group refs by doc_id for cleaner multi-paper display
            docs: dict[str, list[dict]] = {}
            for r in refs:
                did = r.get("doc_id", "unknown")
                docs.setdefault(did, []).append(r)

            for doc_id, doc_refs in docs.items():
                first = doc_refs[0]
                doc_title = first.get("doc_title", "Untitled")
                authors = first.get("authors", "")
                doi = first.get("doi", "")
                source = first.get("source", "")
                pub_date = first.get("pub_date", "") or first.get("pub_year", "")

                # Paper header
                lines.append(f"## {doc_title}")
                meta_parts = []
                if authors:
                    author_list = [a.strip() for a in authors.split(",")]
                    display = ", ".join(author_list[:3])
                    if len(author_list) > 3:
                        display += ", et al."
                    meta_parts.append(display)
                if doi:
                    meta_parts.append(f"DOI: {doi}")
                if source:
                    meta_parts.append(source)
                if pub_date:
                    meta_parts.append(str(pub_date))
                lines.append(" | ".join(meta_parts))
                lines.append(f"`doc_id: {doc_id}`")
                lines.append("")

                # Each citation under this paper
                for r in doc_refs:
                    ref_num = r["ref_num"]
                    line_number = r.get("line_number", "?")
                    content = r.get("content", "")
                    section = r.get("section", "")
                    block_type = r.get("block_type", "")

                    label = f"[{ref_num}] Line {line_number}"
                    if section:
                        label += f" · {section}"
                    if block_type and block_type not in ("paragraph", ""):
                        label += f" ({block_type})"

                    lines.append(f"### {label}")
                    lines.append(f"> {content}")
                    lines.append("")

                lines.append("---")
                lines.append("")

            with open(refs_path, "w") as f:
                f.write("\n".join(lines))
        except Exception as e:
            logger.warning(f"Could not write references.md: {e}")

    async def _cmd_cite(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Register citations and get reference numbers for your response.

        Usage:
            cite LINE [LINE...]                    # Cite lines from current paper
            cite PATH LINE [LINE...]               # Cite lines from a specific path

        Examples:
            cite 21 33 52                          # Cite 3 lines -> returns [1] [2] [3]
            cite supplements/file.lines 5 10       # Cite supplement lines

        Returns [N] reference numbers. Use these in your response:
            "The method was effective [1] across all cohorts [2]."

        References accumulate across turns in the same session.
        """
        if not self.fs:
            return TerminalResult(
                stderr="cite: filesystem not available", exit_code=1, cwd=self.cwd
            )

        if not args:
            return TerminalResult(
                stderr="cite: usage: cite LINE [LINE...]\n"
                "       Returns [N] reference numbers for your response.\n"
                "       Example: cite 21 33 52",
                exit_code=1,
                cwd=self.cwd,
            )

        # Parse current path to get document_id and supplement info (default)
        parsed = self.fs.path_parser.parse(self.cwd)
        doc_id = parsed.document_id
        supplement_filename = None
        line_numbers = []

        # Check if current directory is a supplement
        if parsed.type == "supplement_text":
            supplement_filename = parsed.filename

        # Check if first arg is a path or a number
        first_arg = args[0]
        try:
            # First arg is a line number - use current directory
            line_numbers.append(int(first_arg))
            line_numbers.extend(int(a) for a in args[1:] if a.isdigit())
        except ValueError:
            # First arg is a path
            full_path = self._validate_path(first_arg)
            if full_path:
                parsed = self.fs.path_parser.parse(full_path)
                doc_id = parsed.document_id

                # Check if path points to a supplement file
                if parsed.type == "supplement_text":
                    supplement_filename = parsed.filename
                elif parsed.type in ("supplements_list", "document_dir"):
                    supplement_filename = None

            # Remaining args are line numbers
            for arg in args[1:]:
                try:
                    line_numbers.append(int(arg))
                except ValueError:
                    pass

        if not doc_id:
            return TerminalResult(
                stderr="cite: must be in a paper directory or specify path\n"
                "       cite 42                              # main content\n"
                "       cite /papers/UUID/ 42                # main content\n"
                "       cite /papers/UUID/supplements/f.lines 5  # supplement",
                exit_code=1,
                cwd=self.cwd,
            )

        if not line_numbers:
            return TerminalResult(
                stderr="cite: no line numbers provided\n"
                "       cite LINE or cite PATH LINE [LINE...]",
                exit_code=1,
                cwd=self.cwd,
            )

        # Get citation(s) and register as references — batch query for speed
        refs = self._get_references(session_id)
        try:
            # Split into already-registered and new
            new_lines = []
            outputs = []
            output_order = {}  # line_number -> position in output

            for i, line_number in enumerate(line_numbers):
                existing = self._find_existing_ref(refs, doc_id, line_number)
                if existing:
                    ref_num = existing["ref_num"]
                    content_preview = existing.get("content", "")[:80]
                    section = existing.get("section", "")
                    output_order[line_number] = (
                        f'[{ref_num}] Line {line_number}: "{content_preview}" ({section})'
                    )
                else:
                    new_lines.append(line_number)

            # Batch query via filesystem module (each module handles its own DB schema)
            if new_lines:
                batch_result = await self.fs._batch_cite(
                    doc_id=doc_id,
                    line_numbers=new_lines,
                    supplement_filename=supplement_filename,
                )

                if batch_result is None:
                    return TerminalResult(
                        stderr=f"cite: document {doc_id} not found in database",
                        exit_code=1,
                        cwd=self.cwd,
                    )

                dm = batch_result["doc_meta"]
                for line_number in new_lines:
                    line_data = batch_result["lines"].get(line_number)
                    if line_data is None:
                        output_order[line_number] = (
                            f"⚠ Line {line_number}: not found (skipped)"
                        )
                        continue

                    content = line_data["content"]
                    section = line_data["section"]
                    ci = line_data.get("citation_info", {})

                    ref_num = len(refs) + 1
                    ref_entry = {
                        "ref_num": ref_num,
                        "doc_id": doc_id,
                        "line_number": line_number,
                        "content": content,
                        "section": section or "",
                        "block_type": line_data.get("block_type", ""),
                        "doc_title": dm.get("doc_title", ""),
                        "doi": dm.get("doi", ""),
                        "authors": dm.get("authors", ""),
                        "source": dm.get("source", ""),
                        "month_year": dm.get("month_year", ""),
                        "source_type": ci.get("source_type", ""),
                        "source_path": ci.get("source_path", ""),
                        "xml_id": ci.get("xml_id", ""),
                        "xpath": ci.get("xpath", ""),
                    }
                    refs.append(ref_entry)
                    display = (content[:80] + "...") if len(content) > 80 else content
                    output_order[line_number] = (
                        f'[{ref_num}] Line {line_number}: "{display}" ({section or ""})'
                    )

            # Build output in original order
            registered = 0
            skipped = 0
            for ln in line_numbers:
                if ln in output_order:
                    outputs.append(output_order[ln])
                    if output_order[ln].startswith("⚠"):
                        skipped += 1
                    else:
                        registered += 1

            total = len(refs)
            summary = f"\n{registered} references registered (session total: {total})"
            if skipped > 0:
                summary += f", {skipped} skipped"
            outputs.append(summary)
            outputs.append("Use [N] in your response to cite these.")

            # Persist references.md to session results directory
            self._write_references_md(refs, session_id)

            return TerminalResult(stdout="\n".join(outputs) + "\n", cwd=self.cwd)

        except Exception as e:
            return TerminalResult(
                stderr=f"cite: error getting citation: {e}",
                exit_code=1,
                cwd=self.cwd,
            )

    async def _ask_image_local(
        self, figure_path: str, question: str, session_id: str = "default"
    ) -> dict:
        """Analyze a local image file (e.g. from /.gxl/) using a vision model."""
        import base64

        if not os.path.isfile(figure_path):
            return {"error": f"File not found: {figure_path}"}

        image_bytes = open(figure_path, "rb").read()
        suffix = figure_path.lower().rsplit(".", 1)[-1]
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}
        mime_type = mime_map.get(suffix, "image/jpeg")

        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        message_history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Question: {question}"},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                ],
            }
        ]

        try:
            from gxl_inference_client.client import InferenceClient

            async with InferenceClient(timeout=120.0) as client:
                result = await client.chat(
                    message_history=message_history,
                    model="google/gemini-3-flash-preview",
                    agent_id=f"vision_{session_id[:8]}",
                )

            analysis = ""
            if "response" in result:
                inner = result["response"]
                if isinstance(inner, dict) and "choices" in inner:
                    analysis = inner.get("choices", [{}])[0].get("message", {}).get("content", "")
                elif isinstance(inner, str):
                    analysis = inner
            elif "content" in result:
                analysis = result["content"]

            return {"analysis": analysis, "figure": os.path.basename(figure_path)}
        except Exception as e:
            return {"error": f"Vision model call failed: {e}"}

    def _find_session_figures_dir(self) -> str | None:
        """If cwd is under /.gxl/, find the figures/ directory for the paper."""
        if not self._is_session_files_path(self.cwd):
            return None
        rel = self._session_files_to_sandbox_path(self.cwd)
        # Walk up from cwd to find a figures/ directory
        # e.g. cwd = /.gxl/MyPaper or /.gxl/MyPaper/figures
        parts = rel.strip("/").split("/") if rel.strip("/") else []
        # The paper root is the first path component under /.gxl/
        if not parts:
            return None
        paper_root = parts[0]
        return f"/.gxl/{paper_root}/figures"

    def _parse_figure_path(self, path: str) -> tuple[str | None, str]:
        """Extract (document_id, figure_filename) from a figure path.

        Accepts full paths like /papers/<id>/figures/<file>, relative paths
        like <id>/figures/<file>, or bare filenames like fig1.jpg (uses cwd).
        Returns (document_id, figure_id). document_id is None if not resolved.
        """
        import re as _re

        root_name = "papers"
        if self.fs and hasattr(self.fs, "path_parser"):
            root_name = self.fs.path_parser.root_name

        # Full or relative path: /papers/<id>/figures/<file> or <id>/figures/<file>
        m = _re.match(
            rf"^/?{_re.escape(root_name)}/([^/]+)/(?:figures|supplements)/(.+)$", path
        )
        if m:
            return m.group(1), m.group(2)

        # Path without root: <id>/figures/<file>
        m = _re.match(r"^([^/]+)/(?:figures|supplements)/(.+)$", path)
        if m:
            return m.group(1), m.group(2)

        # Bare filename — fall back to cwd
        if self.fs and hasattr(self.fs, "path_parser"):
            parsed = self.fs.path_parser.parse(self.cwd)
            if parsed.document_id:
                return parsed.document_id, path

        return None, path

    async def _cmd_ask_image(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Analyze a figure from a paper.

        Usage:
            ask_image /papers/<id>/figures/<file> "question"
            ask_image <id>/figures/<file> "question"
            ask_image FIGURE_ID "question"         # when cd'd into a paper dir
            ask_image --list                       # list figures (requires cd)

        Examples:
            ask_image /papers/PMC12345/figures/fig1.jpg "Describe this figure"
            ask_image PMC12345/figures/fig1.jpg "What does this show?"
        """
        # Check if we're in a /.gxl/ paper directory (uploaded PDF)
        session_figures_dir = self._find_session_figures_dir()
        if session_figures_dir:
            return await self._cmd_ask_image_session(
                args, session_figures_dir, session_id=session_id
            )

        if not self.fs or not hasattr(self.fs, "_ask_image"):
            return TerminalResult(
                stderr="ask_image: not available (filesystem module does not implement _ask_image)",
                exit_code=1,
                cwd=self.cwd,
            )

        if not args or args[0] == "--list":
            # --list requires cwd context
            parsed = self.fs.path_parser.parse(self.cwd)
            doc_id = parsed.document_id
            if not doc_id:
                return TerminalResult(
                    stderr="ask_image --list: must be in a paper directory (cd /papers/<id>)",
                    exit_code=1,
                    cwd=self.cwd,
                )
            result = await self.fs._cat(
                path=f"/{self.fs.path_parser.root_name}/{doc_id}/figures/",
                session_id=session_id,
            )
            if isinstance(result, dict) and "contents" in result:
                figures = result["contents"]
                if not figures:
                    return TerminalResult(
                        stdout="No figures found for this paper.\n", cwd=self.cwd
                    )
                output = "Available figures:\n"
                for f in figures:
                    name = f.get("name", f.get("filename", "unknown"))
                    label = f.get("label", "")
                    output += f"  {name}"
                    if label:
                        output += f"  ({label})"
                    output += "\n"
                return TerminalResult(stdout=output, cwd=self.cwd)
            return TerminalResult(stdout=str(result) + "\n", cwd=self.cwd)

        positional = [a for a in args if not a.startswith("-")]

        if not positional:
            return TerminalResult(
                stderr=(
                    'ask_image: usage: ask_image /papers/<id>/figures/<file> "question"\n'
                    "  Example: ask_image /papers/PMC12345/figures/fig1.jpg \"What does this show?\""
                ),
                exit_code=1,
                cwd=self.cwd,
            )

        if len(positional) == 1:
            figure_paths = [positional[0]]
            question = "Describe this figure in detail."
        else:
            figure_paths = positional[:-1]
            question = positional[-1]

        outputs = []
        any_error = False
        for fig_path in figure_paths:
            doc_id, fig_id = self._parse_figure_path(fig_path)
            if not doc_id:
                any_error = True
                outputs.append(
                    f"[{fig_path}] Error: cannot determine paper ID. "
                    f"Use full path: /papers/<id>/figures/<filename>"
                )
                continue
            try:
                result = await self.fs._ask_image(
                    document_id=doc_id,
                    figure_id=fig_id,
                    question=question,
                    session_id=session_id,
                )
                if "error" in result:
                    any_error = True
                    outputs.append(f"[{fig_path}] Error: {result['error']}")
                else:
                    analysis = result.get("analysis", result.get("description", ""))
                    caption = result.get("caption", "")
                    header = f"[{fig_path}]"
                    if caption:
                        header += f" {caption[:100]}..."
                    outputs.append(f"{header}\n{analysis}")
            except Exception as e:
                any_error = True
                outputs.append(f"[{fig_path}] Error: {e}")

        return TerminalResult(
            stdout="\n\n".join(outputs) + "\n",
            exit_code=1 if any_error else 0,
            cwd=self.cwd,
        )

    async def _cmd_ask_image_session(
        self,
        args: list[str],
        figures_vpath: str,
        session_id: str = "default",
    ) -> TerminalResult:
        """Handle ask_image for uploaded PDFs in /.gxl/."""
        # Resolve the figures directory to a local path
        local_dir = self._get_local_session_dir(session_id)
        rel = self._session_files_to_sandbox_path(figures_vpath)
        figures_local = os.path.join(local_dir, rel)

        if not os.path.isdir(figures_local):
            return TerminalResult(
                stderr=f"ask_image: no figures/ directory at {figures_vpath}",
                exit_code=1,
                cwd=self.cwd,
            )

        image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        all_figures = sorted(
            f for f in os.listdir(figures_local)
            if os.path.splitext(f)[1].lower() in image_exts
        )

        if not args or args[0] == "--list":
            if not all_figures:
                return TerminalResult(stdout="No figures found.\n", cwd=self.cwd)
            output = "Available figures:\n"
            for f in all_figures:
                size = os.path.getsize(os.path.join(figures_local, f))
                output += f"  {f}  ({size / 1024:.1f} KB)\n"
            return TerminalResult(stdout=output, cwd=self.cwd)

        # Parse args: figure names and question.
        # Strategy: prefer matching against *known* figure filenames so a
        # prompt that happens to contain a period (e.g. "...dataset.") is
        # never silently mis-classified as a figure id. If nothing matches
        # by name, fall back to the positional contract used by the
        # /papers/ branch (last arg = question, rest = figures).
        positional = [a for a in args if not a.startswith("-")]

        figure_ids: list[str] = []
        question_parts: list[str] = []
        for arg in positional:
            is_known_figure = any(
                arg == f or f.startswith(arg) or arg in f for f in all_figures
            )
            if is_known_figure and not question_parts:
                if arg not in figure_ids:
                    figure_ids.append(arg)
            else:
                question_parts.append(arg)

        if not figure_ids and positional:
            if len(positional) == 1:
                figure_ids = [positional[0]]
                question_parts = []
            else:
                figure_ids = list(positional[:-1])
                question_parts = [positional[-1]]

        if not figure_ids:
            return TerminalResult(
                stderr='ask_image: usage: ask_image FIGURE_NAME [FIGURE_NAME ...] "question"\n'
                f"Available: {', '.join(all_figures[:5])}",
                exit_code=1,
                cwd=self.cwd,
            )

        question = " ".join(question_parts) if question_parts else "Describe this figure in detail."

        outputs = []
        any_error = False
        for fig_id in figure_ids:
            # Resolve: exact match or partial match
            resolved = None
            for f in all_figures:
                if f == fig_id or f.startswith(fig_id) or fig_id in f:
                    resolved = f
                    break
            if not resolved:
                any_error = True
                outputs.append(f"[{fig_id}] Error: figure not found in {figures_vpath}")
                continue

            fig_path = os.path.join(figures_local, resolved)
            try:
                result = await self._ask_image_local(fig_path, question, session_id)
                if "error" in result:
                    any_error = True
                    outputs.append(f"[{resolved}] Error: {result['error']}")
                else:
                    analysis = result.get("analysis", "")
                    outputs.append(f"[{resolved}]\n{analysis}")
            except Exception as e:
                any_error = True
                outputs.append(f"[{resolved}] Error: {e}")

        return TerminalResult(
            stdout="\n\n".join(outputs) + "\n",
            exit_code=1 if any_error else 0,
            cwd=self.cwd,
        )

    async def _cmd_filter(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Filter search results for relevance to the user's query.

        Usage:
            filter --from RESULTS_ID "user's original query"
            filter --from RESULTS_ID --require N "user's original query"

        Options:
            --require N   Fail if fewer than N papers survive filtering.
                          The filtered results are still saved (junk removed),
                          but exit code 1 signals the agent to search more.
        """
        if not self.fs or not hasattr(self.fs, "_filter"):
            return TerminalResult(
                stderr="filter: not available (filesystem module does not implement _filter)",
                exit_code=1,
                cwd=self.cwd,
            )

        from_results = None
        require_count = None
        query_parts = []
        i = 0
        while i < len(args):
            if args[i] == "--from" and i + 1 < len(args):
                from_results = args[i + 1]
                i += 2
            elif args[i] == "--require" and i + 1 < len(args):
                try:
                    require_count = int(args[i + 1])
                except ValueError:
                    return TerminalResult(
                        stderr=f"filter: --require must be an integer, got: {args[i + 1]}",
                        exit_code=1,
                        cwd=self.cwd,
                    )
                i += 2
            else:
                query_parts.append(args[i])
                i += 1

        query = " ".join(query_parts)
        if not from_results:
            return TerminalResult(
                stderr="filter: --from RESULTS_ID is required\n"
                '  usage: filter --from s_abc123 "user query"',
                exit_code=1,
                cwd=self.cwd,
            )
        if not query:
            return TerminalResult(
                stderr="filter: query is required (the user's original question)",
                exit_code=1,
                cwd=self.cwd,
            )

        try:
            result = await self.fs._filter(
                from_results=from_results,
                query=query,
                session_id=session_id,
            )

            if "error" in result:
                return TerminalResult(
                    stderr=f"filter: {result['error']}", exit_code=1, cwd=self.cwd
                )

            pool_size = result.get("original_count", 0)
            total_approved = result.get("filtered_count", 0)
            new_evaluated = result.get("new_evaluated", pool_size)
            newly_approved = result.get("newly_approved", total_approved)
            prev_approved = result.get("previously_approved", 0)
            time_ms = result.get("time_ms", 0)
            results_id = result.get("results_id", from_results)

            if prev_approved > 0:
                output = (
                    f"Filtered: {new_evaluated} new papers evaluated, {newly_approved} passed"
                    f" → {total_approved} total approved (from {pool_size} accumulated) in {time_ms}ms\n"
                    f"  → {total_approved} papers after filtering\n"
                    f"Results ID: {results_id} (updated in place)\n"
                )
            else:
                removed = new_evaluated - newly_approved
                output = (
                    f"Filtered: {pool_size} → {total_approved} papers ({removed} removed as irrelevant) in {time_ms}ms\n"
                    f"  → {total_approved} papers after filtering\n"
                    f"Results ID: {results_id} (updated in place)\n"
                )

            if require_count is not None and total_approved < require_count:
                output += (
                    f"\nERR: {total_approved} papers after filtering, but --require {require_count} was specified.\n"
                    f"  Run additional searches with different/broader terms to increase coverage, then filter again.\n"
                    f"  Previously-evaluated papers are cached — only new papers will be re-filtered.\n"
                )
                return TerminalResult(stdout=output, exit_code=1, cwd=self.cwd)

            return TerminalResult(stdout=output, cwd=self.cwd)
        except Exception as e:
            return TerminalResult(stderr=f"filter: {e}", exit_code=1, cwd=self.cwd)

    async def _cmd_map(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Run a query across multiple papers in parallel.

        Usage:
            map --from RESULTS_ID "query"
            map --from RESULTS_ID --output_schema '{...}' "q"
            map --document-ids ID1,ID2,... "query"
        """
        if not self.fs or not hasattr(self.fs, "_parallel"):
            return TerminalResult(
                stderr="map: not available (filesystem module does not implement _parallel)",
                exit_code=1,
                cwd=self.cwd,
            )

        from_results = None
        output_schema = None
        limit = None
        offset = None
        map_document_ids = None
        query_parts = []
        i = 0
        while i < len(args):
            if args[i] == "--from" and i + 1 < len(args):
                from_results = args[i + 1]
                i += 2
            elif args[i] == "--mode" and i + 1 < len(args):
                i += 2  # Silently ignore legacy --mode flag
            elif args[i] in ("--document-ids", "--document_ids") and i + 1 < len(args):
                map_document_ids = [
                    d.strip() for d in args[i + 1].split(",") if d.strip()
                ]
                i += 2
            elif args[i] == "--output_schema" and i + 1 < len(args):
                try:
                    output_schema = json.loads(args[i + 1])
                except json.JSONDecodeError:
                    return TerminalResult(
                        stderr=f"map: invalid JSON schema: {args[i + 1]}",
                        exit_code=1,
                        cwd=self.cwd,
                    )
                i += 2
            elif args[i] == "--limit" and i + 1 < len(args):
                limit = int(args[i + 1])
                i += 2
            elif args[i] == "--offset" and i + 1 < len(args):
                offset = int(args[i + 1])
                i += 2
            else:
                query_parts.append(args[i])
                i += 1

        query = " ".join(query_parts)

        if not from_results and not map_document_ids:
            from_results = self._last_search_results_id
        if not from_results and not map_document_ids:
            return TerminalResult(
                stderr="map: no search results to map over. Run a search first, or use --from RESULTS_ID.\n"
                '  usage: map --from s_abc123 "query"',
                exit_code=1,
                cwd=self.cwd,
            )
        if not query:
            return TerminalResult(
                stderr="map: query is required", exit_code=1, cwd=self.cwd
            )

        # Build tasks directly from document IDs (repo-scoped map)
        tasks = None
        if map_document_ids:
            root = self.fs.path_parser.root_name if hasattr(self.fs, "path_parser") else "papers"
            if offset:
                map_document_ids = map_document_ids[offset:]
            if limit:
                map_document_ids = map_document_ids[:limit]
            tasks = [
                {"path": f"/{root}/{doc_id}/", "query": query}
                for doc_id in map_document_ids
            ]

        try:
            parallel_kwargs = {
                "query": query,
                "output_schema": output_schema,
                "session_id": session_id,
            }
            if tasks:
                parallel_kwargs["tasks"] = tasks
            else:
                parallel_kwargs["from_results"] = from_results
                parallel_kwargs["limit"] = limit
                parallel_kwargs["offset"] = offset

            result = await self.fs._parallel(**parallel_kwargs)
            if "error" in result:
                return TerminalResult(
                    stderr=f"map: {result['error']}", exit_code=1, cwd=self.cwd
                )
            map_id = result.get("map_id", "unknown")
            n_tasks = result.get("tasks_executed", 0)
            n_ok = result.get("tasks_successful", 0)
            time_ms = result.get("time_ms", 0)
            all_results = result.get("results", [])

            # Build full detailed output for the saved file
            full_lines = [
                f"Map results: {n_ok}/{n_tasks} tasks succeeded in {time_ms}ms",
                f"Results ID: {map_id}",
                f"Query: {query}",
                "",
            ]
            for idx, r in enumerate(all_results, 1):
                title = r.get("title", r.get("path", "unknown"))
                status = r.get("status", "unknown")
                doc_id = r.get("document_id", "")
                full_lines.append(f"--- [{idx}/{n_tasks}] [{status}] {title} ---")
                if doc_id:
                    full_lines.append(f"  doc_id: {doc_id}")
                if r.get("error"):
                    full_lines.append(f"  error: {r['error']}")
                if r.get("output"):
                    full_lines.append(f"  {r['output']}")
                full_lines.append("")

            full_text = "\n".join(full_lines)

            # Save to /.gxl/ scratch directory
            results_filename = f"map_{map_id}.txt"
            saved_path = None
            try:
                local_dir = self._get_local_session_dir(session_id)
                os.makedirs(local_dir, exist_ok=True)
                local_path = os.path.join(local_dir, results_filename)
                with open(local_path, "w") as f:
                    f.write(full_text)
                saved_path = f"/.gxl/{results_filename}"
            except Exception:
                pass

            # Terminal output: summary header + all results inline
            output = f"Map complete: {n_ok}/{n_tasks} tasks succeeded in {time_ms}ms\n"
            output += f"Results ID: {map_id}\n"
            if saved_path:
                output += f"Full results: {saved_path}\n"
            output += "\n"

            for r in all_results:
                title = r.get("title", r.get("path", "unknown"))[:70]
                status = r.get("status", "unknown")
                doc_id = r.get("document_id", "")
                id_tag = f"  ({doc_id})" if doc_id else ""
                output += f"  [{status}] {title}{id_tag}\n"
                if r.get("error"):
                    output += f"    error: {r['error']}\n"
                elif r.get("output"):
                    preview = str(r["output"])[:150].replace("\n", " ")
                    output += f"    {preview}\n"

            output += f'\nUse: paperclip cat {saved_path}  to read full results\n' if saved_path else ""
            output += f'Use: reduce --from {map_id} --strategy summarize "question"\n'

            return TerminalResult(stdout=output, cwd=self.cwd)
        except Exception as e:
            return TerminalResult(stderr=f"map: {e}", exit_code=1, cwd=self.cwd)

    async def _cmd_reduce(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Synthesize results from a map operation.

        Usage:
            reduce --from MAP_ID --strategy STRATEGY "question"

        Strategies: summarize, table, themes, consensus, bullet_points, extract
        """
        if not self.fs or not hasattr(self.fs, "_reduce"):
            return TerminalResult(
                stderr="reduce: not available (filesystem module does not implement _reduce)",
                exit_code=1,
                cwd=self.cwd,
            )

        from_map = None
        strategy = "summarize"
        question_parts = []
        columns = None
        fields = None
        max_items = None
        i = 0
        while i < len(args):
            if args[i] == "--from" and i + 1 < len(args):
                from_map = args[i + 1]
                i += 2
            elif args[i] == "--strategy" and i + 1 < len(args):
                strategy = args[i + 1]
                i += 2
            elif args[i] in ("--limit", "--max") and i + 1 < len(args):
                try:
                    max_items = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
            elif args[i] == "--columns" and i + 1 < len(args):
                columns = [c.strip() for c in args[i + 1].split(",")]
                i += 2
            elif args[i] == "--fields" and i + 1 < len(args):
                fields = [f.strip() for f in args[i + 1].split(",")]
                i += 2
            else:
                question_parts.append(args[i])
                i += 1

        question = " ".join(question_parts) or None
        if not from_map:
            return TerminalResult(
                stderr="reduce: --from MAP_ID is required\n"
                '  usage: reduce --from m_abc123 --strategy summarize "question"',
                exit_code=1,
                cwd=self.cwd,
            )

        reduce_kwargs = dict(
            strategy=strategy,
            from_map=from_map,
            question=question,
            columns=columns,
            fields=fields,
            session_id=session_id,
        )
        if max_items is not None:
            reduce_kwargs["max_items"] = max_items

        try:
            result = await self.fs._reduce(**reduce_kwargs)
            artifact_id = result.get("artifact_id", "")
            n_items = result.get("items_processed", 0)
            time_ms = result.get("time_ms", 0)

            output_data = result.get("output", "")
            if isinstance(output_data, dict):
                output_text = json.dumps(output_data, indent=2, default=str)
            elif isinstance(output_data, list):
                output_text = "\n".join(str(item) for item in output_data)
            else:
                output_text = str(output_data)

            header = f"Reduce ({strategy}): {n_items} items in {time_ms}ms | artifact: {artifact_id}\n\n"
            footer = ""
            if artifact_id:
                # Build a meaningful description from strategy + question
                if question:
                    desc = question[:80]
                elif strategy == "table":
                    desc = f"Comparison table of {n_items} papers"
                elif strategy == "summarize":
                    desc = f"Summary across {n_items} papers"
                elif strategy == "themes":
                    desc = f"Themes from {n_items} papers"
                elif strategy == "consensus":
                    desc = f"Consensus from {n_items} papers"
                elif strategy == "bullet_points":
                    desc = f"Key points from {n_items} papers"
                else:
                    desc = f"{strategy} of {n_items} papers"
                footer = (
                    f"\n\nCite this result in your response with:\n"
                    f'  {{{{"artifact": {{{{"artifact_id": "{artifact_id}", "type": "{strategy}", '
                    f'"source_count": {n_items}, "description": "YOUR_ONE_SENTENCE_SUMMARY"}}}}}}}}'
                )

            full_text = header + output_text + footer + "\n"

            if artifact_id:
                try:
                    local_dir = self._get_local_session_dir(session_id)
                    os.makedirs(local_dir, exist_ok=True)
                    reduce_file = os.path.join(local_dir, f"reduce_{artifact_id}.txt")
                    with open(reduce_file, "w") as f:
                        f.write(full_text)
                    footer += f"\nFull results: /.gxl/reduce_{artifact_id}.txt"
                    full_text = header + output_text + footer + "\n"
                except Exception:
                    pass

            return TerminalResult(
                stdout=full_text, cwd=self.cwd
            )
        except Exception as e:
            return TerminalResult(stderr=f"reduce: {e}", exit_code=1, cwd=self.cwd)

    async def _cmd_references(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Show all registered references for this session.

        Usage:
            references              # List all references
            references --json       # Output as JSON
        """
        refs = self._get_references(session_id)
        if not refs:
            return TerminalResult(
                stdout="No references registered yet. Use: cite LINE [LINE...]\n",
                cwd=self.cwd,
            )

        if args and args[0] == "--json":
            return TerminalResult(
                stdout=json.dumps(refs, indent=2, default=str) + "\n", cwd=self.cwd
            )

        lines = []
        for r in refs:
            content = r.get("content", "")[:60]
            section = r.get("section", "")
            lines.append(
                f'[{r["ref_num"]}] Line {r["line_number"]}: "{content}..." ({section})'
            )
        lines.append(f"\n{len(refs)} references total")
        return TerminalResult(stdout="\n".join(lines) + "\n", cwd=self.cwd)

    async def _cmd_search(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Search across all papers (biorxiv + medrxiv + PMC) in the filesystem.

        Usage:
            search QUERY                  # search primary sources (default; excludes abstracts)
            search pmc QUERY              # PMC only
            search biorxiv QUERY          # bioRxiv only
            search medrxiv QUERY          # medRxiv only
            search arxiv QUERY            # arXiv only
            search pmc biorxiv QUERY      # PMC + bioRxiv
            search --source pmc QUERY     # same as above, explicit flag
            search --include-abstracts QUERY # include abstract-only corpus
            search --include_abstracts QUERY # same as --include-abstracts
            search -s abstracts QUERY     # abstract-only corpus only
            search -s arxiv QUERY         # arXiv preprints
            search -r PATTERN             # regex search across all papers
            search -a AUTHOR              # search by author
            search -t TITLE               # search by title
            search -n LIMIT QUERY         # limit results (default: 20)
            search --tag TOPIC QUERY      # accumulate into named result group
            search --recent QUERY         # papers from last year
            search --quiet QUERY          # minimal output (count + result ID only)

        Filters:
            search --since 30d QUERY      # posted in last 30 days
            search --since 7d QUERY       # posted in last 7 days
            search --since 6m QUERY       # posted in last 6 months
            search --category Neuroscience QUERY  # by bioRxiv category
            search --journal "Nature Medicine" QUERY  # by journal (PMC)
            search --year 2024 QUERY      # by publication year
            search --sort date QUERY      # sort by recency (not relevance)
        Ranking strategy (default: hybrid):
            search --ranking hybrid QUERY     # BM25 + vector fused (default)
            search --ranking bm25 QUERY       # BM25 only (fastest)
            search --ranking vector QUERY     # vector only (conceptual)

        Scope (default: papers from ~2024+):
            search --all QUERY              # search ALL papers (slower)

        Chaining (search then grep full text — preferred for precision):
            search "protein design" | grep "RFdiffusion|ProteinMPNN"
            search "CRISPR delivery" | grep "LNP|lipid nanoparticle"

        Search modes (how strictly terms must match):
            search -m any QUERY           # Any term matches (default, broadest)
            search -m 50% QUERY           # At least half the terms must match
            search -m 75% QUERY           # At least 75% of terms must match
            search -m all QUERY           # All terms must match (strictest)
            search -e QUERY               # Exact phrase (words together in order)

        Examples:
            search "protein folding"                    # Primary sources
            search pmc "CRISPR base editing"            # PMC only
            search biorxiv medrxiv "COVID-19"           # biorxiv + medrxiv
            search --source pmc,biorxiv "Alzheimer"     # comma-separated
            search -s abstracts "drug discovery"        # abstract-only corpus
            search -m all "CRISPR cancer"               # Both words required
            search pmc --since 1y "gene therapy"        # PMC, last year
            search -r "CRISPR.*Cas9"                    # Regex pattern
            search -a "Smith" -n 20                     # Author search
        Count mode (fast, for parallel exploration):
            search -c "CRISPR"                    # Just return count, no results
        """
        if not self.fs:
            return TerminalResult(
                stderr="search: filesystem not available", exit_code=1, cwd=self.cwd
            )

        # Parse flags
        is_regex = "-r" in args or "--regex" in args
        is_author = "-a" in args or "--author" in args
        is_title = "-t" in args or "--title" in args
        is_recent = "--recent" in args
        is_exact = "-e" in args or "--exact" in args
        is_count_only = "-c" in args or "--count" in args
        is_metadata_only = "-M" in args or "--metadata" in args
        is_quiet = "--quiet" in args
        is_all_time = "--all" in args
        is_json = "--json" in args

        # Parse search mode
        search_mode = "phrase" if is_exact else "any"  # Default to "any"
        for i, arg in enumerate(args):
            if arg == "-m" and i + 1 < len(args):
                mode_val = args[i + 1]
                if mode_val in ("any", "all", "50%", "75%", "phrase"):
                    search_mode = mode_val

        # Parse --tag (accumulation group)
        tag_val = None
        for i, arg in enumerate(args):
            if arg == "--tag" and i + 1 < len(args):
                tag_val = args[i + 1].lower().strip()

        # Parse limit (default: 20, max: 1000)
        limit = 20
        for i, arg in enumerate(args):
            if arg == "-n" and i + 1 < len(args):
                try:
                    limit = min(int(args[i + 1]), 1000)
                except ValueError:
                    pass

        # Parse document type filter
        source_type = None
        for i, arg in enumerate(args):
            if arg in ("-T", "--type") and i + 1 < len(args):
                source_type = args[i + 1].lower()

        # Parse --since (e.g. "30d", "7d", "6m", "1y")
        since_val = None
        for i, arg in enumerate(args):
            if arg == "--since" and i + 1 < len(args):
                since_val = args[i + 1]

        # Parse --category (e.g. "Neuroscience")
        category_val = None
        for i, arg in enumerate(args):
            if arg == "--category" and i + 1 < len(args):
                category_val = args[i + 1]

        # Parse --sort (e.g. "date")
        sort_val = None
        for i, arg in enumerate(args):
            if arg == "--sort" and i + 1 < len(args):
                sort_val = args[i + 1]

        # Parse --journal (e.g. "Nature Medicine")
        journal_val = None
        for i, arg in enumerate(args):
            if arg == "--journal" and i + 1 < len(args):
                journal_val = args[i + 1]

        # Parse --article-type / --atype (e.g. "review-article")
        article_type_val = None
        for i, arg in enumerate(args):
            if arg in ("--article-type", "--atype") and i + 1 < len(args):
                article_type_val = args[i + 1].lower()

        # Parse --year (e.g. "2024")
        year_val = None
        for i, arg in enumerate(args):
            if arg == "--year" and i + 1 < len(args):
                year_val = args[i + 1]

        # Parse --ranking (hybrid | bm25 | vector)
        ranking_val = None
        for i, arg in enumerate(args):
            if arg == "--ranking" and i + 1 < len(args):
                rv = args[i + 1].lower()
                if rv in ("hybrid", "bm25", "vector"):
                    ranking_val = rv

        # Parse --include-abstracts (add abstract-only corpus to default sources)
        include_abstracts = "--include-abstracts" in args or "--include_abstracts" in args

        # Parse --document-ids (hard-scope to specific paper IDs, used by repo-scoped search)
        scope_document_ids = None
        for i, arg in enumerate(args):
            if arg in ("--document-ids", "--document_ids") and i + 1 < len(args):
                scope_document_ids = [
                    d.strip() for d in args[i + 1].split(",") if d.strip()
                ]

        # Parse --source / source keywords
        # Supports:  --source pmc,biorxiv  OR  -s abstracts  OR
        #           search pmc "query"  OR  search all "query"
        # Recognised source values mirror the Papers MCP corpus map:
        # biorxiv / medrxiv → biomedrxiv; arxiv; pmc; abstracts → abstract_only.
        SOURCE_KEYWORDS = {
            "pmc", "biorxiv", "medrxiv", "arxiv", "abstracts", "openalex", "all",
        }
        source_filter = None  # None = all sources (default)
        for i, arg in enumerate(args):
            if arg in ("--source", "-s") and i + 1 < len(args):
                raw = args[i + 1].lower()
                sources = [s.strip() for s in raw.replace("+", ",").split(",")]
                source_filter = None if "all" in sources else sources
                break

        # Extract query (non-flag arguments)
        # Source keywords at start of query (e.g. "search pmc CRISPR") are consumed
        query_parts = []
        skip_next = False
        leading_sources = []  # collect source keywords before the real query
        seen_non_source = False
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg in (
                "-r", "--regex", "-a", "--author", "-t", "--title",
                "--recent", "-e", "--exact", "-c", "--count",
                "-M", "--metadata", "--quiet", "--json", "--all",
                "--include-abstracts", "--include_abstracts",
                "--corpus",
            ):
                continue
            if arg in (
                "-n", "-m", "-T", "--type", "--since", "--category",
                "--sort", "--tag", "--source", "-s", "--ranking",
                "--journal", "--article-type", "--atype", "--year",
                "--document-ids", "--document_ids",
            ):
                skip_next = True
                continue
            if not arg.startswith("-"):
                # Consume leading source keywords (before the real query text)
                if not seen_non_source and arg.lower() in SOURCE_KEYWORDS:
                    leading_sources.append(arg.lower())
                else:
                    seen_non_source = True
                    query_parts.append(arg)

        # If source keywords were provided positionally and --source wasn't set yet
        if leading_sources and source_filter is None:
            if "all" not in leading_sources:
                source_filter = leading_sources

        # --include-abstracts: add the abstracts corpus to whatever sources are active
        if include_abstracts:
            if source_filter is None:
                source_filter = ["biorxiv", "medrxiv", "arxiv", "pmc", "abstracts"]
            elif "abstracts" not in source_filter and "openalex" not in source_filter:
                source_filter = list(source_filter) + ["abstracts"]

        query = " ".join(query_parts).strip()

        # Allow empty query if -T is provided (listing mode)
        if not query and not source_type:
            return TerminalResult(
                stderr="search: usage: search QUERY\n"
                "       search -s SOURCE QUERY   # filter by source (pmc, biorxiv, medrxiv, arxiv, abstracts)\n"
                "       search -r PATTERN        # regex\n"
                "       search -n 20 QUERY       # limit results\n"
                "       search -m all QUERY      # all terms must match\n"
                "       search -e QUERY          # exact phrase\n"
                "       search --since 30d QUERY # recent papers\n"
                "       search --all QUERY       # full corpus (slower)\n"
                "  Sources: pmc, biorxiv, medrxiv, arxiv, abstracts (opt-in)\n"
                "  Examples:\n"
                '       search "protein folding"              # default sources\n'
                '       search -s pmc "gene therapy"          # PMC only\n'
                '       search -s abstracts "drug discovery"  # abstract corpus',
                exit_code=1,
                cwd=self.cwd,
            )

        try:
            if is_regex:
                # Use papers_grep for regex search
                result = await self.fs._grep(
                    regex=query,
                    limit=limit,
                    session_id=session_id,
                )
            else:
                # Use papers_find for semantic search
                # Date range for recent
                date_range = "last_year" if is_recent else None

                # Build find arguments
                find_kwargs = {
                    "limit": limit,
                    "session_id": session_id,
                    "search_mode": search_mode,
                }

                if is_author:
                    find_kwargs["author"] = query
                elif is_title:
                    find_kwargs["title"] = query
                else:
                    find_kwargs["query"] = query

                if date_range:
                    find_kwargs["date_range"] = date_range

                if source_type:
                    find_kwargs["source_type"] = source_type

                if is_metadata_only:
                    find_kwargs["metadata_only"] = True

                if since_val:
                    find_kwargs["since"] = since_val
                if category_val:
                    find_kwargs["category"] = category_val
                if journal_val:
                    find_kwargs["journal"] = journal_val
                if article_type_val:
                    find_kwargs["article_type"] = article_type_val
                if year_val:
                    find_kwargs["year"] = year_val
                if sort_val:
                    find_kwargs["sort"] = sort_val
                if source_filter is not None:
                    find_kwargs["source"] = source_filter
                if is_all_time:
                    find_kwargs["all_time"] = True
                if ranking_val:
                    find_kwargs["ranking"] = ranking_val
                if scope_document_ids:
                    find_kwargs["document_ids"] = scope_document_ids

                result = await self.fs._find(**find_kwargs)

            if "error" in result:
                return TerminalResult(
                    stderr=f"search: {result['error']}", exit_code=1, cwd=self.cwd
                )

            # Papers returned from this individual search
            search_papers = result.get("results", result.get("papers", []))
            search_results_id = result.get("results_id", "")
            search_count = result.get("count", len(search_papers))
            search_meta = result.get("search_meta")

            # JSON mode: return structured data for programmatic consumers (CLI, etc.)
            if is_json:
                json_result = {
                    "results_id": search_results_id,
                    "count": search_count,
                    "papers": search_papers,
                }
                return TerminalResult(
                    stdout=json.dumps(json_result, default=str) + "\n",
                    cwd=self.cwd,
                    metadata={"results_id": search_results_id} if search_results_id else None,
                )

            # Count-only mode: just return the count for fast parallel exploration
            if is_count_only:
                if search_count == 0:
                    return TerminalResult(
                        stdout=f"0 results for: {query}\n", cwd=self.cwd
                    )
                return TerminalResult(
                    stdout=f"{search_count} results for: {query}\n", cwd=self.cwd
                )

            if not search_papers:
                type_msg = f" in {source_type}" if source_type else ""
                item_name = "papers" if self.root_path == "/papers/" else "documents"
                return TerminalResult(
                    stdout=f"No {item_name} found{type_msg}.\n", cwd=self.cwd
                )

            # --- Tag-based accumulation ---
            results_id = search_results_id
            total_accumulated = search_count
            new_count = search_count
            truncated = False

            if tag_val and hasattr(self.fs, "results_registry"):
                # Load full results from registry (search_papers may be a preview)
                full_search_data = self.fs.results_registry.load(
                    search_results_id, session_id
                )
                full_papers = (
                    full_search_data["papers"] if full_search_data else search_papers
                )

                accumulator_id = self._search_accumulator_ids.get(tag_val)
                if accumulator_id:
                    pool_id = f"{accumulator_id}__pool"
                    pool_data = self.fs.results_registry.load(pool_id, session_id)
                    if not pool_data:
                        pool_data = self.fs.results_registry.load(
                            accumulator_id, session_id
                        )
                    if pool_data:
                        existing_papers = pool_data.get("papers", [])
                        existing_doc_ids = {
                            p.get("document_id") for p in existing_papers
                        }
                        new_papers = [
                            p
                            for p in full_papers
                            if p.get("document_id") not in existing_doc_ids
                        ]
                        merged_papers = existing_papers + new_papers
                        queries = pool_data.get("queries", [pool_data.get("query", "")])
                        queries.append(query)
                        new_count = len(new_papers)
                    else:
                        merged_papers = full_papers
                        queries = [query]
                        new_count = len(full_papers)
                else:
                    merged_papers = full_papers
                    queries = [query]
                    new_count = len(full_papers)
                    accumulator_id = search_results_id
                    self._search_accumulator_ids[tag_val] = accumulator_id

                # Cap at 1000
                if len(merged_papers) > self._ACCUMULATOR_CAP:
                    merged_papers = merged_papers[: self._ACCUMULATOR_CAP]
                    truncated = True

                pool_save_data = {
                    "papers": merged_papers,
                    "query": "; ".join(queries),
                    "queries": queries,
                }
                self.fs.results_registry.save(
                    data=pool_save_data,
                    session_id=session_id,
                    results_id=accumulator_id,
                )
                self.fs.results_registry.save(
                    data=pool_save_data,
                    session_id=session_id,
                    results_id=f"{accumulator_id}__pool",
                )
                if hasattr(self.fs, "_save_search_artifact"):
                    asyncio.create_task(self.fs._save_search_artifact(
                        accumulator_id, merged_papers, "; ".join(queries), session_id
                    ))

                total_accumulated = len(merged_papers)
                results_id = accumulator_id

            # --- Format output ---
            type_suffix = f" [{source_type}]" if source_type else ""
            item_name = "papers" if self.root_path == "/papers/" else "documents"
            display_papers = search_papers

            output_lines = [
                f"Found {search_count} {item_name}{type_suffix} (showing {len(display_papers)}):"
            ]
            if results_id:
                output_lines[0] += f"  [results_id: {results_id}]"
            if tag_val:
                output_lines[0] += f"  [tag: {tag_val}]"
                if total_accumulated > search_count:
                    output_lines[
                        0
                    ] += f"  [{total_accumulated} accumulated, {new_count} new]"
                if truncated:
                    output_lines.append(
                        f"  ⚠ Accumulated results capped at {self._ACCUMULATOR_CAP} papers."
                    )
            if isinstance(search_meta, dict):
                _backend_line = self._format_search_backends(search_meta)
                if _backend_line:
                    output_lines.append(_backend_line)
            output_lines.append("")

            for i, paper in enumerate(display_papers, 1):
                doc_id = paper.get("document_id", "?")
                title = paper.get("title", "Untitled")
                authors = paper.get("authors", "")
                pub_date = paper.get("pub_date", "") or paper.get("month_year", "")
                doi = paper.get("doi", "")
                source = paper.get("source", "")
                abstract_snippet = paper.get("abstract_snippet", "")

                output_lines.append(f"  {i}. {title}")
                id_line = f"     {doc_id}"
                if source:
                    _src = source.lower()
                    source_label = _SOURCE_DISPLAY.get(_src, source)
                    id_line += f" · {source_label}"
                if pub_date:
                    id_line += f" · {pub_date}"
                output_lines.append(id_line)
                if doi:
                    output_lines.append(f"     https://doi.org/{doi}")
                if authors:
                    output_lines.append(f"     {authors}")
                if abstract_snippet:
                    output_lines.append(f'     "{abstract_snippet}"')
                output_lines.append("")

            if results_id:
                output_lines.append(
                    f'Narrow with full-text grep: grep --from {results_id} "PATTERN"'
                )

            _meta = {"results_id": results_id} if results_id else None

            if results_id:
                self._last_search_results_id = results_id

            if is_quiet:
                quiet_lines = [output_lines[0]]  # "Found N papers ..."
                return TerminalResult(
                    stdout="\n".join(quiet_lines) + "\n", cwd=self.cwd,
                    metadata=_meta,
                )

            return TerminalResult(
                stdout="\n".join(output_lines) + "\n", cwd=self.cwd,
                metadata=_meta,
            )

        except Exception as e:
            return TerminalResult(stderr=f"search: {e}", exit_code=1, cwd=self.cwd)

    async def _cmd_funded_by(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Find papers by funder name in Funding/Acknowledgments sections.

        Usage:
            funded-by "NIH"                         # search for NIH-funded papers
            funded-by "Wellcome Trust"               # funder name search
            funded-by "European Commission"          # any funder
            funded-by -n 50 "NIH"                    # limit results
            funded-by --since 6m "NIH"               # funded papers from last 6 months

        Searches the Funding and Acknowledgments sections of all papers
        for the given funder name or grant identifier.
        """
        if not self.fs:
            return TerminalResult(
                stderr="funded-by: filesystem not available",
                exit_code=1,
                cwd=self.cwd,
            )

        # Parse --tag
        tag_val = None
        for i, arg in enumerate(args):
            if arg == "--tag" and i + 1 < len(args):
                tag_val = args[i + 1].lower().strip()

        # Parse -n limit
        limit = 100
        for i, arg in enumerate(args):
            if arg == "-n" and i + 1 < len(args):
                try:
                    limit = min(int(args[i + 1]), 1000)
                except ValueError:
                    pass

        # Parse --since
        since_val = None
        for i, arg in enumerate(args):
            if arg == "--since" and i + 1 < len(args):
                since_val = args[i + 1]

        # Extract query (non-flag arguments)
        query_parts = []
        skip_next = False
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg in ("-n", "--since", "--tag"):
                skip_next = True
                continue
            if not arg.startswith("-"):
                query_parts.append(arg)

        funder_query = " ".join(query_parts).strip()
        if not funder_query:
            return TerminalResult(
                stderr='funded-by: usage: funded-by "FUNDER_NAME"\n'
                '         funded-by -n 50 "NIH"\n'
                '         funded-by --since 6m "Wellcome Trust"',
                exit_code=1,
                cwd=self.cwd,
            )

        try:
            # Use the document_store helper to resolve funder doc IDs
            ds = self.fs.document_store
            es = None
            try:
                # Access the ES client the same way search_documents does
                from modules.papers.filesystem import (
                    _get_es_client,
                )

                es = _get_es_client()
            except Exception:
                pass

            if not es:
                return TerminalResult(
                    stderr="funded-by: OpenSearch not available",
                    exit_code=1,
                    cwd=self.cwd,
                )

            import asyncio

            # When --since is provided, reverse the query order:
            # 1) Get date-filtered doc IDs from documents index
            # 2) Intersect with funder doc IDs from content index
            # This avoids the 10K aggregation limit missing recent papers.
            if since_val:
                min_year = ds._since_to_pub_year(since_val)
                if not min_year:
                    return TerminalResult(
                        stdout=f"No papers found with funder matching: {funder_query}\n",
                        cwd=self.cwd,
                    )

                date_resp = await asyncio.to_thread(
                    es.search,
                    index="preprints",
                    body={
                        "query": {
                            "bool": {"filter": [{"range": {"pub_year": {"gte": min_year}}}]}
                        },
                        "size": 0,
                        "aggs": {
                            "doc_ids": {
                                "terms": {
                                    "field": "document_id",
                                    "size": 50_000,
                                }
                            }
                        },
                    },
                )
                date_doc_ids = {
                    b["key"] for b in date_resp["aggregations"]["doc_ids"]["buckets"]
                }

                if not date_doc_ids:
                    return TerminalResult(
                        stdout=f"No papers in the given time range.\n",
                        cwd=self.cwd,
                    )

                # Step 2: Resolve funder doc IDs, pre-filtered to dated set
                # Add a document_id filter to the funder content query
                funder_resp = await asyncio.to_thread(
                    es.search,
                    index="preprints",
                    body={
                        "query": {
                            "bool": {
                                "must": [
                                    {
                                        "query_string": {
                                            "query": funder_query,
                                            "default_field": "content",
                                            "default_operator": "AND",
                                        }
                                    }
                                ],
                                "filter": [
                                    {
                                        "bool": {
                                            "should": [
                                                {
                                                    "match": {
                                                        "section": {
                                                            "query": "Funding",
                                                            "operator": "or",
                                                        }
                                                    }
                                                },
                                                {
                                                    "match": {
                                                        "section": {
                                                            "query": "Acknowledgments",
                                                            "operator": "or",
                                                        }
                                                    }
                                                },
                                            ],
                                            "minimum_should_match": 1,
                                        }
                                    },
                                    {"terms": {"document_id": list(date_doc_ids)}},
                                ],
                            }
                        },
                        "size": 0,
                        "aggs": {
                            "doc_ids": {
                                "terms": {"field": "document_id", "size": 10_000}
                            },
                            "total_docs": {"cardinality": {"field": "document_id"}},
                        },
                    },
                )
                funder_buckets = funder_resp["aggregations"]["doc_ids"]["buckets"]
                total_count = funder_resp["aggregations"]["total_docs"]["value"]
                doc_ids = [b["key"] for b in funder_buckets]
            else:
                doc_ids, total_count = ds._resolve_funder_doc_ids(
                    es, funder_query, limit=10_000
                )

            if not doc_ids:
                return TerminalResult(
                    stdout=f"No papers found with funder matching: {funder_query}\n",
                    cwd=self.cwd,
                )

            # Fetch document metadata for the resolved doc IDs
            filter_clauses = [{"terms": {"document_id": doc_ids}}]

            response = await asyncio.to_thread(
                es.search,
                index="preprints",
                body={
                    "query": {"bool": {"filter": filter_clauses}},
                    "size": limit,
                    "sort": [ds._MONTH_YEAR_SORT_SCRIPT, "_score"],
                    "_source": [
                        "document_id",
                        "title",
                        "doi",
                        "authors",
                        "month_year",
                        "source",
                        "abstract",
                    ],
                },
            )

            papers = ds._parse_es_hits(response)

            if not papers:
                return TerminalResult(
                    stdout=f"No papers found with funder matching: {funder_query}\n",
                    cwd=self.cwd,
                )

            # Save result and optionally accumulate by tag
            funder_label = f"funded-by: {funder_query}"
            results_id = ""
            total_accumulated = len(papers)
            truncated = False
            if hasattr(self.fs, "results_registry"):
                individual_id = self.fs.results_registry.save(
                    data={
                        "papers": papers,
                        "query": funder_label,
                        "funder": funder_query,
                        "total_count": total_count,
                    },
                    session_id=session_id,
                    prefix="s",
                )
                results_id = individual_id

                if tag_val:
                    accumulator_id = self._search_accumulator_ids.get(tag_val)
                    if accumulator_id:
                        pool_id = f"{accumulator_id}__pool"
                        pool_data = self.fs.results_registry.load(pool_id, session_id)
                        if not pool_data:
                            pool_data = self.fs.results_registry.load(
                                accumulator_id, session_id
                            )
                        if pool_data:
                            existing_papers = pool_data.get("papers", [])
                            existing_doc_ids = {
                                p.get("document_id") for p in existing_papers
                            }
                            new_papers = [
                                p
                                for p in papers
                                if p.get("document_id") not in existing_doc_ids
                            ]
                            merged_papers = existing_papers + new_papers
                            queries = pool_data.get(
                                "queries", [pool_data.get("query", "")]
                            )
                            queries.append(funder_label)
                        else:
                            merged_papers = papers
                            queries = [funder_label]
                    else:
                        merged_papers = papers
                        queries = [funder_label]
                        accumulator_id = individual_id
                        self._search_accumulator_ids[tag_val] = accumulator_id

                    if len(merged_papers) > self._ACCUMULATOR_CAP:
                        merged_papers = merged_papers[: self._ACCUMULATOR_CAP]
                        truncated = True

                    pool_save_data = {
                        "papers": merged_papers,
                        "query": "; ".join(queries),
                        "queries": queries,
                    }
                    self.fs.results_registry.save(
                        data=pool_save_data,
                        session_id=session_id,
                        results_id=accumulator_id,
                    )
                    self.fs.results_registry.save(
                        data=pool_save_data,
                        session_id=session_id,
                        results_id=f"{accumulator_id}__pool",
                    )
                    if hasattr(self.fs, "_save_search_artifact"):
                        asyncio.create_task(self.fs._save_search_artifact(
                            accumulator_id,
                            merged_papers,
                            "; ".join(queries),
                            session_id,
                        ))
                    results_id = accumulator_id
                    total_accumulated = len(merged_papers)
                else:
                    if hasattr(self.fs, "_save_search_artifact"):
                        asyncio.create_task(self.fs._save_search_artifact(
                            individual_id, papers, funder_label, session_id
                        ))

            output_lines = [
                f'Found {total_count} papers funded by "{funder_query}" (showing {len(papers)}):'
            ]
            if results_id:
                output_lines[0] += f"  [results_id: {results_id}]"
            if tag_val:
                output_lines[0] += f"  [tag: {tag_val}]"
                if total_accumulated > len(papers):
                    output_lines[0] += f"  [{total_accumulated} accumulated]"
                if truncated:
                    output_lines.append(
                        f"  ⚠ Accumulated results capped at {self._ACCUMULATOR_CAP} papers."
                    )
            output_lines.append("")

            for i, paper in enumerate(papers, 1):
                doc_id = paper.get("document_id", "?")
                title = paper.get("title", "Untitled")
                authors = paper.get("authors", "")
                pub_date = paper.get("pub_date", "") or paper.get("pub_year", "")
                doi = paper.get("doi", "")
                source = paper.get("source", "")
                output_lines.append(f"  {i}. {title}")
                id_line = f"     {doc_id}"
                if source:
                    _src = source.lower()
                    source_label = _SOURCE_DISPLAY.get(_src, source)
                    id_line += f" · {source_label}"
                if pub_date:
                    id_line += f" · {pub_date}"
                output_lines.append(id_line)
                if doi:
                    output_lines.append(f"     doi:{doi}")
                if authors:
                    output_lines.append(f"     {authors}")
                abstract_snippet = paper.get("abstract_snippet", "")
                if abstract_snippet:
                    output_lines.append(f'     "{abstract_snippet}"')
                output_lines.append("")

            output_lines.append(f'Cite papers with: {{{{"document_id": "DOC_ID"}}}}')
            if results_id:
                output_lines.append(
                    f'Analyze with: map --from {results_id} "your question"'
                )
                output_lines.append(
                    f'Cite results as table: {{{{"artifact": {{{{"artifact_id": "{results_id}", "type": "table", '
                    f'"source_count": {total_accumulated}, "description": "YOUR_ONE_SENTENCE_SUMMARY"}}}}}}}}'
                )

            return TerminalResult(stdout="\n".join(output_lines) + "\n", cwd=self.cwd)

        except Exception as e:
            return TerminalResult(stderr=f"funded-by: {e}", exit_code=1, cwd=self.cwd)

    async def _cmd_cited_by(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Find papers in the database that cite a given DOI.

        Usage:
            cited-by 10.1101/2024.01.01.000000

        Queries Semantic Scholar for all papers citing the given DOI,
        then filters to those available in the biomedrxiv database.
        Returns one document UUID per line.
        """
        if not self.fs:
            return TerminalResult(
                stderr="cited-by: filesystem not available",
                exit_code=1,
                cwd=self.cwd,
            )

        doi = " ".join(a for a in args if not a.startswith("-")).strip()
        if not doi:
            return TerminalResult(
                stderr="cited-by: usage: cited-by DOI\n"
                "         cited-by 10.1101/2024.01.01.000000",
                exit_code=1,
                cwd=self.cwd,
            )

        # Normalize DOI — strip URL prefixes
        for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/", "DOI:", "doi:"):
            if doi.startswith(prefix):
                doi = doi[len(prefix):]
                break

        try:
            import asyncio

            import aiohttp

            from modules.search.searchers import SemanticScholarSearcher

            client = SemanticScholarSearcher()
            headers = {}
            if client.api_key:
                headers["x-api-key"] = client.api_key

            url = f"{client.base_url}/paper/DOI:{doi}/citations"
            page_size = 1000
            all_items: list[dict] = []
            offset = 0

            async with aiohttp.ClientSession() as http_session:
                while True:
                    params = {
                        "fields": "citingPaper.externalIds",
                        "limit": page_size,
                        "offset": offset,
                    }
                    async with http_session.get(url, params=params, headers=headers) as resp:
                        if resp.status == 404:
                            return TerminalResult(
                                stdout=f"cited-by: paper not found in Semantic Scholar: {doi}\n",
                                cwd=self.cwd,
                            )
                        if resp.status == 429:
                            await asyncio.sleep(1.5)
                            continue
                        if resp.status != 200:
                            error_text = await resp.text()
                            return TerminalResult(
                                stderr=f"cited-by: Semantic Scholar error ({resp.status}): {error_text}",
                                exit_code=1,
                                cwd=self.cwd,
                            )
                        data = await resp.json()
                    items = data.get("data", [])
                    if not items:
                        break
                    all_items.extend(items)
                    offset += len(items)
                    if len(items) < page_size:
                        break

            # Collect DOIs from citing papers
            citing_dois = []
            for item in all_items:
                ext_ids = (item.get("citingPaper") or {}).get("externalIds") or {}
                doi_val = ext_ids.get("DOI")
                if doi_val:
                    citing_dois.append(doi_val)

            if not citing_dois:
                return TerminalResult(
                    stdout=(
                        f"cited-by: {len(all_items)} citing papers on Semantic Scholar, "
                        f"none with resolvable DOIs for database lookup\n"
                    ),
                    cwd=self.cwd,
                )

            # Look up full metadata for citing papers that are in the database
            from modules.papers.filesystem import _get_db_connection

            conn = _get_db_connection()
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT DISTINCT ON (doi) document_id::text, title, doi, authors, month_year, source
                       FROM documents
                       WHERE doi = ANY(%s)
                       ORDER BY doi, created_at DESC""",
                    (citing_dois,),
                )
                rows = cur.fetchall()

            if not rows:
                return TerminalResult(
                    stdout=(
                        f"cited-by {doi}: {len(all_items)} citations on Semantic Scholar, "
                        f"0 available in database\n"
                    ),
                    cwd=self.cwd,
                )

            papers = [
                {
                    "document_id": row[0],
                    "title": row[1],
                    "doi": row[2],
                    "authors": row[3],
                    "month_year": row[4],
                    "source": row[5],
                    "path": f"/papers/{row[0]}/",
                }
                for row in rows
            ]

            # Save to results registry so map/reduce can consume it
            results_id = ""
            if hasattr(self.fs, "results_registry"):
                label = f"cited-by: {doi}"
                results_id = self.fs.results_registry.save(
                    data={"papers": papers, "query": label, "doi": doi},
                    session_id=session_id,
                    prefix="s",
                )
                if hasattr(self.fs, "_save_search_artifact"):
                    asyncio.create_task(self.fs._save_search_artifact(results_id, papers, label, session_id))

            output_lines = [
                f"cited-by {doi}: {len(papers)} papers in database "
                f"(out of {len(all_items)} total citations on Semantic Scholar)",
            ]
            if results_id:
                output_lines[0] += f"  [results_id: {results_id}]"
            output_lines.append("")

            for paper in papers:
                output_lines.append(paper["document_id"])

            if results_id:
                output_lines += [
                    "",
                    f'Analyze with: map --from {results_id} "your question"',
                    f'Cite results as table: {{{{"artifact": {{"artifact_id": "{results_id}", "type": "table", '
                    f'"source_count": {len(papers)}, "description": "YOUR_ONE_SENTENCE_SUMMARY"}}}}}}',
                ]

            return TerminalResult(stdout="\n".join(output_lines) + "\n", cwd=self.cwd)

        except Exception as e:
            logger.error(f"cited-by error: {e}")
            return TerminalResult(stderr=f"cited-by: {e}", exit_code=1, cwd=self.cwd)

    async def _cmd_lookup_citation(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Find in-text mentions of a reference in a paper.

        Usage (from within a paper directory):
            lookup-citation "Smith 2020"
            lookup-citation "10.1101/2021.01.01.000000"
            lookup-citation "[15]"
            lookup-citation 15

        Usage from anywhere (explicit UUID):
            lookup-citation <uuid> "Smith 2020"

        Searches the References section for the query to identify the reference
        entry, derives a compact citation key (e.g. "[15]" or "Smith"), then finds
        every passage in the body text where that key appears.  Returns each
        passage with ±2 lines of context and block IDs for citation.
        """
        if not self.fs:
            return TerminalResult(
                stderr="lookup-citation: filesystem not available",
                exit_code=1,
                cwd=self.cwd,
            )

        _DOC_ID_RE = re.compile(
            r"^(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|(?:bio|med)_[0-9a-f]{12})$",
            re.IGNORECASE,
        )

        if not args:
            return TerminalResult(
                stderr=(
                    "lookup-citation: usage: lookup-citation [ID] QUERY\n"
                    '         lookup-citation "Smith 2020"\n'
                    '         lookup-citation "10.1101/2021.01.01.000000"\n'
                    '         lookup-citation "[15]"\n'
                    '         lookup-citation <paper_id> "Smith 2020"'
                ),
                exit_code=1,
                cwd=self.cwd,
            )

        # Determine doc_id from explicit ID arg or current CWD
        if _DOC_ID_RE.match(args[0]):
            doc_id = args[0]
            query = " ".join(args[1:]).strip()
        else:
            m = re.search(
                r"/papers/((?:bio|med)_[0-9a-f]{12}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
                self.cwd,
                re.IGNORECASE,
            )
            if not m:
                return TerminalResult(
                    stderr=(
                        "lookup-citation: not in a paper directory. "
                        "Either cd into a paper or supply UUID explicitly:\n"
                        "  lookup-citation <uuid> QUERY"
                    ),
                    exit_code=1,
                    cwd=self.cwd,
                )
            doc_id = m.group(1)
            query = " ".join(args).strip()

        if not query:
            return TerminalResult(
                stderr="lookup-citation: QUERY required",
                exit_code=1,
                cwd=self.cwd,
            )

        # Strip surrounding quotes
        if len(query) >= 2 and query[0] == query[-1] and query[0] in ('"', "'"):
            query = query[1:-1]

        try:
            from modules.papers.filesystem import _get_db_connection

            conn = _get_db_connection()

            # Step 1: Locate the reference entry in the References section
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT id, line_number, content
                       FROM content_blocks
                       WHERE document_id::text = %s
                         AND (section ILIKE '%reference%'
                              OR section ILIKE '%bibliograph%'
                              OR section ILIKE '%literature%')
                         AND content ILIKE %s
                       ORDER BY line_number
                       LIMIT 5""",
                    (doc_id, f"%{query}%"),
                )
                ref_rows = cur.fetchall()

            if not ref_rows:
                return TerminalResult(
                    stdout=(
                        f'lookup-citation: no reference found matching "{query}" '
                        f"in the References section of {doc_id}.\n"
                        "Try a different term (author name, title fragment, DOI).\n"
                    ),
                    cwd=self.cwd,
                )

            ref_id, _ref_line, ref_content = ref_rows[0]

            # Step 2: Derive a compact citation key for searching body text
            search_key = _extract_citation_key(ref_content, query)

            # Step 3: Find every in-text mention (outside References section)
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT id, line_number, content, section
                       FROM content_blocks
                       WHERE document_id::text = %s
                         AND section NOT ILIKE '%reference%'
                         AND section NOT ILIKE '%bibliograph%'
                         AND content ILIKE %s
                       ORDER BY line_number""",
                    (doc_id, f"%{search_key}%"),
                )
                body_rows = cur.fetchall()

            # Step 4: Fetch context blocks (±2 lines per match) in one query
            all_ctx: dict[int, tuple[int, str]] = {}
            if body_rows:
                context_lines = set()
                for _, match_ln, _, _ in body_rows:
                    for offset in range(-2, 3):
                        context_lines.add(match_ln + offset)
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT id, line_number, content
                           FROM content_blocks
                           WHERE document_id::text = %s
                             AND line_number = ANY(%s)
                             AND section NOT ILIKE '%reference%'
                             AND section NOT ILIKE '%bibliograph%'
                           ORDER BY line_number""",
                        (doc_id, list(context_lines)),
                    )
                    all_ctx = {row[1]: (row[0], row[2]) for row in cur.fetchall()}

            # Build output
            ref_display = ref_content[:150] + ("..." if len(ref_content) > 150 else "")
            output_lines = [
                f"Reference [{ref_id}]: {ref_display}",
                f'Citation key: "{search_key}"',
                "",
            ]

            if not body_rows:
                output_lines += [
                    "No in-text mentions found.",
                    "Note: superscript-only citation markers may be stripped during XML extraction.",
                ]
                return TerminalResult(stdout="\n".join(output_lines) + "\n", cwd=self.cwd)

            output_lines.append(f"{len(body_rows)} in-text mention(s):")
            output_lines.append("")

            for i, (match_id, match_ln, _match_content, match_section) in enumerate(
                body_rows, 1
            ):
                output_lines.append(f"[{i}] Section: {match_section or '(unknown)'}")
                for ctx_ln in range(match_ln - 2, match_ln + 3):
                    if ctx_ln in all_ctx:
                        ctx_id, ctx_text = all_ctx[ctx_ln]
                        marker = ">>>" if ctx_ln == match_ln else "   "
                        truncated = ctx_text[:120] + ("..." if len(ctx_text) > 120 else "")
                        output_lines.append(f"  {marker} [{ctx_id}] {truncated}")
                output_lines.append("")

            output_lines.append('Cite with: document_id + line number (e.g. "arx_2402.02008, L42")')
            return TerminalResult(stdout="\n".join(output_lines) + "\n", cwd=self.cwd)

        except Exception as e:
            logger.error(f"lookup-citation error: {e}")
            return TerminalResult(
                stderr=f"lookup-citation: {e}", exit_code=1, cwd=self.cwd
            )

    @staticmethod
    def _format_search_backends(meta: dict) -> str:
        """One-line compact summary of which search backends served the query.

        Reads the ``search_meta`` dict surfaced by ``PapersStore.search_documents``
        and renders something like::

            [os 87ms/25 · qdrant 203ms(embed=180+search=23)/25 · hydrate=110ms · ranking=hybrid]

        Breaks the qdrant number into ``embed`` (Gemini API) vs ``search``
        (pure Qdrant HTTP) so you can see whether the bottleneck is the
        embedding call or the vector search itself. ``hydrate`` is the
        postgres-backed TL;DR/metadata fill. Shows ``skip`` / ``ERR(...)``
        when a backend was disabled (via ``PAPERS_DISABLE_OPENSEARCH`` /
        ``PAPERS_DISABLE_QDRANT``) or errored.
        """
        parts: list[str] = []
        os_sub = meta.get("opensearch") or {}
        if os_sub.get("error"):
            parts.append(f"os ERR({str(os_sub['error'])[:40]})")
        elif os_sub.get("enabled") is False or os_sub.get("ms") is None:
            parts.append("os skip")
        else:
            parts.append(f"os {os_sub['ms']}ms/{os_sub.get('hits', 0)}")

        q_sub = meta.get("qdrant") or {}
        if q_sub.get("error"):
            parts.append(f"qdrant ERR({str(q_sub['error'])[:40]})")
        elif q_sub.get("enabled") is False or q_sub.get("ms") is None:
            parts.append("qdrant skip")
        else:
            embed = q_sub.get("embed_ms")
            sms = q_sub.get("search_ms")
            breakdown = ""
            if embed is not None and sms is not None:
                breakdown = f"(embed={embed}+search={sms})"
            parts.append(
                f"qdrant {q_sub['ms']}ms{breakdown}/{q_sub.get('hits', 0)}"
            )

        hyd = meta.get("hydrate_ms")
        if hyd is not None:
            parts.append(f"hydrate={hyd}ms")
        total = meta.get("total_ms")
        if total is not None:
            parts.append(f"server={total}ms")
        ranking = meta.get("ranking") or "?"
        parts.append(f"ranking={ranking}")
        return f"  [{' · '.join(parts)}]"

    @staticmethod
    def _format_sql_table(columns: list[str], rows: list) -> str:
        """Format columns + rows as an aligned text table."""
        str_rows = [[str(v) if v is not None else "NULL" for v in row] for row in rows]
        max_col_width = 60
        col_widths = []
        for i, col in enumerate(columns):
            width = len(col)
            for row in str_rows:
                if i < len(row):
                    width = max(width, len(row[i]))
            col_widths.append(min(width, max_col_width))
        header = " | ".join(col.ljust(col_widths[i]) for i, col in enumerate(columns))
        separator = "-+-".join("-" * w for w in col_widths)
        lines = [header, separator]
        for row in str_rows:
            cells = []
            for i, val in enumerate(row):
                w = col_widths[i] if i < len(col_widths) else max_col_width
                if len(val) > w:
                    val = val[:w - 3] + "..."
                cells.append(val.ljust(w))
            lines.append(" | ".join(cells))
        return "\n".join(lines)

    async def _cmd_sql(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Execute a read-only SQL query against the unified papers database.

        Usage:
            sql "SELECT COUNT(*) FROM documents"
            sql "SELECT source, COUNT(*) FROM documents GROUP BY source"
            sql "SELECT title, doi FROM documents WHERE authors ILIKE '%Doudna%' LIMIT 5"
            sql "SELECT journal_title, COUNT(*) FROM documents WHERE source = 'pmc' ORDER BY 1 LIMIT 10"

        Options:
            --source pmc|biorxiv   Target a single source (default: queries all)
            --json                 Return raw JSON instead of table

        Unified `documents` columns:
            id               — paper identifier (UUID or PMC ID)
            title            — paper title
            doi              — Digital Object Identifier
            authors          — comma-separated author list
            source           — 'biorxiv', 'medrxiv', or 'pmc'
            abstract_text    — paper abstract
            pub_date         — publication date
            journal_title    — journal name (PMC only)
            article_type     — e.g. 'research-article' (PMC only)
            pmid             — PubMed ID (PMC only)
            keywords         — JSONB array (PMC only)
            categories       — JSONB array (PMC only)
            pub_year         — publication year (PMC only)
            created_at       — when indexed

        Only the `documents` table is available. SELECT only. 15s timeout, 200-row limit.
        """
        if not self.fs:
            return TerminalResult(
                stderr="sql: filesystem not available", exit_code=1, cwd=self.cwd
            )

        if not hasattr(self.fs, "_raw_sql"):
            return TerminalResult(
                stderr="sql: not supported by this filesystem module",
                exit_code=1,
                cwd=self.cwd,
            )

        # Parse flags
        source = "all"
        is_json = False
        clean_args = []
        skip_next = False
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg in ("--source", "-s") and i + 1 < len(args):
                source = args[i + 1].lower()
                skip_next = True
                continue
            if arg == "--json":
                is_json = True
                continue
            clean_args.append(arg)

        # Join all args into one query string
        query = " ".join(clean_args).strip()

        # Also accept piped stdin as query
        if not query and stdin:
            query = stdin.strip()

        if not query:
            return TerminalResult(
                stderr='sql: usage: sql "SELECT ..."\n'
                "     Queries all sources by default. Use --source pmc|biorxiv to filter.\n"
                "     Only SELECT on the `documents` table. 15s timeout, 200-row limit.\n"
                "\n"
                "     Key columns: id, title, doi, authors, source, abstract_text, pub_date\n"
                "     PMC-only:    journal_title, article_type, pmid, pub_year, keywords, categories\n"
                "\n"
                "     Examples:\n"
                '       sql "SELECT COUNT(*) FROM documents"\n'
                '       sql "SELECT title, doi, source FROM documents WHERE authors ILIKE \'%Doudna%\' LIMIT 5"\n'
                '       sql "SELECT journal_title, COUNT(*) FROM documents WHERE source = \'pmc\' ORDER BY 1 LIMIT 10"',
                exit_code=1,
                cwd=self.cwd,
            )

        try:
            result = await self.fs._raw_sql(
                query=query,
                session_id=session_id,
                source=source,
            )

            if "error" in result:
                return TerminalResult(
                    stderr=f"sql: {result['error']}",
                    exit_code=1,
                    cwd=self.cwd,
                )

            # Handle multi-section results (different columns per DB)
            if "sections" in result:
                output_lines = []
                time_ms = result.get("time_ms", 0)
                for sec in result["sections"]:
                    label = sec.get("label", "?")
                    cols = sec.get("columns", [])
                    sec_rows = sec.get("rows", [])
                    output_lines.append(f"── {label} ──")
                    output_lines.append(self._format_sql_table(cols, sec_rows))
                    output_lines.append("")
                output_lines.append(f"({time_ms}ms)")
                return TerminalResult(stdout="\n".join(output_lines) + "\n", cwd=self.cwd)

            columns = result.get("columns", [])
            rows = result.get("rows", [])
            count = result.get("count", 0)
            time_ms = result.get("time_ms", 0)
            truncated = result.get("truncated", False)
            sources_detail = result.get("sources_detail", "")

            if not rows:
                detail = f" [{sources_detail}]" if sources_detail else ""
                return TerminalResult(
                    stdout=f"(0 rows, {time_ms}ms){detail}\n",
                    cwd=self.cwd,
                )

            table_text = self._format_sql_table(columns, rows)
            output_lines = table_text.split("\n")

            footer = f"({count} row{'s' if count != 1 else ''}, {time_ms}ms)"
            if sources_detail:
                footer += f" [{sources_detail}]"
            if truncated:
                footer += f" [truncated to {count} rows]"
            output_lines.append(footer)

            # Auto-save as results artifact when an id column is present.
            # NB: PapersModule.(_BIO|_PMC)_COLUMN_MAP rewrites the raw DB
            # columns (``document_id`` / ``pmc_id``) to the unified ``id``
            # before the sql handler sees them, so ``id`` is the name we
            # get in practice. We still accept the raw names for any caller
            # that bypasses the unified schema (e.g. FDA filesystem).
            id_col_name = (
                "document_id" if "document_id" in columns
                else "pmc_id" if "pmc_id" in columns
                else "id" if "id" in columns
                else None
            )
            if (
                id_col_name
                and self.fs
                and hasattr(self.fs, "results_registry")
            ):
                doc_idx = columns.index(id_col_name)
                col_map = {c: i for i, c in enumerate(columns)}
                papers = []
                for row in rows:
                    doc_id = str(row[doc_idx]) if row[doc_idx] is not None else ""
                    if not doc_id:
                        continue
                    p = {
                        "document_id": doc_id,
                        "title": str(row[col_map["title"]]) if "title" in col_map and row[col_map["title"]] else "",
                        "doi": str(row[col_map["doi"]]) if "doi" in col_map and row[col_map["doi"]] else "",
                        "authors": str(row[col_map["authors"]]) if "authors" in col_map and row[col_map["authors"]] else "",
                        "path": f"/papers/{doc_id}/",
                    }
                    if "month_year" in col_map and row[col_map["month_year"]]:
                        p["month_year"] = str(row[col_map["month_year"]])
                    if "pub_year" in col_map and row[col_map["pub_year"]]:
                        p["pub_year"] = str(row[col_map["pub_year"]])
                    if "source" in col_map and row[col_map["source"]]:
                        p["source"] = str(row[col_map["source"]])
                    if "journal_title" in col_map and row[col_map["journal_title"]]:
                        p["journal"] = str(row[col_map["journal_title"]])
                    papers.append(p)
                if papers:
                    results_id = self.fs.results_registry.save(
                        data={"papers": papers, "query": query[:200]},
                        session_id=session_id,
                        prefix="s",
                    )
                    if hasattr(self.fs, "_save_search_artifact"):
                        asyncio.create_task(self.fs._save_search_artifact(
                            results_id, papers, query[:200], session_id
                        ))
                    # Put results_id FIRST so it survives output truncation
                    output_lines.insert(0, f"→ Saved as {results_id} ({len(papers)} papers). Use: map --from {results_id} \"your question\"\n")

            return TerminalResult(
                stdout="\n".join(output_lines) + "\n",
                cwd=self.cwd,
            )

        except Exception as e:
            return TerminalResult(stderr=f"sql: {e}", exit_code=1, cwd=self.cwd)

    async def _cmd_export(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Export SQL query results to a CSV file and table artifact (up to 1K rows).

        Usage:
            export "SELECT title, doi, authors, month_year FROM documents WHERE source = 'biorxiv' ORDER BY created_at DESC"
            export --desc "NIH-funded biorxiv papers 2024" "SELECT d.title, d.doi, d.authors FROM documents d JOIN content_blocks cb ON d.document_id = cb.document_id WHERE cb.content ILIKE '%NIH%' AND cb.section IN ('Funding', 'Acknowledgments') GROUP BY d.document_id, d.title, d.doi, d.authors"

        Options:
            --desc "description"    Human-readable description for the export

        Output:
            - CSV file saved in session artifacts
            - Table artifact viewable in the table tab
            - Returns artifact_id for citation in your response

        To cite in your response:
            {{"artifact": {{"artifact_id": "a_xxx", "type": "table", "source_count": N, "description": "one-sentence summary of what the table contains"}}}}
        """
        if not self.fs:
            return TerminalResult(
                stderr="export: filesystem not available", exit_code=1, cwd=self.cwd
            )

        if not hasattr(self.fs, "_export_sql"):
            return TerminalResult(
                stderr="export: not supported by this filesystem module",
                exit_code=1,
                cwd=self.cwd,
            )

        # Parse --desc flag
        description = ""
        clean_args = []
        i = 0
        while i < len(args):
            if args[i] == "--desc" and i + 1 < len(args):
                description = args[i + 1]
                i += 2
            else:
                clean_args.append(args[i])
                i += 1

        query = " ".join(clean_args).strip()

        if not query and stdin:
            query = stdin.strip()

        if not query:
            return TerminalResult(
                stderr='export: usage: export [--desc "description"] "SELECT ..."\n'
                "     Exports SQL results to CSV + table artifact (up to 1K rows).\n"
                "\n"
                "     Tables:\n"
                "       documents       — document_id, title, doi, source, authors, month_year, abstract_text, created_at\n"
                "       content_blocks  — id, document_id, line_number, content, section, block_type\n"
                "       figures         — document_id, graphic, source_path\n"
                "\n"
                "     Examples:\n"
                "       export \"SELECT title, doi, authors FROM documents WHERE source = 'biorxiv' ORDER BY created_at DESC\"\n"
                "       export --desc \"CRISPR papers 2024\" \"SELECT DISTINCT d.title, d.doi FROM documents d JOIN content_blocks cb ON d.document_id = cb.document_id WHERE cb.content ILIKE '%CRISPR%' AND d.month_year >= '2024-01'\"",
                exit_code=1,
                cwd=self.cwd,
            )

        # Catch obvious non-SQL input early so we fail fast with a clear error
        # rather than sending garbage to Postgres and potentially hanging.
        stripped = query.lstrip()
        if not stripped.upper().startswith("SELECT") and not stripped.upper().startswith("WITH"):
            return TerminalResult(
                stderr=f"export: argument must be a SQL query starting with SELECT or WITH.\n"
                f"  Got: {query[:120]!r}\n"
                "\n"
                "  To fetch fields for papers from a search result, write a query like:\n"
                "    export \"SELECT title, doi, month_year, abstract_text FROM documents\n"
                "             WHERE document_id IN (...) ORDER BY month_year DESC LIMIT 30\"",
                exit_code=1,
                cwd=self.cwd,
            )

        try:
            result = await self.fs._export_sql(
                query=query,
                description=description,
                session_id=session_id,
            )

            if "error" in result:
                return TerminalResult(
                    stderr=f"export: {result['error']}",
                    exit_code=1,
                    cwd=self.cwd,
                )

            count = result.get("count", 0)
            if count == 0:
                return TerminalResult(
                    stdout="Query returned 0 rows. Nothing to export.\n",
                    cwd=self.cwd,
                )

            artifact_id = result.get("artifact_id", "")
            columns = result.get("columns", [])
            csv_path = result.get("csv_path", "")
            time_ms = result.get("time_ms", 0)
            truncated = result.get("truncated", False)
            desc = result.get("description", "")

            output_lines = [
                f"Exported {count} rows to table artifact and CSV.",
            ]
            if truncated:
                output_lines[0] = (
                    f"Exported {count} rows (capped at 1,000) to table artifact and CSV."
                )
            output_lines.append(f"  artifact_id: {artifact_id}")
            output_lines.append(f"  csv_file:    {csv_path}")
            output_lines.append(f"  columns:     {', '.join(columns)}")
            output_lines.append(f"  time:        {time_ms}ms")
            if desc:
                output_lines.append(f"  description: {desc}")
            output_lines.append("")
            output_lines.append("To cite this table in your response:")
            output_lines.append(
                f'  {{{{"artifact": {{{{"artifact_id": "{artifact_id}", "type": "table", '
                f'"source_count": {count}, "description": "YOUR_ONE_SENTENCE_SUMMARY"}}}}}}}}'
            )

            return TerminalResult(
                stdout="\n".join(output_lines) + "\n",
                cwd=self.cwd,
            )

        except Exception as e:
            return TerminalResult(stderr=f"export: {e}", exit_code=1, cwd=self.cwd)

    async def _cmd_searches(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Run multiple searches in parallel and save results to durable files.

        Usage:
            searches "query1" "query2" "query3" ...
            searches --tag TOPIC "query1" "query2" ...   # accumulate + dedup
            searches --quiet --tag TOPIC "q1" "q2" ...   # minimal output (pipeline mode)
            searches -m all "CRISPR delivery" "gene therapy" "cancer treatment"
            searches --recent "AI diagnostics" "machine learning"
            searches "q1" "q2"                         # multi-query combined search

        Returns a summary table with query, count, and filepath for each search.
        With --quiet, returns only the accumulated count and result ID.
        Use cat /session_files/searches/FILENAME to view full results.

        Example:
            searches --quiet --tag crispr "CRISPR delivery" "gene editing cancer" "viral vectors"

        Output (--quiet with --tag):
            All results accumulated into: s_abc123  [tag: crispr]  (58 unique papers)
            Cite search results as table: ...
        """
        import asyncio
        import hashlib

        if not self.fs:
            return TerminalResult(
                stderr="searches: filesystem not available", exit_code=1, cwd=self.cwd
            )

        # Parse flags (apply to all searches)
        is_recent = "--recent" in args
        is_exact = "-e" in args or "--exact" in args
        is_quiet = "--quiet" in args

        # Parse --tag
        tag_val = None
        for i, arg in enumerate(args):
            if arg == "--tag" and i + 1 < len(args):
                tag_val = args[i + 1].lower().strip()

        # Parse search mode
        search_mode = "phrase" if is_exact else "any"
        for i, arg in enumerate(args):
            if arg == "-m" and i + 1 < len(args):
                mode_val = args[i + 1]
                if mode_val in ("any", "all", "50%", "75%", "phrase"):
                    search_mode = mode_val

        # Parse limit per search (default: 50, max: 1000)
        limit = 50
        for i, arg in enumerate(args):
            if arg == "-n" and i + 1 < len(args):
                try:
                    limit = min(int(args[i + 1]), 1000)
                except ValueError:
                    pass

        # Extract queries (non-flag arguments)
        queries = []
        skip_next = False
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg in ("--recent", "-e", "--exact", "--quiet"):
                continue
            if arg in ("-n", "-m", "--tag"):
                skip_next = True
                continue
            if not arg.startswith("-"):
                queries.append(arg)

        if not queries:
            return TerminalResult(
                stderr='searches: usage: searches "query1" "query2" ...\n'
                "         Run multiple searches in parallel, save results to /session_files/searches/\n"
                "         Options: -m MODE, -n LIMIT, --recent, -e\n"
                '         Example: searches "CRISPR" "gene therapy" "viral delivery"',
                exit_code=1,
                cwd=self.cwd,
            )

        # Run searches
        import time as time_module

        start_time = time_module.perf_counter()
        search_results = []
        used_combined = False

        # Primary path: combined search (single ES bool.should query, ranked by relevance)
        has_combined = hasattr(self.fs, "document_store") and hasattr(
            self.fs.document_store, "combined_search_documents"
        )
        if has_combined:
            try:
                combined = await self.fs.document_store.combined_search_documents(
                    queries=queries,
                    search_mode=search_mode,
                    limit=limit,
                )
                duration_ms = round((time_module.perf_counter() - start_time) * 1000)
                if not combined.get("error"):
                    used_combined = True
                    search_results = [
                        {
                            "query": " | ".join(queries),
                            "result": {
                                "papers": combined["papers"],
                                "total": combined["total"],
                            },
                            "error": None,
                            "start_ms": 0,
                            "end_ms": duration_ms,
                            "duration_ms": duration_ms,
                        }
                    ]
            except Exception:
                pass

        # Fallback: msearch (separate query per term) or sequential
        if not used_combined:
            has_msearch = hasattr(self.fs, "document_store") and hasattr(
                self.fs.document_store, "msearch_documents"
            )
            if has_msearch:
                try:
                    msearch_results = await self.fs.document_store.msearch_documents(
                        queries=queries,
                        search_mode=search_mode,
                        limit=limit,
                    )
                    msearch_duration = round(
                        (time_module.perf_counter() - start_time) * 1000
                    )
                    for res in msearch_results:
                        search_results.append(
                            {
                                "query": res["query"],
                                "result": (
                                    {"papers": res["papers"], "total": res["total"]}
                                    if not res["error"]
                                    else None
                                ),
                                "error": res["error"],
                                "start_ms": 0,
                                "end_ms": msearch_duration,
                                "duration_ms": msearch_duration,
                            }
                        )
                except Exception:
                    search_results = []

            if not search_results:
                for query in queries:
                    search_start = time_module.perf_counter()
                    try:
                        result = await self.fs._find(
                            query=query,
                            search_mode=search_mode,
                            limit=limit,
                            session_id=session_id,
                        )
                        search_end = time_module.perf_counter()
                        search_results.append(
                            {
                                "query": query,
                                "result": result,
                                "error": None,
                                "start_ms": round((search_start - start_time) * 1000),
                                "end_ms": round((search_end - start_time) * 1000),
                                "duration_ms": round(
                                    (search_end - search_start) * 1000
                                ),
                            }
                        )
                    except Exception as e:
                        search_end = time_module.perf_counter()
                        search_results.append(
                            {
                                "query": query,
                                "result": None,
                                "error": str(e),
                                "start_ms": round((search_start - start_time) * 1000),
                                "end_ms": round((search_end - start_time) * 1000),
                                "duration_ms": round(
                                    (search_end - search_start) * 1000
                                ),
                            }
                        )

        total_time_ms = round((time_module.perf_counter() - start_time) * 1000)

        # Process results and save to files
        output_rows = []
        for sr in search_results:
            query = sr["query"]
            timing = (
                sr.get("start_ms", 0),
                sr.get("end_ms", 0),
                sr.get("duration_ms", 0),
            )

            if sr["error"]:
                output_rows.append(
                    {
                        "query": query,
                        "count": 0,
                        "file": "-",
                        "error": sr["error"],
                        "timing": timing,
                    }
                )
                continue

            result = sr["result"]
            if "error" in result:
                output_rows.append(
                    {
                        "query": query,
                        "count": 0,
                        "file": "-",
                        "error": result["error"],
                        "timing": timing,
                    }
                )
                continue

            papers = result.get("results", result.get("papers", []))
            total = min(result.get("total", len(papers)), len(papers))

            if total == 0:
                output_rows.append(
                    {
                        "query": query,
                        "count": 0,
                        "file": "-",
                        "error": None,
                        "timing": timing,
                    }
                )
                continue

            # Generate filename based on query hash
            query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
            filename = f"s_{query_hash}.txt"
            legacy_filepath = f"/tmp/searches/{filename}"
            filepath = f"/session_files/searches/{filename}"

            # Format results for file
            item_name = "papers" if self.root_path == "/papers/" else "documents"
            file_lines = [
                f"Search: {query}",
                f"Results: {total} {item_name} (showing {len(papers)})",
                "",
            ]
            for i, paper in enumerate(papers, 1):
                doc_id = paper.get("document_id", "?")
                title = paper.get("title", "Untitled")
                authors = paper.get("authors", "")
                pub_date = paper.get("pub_date", "") or paper.get("pub_year", "")
                doc_source_type = paper.get("source_type", "")

                # Use /youtube/ path for YouTube videos
                if doc_source_type == "youtube":
                    doc_path = f"/youtube/{doc_id}/"
                else:
                    doc_path = f"{self.root_path}{doc_id}/"

                file_lines.append(f"{i}. {title}")
                file_lines.append(f"   {doc_path}")
                if authors:
                    file_lines.append(f"   {authors}")
                if pub_date:
                    file_lines.append(f"   {pub_date}")
                file_lines.append("")

            # Add "[and N more]" if there are more results
            remaining = total - len(papers)
            if remaining > 0:
                file_lines.append(f"[and {remaining} more]")

            # Store in virtual filesystem cache
            file_content = "\n".join(file_lines)
            if not hasattr(self, "_search_results_cache"):
                self._search_results_cache = {}
            self._search_results_cache[legacy_filepath] = file_content
            try:
                await self._session_files_write(filepath, file_content, session_id)
            except Exception:
                # Keep the in-memory alias working even if durable write fails.
                pass

            output_rows.append(
                {
                    "query": query,
                    "count": total,
                    "file": filepath,
                    "error": None,
                    "timing": timing,
                }
            )

        # Accumulate results and save artifact (always, so the frontend can display tables)
        accumulated_id = ""
        accumulated_total = 0
        truncated = False
        if hasattr(self.fs, "results_registry"):
            all_search_papers = []
            for sr in search_results:
                if sr["error"]:
                    continue
                result = sr.get("result")
                if not result or "error" in result:
                    continue
                papers = result.get("results", result.get("papers", []))
                all_search_papers.extend(papers)

            # Dedup and cap at limit BEFORE accumulator logic.
            # msearch returns up to `limit` per query term; combined search
            # returns up to `limit` total.  Either way, the user asked for
            # at most `limit` papers from this single searches call.
            seen_pre = {}
            capped = []
            for p in all_search_papers:
                did = p.get("document_id")
                if did and did not in seen_pre:
                    seen_pre[did] = True
                    capped.append(p)
                    if len(capped) >= limit:
                        break
            all_search_papers = capped

            if all_search_papers:
                if tag_val:
                    accumulator_id = self._search_accumulator_ids.get(tag_val)
                else:
                    accumulator_id = None

                if accumulator_id:
                    pool_id = f"{accumulator_id}__pool"
                    pool_data = self.fs.results_registry.load(pool_id, session_id)
                    if not pool_data:
                        pool_data = self.fs.results_registry.load(
                            accumulator_id, session_id
                        )
                    if pool_data:
                        existing_papers = pool_data.get("papers", [])
                        existing_doc_ids = {
                            p.get("document_id") for p in existing_papers
                        }
                        new_papers = [
                            p
                            for p in all_search_papers
                            if p.get("document_id") not in existing_doc_ids
                        ]
                        merged_papers = existing_papers + new_papers
                        acc_queries = pool_data.get(
                            "queries", [pool_data.get("query", "")]
                        )
                        acc_queries.extend(queries)
                    else:
                        seen, deduped = {}, []
                        for p in all_search_papers:
                            did = p.get("document_id")
                            if did not in seen:
                                seen[did] = True
                                deduped.append(p)
                        merged_papers = deduped
                        acc_queries = list(queries)
                else:
                    seen, deduped = {}, []
                    for p in all_search_papers:
                        did = p.get("document_id")
                        if did not in seen:
                            seen[did] = True
                            deduped.append(p)
                    merged_papers = deduped
                    acc_queries = list(queries)
                    from .cache import _generate_id

                    accumulator_id = _generate_id("s")
                    if tag_val:
                        self._search_accumulator_ids[tag_val] = accumulator_id

                # When accumulating with --tag, only apply the hard cap so
                # multi-round searches can grow the pool.  Without --tag,
                # respect the per-search limit.
                effective_cap = self._ACCUMULATOR_CAP if tag_val else limit
                if len(merged_papers) > effective_cap:
                    merged_papers = merged_papers[:effective_cap]
                    truncated = True

                pool_save_data = {
                    "papers": merged_papers,
                    "query": "; ".join(acc_queries),
                    "queries": acc_queries,
                }
                self.fs.results_registry.save(
                    data=pool_save_data,
                    session_id=session_id,
                    results_id=accumulator_id,
                )
                self.fs.results_registry.save(
                    data=pool_save_data,
                    session_id=session_id,
                    results_id=f"{accumulator_id}__pool",
                )
                if hasattr(self.fs, "_save_search_artifact"):
                    asyncio.create_task(self.fs._save_search_artifact(
                        accumulator_id,
                        merged_papers,
                        "; ".join(acc_queries),
                        session_id,
                    ))
                accumulated_id = accumulator_id
                accumulated_total = len(merged_papers)

        # Format output table
        max_query_len = max(len(r["query"]) for r in output_rows)
        max_query_len = min(max(max_query_len, 20), 40)

        output_lines = []
        header = f"{'Query':<{max_query_len + 2}}  {'Results':>7}  File"
        output_lines.append(header)
        output_lines.append("-" * len(header))

        for row in output_rows:
            query_display = f'"{row["query"]}"'
            if len(query_display) > max_query_len + 2:
                query_display = query_display[: max_query_len - 1] + '..."'

            count_str = str(row["count"]) if row["count"] > 0 else "0"
            file_str = row["file"]

            line = f"{query_display:<{max_query_len + 2}}  {count_str:>7}  {file_str}"
            if row.get("error"):
                line += f"  (error: {row['error'][:30]})"
            output_lines.append(line)

        output_lines.append("")
        if accumulated_id:
            tag_suffix = f"  [tag: {tag_val}]" if tag_val else ""
            output_lines.append(
                f"All results accumulated into: {accumulated_id}{tag_suffix}  ({accumulated_total} unique papers)"
            )
            if used_combined:
                output_lines.append(
                    f"  → {accumulated_total} papers (ranked by relevance across {len(queries)} search terms)"
                )
            else:
                output_lines.append(f"  → {accumulated_total} papers")
            if truncated:
                cap_used = self._ACCUMULATOR_CAP if tag_val else limit
                output_lines.append(
                    f"  ⚠ Accumulated results capped at {cap_used} papers."
                )
            output_lines.append(
                f'Cite search results as table: {{{{"artifact": {{{{"artifact_id": "{accumulated_id}", "type": "table", '
                f'"source_count": {accumulated_total}, "description": "YOUR_ONE_SENTENCE_SUMMARY"}}}}}}}}'
            )
        else:
            output_lines.append("Use: cat FILEPATH to view full results")

        if is_quiet:
            quiet_lines = []
            if accumulated_id:
                tag_suffix = f"  [tag: {tag_val}]" if tag_val else ""
                quiet_lines.append(
                    f"All results accumulated into: {accumulated_id}{tag_suffix}  ({accumulated_total} unique papers)"
                )
                if used_combined:
                    quiet_lines.append(
                        f"  → {accumulated_total} papers (ranked by relevance across {len(queries)} search terms)"
                    )
                else:
                    quiet_lines.append(f"  → {accumulated_total} papers")
                quiet_lines.append(
                    f'Cite search results as table: {{{{"artifact": {{{{"artifact_id": "{accumulated_id}", "type": "table", '
                    f'"source_count": {accumulated_total}, "description": "YOUR_ONE_SENTENCE_SUMMARY"}}}}}}}}'
                )
            else:
                # No --tag: save combined results into a one-off registry entry
                # so the model can still use --from after --quiet searches.
                all_papers = []
                seen_ids: set = set()
                for row in output_rows:
                    row_id = row.get("results_id", "")
                    if row_id and hasattr(self.fs, "results_registry"):
                        saved = self.fs.results_registry.load(row_id, session_id)
                        for p in (saved or {}).get("papers", []):
                            did = p.get("document_id", "")
                            if did and did not in seen_ids:
                                seen_ids.add(did)
                                all_papers.append(p)

                if all_papers and hasattr(self.fs, "results_registry"):
                    combo_id = self.fs.results_registry.save(
                        data={"papers": all_papers, "query": "; ".join(queries)},
                        session_id=session_id,
                        prefix="s",
                    )
                    quiet_lines.append(
                        f"Searched {len(queries)} queries → {len(all_papers)} papers"
                    )
                    quiet_lines.append(f"  results_id: {combo_id}")
                    quiet_lines.append(
                        f"  Use: map --from {combo_id} \"question\""
                    )
                else:
                    total_hits = sum(r["count"] for r in output_rows)
                    quiet_lines.append(
                        f"Searched {len(queries)} queries, {total_hits} total hits"
                    )
                    quiet_lines.append(f"  → {total_hits} papers (use --tag to save for map)")
            return TerminalResult(stdout="\n".join(quiet_lines) + "\n", cwd=self.cwd)

        if accumulated_id:
            self._last_search_results_id = accumulated_id

        return TerminalResult(stdout="\n".join(output_lines) + "\n", cwd=self.cwd)

    async def _cmd_merge(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Merge multiple search results into a single deduplicated result.

        Usage:
            merge s_abc123 s_def456              # Merge two search results
            merge s_abc123 s_def456 s_ghi789     # Merge three or more

        Deduplicates papers by document_id, keeping the first occurrence.
        Returns a new combined search_id (s_ prefix) with a table artifact.
        """
        if not self.fs:
            return TerminalResult(
                stderr="merge: filesystem not available", exit_code=1, cwd=self.cwd
            )

        # Extract search IDs from args
        search_ids = [arg for arg in args if arg.startswith("s_")]

        if len(search_ids) < 2:
            return TerminalResult(
                stderr="merge: usage: merge s_ID1 s_ID2 [s_ID3 ...]\n"
                "       Merge multiple search results into one deduplicated result.\n"
                "       Requires at least 2 search result IDs (s_ prefix).\n"
                "       Example: merge s_abc123 s_def456 s_ghi789",
                exit_code=1,
                cwd=self.cwd,
            )

        # Load all search results
        seen_doc_ids = set()
        merged_papers = []
        queries = []
        errors = []

        for sid in search_ids:
            data = self.fs.results_registry.load(sid, session_id)
            if data is None:
                errors.append(f"not found: {sid}")
                continue

            papers = data.get("papers", [])
            query = data.get("query", "")
            if query:
                queries.append(query)

            for paper in papers:
                doc_id = paper.get("document_id", "")
                if doc_id and doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    merged_papers.append(paper)

        if errors:
            return TerminalResult(
                stderr=f"merge: {'; '.join(errors)}", exit_code=1, cwd=self.cwd
            )

        # Save merged results with a new s_ ID
        merged_query = " + ".join(queries) if queries else "merged search"
        merged_data = {"papers": merged_papers, "query": merged_query}
        merged_id = self.fs.results_registry.save(
            data=merged_data,
            session_id=session_id,
            prefix="s",
        )

        # Save as a table artifact so it can be cited/viewed (fire-and-forget)
        asyncio.create_task(self.fs._save_search_artifact(
            merged_id, merged_papers, merged_query, session_id
        ))

        # Compute stats
        total_before = sum(
            len(
                (self.fs.results_registry.load(sid, session_id) or {}).get("papers", [])
            )
            for sid in search_ids
        )
        duplicates_removed = total_before - len(merged_papers)

        # Format output
        item_name = "papers" if self.root_path == "/papers/" else "documents"
        output_lines = [
            f"Merged {len(search_ids)} searches → {len(merged_papers)} unique {item_name}  [results_id: {merged_id}]",
        ]
        if duplicates_removed > 0:
            output_lines.append(
                f"  ({total_before} total, {duplicates_removed} duplicates removed)"
            )
        output_lines.append(f"  Sources: {', '.join(search_ids)}")
        if queries:
            output_lines.append(f'  Queries: {"; ".join(queries)}')
        output_lines.append("")

        # Show first few papers as preview
        preview_count = min(5, len(merged_papers))
        for i, paper in enumerate(merged_papers[:preview_count], 1):
            doc_id = paper.get("document_id", "?")
            title = paper.get("title", "Untitled")
            output_lines.append(f"  {i}. {title}")
            output_lines.append(f"     doc_id: {doc_id}")
            output_lines.append("")

        if len(merged_papers) > preview_count:
            output_lines.append(f"  ... and {len(merged_papers) - preview_count} more")
            output_lines.append("")

        output_lines.append(
            f'Cite merged results as table: {{{{"artifact": {{{{"artifact_id": "{merged_id}", "type": "table", '
            f'"source_count": {len(merged_papers)}, "description": "YOUR_ONE_SENTENCE_SUMMARY"}}}}}}}}'
        )
        output_lines.append(
            f'Use merged results with map: map --from {merged_id} "your question"'
        )

        return TerminalResult(stdout="\n".join(output_lines) + "\n", cwd=self.cwd)

    async def _cmd_lookup(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Look up papers by metadata field.

        Usage:
            lookup FIELD VALUE           # Exact/partial match
            lookup doi 10.1101/...       # By DOI
            lookup author "Smith"        # By author name
            lookup title "CRISPR"        # By title
            lookup month_year 2024       # By publication date
            lookup pmc PMC7194329        # By PMC ID
            lookup pmid 32943797         # By PubMed ID
            lookup -n 50 author Smith    # Limit results

        Fields: doi, author, title, month_year, source, abstract, pmc, pmid
        """
        if not self.fs:
            return TerminalResult(
                stderr="lookup: filesystem not available", exit_code=1, cwd=self.cwd
            )

        # Parse flags
        limit = 25
        is_json = "--json" in args
        for i, arg in enumerate(args):
            if arg == "-n" and i + 1 < len(args):
                try:
                    limit = int(args[i + 1])
                except ValueError:
                    pass

        # Remove flags from args
        clean_args = []
        skip_next = False
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg in ("-n",):
                skip_next = True
                continue
            if arg in ("--json",):
                continue
            if not arg.startswith("-"):
                clean_args.append(arg)

        if len(clean_args) < 2:
            if self.root_path == "/fda/":
                return TerminalResult(
                    stderr=(
                        "lookup: usage: lookup FIELD VALUE\n"
                        "\n"
                        "  FDA fields:\n"
                        "    url (public_url)    - FDA document URL or partial URL\n"
                        "    tradename (name)    - Drug/product trade name\n"
                        "    identifier (application) - Application number\n"
                        "    source_type         - Document source (toc_review, cber, etc.)\n"
                        "    document_type (type) - Document type\n"
                        "    source_file         - Source filename\n"
                        "    title               - Alias for tradename\n"
                        "\n"
                        "  Examples:\n"
                        "    lookup url https://www.fda.gov/media/192156/download\n"
                        "    lookup url fda.gov/media/192156\n"
                        "    lookup tradename \"Keytruda\"\n"
                        "    lookup identifier BLA761388\n"
                        "    lookup source_type toc_review\n"
                        "    lookup -n 50 tradename ibrutinib"
                    ),
                    exit_code=1,
                    cwd=self.cwd,
                )
            return TerminalResult(
                stderr=(
                    "lookup: usage: lookup FIELD VALUE\n"
                    "\n"
                    "  Shared (bioRxiv + PMC):\n"
                    "    doi, title, author, abstract, source\n"
                    "\n"
                    "  bioRxiv/medRxiv only:\n"
                    "    month_year (alias: date, month)\n"
                    "\n"
                    "  PMC only:\n"
                    "    pmc, pmid, journal, publisher, type, keywords, category,\n"
                    "    license, year, volume, issue, issn\n"
                    "\n"
                    "  Examples:\n"
                    "    lookup doi 10.1101/2024.09.16.613278\n"
                    "    lookup author \"James Zou\"\n"
                    "    lookup journal \"Nature Medicine\"\n"
                    "    lookup keywords \"CRISPR\"\n"
                    "    lookup type review-article\n"
                    "    lookup year 2024\n"
                    "    lookup -n 50 title CRISPR"
                ),
                exit_code=1,
                cwd=self.cwd,
            )

        field = clean_args[0].lower()
        value = " ".join(clean_args[1:])

        # FDA-specific fields (when root_path is /fda/)
        FDA_FIELD_MAP = {
            "url": "public_url",
            "public_url": "public_url",
            "tradename": "tradename",
            "name": "tradename",
            "identifier": "identifier",
            "application": "identifier",
            "source_type": "source_type",
            "document_type": "document_type",
            "type": "document_type",
            "source_file": "source_file",
        }

        # Map user-friendly field names → DB column names (or routing keys for PMC)
        field_map = {
            # ── Shared fields (biomedrxiv + PMC) ─────────────────────────────
            "doi": "doi",
            "title": "title",
            "author": "authors",
            "authors": "authors",
            "source": "source",
            "abstract": "abstract_text",
            "abstract_text": "abstract_text",
            # ── bioRxiv / medRxiv only ────────────────────────────────────────
            "month_year": "month_year",
            "date": "month_year",
            "month": "month_year",
            # ── arXiv (queries document_id with arx_ prefix) ─────────────────
            "arxiv": "arxiv",
            "arxiv_id": "arxiv",
            # ── PMC-specific (routed to PMC DB) ──────────────────────────────
            "pmc": "pmc",
            "pmc_id": "pmc",
            "pmid": "pmid",
            "journal": "journal_title",
            "journal_title": "journal_title",
            "publisher": "publisher_name",
            "publisher_name": "publisher_name",
            "type": "article_type",
            "article_type": "article_type",
            "keywords": "keywords",
            "keyword": "keywords",
            "category": "categories",
            "categories": "categories",
            "license": "license_type",
            "license_type": "license_type",
            "year": "pub_year",
            "pub_year": "pub_year",
            "volume": "volume",
            "issue": "issue",
            "issn": "issn",
        }

        # If we're in FDA mode, use FDA field map (with fallback to shared)
        is_fda = self.root_path == "/fda/"
        if is_fda:
            if field in FDA_FIELD_MAP:
                db_field = FDA_FIELD_MAP[field]
            elif field in field_map:
                # Allow "title" → "tradename" for FDA
                if field == "title":
                    db_field = "tradename"
                else:
                    return TerminalResult(
                        stderr=(
                            f"lookup: field '{field}' is not available for FDA documents.\n"
                            f"  Valid FDA fields: url, tradename, name, identifier, application,\n"
                            f"                    source_type, document_type, type, source_file"
                        ),
                        exit_code=1,
                        cwd=self.cwd,
                    )
            else:
                return TerminalResult(
                    stderr=(
                        f"lookup: unknown field '{field}'\n"
                        f"  Valid FDA fields: url, tradename, name, identifier, application,\n"
                        f"                    source_type, document_type, type, source_file"
                    ),
                    exit_code=1,
                    cwd=self.cwd,
                )
        else:
            if field not in field_map:
                return TerminalResult(
                    stderr=(
                        f"lookup: unknown field '{field}'\n"
                        f"  Valid: doi, arxiv, title, author, abstract, source, month_year\n"
                        f"  PMC:   pmc, pmid, journal, publisher, type, keywords,\n"
                        f"         category, license, year, volume, issue, issn"
                    ),
                    exit_code=1,
                    cwd=self.cwd,
                )
            db_field = field_map[field]

        # arXiv IDs are stored as bare IDs in document_id (e.g. "2404.10198").
        # Strip any arx_/arxiv_ prefix the user may have included.
        if db_field == "arxiv":
            value = re.sub(r"^(arxiv|arx)_", "", value.strip())
            db_field = "document_id"

        # PMC-only fields that must route to the PMC database
        PMC_ONLY_FIELDS = {
            "pmc", "pmc_id", "pmid",
            "journal_title", "publisher_name", "article_type",
            "keywords", "categories", "license_type",
            "pub_year", "volume", "issue", "issn",
        }
        force_pmc = (not is_fda) and (db_field in PMC_ONLY_FIELDS)

        try:
            result = await self.fs._lookup(
                field=db_field,
                value=value,
                limit=limit,
                session_id=session_id,
                force_pmc=force_pmc,
            )

            if "error" in result:
                return TerminalResult(
                    stderr=f"lookup: {result['error']}", exit_code=1, cwd=self.cwd
                )

            papers = result.get("results", [])
            total = result.get("total", len(papers))
            results_id = result.get("results_id", "")

            item_name = "papers" if self.root_path == "/papers/" else "documents"
            if not papers:
                if is_json:
                    return TerminalResult(
                        stdout=json.dumps({"total": 0, "results": []}) + "\n",
                        cwd=self.cwd,
                    )
                return TerminalResult(
                    stdout=f"No {item_name} found where {field} contains '{value}'.\n",
                    cwd=self.cwd,
                )

            if is_json:
                return TerminalResult(
                    stdout=json.dumps({"total": total, "results": papers, "results_id": results_id}, default=str) + "\n",
                    cwd=self.cwd,
                )

            output_lines = [
                f"Found {total} {item_name} where {field} contains '{value}' (showing {len(papers)}):"
            ]
            output_lines.append("")

            for i, paper in enumerate(papers, 1):
                doc_id = paper.get("document_id", "?")
                title = paper.get("title", "Untitled")

                output_lines.append(f"  {i}. {title}")
                output_lines.append(f"     doc_id: {doc_id}")

                if is_fda:
                    source_type = paper.get("source_type", "")
                    identifier = paper.get("identifier", "")
                    document_type = paper.get("document_type", "")
                    public_url = paper.get("public_url", "")
                    total_pages = paper.get("total_pages")
                    if identifier:
                        output_lines.append(f"     {identifier}")
                    if source_type or document_type:
                        parts = [p for p in [source_type, document_type] if p]
                        output_lines.append(f"     {' · '.join(parts)}")
                    if public_url:
                        output_lines.append(f"     {public_url}")
                    if total_pages:
                        output_lines.append(f"     {total_pages} pages")
                else:
                    authors = paper.get("authors", "")
                    doi = paper.get("doi", "")
                    pub_date = paper.get("pub_date", "") or paper.get("pub_year", "")
                    journal = paper.get("journal", "")
                    article_type = paper.get("article_type", "")

                    if authors:
                        output_lines.append(f"     {authors}")
                    if doi:
                        output_lines.append(f"     DOI: {doi}")
                    if journal:
                        line = f"     {journal}"
                        if article_type:
                            line += f" · {article_type}"
                        if pub_date:
                            line += f" ({pub_date})"
                        output_lines.append(line)
                    elif pub_date:
                        output_lines.append(f"     {pub_date}")
                output_lines.append("")

            # Auto-cd when exactly 1 result — saves a round-trip
            if len(papers) == 1:
                doc_id = papers[0].get("document_id", "")
                auto_path = f"{self.root_path}{doc_id}/"
                self.cwd = auto_path
                self.env["PWD"] = self.cwd
                output_lines.append(f"(auto-cd to {auto_path})")
                try:
                    ls_result = await self._cmd_ls(["-la"], session_id=session_id)
                    if ls_result.stdout:
                        output_lines.append("")
                        output_lines.append(ls_result.stdout.rstrip("\n"))
                except Exception:
                    pass

            return TerminalResult(stdout="\n".join(output_lines) + "\n", cwd=self.cwd)

        except Exception as e:
            return TerminalResult(stderr=f"lookup: {e}", exit_code=1, cwd=self.cwd)

    async def _cmd_sort(
        self, args: list[str], stdin: str = "", **kwargs
    ) -> TerminalResult:
        """Sort lines."""
        reverse = "-r" in args
        numeric = "-n" in args
        unique = "-u" in args

        if not stdin:
            return TerminalResult(stdout="", cwd=self.cwd)

        lines = stdin.strip().split("\n")

        if numeric:

            def sort_key(x):
                try:
                    return float(re.search(r"-?\d+\.?\d*", x).group())
                except (AttributeError, ValueError):
                    return 0

            lines.sort(key=sort_key, reverse=reverse)
        else:
            lines.sort(reverse=reverse)

        if unique:
            lines = list(dict.fromkeys(lines))

        return TerminalResult(stdout="\n".join(lines) + "\n", cwd=self.cwd)

    async def _cmd_uniq(
        self, args: list[str], stdin: str = "", **kwargs
    ) -> TerminalResult:
        """Filter adjacent duplicate lines."""
        count = "-c" in args
        duplicates_only = "-d" in args
        unique_only = "-u" in args

        if not stdin:
            return TerminalResult(stdout="", cwd=self.cwd)

        lines = stdin.strip().split("\n")
        result = []
        prev = None
        prev_count = 0

        for line in lines:
            if line == prev:
                prev_count += 1
            else:
                if prev is not None:
                    if count:
                        result.append(f"{prev_count:7} {prev}")
                    elif duplicates_only:
                        if prev_count > 1:
                            result.append(prev)
                    elif unique_only:
                        if prev_count == 1:
                            result.append(prev)
                    else:
                        result.append(prev)
                prev = line
                prev_count = 1

        # Don't forget the last line
        if prev is not None:
            if count:
                result.append(f"{prev_count:7} {prev}")
            elif duplicates_only:
                if prev_count > 1:
                    result.append(prev)
            elif unique_only:
                if prev_count == 1:
                    result.append(prev)
            else:
                result.append(prev)

        return TerminalResult(
            stdout="\n".join(result) + "\n" if result else "", cwd=self.cwd
        )

    async def _cmd_awk(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Basic awk implementation supporting common patterns.

        Supported:
            awk '/pattern/'                    # Print matching lines (like grep)
            awk '/start/,/end/'                # Print range between patterns
            awk '{print $N}'                   # Print Nth field (space-delimited)
            awk -F: '{print $1}'               # Print field with custom delimiter
            awk 'NR>=5 && NR<=10'              # Print line range
            awk '{print NR, $0}'               # Print with line numbers
        """
        import re as re_module

        # Parse flags and program
        field_sep = None
        program = ""
        file_args = []
        i = 0
        while i < len(args):
            if args[i] == "-F" and i + 1 < len(args):
                field_sep = args[i + 1]
                i += 2
            elif args[i].startswith("-F"):
                field_sep = args[i][2:]
                i += 1
            elif not program and (
                args[i].startswith("'")
                or args[i].startswith('"')
                or args[i].startswith("{")
                or args[i].startswith("/")
            ):
                program = args[i].strip("'\"")
                i += 1
            elif program:
                file_args.append(args[i])
                i += 1
            else:
                program = args[i].strip("'\"")
                i += 1

        # Get input
        text = stdin
        if not text and file_args:
            cat_result = await self._cmd_cat(file_args, session_id=session_id)
            text = cat_result.stdout
        if not text:
            return TerminalResult(
                stderr="vsh: awk: no input",
                exit_code=1,
                cwd=self.cwd,
            )

        if not program:
            return TerminalResult(stdout=text, cwd=self.cwd)

        lines = text.split("\n")
        sep = field_sep or r"\s+"
        output_lines = []

        # Pattern: /start/,/end/ (range between patterns)
        range_match = re_module.match(r"^/(.*)/,/(.*)/\s*$", program)
        if range_match:
            start_pat = re_module.compile(range_match.group(1))
            end_pat = re_module.compile(range_match.group(2))
            in_range = False
            for line in lines:
                if not in_range and start_pat.search(line):
                    in_range = True
                if in_range:
                    output_lines.append(line)
                    if end_pat.search(line):
                        in_range = False
            return TerminalResult(stdout="\n".join(output_lines), cwd=self.cwd)

        # Pattern: /pattern/ (grep-like)
        grep_match = re_module.match(r"^/(.*)/\s*$", program)
        if grep_match:
            pat = re_module.compile(grep_match.group(1))
            output_lines = [l for l in lines if pat.search(l)]
            return TerminalResult(stdout="\n".join(output_lines), cwd=self.cwd)

        # Pattern: NR>=N && NR<=M or NR==N (line range)
        nr_match = re_module.match(r"^NR\s*>=\s*(\d+)\s*&&\s*NR\s*<=\s*(\d+)$", program)
        if nr_match:
            start = int(nr_match.group(1))
            end = int(nr_match.group(2))
            output_lines = [l for i, l in enumerate(lines, 1) if start <= i <= end]
            return TerminalResult(stdout="\n".join(output_lines), cwd=self.cwd)

        # Pattern: {print ...} (field extraction)
        print_match = re_module.match(r"^\{print\s+(.*)\}$", program)
        if print_match:
            expr = print_match.group(1).strip()
            for line in lines:
                if not line:
                    continue
                fields = re_module.split(sep, line) if sep != r"\s+" else line.split()
                # Build output from expression
                parts = []
                for token in re_module.split(r"[,\s]+", expr):
                    token = token.strip().strip('"')
                    if token == "$0":
                        parts.append(line)
                    elif token == "NR":
                        parts.append(str(lines.index(line) + 1))
                    elif token.startswith("$") and token[1:].isdigit():
                        idx = int(token[1:])
                        if idx == 0:
                            parts.append(line)
                        elif idx <= len(fields):
                            parts.append(fields[idx - 1])
                        else:
                            parts.append("")
                    else:
                        parts.append(token)
                output_lines.append(" ".join(parts))
            return TerminalResult(stdout="\n".join(output_lines), cwd=self.cwd)

        return TerminalResult(
            stderr=f"vsh: awk: unsupported program: {program}",
            exit_code=1,
            cwd=self.cwd,
        )

    async def _cmd_sed(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Basic sed implementation supporting common patterns.

        Supported:
            sed -n 'START,ENDp'           # Print line range
            sed -n 'Np'                   # Print single line
            sed 's/pattern/replacement/'  # Substitute first occurrence
            sed 's/pattern/replacement/g' # Substitute all occurrences
            sed '/pattern/d'              # Delete matching lines
            sed '/pattern/!d'             # Keep only matching lines (like grep)
        """
        import re as re_module

        # Get input text
        text = stdin
        if not text:
            # Check for file argument
            files = [
                a
                for a in args
                if not a.startswith("-")
                and not a.startswith("'")
                and not a.startswith('"')
                and "/" in a
                or "." in a
            ]
            if files:
                cat_result = await self._cmd_cat(files, session_id=session_id)
                text = cat_result.stdout
            if not text:
                return TerminalResult(
                    stderr="vsh: sed: no input (pipe data or specify file)",
                    exit_code=1,
                    cwd=self.cwd,
                )

        # Extract the sed expression (may be after -n flag)
        expr = ""
        is_quiet = "-n" in args
        for a in args:
            if a == "-n" or a == "sed":
                continue
            if a.startswith("-"):
                continue
            # Skip file args
            if "/" in a or "." in a:
                continue
            expr = a.strip("'\"")
            break

        if not expr:
            return TerminalResult(stdout=text, cwd=self.cwd)

        lines = text.split("\n")

        # Pattern: Np or N,Mp (print line range)
        range_match = re_module.match(r"^(\d+)(?:,(\d+))?p$", expr)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2)) if range_match.group(2) else start
            selected = lines[max(0, start - 1) : end]
            return TerminalResult(stdout="\n".join(selected), cwd=self.cwd)

        # Pattern: s/old/new/ or s/old/new/g
        sub_match = re_module.match(r"^s(.)(.+?)\1(.*?)\1(g?)$", expr)
        if sub_match:
            pattern = sub_match.group(2)
            replacement = sub_match.group(3)
            global_flag = sub_match.group(4) == "g"
            result_lines = []
            for line in lines:
                if global_flag:
                    result_lines.append(re_module.sub(pattern, replacement, line))
                else:
                    result_lines.append(
                        re_module.sub(pattern, replacement, line, count=1)
                    )
            return TerminalResult(stdout="\n".join(result_lines), cwd=self.cwd)

        # Pattern: /pattern/d (delete matching)
        del_match = re_module.match(r"^/(.*)/d$", expr)
        if del_match:
            pattern = del_match.group(1)
            result_lines = [l for l in lines if not re_module.search(pattern, l)]
            return TerminalResult(stdout="\n".join(result_lines), cwd=self.cwd)

        # Pattern: /pattern/!d (keep only matching, like grep)
        keep_match = re_module.match(r"^/(.*)/!d$", expr)
        if keep_match:
            pattern = keep_match.group(1)
            result_lines = [l for l in lines if re_module.search(pattern, l)]
            return TerminalResult(stdout="\n".join(result_lines), cwd=self.cwd)

        return TerminalResult(
            stderr=f"vsh: sed: unsupported expression: {expr}",
            exit_code=1,
            cwd=self.cwd,
        )

    async def _cmd_cut(
        self, args: list[str], stdin: str = "", **kwargs
    ) -> TerminalResult:
        """Extract columns/fields."""
        delimiter = "\t"
        fields = None

        for i, arg in enumerate(args):
            if arg == "-d" and i + 1 < len(args):
                delimiter = args[i + 1]
            elif arg.startswith("-d"):
                delimiter = arg[2:]
            elif arg == "-f" and i + 1 < len(args):
                fields = args[i + 1]
            elif arg.startswith("-f"):
                fields = arg[2:]

        if not stdin or not fields:
            return TerminalResult(stdout=stdin, cwd=self.cwd)

        # Parse field spec (e.g., "1,3" or "1-3")
        field_indices = set()
        for spec in fields.split(","):
            if "-" in spec:
                start, end = spec.split("-", 1)
                start = int(start) if start else 1
                end = int(end) if end else 100
                field_indices.update(range(start, end + 1))
            else:
                field_indices.add(int(spec))

        result = []
        for line in stdin.strip().split("\n"):
            parts = line.split(delimiter)
            selected = [
                parts[i - 1] for i in sorted(field_indices) if 0 < i <= len(parts)
            ]
            result.append(delimiter.join(selected))

        return TerminalResult(stdout="\n".join(result) + "\n", cwd=self.cwd)

    def _expand_char_range(self, spec: str) -> str:
        """Expand character ranges like a-z, A-Z, 0-9."""
        # Handle character classes
        if spec == "[:upper:]":
            return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        elif spec == "[:lower:]":
            return "abcdefghijklmnopqrstuvwxyz"
        elif spec == "[:digit:]":
            return "0123456789"
        elif spec == "[:alpha:]":
            return "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        elif spec == "[:alnum:]":
            return "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        # Expand ranges like a-z
        result = []
        i = 0
        while i < len(spec):
            if i + 2 < len(spec) and spec[i + 1] == "-":
                # Range expansion
                start_char = spec[i]
                end_char = spec[i + 2]
                for c in range(ord(start_char), ord(end_char) + 1):
                    result.append(chr(c))
                i += 3
            else:
                result.append(spec[i])
                i += 1

        return "".join(result)

    async def _cmd_tr(
        self, args: list[str], stdin: str = "", **kwargs
    ) -> TerminalResult:
        """Translate characters."""
        delete = "-d" in args
        squeeze = "-s" in args

        char_args = [a for a in args if not a.startswith("-")]

        if not stdin:
            return TerminalResult(stdout="", cwd=self.cwd)

        if delete and char_args:
            # Delete characters
            chars_to_delete = self._expand_char_range(char_args[0])
            result = stdin.translate(str.maketrans("", "", chars_to_delete))
        elif len(char_args) >= 2:
            # Translate
            set1 = self._expand_char_range(char_args[0])
            set2 = self._expand_char_range(char_args[1])

            # Pad set2 to match set1 length (like real tr)
            if len(set2) < len(set1):
                set2 = set2 + set2[-1] * (len(set1) - len(set2))

            result = stdin.translate(str.maketrans(set1, set2[: len(set1)]))
        else:
            result = stdin

        if squeeze and char_args:
            # Squeeze repeated characters
            squeeze_chars = self._expand_char_range(char_args[-1])
            for c in squeeze_chars:
                while c + c in result:
                    result = result.replace(c + c, c)

        return TerminalResult(stdout=result, cwd=self.cwd)

    async def _cmd_jq(
        self, args: list[str], stdin: str = "", **kwargs
    ) -> TerminalResult:
        """JSON processor."""
        filter_expr = args[0] if args else "."

        if not stdin:
            return TerminalResult(
                stderr="vsh: jq: requires JSON input",
                exit_code=1,
                cwd=self.cwd,
            )

        try:
            data = json.loads(stdin)
        except json.JSONDecodeError as e:
            return TerminalResult(
                stderr=f"vsh: jq: parse error: {e}",
                exit_code=1,
                cwd=self.cwd,
            )

        # Simple jq filters
        if filter_expr == ".":
            result = json.dumps(data, indent=2)
        elif filter_expr.startswith("."):
            # Field access
            keys = filter_expr[1:].split(".")
            current = data
            for key in keys:
                if not key:
                    continue
                if isinstance(current, dict) and key in current:
                    current = current[key]
                elif isinstance(current, list) and key.isdigit():
                    current = current[int(key)]
                else:
                    current = None
                    break
            result = json.dumps(current, indent=2)
        elif filter_expr == "keys":
            if isinstance(data, dict):
                result = json.dumps(list(data.keys()), indent=2)
            else:
                result = "[]"
        elif filter_expr == "length":
            result = str(len(data))
        else:
            result = json.dumps(data, indent=2)

        return TerminalResult(stdout=result + "\n", cwd=self.cwd)

    async def _cmd_echo(
        self, args: list[str], stdin: str = "", **kwargs
    ) -> TerminalResult:
        """Print text."""
        no_newline = "-n" in args
        text_args = [a for a in args if a != "-n"]
        output = " ".join(text_args)

        if not no_newline:
            output += "\n"

        return TerminalResult(stdout=output, cwd=self.cwd)

    async def _cmd_env(
        self, args: list[str], stdin: str = "", **kwargs
    ) -> TerminalResult:
        """Print environment variables."""
        lines = [f"{k}={v}" for k, v in self.env.items()]
        return TerminalResult(stdout="\n".join(lines) + "\n", cwd=self.cwd)

    async def _cmd_history(
        self, args: list[str], stdin: str = "", **kwargs
    ) -> TerminalResult:
        """Show command history."""
        n = 20
        if args and args[0].isdigit():
            n = int(args[0])

        recent = self.history[-n:]
        lines = [f"{i+1:5}  {cmd}" for i, cmd in enumerate(recent)]
        return TerminalResult(stdout="\n".join(lines) + "\n", cwd=self.cwd)

    async def _cmd_mode(
        self, args: list[str], stdin: str = "", **kwargs
    ) -> TerminalResult:
        """Switch data source mode.

        Usage:
            mode              Show current mode
            mode all          Search across all sources (default)
            mode pmc          Restrict to PMC articles only
            mode biorxiv      Restrict to bioRxiv preprints only
            mode medrxiv      Restrict to medRxiv preprints only
            mode biomedrxiv   Restrict to bioRxiv + medRxiv
        """
        valid_modes = {"all", "pmc", "biorxiv", "medrxiv", "biomedrxiv"}
        current = self.env.get("PAPERS_MODE", "all")

        if not args:
            return TerminalResult(
                stdout=f"Current mode: {current}\nAvailable: {', '.join(sorted(valid_modes))}\n",
                cwd=self.cwd,
            )

        new_mode = args[0].lower()
        if new_mode not in valid_modes:
            return TerminalResult(
                stderr=f"Unknown mode: {new_mode}. Available: {', '.join(sorted(valid_modes))}\n",
                exit_code=1,
                cwd=self.cwd,
            )

        self.env["PAPERS_MODE"] = new_mode
        label = {
            "all": "all sources (PMC + bioRxiv + medRxiv)",
            "pmc": "PMC articles only",
            "biorxiv": "bioRxiv preprints only",
            "medrxiv": "medRxiv preprints only",
            "biomedrxiv": "bioRxiv + medRxiv preprints",
        }
        return TerminalResult(
            stdout=f"Switched to: {label[new_mode]}\n",
            cwd=self.cwd,
        )

    async def _cmd_help(
        self, args: list[str], stdin: str = "", **kwargs
    ) -> TerminalResult:
        """Show help."""
        if args:
            cmd = args[0]
            help_text = COMMAND_HELP.get(cmd, f"No help available for '{cmd}'")
            return TerminalResult(stdout=help_text + "\n", cwd=self.cwd)

        help_text = """vsh - Virtual Shell for BioMedRxiv Filesystem

NAVIGATION:
  cd [dir]     Change directory
  pwd          Print working directory
  ls [opts]    List directory (-l, -a)
  tree [dir]   Show directory tree

READING:
  cat [file]   Display file contents (-n for line numbers)
  head [file]  Show first lines (-n N)
  tail [file]  Show last lines (-n N)
  wc [file]    Count lines/words/chars (-l, -w, -c)

SEARCHING:
  search QUERY          Search all papers (semantic, default: 25 results)
  search -r PATTERN     Regex search across all papers
  search -a AUTHOR      Search by author
  search -n 50 QUERY    Limit results (default: 25)
  searches "q1" "q2"... Run multiple searches in parallel, save to /session_files/searches/
  lookup FIELD VALUE    Look up by metadata (doi, author, title, month_year)
  grep PATTERN [file]   Search within file (-i, -n, -c, -v)

ANALYSIS:
  ask_image FIG "q"     Analyze a paper figure
  ask_image --list      List available figures
  map --from ID "q"     Map query across papers
  reduce --from ID ...  Synthesize map results

TEXT PROCESSING:
  sort         Sort lines (-r, -n, -u)
  uniq         Remove duplicates (-c, -d, -u)
  cut          Extract columns (-d DELIM, -f FIELDS)
  tr           Translate characters (-d, -s)
  jq           JSON processor

PIPES:
  cmd1 | cmd2        Pipe output
  
Type 'help COMMAND' for detailed help.
"""
        return TerminalResult(stdout=help_text, cwd=self.cwd)

    _SKILL_TEXT: str | None = None

    @classmethod
    def _load_skill_text(cls) -> str:
        """Load the canonical skill.md from disk (cached after first read)."""
        if cls._SKILL_TEXT is not None:
            return cls._SKILL_TEXT

        from pathlib import Path

        here = Path(__file__).resolve().parent
        candidates = [
            here.parents[3] / "packages" / "gxl-paperclip" / "skills" / "staging" / "skill.md",
        ]
        for p in candidates:
            if p.is_file():
                cls._SKILL_TEXT = p.read_text().strip()
                return cls._SKILL_TEXT

        cls._SKILL_TEXT = "skill.md not found. Run `help` for basic usage."
        return cls._SKILL_TEXT

    async def _cmd_skill(
        self, args: list[str], stdin: str = "", **kwargs
    ) -> TerminalResult:
        """Return the full paperclip skill documentation."""
        return TerminalResult(stdout=self._load_skill_text() + "\n", cwd=self.cwd)

    async def _cmd_scan(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Multi-pattern grep: scan FILE "pat1" "pat2" ... — results grouped by pattern.

        Usage:
            scan FILE "pattern1" "pattern2" "pattern3"
            scan -C 3 FILE "pat1" "pat2"          # 3 lines context per match
            scan -i FILE "pat1" "pat2"             # case insensitive

        Returns matches grouped by pattern with automatic ±5 lines of context.
        Replaces multiple sequential grep calls with a single scan.
        """
        ignore_case = "-i" in args
        context_lines = 5  # default

        # Parse -C flag
        skip_indices: set[int] = set()
        for i, arg in enumerate(args):
            if arg == "-C" and i + 1 < len(args):
                try:
                    context_lines = int(args[i + 1])
                    skip_indices.add(i)
                    skip_indices.add(i + 1)
                except ValueError:
                    pass

        # Separate file and patterns from args
        positional = [
            a
            for i, a in enumerate(args)
            if not a.startswith("-") and i not in skip_indices
        ]

        if len(positional) < 2:
            return TerminalResult(
                stderr='scan: usage: scan FILE "pattern1" "pattern2" ...\n'
                "  Multi-pattern grep with grouped results.\n"
                "  Flags: -i (case insensitive), -C N (context lines, default 5)",
                exit_code=1,
                cwd=self.cwd,
            )

        file_path = positional[0]
        patterns = positional[1:]

        # Read the file content once
        full_path = self._validate_path(file_path)
        if full_path is None:
            return TerminalResult(
                stderr=f"vsh: scan: {file_path}: Permission denied",
                exit_code=1,
                cwd=self.cwd,
            )

        if not self.fs:
            return TerminalResult(
                stderr="scan: filesystem not available", exit_code=1, cwd=self.cwd
            )

        try:
            result = await self.fs._cat(path=full_path, session_id=session_id)
            if "error" in result:
                return TerminalResult(
                    stderr=f"vsh: scan: {file_path}: {result['error']}",
                    exit_code=1,
                    cwd=self.cwd,
                )

            lines_data = result.get("lines", [])
            if not lines_data:
                return TerminalResult(stdout="(empty file)\n", cwd=self.cwd)

            all_lines: list[tuple[int, str]] = []
            for line in lines_data:
                if isinstance(line, dict):
                    all_lines.append(
                        (
                            line.get("line", len(all_lines) + 1),
                            str(line.get("content", "")),
                        )
                    )
                else:
                    all_lines.append((len(all_lines) + 1, str(line)))

            flags = re.IGNORECASE if ignore_case else 0
            total_matches = 0
            output_parts: list[str] = []

            for pattern in patterns:
                try:
                    regex = re.compile(pattern, flags)
                except re.error as e:
                    output_parts.append(f"\n[{pattern}] ERROR: invalid regex: {e}")
                    continue

                match_indices = [
                    i
                    for i, (_, content) in enumerate(all_lines)
                    if regex.search(content)
                ]

                if not match_indices:
                    output_parts.append(f"\n[{pattern}] 0 matches")
                    continue

                total_matches += len(match_indices)
                output_parts.append(f"\n[{pattern}] {len(match_indices)} match(es)")

                # Build context-aware output
                lines_to_show: set[int] = set()
                for idx in match_indices:
                    for c in range(
                        max(0, idx - context_lines),
                        min(len(all_lines), idx + context_lines + 1),
                    ):
                        lines_to_show.add(c)

                match_set = set(match_indices)
                prev_idx = -2
                for idx in sorted(lines_to_show):
                    if prev_idx >= 0 and idx > prev_idx + 1:
                        output_parts.append("  --")
                    prev_idx = idx

                    line_num, content = all_lines[idx]
                    is_match = idx in match_set
                    marker = ">" if is_match else " "
                    output_parts.append(f"  {marker} L{line_num}: {content}")

            header = f"{len(patterns)} patterns, {total_matches} total matches in {file_path}"
            return TerminalResult(
                stdout=header + "\n".join(output_parts) + "\n",
                cwd=self.cwd,
            )

        except Exception as e:
            return TerminalResult(
                stderr=f"vsh: scan: {file_path}: {e}", exit_code=1, cwd=self.cwd
            )

    async def _cmd_tee(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> TerminalResult:
        """Write stdin to a file in /.gxl/ and pass through to stdout.

        Usage: echo "data" | tee /.gxl/output.txt
               command | tee -a /.gxl/output.txt  (append)
        """
        append = "-a" in args
        files = [a for a in args if not a.startswith("-")]

        if not files:
            return TerminalResult(
                stderr="vsh: tee: missing file operand\n"
                "Usage: command | tee /.gxl/filename",
                exit_code=1,
                cwd=self.cwd,
            )

        for file_path in files:
            full_path = self._validate_path(file_path)
            if full_path is None:
                return TerminalResult(
                    stderr=f"vsh: tee: {file_path}: Permission denied",
                    exit_code=1,
                    cwd=self.cwd,
                )
            if not self._is_session_files_path(full_path):
                return TerminalResult(
                    stderr=f"vsh: tee: {file_path}: Can only write to /.gxl/",
                    exit_code=1,
                    cwd=self.cwd,
                )
            result = await self._session_files_write(
                full_path, stdin, session_id, append=append
            )
            if result.exit_code != 0:
                return result

        return TerminalResult(stdout=stdin, cwd=self.cwd)


# =============================================================================
# Help Text
# =============================================================================

COMMAND_HELP = {
    "cd": """cd - change directory

Usage: cd [DIR]

Change the current working directory to DIR.
If DIR is omitted, changes to home directory (/papers/).

Examples:
  cd /papers/abc-123/
  cd sections/
  cd ..
  cd ~""",
    "ls": """ls - list directory contents

Usage: ls [OPTIONS] [PATH]

Options:
  -l    Long format
  -a    Show hidden files

Examples:
  ls
  ls -la /papers/
  ls sections/""",
    "cat": """cat - concatenate and display files

Usage: cat [OPTIONS] [FILE...]

Options:
  -n    Number output lines

Examples:
  cat Methods.lines
  cat -n abstract.lines
  echo "hello" | cat""",
    "head": """head - display first lines of a file

Usage: head [OPTIONS] [FILE...]

Options:
  -n N    Show first N lines (default: 10)
  -N      Shorthand for -n N (e.g. -40)

Examples:
  head /papers/PMC12345/content.lines
  head -40 /papers/bio_abc123/content.lines
  head -n 5 abstract.lines""",
    "tail": """tail - display last lines of a file

Usage: tail [OPTIONS] [FILE...]

Options:
  -n N    Show last N lines (default: 10)
  -N      Shorthand for -n N (e.g. -20)

Examples:
  tail /papers/PMC12345/content.lines
  tail -20 content.lines
  tail -n 5 abstract.lines""",
    "grep": """grep - search for patterns

Usage: grep [OPTIONS] PATTERN [FILE...]

Options:
  -i          Ignore case
  -n          Show line numbers
  -c          Count matches only
  -v          Invert match (show non-matching lines)
  -o          Print only the matching part of lines
  -w          Match whole words only
  -l          List only filenames with matches
  -h          Suppress filename prefix
  -m NUM      Stop after NUM matches
  -e PATTERN  Explicit pattern (can repeat for multi-pattern OR)
  -F          Fixed strings (literal match, no regex)
  -A NUM      Show NUM lines after each match
  -B NUM      Show NUM lines before each match
  -C NUM      Show NUM lines before and after each match

Examples:
  grep "learning rate" Methods.lines
  grep -i "off-target" /papers/bio_abc123/content.lines
  grep -c "CRISPR" /papers/PMC12345/content.lines
  grep -A 3 -B 1 "IC50" content.lines
  cat abstract.lines | grep -i "protein"
  grep -n "error" *.lines""",
    "search": """search - semantic + keyword search across papers

Usage: search [OPTIONS] QUERY

Sources (positional or flag):
  pmc, biorxiv, medrxiv, arxiv    Restrict to specific source(s)
  -s/--source SOURCE              Explicit source flag (pmc, biorxiv, medrxiv, arxiv, abstracts)

Options:
  -n/--limit N      Max results (default: 100)
  -e/--exact        Exact phrase match
  -r/--regex        Regex search across all papers
  -a/--author       Search by author name
  -t/--title        Search by title
  -c/--count        Count only (no results)
  --since PERIOD    Filter by recency (e.g. 7d, 30d, 6m, 1y)
  --sort MODE       Sort order: relevance (default), date
  --journal NAME    Filter by journal (PMC)
  --year YEAR       Filter by publication year
  --category CAT    Filter by bioRxiv category
  --ranking MODE    Ranking: hybrid (default), bm25, vector
  --all             Search all papers (not just recent)
  -m/--mode MODE    Match mode: any (default), all, 50%, 75%
  --quiet           Minimal output (count + result ID only)
  --recent          Papers from last year

Examples:
  search "CRISPR base editing delivery"
  search -s arxiv "transformer attention"
  search biorxiv "single cell RNA-seq"
  search -n 10 --since 30d "protein design"
  search --all "gene therapy"
  search "protein design" | grep "diffusion"   # chain with grep""",
    "lookup": """lookup - find papers by metadata field

Usage: lookup [OPTIONS] FIELD VALUE

Fields: doi, author, title, abstract, source, date, pmc, pmid, arxiv,
        journal, publisher, type, keywords, category, license, year,
        volume, issue, issn

Options:
  -n N      Limit results (default: 25)
  --json    Output as JSON

Examples:
  lookup doi 10.1101/2024.01.15.575613
  lookup pmc PMC7194329
  lookup pmid 32943797
  lookup arxiv 2403.03507
  lookup author "James Zou" -n 10
  lookup title "CRISPR base editing"
  lookup journal "Nature Medicine" """,
    "scan": """scan - multi-pattern search in a file

Usage: scan [OPTIONS] FILE "pattern1" "pattern2" ...

Options:
  -i      Case insensitive
  -C N    Context lines per match (default: 5)

Results are grouped by pattern with automatic context lines.
Replaces multiple sequential grep calls with a single scan.

Examples:
  scan /papers/PMC12345/content.lines "CRISPR" "off-target"
  scan -i content.lines "IC50" "EC50" "dose"
  scan -C 3 content.lines "methods" "results" """,
    "map": """map - run a query across multiple papers in parallel

Usage: map --from RESULTS_ID [OPTIONS] "query"

Options:
  --from ID           Results ID from a previous search (required)
  --output_schema JSON  Structured output schema
  -n/--limit N        Limit number of papers to process
  --offset N          Skip first N papers

Tips:
  - Be specific: "What delivery vector, cell type, and efficiency?" not "Summarize"
  - Keep to 3-10 papers for speed (-n 5 or -n 10 on search)
  - After map, synthesize directly — don't re-read papers individually

Examples:
  search "protein design" -n 10
  map --from s_abc123 "What methods were used for protein design?"
  map --from s_abc123 "From Methods: what model, dataset size, and benchmark?" """,
    "reduce": """reduce - synthesize results from a map operation

Usage: reduce --from MAP_ID [OPTIONS] "question"

Options:
  --from ID           Map results ID (required)
  --strategy STRAT    Synthesis strategy (default: summarize)

Strategies:
  summarize       Narrative summary of findings
  table           Structured comparison table
  themes          Identify emerging themes
  consensus       Find points of agreement/disagreement
  bullet_points   Concise bullet-point summary
  extract         Extract specific data points

Examples:
  reduce --from m_abc123 --strategy summarize "What are the main findings?"
  reduce --from m_abc123 --strategy table "Compare methods and results"
  reduce --from m_abc123 --strategy themes "What themes emerge?" """,
    "ask-image": """ask-image - analyze a figure with vision

Usage: ask-image PATH "question"
       ask-image --list

Options:
  --fn describe       Describe the figure
  --fn extract-data   Extract data from the figure
  --list              List available figures (requires cd into paper)

Examples:
  ls /papers/PMC12345/figures/
  ask-image /papers/PMC12345/figures/fig1.jpg "What does this figure show?"
  ask-image /papers/PMC12345/figures/fig1.jpg --fn describe
  ask-image /papers/PMC12345/figures/fig1.jpg --fn extract-data""",
    "ask_image": """ask-image - analyze a figure with vision

Usage: ask-image PATH "question"
       ask-image --list

Options:
  --fn describe       Describe the figure
  --fn extract-data   Extract data from the figure
  --list              List available figures (requires cd into paper)

Examples:
  ask-image /papers/PMC12345/figures/fig1.jpg "What does this figure show?"
  ask-image /papers/PMC12345/figures/fig1.jpg --fn describe""",
    "wc": """wc - count lines, words, and characters

Usage: wc [OPTIONS] [FILE...]

Options:
  -l    Count lines only
  -w    Count words only
  -c    Count characters only

Examples:
  wc /papers/PMC12345/content.lines
  wc -l content.lines""",
    "export": """export - export SQL results to CSV + table artifact (up to 1K rows)

Usage: export [--desc "description"] "SELECT ..."

Runs a SQL query and saves results as:
  1. CSV file in session storage (for download)
  2. Table artifact (opens in the table visualization tab)

Returns an artifact_id that you MUST cite in your response using:
  {{"artifact": {{"artifact_id": "a_xxx", "type": "table", "source_count": N, "description": "one-sentence summary of what the table contains"}}}}

Options:
  --desc "text"    Human-readable description for the export

Examples:
  export "SELECT title, doi, authors FROM documents WHERE source = 'biorxiv' ORDER BY created_at DESC"
  export --desc "NIH-funded papers 2024" "SELECT DISTINCT d.title, d.doi FROM documents d JOIN content_blocks cb ON d.document_id = cb.document_id WHERE cb.content ILIKE '%NIH%'"
""",
    "sql": """sql - execute a read-only SQL query (fallback)

Usage: sql "SELECT ..."

Only SELECT statements are allowed. 15s timeout, 200-row limit enforced.
Use as a FALLBACK when search/lookup/grep can't express your query.

Tables:
  documents       — document_id (UUID), title, doi, source, authors, month_year,
                    abstract_text, created_at
  content_blocks  — id, document_id, line_number, content, section,
                    block_type, citation_info (JSONB)
  figures         — document_id, graphic, source_path

Examples:
  sql "SELECT COUNT(*) FROM documents WHERE source = 'biorxiv'"
  sql "SELECT month_year, COUNT(*) FROM documents GROUP BY month_year ORDER BY month_year DESC LIMIT 12"
  sql "SELECT d.title, d.doi FROM documents d JOIN content_blocks cb ON d.document_id = cb.document_id WHERE cb.content ILIKE '%CRISPR%' AND cb.section = 'Methods' GROUP BY d.document_id, d.title, d.doi LIMIT 10"
""",
}
