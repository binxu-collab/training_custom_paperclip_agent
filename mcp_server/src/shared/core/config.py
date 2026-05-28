"""
Configuration management for the MCP server with E2B sandbox integration.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from shared.core.environment import get_resource_limit, get_timeout

logger = logging.getLogger(__name__)


@dataclass
class E2BConfig:
    """E2B sandbox API configuration."""

    api_key: str | None = None
    default_template: str = "gxl-sandbox-gcs"
    timeout_seconds: int = 300
    max_file_size_mb: int = 10
    gcs_bucket_name: str | None = None
    gcs_template: str | None = None

    def __post_init__(self):
        # Try to get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv("E2B_API_KEY")
        if not self.gcs_bucket_name:
            self.gcs_bucket_name = os.getenv(
                "SANDBOX_GCS_BUCKET_NAME", "gxl_sandbox_staging"
            )
        if not self.gcs_template:
            self.gcs_template = os.getenv("E2B_GCS_TEMPLATE", "gxl-sandbox-gcs")


@dataclass
class CodeSandboxConfig:
    """CodeSandbox API configuration (DEPRECATED - use E2BConfig)."""

    api_key: str | None = None
    base_url: str = "https://codesandbox.io/api/v1"
    default_template: str = "protein-1"
    timeout_seconds: int = 30
    max_file_size_mb: int = 10

    def __post_init__(self):
        # Try to get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv("CODESANDBOX_API_KEY")


@dataclass
class SessionConfig:
    """Session management configuration."""

    timeout_seconds: int = 1800  # 30 minutes
    cleanup_interval_seconds: int = 120  # 2 minutes
    max_sessions_per_instance: int = 10000


@dataclass
class ServerConfig:
    """Overall server configuration."""

    log_level: str = "INFO"
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8080")))
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))


@dataclass
class SearchConfig:
    """Search tools configuration."""

    ncbi_api_key: str | None = field(default_factory=lambda: os.getenv("NCBI_API_KEY"))
    ncbi_email: str = field(
        default_factory=lambda: os.getenv("NCBI_EMAIL", "researcher@example.com")
    )
    scrapingdog_api_key: str | None = field(
        default_factory=lambda: os.getenv("SCRAPINGDOG_API_KEY")
    )
    pubmed_rate_limit: int = 3  # requests per second (10 with API key)
    patents_enabled: bool = True
    scholar_enabled: bool = True


@dataclass
class AuthConfig:
    """Authentication configuration."""

    require_api_key: bool = field(
        default_factory=lambda: os.getenv("REQUIRE_API_KEY", "true").lower() == "true"
    )
    allowed_api_keys: str = field(
        default_factory=lambda: os.getenv("ALLOWED_API_KEYS", "")
    )
    rate_limit_per_key: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_KEY", "1000"))
    )
    enable_rate_limiting: bool = field(
        default_factory=lambda: os.getenv("ENABLE_RATE_LIMITING", "false").lower()
        == "true"
    )
    log_auth_attempts: bool = field(
        default_factory=lambda: os.getenv("LOG_AUTH_ATTEMPTS", "true").lower() == "true"
    )


@dataclass
class MCPConfig:
    """Complete MCP server configuration."""

    e2b: E2BConfig = field(default_factory=E2BConfig)
    codesandbox: CodeSandboxConfig = field(
        default_factory=CodeSandboxConfig
    )  # Deprecated
    session: SessionConfig = field(default_factory=SessionConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)

    # External API configurations
    tavily_api_key: str | None = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY")
    )
    tamarind_api_key: str | None = field(
        default_factory=lambda: os.getenv("TAMARIND_API_KEY")
    )
    neurosnap_api_key: str | None = field(
        default_factory=lambda: os.getenv("NEUROSNAP_API_KEY")
    )
    use_cached_neurosnap_jobs: bool = field(
        default_factory=lambda: os.getenv("USE_CACHED_NEUROSNAP_JOBS", "false").lower()
        == "true"
    )

    # Response management configuration
    max_response_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_RESPONSE_TOKENS", "12500"))
    )

    @classmethod
    def from_file(cls, config_path: str) -> "MCPConfig":
        """Load configuration from a JSON file."""
        try:
            with open(config_path) as f:
                config_data = json.load(f)

            return cls(
                e2b=E2BConfig(**config_data.get("e2b", {})),
                codesandbox=CodeSandboxConfig(**config_data.get("codesandbox", {})),
                session=SessionConfig(**config_data.get("session", {})),
                server=ServerConfig(**config_data.get("server", {})),
                search=SearchConfig(**config_data.get("search", {})),
                auth=AuthConfig(**config_data.get("auth", {})),
            )
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file {config_path}: {e}")
            return cls()
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return cls()

    @classmethod
    def from_env(cls) -> "MCPConfig":
        """Load configuration from environment variables."""
        return cls(
            e2b=E2BConfig(
                api_key=os.getenv("E2B_API_KEY"),
                default_template=os.getenv("E2B_DEFAULT_TEMPLATE", "python"),
                timeout_seconds=int(
                    float(
                        os.getenv(
                            "E2B_TIMEOUT_SECONDS",
                            str(int(get_timeout("E2B_TIMEOUT_SECONDS", 300.0))),
                        )
                    )
                ),
                max_file_size_mb=int(os.getenv("E2B_MAX_FILE_SIZE_MB", "10")),
                gcs_bucket_name=os.getenv(
                    "SANDBOX_GCS_BUCKET_NAME", "gxl_sandbox_staging"
                ),
                gcs_template=os.getenv("E2B_GCS_TEMPLATE"),
            ),
            codesandbox=CodeSandboxConfig(
                api_key=os.getenv("CODESANDBOX_API_KEY"),
                base_url=os.getenv(
                    "CODESANDBOX_BASE_URL", "https://codesandbox.io/api/v1"
                ),
                default_template=os.getenv("CODESANDBOX_DEFAULT_TEMPLATE", "protein-1"),
                timeout_seconds=int(
                    float(
                        os.getenv(
                            "CODESANDBOX_TIMEOUT_SECONDS",
                            str(int(get_timeout("CODESANDBOX_TIMEOUT_SECONDS", 30.0))),
                        )
                    )
                ),
                max_file_size_mb=int(os.getenv("CODESANDBOX_MAX_FILE_SIZE_MB", "10")),
            ),
            session=SessionConfig(
                timeout_seconds=int(
                    float(
                        os.getenv(
                            "SESSION_TIMEOUT_SECONDS",
                            str(int(get_timeout("SESSION_TIMEOUT_SECONDS", 3600.0))),
                        )
                    )
                ),
                cleanup_interval_seconds=int(
                    float(
                        os.getenv(
                            "SESSION_CLEANUP_INTERVAL_SECONDS",
                            str(
                                int(
                                    get_timeout(
                                        "SESSION_CLEANUP_INTERVAL_SECONDS", 300.0
                                    )
                                )
                            ),
                        )
                    )
                ),
                max_sessions_per_instance=int(
                    float(
                        os.getenv(
                            "MAX_SESSIONS_PER_INSTANCE",
                            str(get_resource_limit("MAX_SESSIONS_PER_INSTANCE", 10000)),
                        )
                    )
                ),
            ),
            server=ServerConfig(
                log_level=os.getenv("LOG_LEVEL", "INFO"),
                port=int(os.getenv("PORT", "8080")),
                host=os.getenv("HOST", "0.0.0.0"),
            ),
            search=SearchConfig(
                ncbi_api_key=os.getenv("NCBI_API_KEY"),
                ncbi_email=os.getenv("NCBI_EMAIL", "researcher@example.com"),
                scrapingdog_api_key=os.getenv("SCRAPINGDOG_API_KEY"),
                pubmed_rate_limit=10 if os.getenv("NCBI_API_KEY") else 3,
                patents_enabled=os.getenv("PATENTS_ENABLED", "true").lower() == "true",
                scholar_enabled=os.getenv("SCHOLAR_ENABLED", "true").lower() == "true",
            ),
            auth=AuthConfig(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "e2b": {
                "api_key": "***" if self.e2b.api_key else None,  # Don't expose API key
                "default_template": self.e2b.default_template,
                "timeout_seconds": self.e2b.timeout_seconds,
                "max_file_size_mb": self.e2b.max_file_size_mb,
                "gcs_bucket_name": self.e2b.gcs_bucket_name,
                "gcs_template": self.e2b.gcs_template,
            },
            "codesandbox": {
                "api_key": (
                    "***" if self.codesandbox.api_key else None
                ),  # Don't expose API key
                "base_url": self.codesandbox.base_url,
                "default_template": self.codesandbox.default_template,
                "timeout_seconds": self.codesandbox.timeout_seconds,
                "max_file_size_mb": self.codesandbox.max_file_size_mb,
            },
            "session": {
                "timeout_seconds": self.session.timeout_seconds,
                "cleanup_interval_seconds": self.session.cleanup_interval_seconds,
                "max_sessions_per_instance": self.session.max_sessions_per_instance,
            },
            "server": {
                "log_level": self.server.log_level,
                "port": self.server.port,
                "host": self.server.host,
            },
            "search": {
                "ncbi_api_key": "***" if self.search.ncbi_api_key else None,
                "ncbi_email": self.search.ncbi_email,
                "scrapingdog_api_key": (
                    "***" if self.search.scrapingdog_api_key else None
                ),
                "pubmed_rate_limit": self.search.pubmed_rate_limit,
                "patents_enabled": self.search.patents_enabled,
                "scholar_enabled": self.search.scholar_enabled,
            },
            "external_apis": {
                "tavily_api_key": "***" if self.tavily_api_key else None,
                "tamarind_api_key": "***" if self.tamarind_api_key else None,
                "neurosnap_api_key": "***" if self.neurosnap_api_key else None,
            },
            "auth": {
                "require_api_key": self.auth.require_api_key,
                "allowed_api_keys": "***" if self.auth.allowed_api_keys else None,
                "rate_limit_per_key": self.auth.rate_limit_per_key,
                "enable_rate_limiting": self.auth.enable_rate_limiting,
                "log_auth_attempts": self.auth.log_auth_attempts,
            },
        }

    def save_to_file(self, config_path: str) -> bool:
        """Save configuration to a JSON file (without API keys)."""
        try:
            config_data = {
                "e2b": {
                    "default_template": self.e2b.default_template,
                    "timeout_seconds": self.e2b.timeout_seconds,
                    "max_file_size_mb": self.e2b.max_file_size_mb,
                    "gcs_bucket_name": self.e2b.gcs_bucket_name,
                    "gcs_template": self.e2b.gcs_template,
                },
                "codesandbox": {
                    "base_url": self.codesandbox.base_url,
                    "default_template": self.codesandbox.default_template,
                    "timeout_seconds": self.codesandbox.timeout_seconds,
                    "max_file_size_mb": self.codesandbox.max_file_size_mb,
                },
                "session": {
                    "timeout_seconds": self.session.timeout_seconds,
                    "cleanup_interval_seconds": self.session.cleanup_interval_seconds,
                    "max_sessions_per_instance": self.session.max_sessions_per_instance,
                },
                "server": {
                    "log_level": self.server.log_level,
                    "port": self.server.port,
                    "host": self.server.host,
                },
                "search": {
                    "ncbi_email": self.search.ncbi_email,
                    "pubmed_rate_limit": self.search.pubmed_rate_limit,
                    "patents_enabled": self.search.patents_enabled,
                    "scholar_enabled": self.search.scholar_enabled,
                },
                "auth": {
                    "require_api_key": self.auth.require_api_key,
                    "rate_limit_per_key": self.auth.rate_limit_per_key,
                    "enable_rate_limiting": self.auth.enable_rate_limiting,
                    "log_auth_attempts": self.auth.log_auth_attempts,
                },
            }

            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
            return False


# Global configuration instance
_config: MCPConfig | None = None


def get_config() -> MCPConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        # Try to load from config file first, then environment variables
        config_file = os.getenv("MCP_CONFIG_FILE", "mcp_config.json")
        _config = (
            MCPConfig.from_file(config_file)
            if os.path.exists(config_file)
            else MCPConfig.from_env()
        )

        # Set up logging level
        logging.getLogger().setLevel(
            getattr(logging, _config.server.log_level.upper(), logging.INFO)
        )

    return _config


def set_config(config: MCPConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config

    # Update logging level
    logging.getLogger().setLevel(
        getattr(logging, config.server.log_level.upper(), logging.INFO)
    )


def validate_config() -> dict[str, Any]:
    """Validate the current configuration and return status."""
    config = get_config()
    status = {"valid": True, "warnings": [], "errors": []}

    # Check E2B configuration
    if not config.e2b.api_key:
        status["warnings"].append(
            "E2B API key not configured - sandbox features will be limited"
        )

    if config.e2b.timeout_seconds <= 0:
        status["errors"].append("E2B timeout must be positive")
        status["valid"] = False

    if config.e2b.max_file_size_mb <= 0:
        status["errors"].append("E2B max file size must be positive")
        status["valid"] = False

    # Check legacy CodeSandbox configuration (deprecated)
    if config.codesandbox.api_key:
        status["warnings"].append(
            "CodeSandbox configuration is deprecated - please migrate to E2B"
        )

    if config.codesandbox.timeout_seconds <= 0:
        status["errors"].append("CodeSandbox timeout must be positive")
        status["valid"] = False

    if config.codesandbox.max_file_size_mb <= 0:
        status["errors"].append("CodeSandbox max file size must be positive")
        status["valid"] = False

    # Check session configuration
    if config.session.timeout_seconds <= 0:
        status["errors"].append("Session timeout must be positive")
        status["valid"] = False

    if config.session.cleanup_interval_seconds <= 0:
        status["errors"].append("Session cleanup interval must be positive")
        status["valid"] = False

    if config.session.max_sessions_per_instance <= 0:
        status["errors"].append("Max sessions per instance must be positive")
        status["valid"] = False

    # Check search API keys
    if not config.search.ncbi_api_key:
        status["warnings"].append(
            "NCBI API key not configured - PubMed search rate limited to 3/sec"
        )

    if not config.search.scrapingdog_api_key:
        if config.search.patents_enabled:
            status["warnings"].append(
                "ScrapingDog API key not configured - Google Patents search will be disabled"
            )
        if config.search.scholar_enabled:
            status["warnings"].append(
                "ScrapingDog API key not configured - Google Scholar search will be disabled"
            )
    else:
        status["info"] = status.get("info", [])
        status["info"].append(
            "Using ScrapingDog API for Google Scholar and Patents search"
        )

    # Check external API keys
    if not config.tavily_api_key:
        status["warnings"].append(
            "Tavily API key not configured - search features will be limited"
        )

    if not config.tamarind_api_key:
        status["warnings"].append(
            "Tamarind API key not configured - Tamarind.bio tools will be limited"
        )

    if not config.neurosnap_api_key:
        status["warnings"].append(
            "Neurosnap API key not configured - Neurosnap.ai tools will be limited"
        )

    return status
