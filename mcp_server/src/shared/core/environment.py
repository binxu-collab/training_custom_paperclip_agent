"""
Environment management for GXL services.

This module provides utilities for loading environment variables from .env files
based on the current environment (dev, staging, prod).
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

# Try to import GCP Secret Manager (optional dependency for cloud environments)
try:
    from google.cloud import secretmanager

    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False


class Environment(str, Enum):
    """Environment types."""

    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"


def get_environment() -> Environment:
    """
    Get current environment from GXL_ENVIRONMENT variable.

    Returns:
        Environment: The current environment (dev, staging, or prod)

    Raises:
        ValueError: If GXL_ENVIRONMENT is not set or invalid
    """
    env = os.getenv("GXL_ENVIRONMENT", "").lower()

    # Map 'production' to 'prod' if user types the full word
    if env == "production":
        env = "prod"

    # Default to dev if not set
    if not env:
        env = "dev"

    if env not in [e.value for e in Environment]:
        valid_envs = [e.value for e in Environment]
        raise ValueError(f"GXL_ENVIRONMENT must be one of {valid_envs}. Got: '{env}'")

    return Environment(env)


def get_project_id() -> str:
    """Get GCP project ID for current environment."""
    # Allow override via environment variable
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if project_id:
        return project_id

    # Default project IDs by environment
    # Maps dev/staging -> gxl-dev, prod -> gxl-prod
    return "gxl-prod" if get_environment() == Environment.PRODUCTION else "gxl-dev"


def is_production() -> bool:
    """Check if running in production."""
    try:
        return get_environment() == Environment.PRODUCTION
    except ValueError:
        return False


def is_cloud_environment() -> bool:
    """Check if running in cloud (staging or production)."""
    try:
        return get_environment() in [Environment.STAGING, Environment.PRODUCTION]
    except ValueError:
        return False


logger = logging.getLogger(__name__)


def load_secrets_from_gcp(secret_mappings: dict[str, str] | None = None):
    """
    Load secrets from GCP Secret Manager at runtime.

    This function is designed for Cloud Run deployments where secrets are stored
    in GCP Secret Manager and loaded at container startup via service account authentication.

    Args:
        secret_mappings: Optional dict mapping environment variable names to GCP secret names.
                        If None, uses default mappings for common secrets.

    Behavior:
        - Only runs in staging/production environments (skips dev)
        - Fetches secrets in parallel for faster startup
        - Requires GOOGLE_CLOUD_PROJECT environment variable
        - Requires service account with secretmanager.secretAccessor role

    Example:
        In Cloud Run, secrets are named: anthropic-api-key, openrouter-api-key, etc.
        The same secrets are used across all environments (staging/prod).
    """
    if not GOOGLE_CLOUD_AVAILABLE:
        logger.warning(
            "⚠ google-cloud-secret-manager not available, skipping GCP secret loading"
        )
        return

    env = get_environment()

    # Skip for local development
    if env == Environment.DEV:
        logger.debug("Dev environment detected, skipping GCP secret loading")
        return

    project_id = get_project_id()
    if not project_id:
        logger.warning("⚠ GOOGLE_CLOUD_PROJECT not set, cannot load secrets from GCP")
        return

    # Default secret mappings (env var name -> GCP secret name)
    # GCP secrets use UPPERCASE_SNAKE_CASE naming convention
    if secret_mappings is None:
        secret_mappings = {
            "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY": "OPENROUTER_API_KEY",
            "OPENAI_API_KEY": "OPENAI_API_KEY",
            "GEMINI_API_KEY": "GEMINI_API_KEY",
            "MCP_API_KEY": "MCP_API_KEY",
            "LANGFUSE_SECRET_KEY": "LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY": "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_BASE_URL": "LANGFUSE_BASE_URL",
            "INFERENCE_DB_URL": "INFERENCE_DB_URL",
            "FDA_DOCS_DB_URL": "FDA_DOCS_DB_URL",
            "CTGOV_DB_URL": "CTGOV_DB_URL",
            "VITE_FIREBASE_API_KEY": "VITE_FIREBASE_API_KEY",
            "E2B_API_KEY": "E2B_API_KEY",
            "SANDBOX_GCS_SA_KEY": "SANDBOX_GCS_SA_KEY",
        }

    logger.info(
        f"Loading {len(secret_mappings)} secrets from GCP Secret Manager "
        f"(project: {project_id}, env: {env.value}, parallel fetch)..."
    )

    client = secretmanager.SecretManagerServiceClient()

    def fetch_secret(
        env_var: str, secret_name: str
    ) -> tuple[str, str | None, str | None]:
        """Fetch a single secret, skipping if Cloud Run already injected it."""
        # Cloud Run injects secrets as env vars via --set-secrets before the
        # container process starts. Re-fetching them is redundant and adds
        # startup latency + 504 risk under load. Skip the network call when
        # the value is already present.
        if existing := os.environ.get(env_var):
            logger.debug(f"✓ {env_var} already set (skipping Secret Manager fetch)")
            return (env_var, existing, None)
        try:
            secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": secret_path})
            value = response.payload.data.decode("UTF-8")
            return (env_var, value, None)
        except Exception as e:
            return (env_var, None, str(e))

    # Fetch secrets in parallel for faster startup
    success_count = 0
    failed_secrets = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_secret, env_var, secret_name): env_var
            for env_var, secret_name in secret_mappings.items()
        }

        for future in as_completed(futures):
            env_var, value, error = future.result()
            if error:
                logger.warning(f"✗ Failed to load {env_var}: {error}")
                failed_secrets.append(env_var)
            else:
                if value is not None:
                    os.environ[env_var] = value
                    success_count += 1
                    logger.debug(f"✓ Loaded {env_var}")

    logger.info(
        f"✓ Loaded {success_count}/{len(secret_mappings)} secrets from GCP Secret Manager"
    )

    if failed_secrets:
        logger.warning(
            f"⚠ Failed to load {len(failed_secrets)} secrets: {', '.join(failed_secrets)}"
        )


def load_environment():
    """
    Load environment variables based on GXL_ENVIRONMENT.

    Behavior:
    - For dev environment: Loads from .env file (local development)
    - For staging/prod: Loads secrets from GCP Secret Manager (Cloud Run)

    This function is called at service startup to bootstrap configuration.
    """
    try:
        env = get_environment()

        if env == Environment.DEV:
            # Local development: Load from .env file
            env_filename = ".env"

            # Search for config file in possible locations
            possible_roots = [
                Path("/workspaces/gxl"),  # Dev container
                Path(__file__).parents[5],  # Local relative path from this file
                Path(os.getcwd()),  # Current working directory
            ]

            env_file_path = None
            for root in possible_roots:
                candidate = root / env_filename
                if candidate.exists():
                    env_file_path = candidate
                    break

            if env_file_path:
                # Force override to ensure .env values take precedence
                load_dotenv(env_file_path, override=True)
                logger.info(f"✓ Loaded config from {env_file_path}")
            else:
                logger.error(
                    "⚠ .env file is required for local development. "
                    "Please create it from .env.example "
                    "(or copy .env.dev.example to .env if it exists)"
                )
        else:
            # Cloud environments (staging/prod): Load from GCP Secret Manager
            logger.info(
                f"Cloud environment ({env.value}) detected, loading secrets from GCP Secret Manager"
            )
            load_secrets_from_gcp()

    except ValueError as e:
        logger.warning(f"⚠ Environment configuration error: {e}")
        # For dev environment, always require .env
        env_value = os.getenv("GXL_ENVIRONMENT", "dev").lower()
        if env_value == "dev":
            logger.error(
                "⚠ Invalid GXL_ENVIRONMENT. For local development, ensure GXL_ENVIRONMENT=dev "
                "and .env file exists."
            )
        else:
            logger.warning(
                "⚠ Could not determine environment, trying GCP secret loading as fallback"
            )
            load_secrets_from_gcp()


# =============================================================================
# Service URL Configuration
# =============================================================================
# All service URLs are read from environment variables with sensible defaults
# for the unified deployment (localhost with different ports).
#
# For devcontainer/docker-compose, set the env vars to use container names.
#
# Default ports (unified deployment):
#   - Widgets:              8000 (externally exposed)
#   - Inference Engine:     8001
#   - FDA MCP:              8082
#   - CTGov MCP:            8084
#   - Protein Design MCP:   8081
# =============================================================================


def get_inference_port() -> int:
    """Get inference engine port from environment."""
    return int(os.getenv("GXL_INFERENCE_PORT", "8000"))


def get_widgets_port() -> int:
    """Get widgets frontend port from environment."""
    return int(os.getenv("GXL_WIDGETS_PORT", "8002"))


def get_inference_url() -> str:
    """Get inference engine URL from environment.

    Reads from INFERENCE_ENGINE_URL if set, otherwise constructs from GXL_INFERENCE_PORT.
    """
    if url := os.getenv("INFERENCE_ENGINE_URL"):
        return url
    return f"http://localhost:{get_inference_port()}"


def get_widgets_url() -> str:
    """Get widgets service URL from environment.

    Reads from WIDGETS_URL if set, otherwise constructs from GXL_WIDGETS_PORT.
    """
    if url := os.getenv("WIDGETS_URL"):
        return url
    return f"http://localhost:{get_widgets_port()}"


def get_fda_mcp_url() -> str:
    """Get FDA MCP server URL from environment."""
    if url := os.getenv("FDA_MCP_URL"):
        return url
    port = os.getenv("GXL_FDA_MCP_PORT", "8082")
    return f"http://localhost:{port}"


def get_ctgov_mcp_url() -> str:
    """Get CTGov MCP server URL from environment."""
    if url := os.getenv("CTGOV_MCP_URL"):
        return url
    port = os.getenv("GXL_CTGOV_MCP_PORT", "8084")
    return f"http://localhost:{port}"


def get_protein_design_mcp_url() -> str:
    """Get Protein Design MCP server URL from environment."""
    if url := os.getenv("PROTEIN_DESIGN_MCP_URL"):
        return url
    port = os.getenv("GXL_PROTEIN_DESIGN_MCP_PORT", "8081")
    return f"http://localhost:{port}"


def get_all_mcp_urls() -> dict[str, str]:
    """Get all MCP server URLs as a dict."""
    return {
        "fda": get_fda_mcp_url(),
        "ctgov": get_ctgov_mcp_url(),
        "protein_design": get_protein_design_mcp_url(),
    }


# =============================================================================
# Timeout and Resource Limit Helpers
# =============================================================================


def get_timeout(name: str, default: float) -> float:
    """
    Get a timeout value from environment with fallback.

    Args:
        name: Environment variable name
        default: Default value if not set

    Returns:
        Timeout value as float
    """
    return float(os.getenv(name, str(default)))


def get_resource_limit(name: str, default: int) -> int:
    """
    Get a resource limit from environment with fallback.

    Args:
        name: Environment variable name
        default: Default value if not set

    Returns:
        Resource limit as int
    """
    return int(os.getenv(name, str(default)))
