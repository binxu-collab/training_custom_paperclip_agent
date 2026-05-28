"""
Authentication module for the MCP server.
Provides API key validation and authentication utilities.
"""

import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AuthConfig:
    """Authentication configuration."""

    require_api_key: bool = field(
        default_factory=lambda: os.getenv("REQUIRE_API_KEY", "true").lower() == "true"
    )
    allowed_api_keys: set[str] = field(default_factory=set)
    rate_limit_per_key: int = 1000  # requests per hour
    enable_rate_limiting: bool = False
    log_auth_attempts: bool = True

    def __post_init__(self):
        # Load API keys from environment variable
        # For dev: Set in .env file
        # For staging/prod: Injected by Cloud Run secret mounting
        if not self.allowed_api_keys:
            keys_str = os.getenv("ALLOWED_API_KEYS", "")
            if keys_str:
                self.allowed_api_keys = {key.strip() for key in keys_str.split(",") if key.strip()}
                logger.info(f"Loaded {len(self.allowed_api_keys)} API keys from environment")


class ApiKeyValidator:
    """Handles API key validation and rate limiting."""

    def __init__(self, auth_config: AuthConfig):
        self.config = auth_config
        self._rate_limit_data: dict[str, dict[str, Any]] = {}

    def validate_api_key(self, api_key: str | None, client_info: str = "unknown") -> bool:
        """
        Validate an API key using timing-safe comparison.

        Args:
            api_key: The API key to validate
            client_info: Client information for logging

        Returns:
            True if the API key is valid, False otherwise
        """
        if not self.config.require_api_key:
            return True

        if not api_key:
            if self.config.log_auth_attempts:
                logger.warning(f"Authentication failed - no API key provided from {client_info}")
            return False

        if not self.config.allowed_api_keys:
            logger.error("No allowed API keys configured - all requests will be rejected")
            return False

        # Use timing-safe comparison to prevent timing attacks
        is_valid = any(
            self._secure_compare(api_key, allowed_key)
            for allowed_key in self.config.allowed_api_keys
        )

        if self.config.log_auth_attempts:
            key_hash = self._hash_api_key(api_key)
            if is_valid:
                logger.info(f"Authentication successful - API key {key_hash} from {client_info}")
            else:
                logger.warning(
                    f"Authentication failed - invalid API key {key_hash} from {client_info}"
                )

        # Check rate limiting if enabled
        if is_valid and self.config.enable_rate_limiting:
            return self._check_rate_limit(api_key)

        return is_valid

    def _secure_compare(self, provided_key: str, allowed_key: str) -> bool:
        """Perform timing-safe string comparison."""
        return hmac.compare_digest(provided_key, allowed_key)

    def _hash_api_key(self, api_key: str) -> str:
        """Create a hash of the API key for safe logging."""
        return hashlib.sha256(api_key.encode()).hexdigest()[:8]

    def _check_rate_limit(self, api_key: str) -> bool:
        """Check if the API key has exceeded rate limits."""
        current_time = time.time()
        current_hour = int(current_time // 3600)

        if api_key not in self._rate_limit_data:
            self._rate_limit_data[api_key] = {"hour": current_hour, "count": 1}
            return True

        key_data = self._rate_limit_data[api_key]

        # Reset counter if we're in a new hour
        if key_data["hour"] != current_hour:
            key_data["hour"] = current_hour
            key_data["count"] = 1
            return True

        # Check if limit exceeded
        if key_data["count"] >= self.config.rate_limit_per_key:
            logger.warning(f"Rate limit exceeded for API key {self._hash_api_key(api_key)}")
            return False

        # Increment counter
        key_data["count"] += 1
        return True

    def get_rate_limit_info(self, api_key: str) -> dict[str, Any]:
        """Get rate limit information for an API key."""
        if not self.config.enable_rate_limiting or api_key not in self._rate_limit_data:
            return {
                "limit": self.config.rate_limit_per_key,
                "remaining": self.config.rate_limit_per_key,
                "reset_time": int(time.time() // 3600 + 1) * 3600,
            }

        key_data = self._rate_limit_data[api_key]
        current_hour = int(time.time() // 3600)

        if key_data["hour"] != current_hour:
            remaining = self.config.rate_limit_per_key
        else:
            remaining = max(0, self.config.rate_limit_per_key - key_data["count"])

        return {
            "limit": self.config.rate_limit_per_key,
            "remaining": remaining,
            "reset_time": (current_hour + 1) * 3600,
        }


# Global validator instance
_validator: ApiKeyValidator | None = None


def get_api_validator() -> ApiKeyValidator:
    """Get the global API key validator instance."""
    global _validator
    if _validator is None:
        auth_config = AuthConfig()
        _validator = ApiKeyValidator(auth_config)
    return _validator


def set_api_validator(validator: ApiKeyValidator) -> None:
    """Set the global API key validator instance."""
    global _validator
    _validator = validator


def validate_api_key(api_key: str | None, client_info: str = "unknown") -> bool:
    """Convenience function to validate an API key."""
    return get_api_validator().validate_api_key(api_key, client_info)


def is_authentication_required() -> bool:
    """Check if API key authentication is required."""
    return get_api_validator().config.require_api_key


def is_api_key_auth_enabled() -> bool:
    """Check if static API key authentication is allowed in this environment.

    API key auth is enabled for dev and staging to support direct token-based
    access (e.g. from Cursor MCP config). Disabled in production where only
    Firebase/OAuth tokens are accepted.
    """
    try:
        from shared.core.environment import get_environment, Environment
        return get_environment() != Environment.PRODUCTION
    except Exception:
        return True


def get_rate_limit_info(api_key: str) -> dict[str, Any]:
    """Get rate limit information for an API key."""
    return get_api_validator().get_rate_limit_info(api_key)
