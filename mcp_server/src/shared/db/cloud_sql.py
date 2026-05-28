"""Cloud SQL connection utilities using IAM authentication.

This module provides database connections to Cloud SQL instances using
the Cloud SQL Python Connector with IAM authentication. No passwords needed.

Environment Variables:
    GOOGLE_IMPERSONATE_SERVICE_ACCOUNT: Optional service account to impersonate

Instance Configuration:
    FDA_DOCS_INSTANCE: Cloud SQL instance for FDA Docs (default: gxl-dev:us-central1:gxl-fda)
    CTGOV_INSTANCE: Cloud SQL instance for CTGov (default: gxl-dev:us-central1:gxl-ctgov)

Fallback (optional - for local dev without GCP auth):
    FDA_DOCS_DB_URL: Direct PostgreSQL URL for FDA Docs
    CTGOV_DB_URL: Direct PostgreSQL URL for CTGov
"""

import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

logger = logging.getLogger(__name__)

# Default statement timeout (2 minutes) applied to all connections.
# Prevents any single query from running indefinitely.
_DEFAULT_STATEMENT_TIMEOUT_MS = 120_000


def ensure_ssl_mode(db_url: str, default_mode: str = "disable") -> str:
    """Ensure the database URL has sslmode set.

    If sslmode is not specified in the URL, appends ?sslmode=require.
    This is needed for Cloud SQL connections that require SSL.

    Args:
        db_url: PostgreSQL connection URL
        default_mode: SSL mode to use if not specified (default: "require")

    Returns:
        URL with sslmode parameter
    """
    if not db_url:
        return db_url

    # Check if sslmode is already in the URL
    if "sslmode=" in db_url:
        return db_url

    # Add sslmode parameter
    if "?" in db_url:
        return f"{db_url}&sslmode={default_mode}"
    else:
        return f"{db_url}?sslmode={default_mode}"


# Global Cloud SQL connector singleton
_cloud_sql_connector = None


@dataclass
class CloudSQLConfig:
    """Configuration for a Cloud SQL connection."""

    instance_connection_name: str
    database: str
    user: str
    use_iam_auth: bool = True


# Default configurations for each database
FDA_DOCS_CONFIG = CloudSQLConfig(
    instance_connection_name=os.getenv(
        "FDA_DOCS_INSTANCE", "gxl-chat:us-central1:gxl-fda"
    ),
    database=os.getenv("FDA_DOCS_DATABASE", "fda_docs"),
    user=os.getenv("FDA_DOCS_USER", "postgres"),
)

CTGOV_CONFIG = CloudSQLConfig(
    instance_connection_name=os.getenv(
        "CTGOV_INSTANCE", "gxl-dev:us-central1:gxl-ctgov"
    ),
    database=os.getenv("CTGOV_DATABASE", "postgres"),
    user=os.getenv("CTGOV_USER", "postgres"),
)

# FDA Metadata database (OpenFDA) - uses direct URL connection only
FDA_METADATA_CONFIG = CloudSQLConfig(
    instance_connection_name=os.getenv(
        "FDA_METADATA_INSTANCE", "gxl-chat:us-central1:gxl-fda-metadata"
    ),
    database=os.getenv("FDA_METADATA_DATABASE", "openfda_v"),
    user=os.getenv("FDA_METADATA_USER", "postgres"),
)

# BioMedRxiv database (bioRxiv + medRxiv preprints)
BIOMEDRXIV_CONFIG = CloudSQLConfig(
    instance_connection_name=os.getenv(
        "BIOMEDRXIV_INSTANCE", "gxl-prod:us-central1:biomedrxiv"
    ),
    database=os.getenv("BIOMEDRXIV_DATABASE", "biomedrxiv"),
    user=os.getenv("BIOMEDRXIV_USER", "postgres"),
)


class Pg8000CursorWrapper:
    """Wrapper to make pg8000 cursor work as context manager."""

    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self._cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cursor.close()
        return False

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class Pg8000ConnectionWrapper:
    """Wrapper to make pg8000 connection compatible with psycopg2 interface."""

    def __init__(self, conn):
        self._conn = conn
        self.autocommit = False

    def cursor(self, cursor_factory=None):
        """Return a wrapped cursor. cursor_factory is ignored (pg8000 doesn't support it)."""
        return Pg8000CursorWrapper(self._conn.cursor())

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    @property
    def closed(self):
        """Check if connection is closed."""
        # pg8000 doesn't have a closed property, so we try a simple operation
        try:
            # Check if the connection's socket is still valid
            return self._conn._sock is None
        except Exception:
            return True

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()
        return False


def get_cloud_sql_connector():
    """Get or create Cloud SQL connector singleton with IAM authentication."""
    global _cloud_sql_connector

    if _cloud_sql_connector is None:
        try:
            import google.auth
            from google.auth import impersonated_credentials
            from google.auth.transport.requests import Request
            from google.cloud.sql.connector import Connector

            # Get base credentials
            base_credentials, project = google.auth.default()

            # Refresh base credentials first
            if hasattr(base_credentials, "refresh"):
                base_credentials.refresh(Request())

            # Check if we should impersonate a service account
            impersonate_sa = os.environ.get("GOOGLE_IMPERSONATE_SERVICE_ACCOUNT")
            if impersonate_sa:
                logger.info(f"Using impersonated credentials for: {impersonate_sa}")
                credentials = impersonated_credentials.Credentials(
                    source_credentials=base_credentials,
                    target_principal=impersonate_sa,
                    target_scopes=["https://www.googleapis.com/auth/sqlservice.admin"],
                )
            else:
                credentials = base_credentials

            _cloud_sql_connector = Connector(credentials=credentials)
            logger.info("Cloud SQL connector initialized with IAM authentication")
        except ImportError as e:
            logger.error(f"Failed to import Cloud SQL connector: {e}")
            logger.error("Install with: pip install cloud-sql-python-connector[pg8000]")
            raise

    return _cloud_sql_connector


def get_cloud_sql_connection(config: CloudSQLConfig) -> Pg8000ConnectionWrapper:
    """Create a connection to Cloud SQL using the Python Connector.

    Args:
        config: CloudSQLConfig with instance name, database, and user

    Returns:
        Pg8000ConnectionWrapper that's compatible with psycopg2 interface
    """
    connector = get_cloud_sql_connector()

    conn_kwargs = {
        "user": config.user,
        "db": config.database,
    }

    if config.use_iam_auth:
        conn_kwargs["enable_iam_auth"] = True

    logger.info(
        f"Connecting to Cloud SQL: {config.instance_connection_name}/{config.database}"
    )

    conn = connector.connect(config.instance_connection_name, "pg8000", **conn_kwargs)

    # Set default statement_timeout to prevent runaway queries
    cursor = conn.cursor()
    cursor.execute(f"SET statement_timeout = {_DEFAULT_STATEMENT_TIMEOUT_MS}")
    cursor.close()

    logger.info(f"✓ Connected to Cloud SQL: {config.instance_connection_name}")
    return Pg8000ConnectionWrapper(conn)


def close_cloud_sql_connector() -> None:
    """Close the global Cloud SQL connector (cleanup on shutdown)."""
    global _cloud_sql_connector

    if _cloud_sql_connector is not None:
        logger.info("Closing Cloud SQL connector")
        _cloud_sql_connector.close()
        _cloud_sql_connector = None


def is_cloud_environment() -> bool:
    """Check if running in a cloud environment (staging/prod)."""
    env = os.environ.get("GXL_ENVIRONMENT", "dev").lower()
    return env in ("staging", "prod")


def get_fda_docs_connection():
    """Get a connection to the FDA Docs database.

    Priority:
    1. FDA_DOCS_DB_URL if set (direct connection with password)
    2. Cloud SQL connector with IAM auth
    """
    # Check for direct URL first (recommended for now - IAM auth requires DB setup)
    if url := os.getenv("FDA_DOCS_DB_URL"):
        logger.info("Using FDA_DOCS_DB_URL for direct connection")
        import psycopg2

        # Ensure SSL mode is set for Cloud SQL connections
        url = ensure_ssl_mode(url)
        conn = psycopg2.connect(
            url,
            options=f"-c statement_timeout={_DEFAULT_STATEMENT_TIMEOUT_MS}",
        )
        conn.autocommit = True
        return conn

    # Fall back to Cloud SQL connector (requires IAM database auth enabled)
    logger.info("No FDA_DOCS_DB_URL set, trying Cloud SQL connector with IAM auth...")
    return get_cloud_sql_connection(FDA_DOCS_CONFIG)


def get_ctgov_connection():
    """Get a connection to the CTGov database.

    Priority:
    1. CTGOV_DB_URL if set (direct connection with password)
    2. Cloud SQL connector with IAM auth
    """
    # Check for direct URL first (recommended for now - IAM auth requires DB setup)
    if url := os.getenv("CTGOV_DB_URL"):
        logger.info("Using CTGOV_DB_URL for direct connection")
        import psycopg2

        # Ensure SSL mode is set for Cloud SQL connections
        url = ensure_ssl_mode(url)
        conn = psycopg2.connect(
            url,
            options=f"-c statement_timeout={_DEFAULT_STATEMENT_TIMEOUT_MS}",
        )
        conn.autocommit = True
        return conn

    # Fall back to Cloud SQL connector (requires IAM database auth enabled)
    logger.info("No CTGOV_DB_URL set, trying Cloud SQL connector with IAM auth...")
    return get_cloud_sql_connection(CTGOV_CONFIG)


def get_trials_connection():
    """Get a connection to the Trials database.

    The Trials DB is the same underlying Cloud SQL instance as CTGov
    (everything is in one physical database: postgres @ gxl-ctgov). It
    contains all clinical-trial registry data (ctgov/ct_jpn/ct_eu/ct_cn),
    all non-US FDA-equivalent regulatory data (fda_jpn/fda_eu/fda_cn),
    and unified document views (public.all_documents /
    public.all_content_blocks) that the Trials VFS queries.

    Priority:
    1. TRIALS_DB_URL if set (allows pointing at a dedicated replica)
    2. CTGOV_DB_URL (same physical DB)
    3. Cloud SQL connector with IAM auth (falls back to CTGOV_CONFIG)
    """
    if url := (os.getenv("TRIALS_DB_URL") or os.getenv("CTGOV_DB_URL")):
        logger.info("Using TRIALS_DB_URL/CTGOV_DB_URL for direct connection")
        import psycopg2

        url = ensure_ssl_mode(url)
        conn = psycopg2.connect(
            url,
            options=f"-c statement_timeout={_DEFAULT_STATEMENT_TIMEOUT_MS}",
        )
        conn.autocommit = True
        return conn

    logger.info("No TRIALS_DB_URL set, trying Cloud SQL connector with IAM auth...")
    return get_cloud_sql_connection(CTGOV_CONFIG)


def get_fda_metadata_connection():
    """Get a connection to the FDA Metadata (OpenFDA) database.

    Priority:
    1. FDA_METADATA_DB_URL if set (direct connection with password)
    2. Cloud SQL connector with IAM auth

    This database contains OpenFDA drug/device metadata including:
    - drug_drugsfda: Drugs@FDA application data
    - device_510k: 510(k) premarket notifications
    - device_pma: Premarket approvals
    - And other FDA datasets

    Note: Uses the 'openfda_v' schema by default.
    """
    # Check for direct URL first (recommended for now - IAM auth requires DB setup)
    if url := os.getenv("FDA_METADATA_DB_URL"):
        logger.info("Using FDA_METADATA_DB_URL for direct connection")
        import psycopg2

        # Ensure SSL mode is set for Cloud SQL connections
        url = ensure_ssl_mode(url)
        conn = psycopg2.connect(
            url,
            options=f"-c statement_timeout={_DEFAULT_STATEMENT_TIMEOUT_MS}",
        )
        conn.autocommit = True

        # Set search_path to openfda_v schema
        with conn.cursor() as cur:
            cur.execute("SET search_path TO openfda_v, public")
        logger.info("Set search_path to openfda_v schema")
        return conn

    # Fall back to Cloud SQL connector (requires IAM database auth enabled)
    logger.info(
        "No FDA_METADATA_DB_URL set, trying Cloud SQL connector with IAM auth..."
    )
    conn = get_cloud_sql_connection(FDA_METADATA_CONFIG)
    # Set search_path to openfda_v schema
    with conn.cursor() as cur:
        cur.execute("SET search_path TO openfda_v, public")
    logger.info("Set search_path to openfda_v schema")
    return conn


def get_biomedrxiv_connection():
    """Get a connection to the BioMedRxiv database.

    Priority:
    1. BIOMEDRXIV_DB_URL if set (direct connection with password)
    2. Direct connection using BIOMEDRXIV_DB_* env vars
    3. Cloud SQL connector with IAM auth

    This database contains bioRxiv and medRxiv preprint articles:
    - documents: 453K+ article metadata
    - content_blocks: 70M+ full-text content blocks
    """
    # Check for direct URL first
    if url := os.getenv("BIOMEDRXIV_DB_URL"):
        logger.info("Using BIOMEDRXIV_DB_URL for direct connection")
        import psycopg2

        url = ensure_ssl_mode(url)
        conn = psycopg2.connect(
            url,
            options=f"-c statement_timeout={_DEFAULT_STATEMENT_TIMEOUT_MS}",
        )
        conn.autocommit = True
        return conn

    # Check for direct connection parameters
    if os.getenv("BIOMEDRXIV_DB_PASSWORD"):
        logger.info("Using BIOMEDRXIV_DB_* parameters for direct connection")
        import psycopg2

        conn = psycopg2.connect(
            host=os.getenv("BIOMEDRXIV_DB_HOST", "35.225.183.0"),
            port=int(os.getenv("BIOMEDRXIV_DB_PORT", "5432")),
            database=os.getenv("BIOMEDRXIV_DB_NAME", "biomedrxiv"),
            user=os.getenv("BIOMEDRXIV_DB_USER", "postgres"),
            password=os.getenv("BIOMEDRXIV_DB_PASSWORD"),
            options=f"-c statement_timeout={_DEFAULT_STATEMENT_TIMEOUT_MS}",
        )
        conn.autocommit = True
        return conn

    # Fall back to Cloud SQL connector
    logger.info(
        "No BIOMEDRXIV_DB_URL set, trying Cloud SQL connector with IAM auth..."
    )
    return get_cloud_sql_connection(BIOMEDRXIV_CONFIG)
