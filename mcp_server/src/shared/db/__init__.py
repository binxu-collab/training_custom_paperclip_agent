"""Shared database connection utilities for MCP tools."""

from .cloud_sql import (
    get_cloud_sql_connection,
    get_cloud_sql_connector,
    close_cloud_sql_connector,
    get_fda_docs_connection,
    get_ctgov_connection,
    CloudSQLConfig,
    FDA_DOCS_CONFIG,
    CTGOV_CONFIG,
)

__all__ = [
    "get_cloud_sql_connection",
    "get_cloud_sql_connector", 
    "close_cloud_sql_connector",
    "get_fda_docs_connection",
    "get_ctgov_connection",
    "CloudSQLConfig",
    "FDA_DOCS_CONFIG",
    "CTGOV_CONFIG",
]

