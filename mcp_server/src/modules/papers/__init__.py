"""Papers virtual filesystem module — canonical import path."""

from .filesystem import (
    PapersModule,
    PapersTerminal,
    PapersPathParser,
    PapersStore,
)
from .short_ids import resolve, shorten, shorten_result, shorten_results

__all__ = [
    "PapersModule",
    "PapersTerminal",
    "PapersPathParser",
    "PapersStore",
    "resolve",
    "shorten",
    "shorten_result",
    "shorten_results",
]
