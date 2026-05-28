"""
Virtual Filesystem Module - Generalized document exploration framework.

This module provides a reusable framework for building virtual filesystem interfaces
over document collections. It can be extended for different document types:
- BioMedRxiv papers
- FDA documents
- Clinical trial records
- Any document collection with structured content

Key Components:
- VirtualFilesystemModule: Base class with parallel execution, caching, reduce strategies
- PathParser: Configurable path parsing for different document structures
- DocumentStore: Abstract interface for document storage backends
- ParallelExecutor: Subagent-based parallel task execution
"""

from .base import VirtualFilesystemModule, PathParser, DocumentStore
from .parallel import ParallelExecutor, ReduceStrategies
from .cache import ResultsRegistry

__all__ = [
    "VirtualFilesystemModule",
    "PathParser",
    "DocumentStore",
    "ParallelExecutor",
    "ReduceStrategies",
    "ResultsRegistry",
]
