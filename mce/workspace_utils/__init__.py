"""Workspace utilities for MCE agents.

This package provides helper functions and utilities that are copied to each
workspace and can be used by agents during context learning.

Available modules:
- llm: LLM calling utilities with structured output
- embedding: Embedding and similarity computation utilities
- validate_base: Validation script for base agent implementation
- validate_meta: Validation script for meta agent implementation
"""

__all__ = [
    "llm",
    "embedding",
    "validate_base",
    "validate_meta",
]

