"""
MCE Task Environments.

This module provides the environment registry and base classes.
Tasks define interface signatures, MCE validates agent implementations.
"""

from env.base import (
    InterfaceSignature,
    Sample,
    EnvironmentResult,
    TaskEnvironment,
)
from env.registry import EnvironmentRegistry

__all__ = [
    "InterfaceSignature",
    "Sample", 
    "EnvironmentResult",
    "TaskEnvironment",
    "EnvironmentRegistry",
]
