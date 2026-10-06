"""Shared GUI interactive analysis contracts (no Qt or domain policy)."""

from .plugin import Action, Command, PluginDefinition
from .session import Session

__all__ = ["Action", "Command", "PluginDefinition", "Session"]
