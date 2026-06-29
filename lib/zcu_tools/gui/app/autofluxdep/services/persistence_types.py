"""Persistence memento types for autofluxdep-gui workflow state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

APP_STATE_VERSION = 1


class PersistenceError(RuntimeError):
    """Expected failure while reading or writing autofluxdep GUI state."""


class PersistedNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    type_name: str
    name: str
    cfg_raw: dict[str, Any] = Field(default_factory=dict)


class PersistedWorkflow(BaseModel):
    model_config = ConfigDict(frozen=True)

    nodes: tuple[PersistedNode, ...] = ()


class PersistedFluxSweep(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_expr: str = "2e-3"
    stop_expr: str = "-0.2e-3"
    npts_expr: str = "101"
    values: tuple[float, ...] = ()


class AppPersistedState(BaseModel):
    """The on-disk autofluxdep GUI snapshot: workflow + flux sweep only."""

    model_config = ConfigDict(frozen=True)

    version: int = APP_STATE_VERSION
    workflow: PersistedWorkflow = Field(default_factory=PersistedWorkflow)
    flux: PersistedFluxSweep = Field(default_factory=PersistedFluxSweep)


@dataclass(frozen=True)
class RestoreIssue:
    """One rejected persisted item during workflow restore."""

    subject: str
    message: str


@dataclass(frozen=True)
class RestoreReport:
    """Outcome of applying a persisted autofluxdep workflow snapshot."""

    restored_nodes: int
    rejected_nodes: tuple[RestoreIssue, ...] = ()


__all__ = [
    "APP_STATE_VERSION",
    "AppPersistedState",
    "PersistedFluxSweep",
    "PersistedNode",
    "PersistedWorkflow",
    "PersistenceError",
    "RestoreIssue",
    "RestoreReport",
]
