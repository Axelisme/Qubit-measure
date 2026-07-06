"""Persistence memento types for autofluxdep-gui workflow state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from zcu_tools.gui.session.services.startup import PersistedStartup

APP_STATE_VERSION = 1


class PersistenceError(RuntimeError):
    """Expected failure while reading or writing autofluxdep GUI state."""


class PersistedNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    type_name: str
    name: str
    enabled: bool = True
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


class PersistedPredictorModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    EJ: float
    EC: float
    EL: float
    flux_half: float
    flux_period: float
    flux_bias: float = 0.0
    path: str | None = None


class PersistedPredictorDialogState(BaseModel):
    model_config = ConfigDict(frozen=True)

    tracked_transitions: tuple[tuple[int, int], ...] = (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
    )
    tab_index: int = 0
    params_path_text: str = ""


class PersistedUiPrefs(BaseModel):
    model_config = ConfigDict(frozen=True)

    auto_follow_tabs: bool = True
    predictor_dialog: PersistedPredictorDialogState = Field(
        default_factory=PersistedPredictorDialogState
    )


class AppPersistedState(BaseModel):
    """The on-disk autofluxdep GUI snapshot: workflow + flux sweep + UI prefs."""

    model_config = ConfigDict(frozen=True)

    version: int = APP_STATE_VERSION
    startup: PersistedStartup = Field(default_factory=PersistedStartup)
    workflow: PersistedWorkflow = Field(default_factory=PersistedWorkflow)
    flux: PersistedFluxSweep = Field(default_factory=PersistedFluxSweep)
    predictor: PersistedPredictorModel | None = None
    ui: PersistedUiPrefs = Field(default_factory=PersistedUiPrefs)


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
    restored_predictor: bool = False
    predictor_issue: RestoreIssue | None = None


__all__ = [
    "APP_STATE_VERSION",
    "AppPersistedState",
    "PersistedFluxSweep",
    "PersistedNode",
    "PersistedPredictorDialogState",
    "PersistedPredictorModel",
    "PersistedStartup",
    "PersistedUiPrefs",
    "PersistedWorkflow",
    "PersistenceError",
    "RestoreIssue",
    "RestoreReport",
]
