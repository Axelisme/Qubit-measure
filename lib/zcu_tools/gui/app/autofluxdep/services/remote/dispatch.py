"""Method dispatcher for the autofluxdep RemoteControlAdapter.

Every handler is a pure synchronous function ``(adapter, params) -> Mapping`` that
runs on the Qt main thread. The adapter layer is responsible for marshalling —
handlers must not touch threading or Qt directly. Handlers reach the autofluxdep
command façade via ``adapter.ctrl`` (a ``Controller``).

The whole surface is READ-ONLY: the agent observes a workflow the user drives.
There are deliberately NO mutating handlers (add-node / set-flux / run / stop):
those are user actions in the GUI. Building the node graph and judging the live
fits need the human's eye on the plot, which the agent does not have.

Reading the run-lived ``run_results`` on the main thread is safe even mid-run: the
worker only fills pre-allocated numpy rows in place (not a State semantic write),
so the handler sees a consistent snapshot (rows not yet measured stay nan).

Adding a method:
  1. Implement ``def _h_<dotted_name>(adapter, params): ...`` (returns wire dict).
  2. Register it in ``_HANDLERS`` below; declare its contract in ``method_specs``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    # Type-only: a runtime import of the adapter would cycle (service.py imports
    # this module). String annotations keep pyright checking the call sites.
    from .service import RemoteControlAdapter

from zcu_tools.gui.app.autofluxdep.nodes.result import (
    QubitFreqResult,
    Sweep1DResult,
    Sweep2DResult,
)
from zcu_tools.gui.app.autofluxdep.state import AutoFluxDepState
from zcu_tools.gui.project import DEFAULT_CHIP, DEFAULT_QUBIT
from zcu_tools.gui.remote.method_spec import BoundMethod, build_method_registry
from zcu_tools.gui.remote.readonly_handlers import h_resources_versions

from .method_specs import METHOD_SPECS

logger = logging.getLogger(__name__)

# Precise per-app handler alias (assignable to the shared, unconstrained
# ``method_spec.Handler``): every handler takes this app's RemoteControlAdapter.
Handler = Callable[["RemoteControlAdapter", Mapping[str, object]], Mapping[str, object]]


# ---------------------------------------------------------------------------
# Project — note autofluxdep's ProjectInfo differs from gui.project.ProjectInfo
# (it also carries ``params_path`` and may be None), so the shared
# project_info_payload / is_real_project helpers do NOT apply; only the
# DEFAULT_CHIP / DEFAULT_QUBIT placeholder constants are reused.
# ---------------------------------------------------------------------------


def _has_real_project(state: AutoFluxDepState) -> bool:
    """Whether the user has set a real chip/qubit project (not the placeholders)."""
    project = state.project
    return bool(
        project is not None
        and project.chip_name
        and project.qub_name
        and (project.chip_name, project.qub_name) != (DEFAULT_CHIP, DEFAULT_QUBIT)
    )


def _h_project_info(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    project = adapter.ctrl.state.project
    if project is None:
        return {
            "chip_name": None,
            "qub_name": None,
            "result_dir": None,
            "database_path": None,
            "params_path": None,
        }
    return {
        "chip_name": project.chip_name,
        "qub_name": project.qub_name,
        "result_dir": project.result_dir,
        "database_path": project.database_path,
        "params_path": project.params_path,
    }


# ---------------------------------------------------------------------------
# Workflow definition
# ---------------------------------------------------------------------------


def _h_workflow_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    state = adapter.ctrl.state
    results = state.run_results
    return {
        "nodes": [
            {
                "name": node.name,
                "type": node.type_name,
                "enabled": node.enabled,
                "provides": list(node.provides),
                "provides_modules": list(node.provides_modules),
                "requires": [dep.key for dep in node.all_dependencies()],
                "has_result": node.name in results,
            }
            for node in state.nodes
        ]
    }


def _h_node_cfg(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # ``name`` is already validated as a required non-empty string by the service
    # (it runs validate_params against the method's ParamSpec before dispatch).
    name = params["name"]
    for node in adapter.ctrl.state.nodes:
        if node.name == name:
            return {
                "name": node.name,
                "type": node.type_name,
                "knobs": node.schema.read_knobs(),
            }
    raise KeyError(f"No placed node named {name!r}")


# ---------------------------------------------------------------------------
# Per-node run results — progress summary only, never the raw 2D arrays.
# ---------------------------------------------------------------------------


def _count_measured(per_flux: Any) -> int:
    """Number of flux rows already measured = the non-nan entries of a (n_flux,) fit."""
    return int(np.count_nonzero(~np.isnan(np.asarray(per_flux, dtype=np.float64))))


def _last_finite(per_flux: Any) -> float | None:
    """The last non-nan value of a (n_flux,) fit array, or None if none measured yet."""
    arr = np.asarray(per_flux, dtype=np.float64)
    finite = arr[~np.isnan(arr)]
    return float(finite[-1]) if finite.size else None


def _summarise_result(result: Any) -> dict[str, Any]:
    """A tiny, JSON-friendly progress + fit summary for one node's Result.

    Reports the result kind, n_flux, n_measured (rows whose primary fit is filled)
    and a minimal ``fit_summary`` (a count + the latest fitted scalar) — never the
    raw 2D signal arrays (those are large and the agent does not plot).
    """
    if isinstance(result, QubitFreqResult):
        n_measured = _count_measured(result.fit_freq)
        return {
            "kind": "qubit_freq",
            "n_flux": result.n_flux,
            "n_measured": n_measured,
            "fit_summary": {
                "n_fitted": n_measured,
                "last_fit_freq": _last_finite(result.fit_freq),
            },
        }
    if isinstance(result, Sweep1DResult):
        n_measured = _count_measured(result.fit_value)
        return {
            "kind": "sweep1d",
            "n_flux": result.n_flux,
            "n_measured": n_measured,
            "fit_summary": {
                "x_label": result.x_label,
                "n_fitted": n_measured,
                "last_fit_value": _last_finite(result.fit_value),
            },
        }
    if isinstance(result, Sweep2DResult):
        n_measured = _count_measured(result.best_freq)
        return {
            "kind": "sweep2d",
            "n_flux": result.n_flux,
            "n_measured": n_measured,
            "fit_summary": {
                "n_fitted": n_measured,
                "last_best_freq": _last_finite(result.best_freq),
                "last_best_gain": _last_finite(result.best_gain),
            },
        }
    # The run only allocates the three Result dataclasses above; anything else is a
    # contract breach (a new Result type added without teaching this summary).
    raise TypeError(f"Unknown result type {type(result).__name__} in run_results")


def _h_result_summary(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    results = adapter.ctrl.state.run_results
    return {
        "results": [
            {"name": name, **_summarise_result(result)}
            for name, result in results.items()
        ]
    }


# ---------------------------------------------------------------------------
# State handler (app-specific; resources.versions is shared, see
# zcu_tools.gui.remote.readonly_handlers; project.info is app-local because
# autofluxdep's ProjectInfo shape differs from the shared one).
# ---------------------------------------------------------------------------


def _h_state_check(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    state = adapter.ctrl.state
    return {
        "has_project": _has_real_project(state),
        "has_soc": state.exp_context.has_soc(),
        "node_count": len(state.nodes),
        "flux_count": len(state.flux_values),
        "has_flux_device": state.flux_device_name is not None,
        "is_running": adapter.ctrl.is_running,
        "has_results": bool(state.run_results),
        # Predictor flags (D4): only presence, no calibration curves. The loaded
        # raw FluxoniumPredictor lives in exp_context; the adaptive per-run
        # predictor (Simple stand-in or Fluxonium-backed) is run-lived state.
        "has_loaded_predictor": state.exp_context.predictor is not None,
        "has_run_predictor": state.run_predictor is not None,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_HANDLERS: dict[str, Handler] = {
    "project.info": _h_project_info,
    "workflow.list": _h_workflow_list,
    "node.cfg": _h_node_cfg,
    "result.summary": _h_result_summary,
    "resources.versions": h_resources_versions,
    "state.check": _h_state_check,
}

METHOD_REGISTRY: dict[str, BoundMethod] = build_method_registry(_HANDLERS, METHOD_SPECS)
