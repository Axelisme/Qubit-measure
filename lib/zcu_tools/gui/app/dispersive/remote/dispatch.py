"""Method dispatcher for the dispersive RemoteControlAdapter.

Every handler is a pure synchronous function ``(adapter, params) -> dict`` that
runs on the Qt main thread. Handlers reach the dispersive command façade via
``adapter.ctrl`` (a ``Controller``). The whole method set is read-only — the agent
observes, the user drives the analysis in the GUI.

Adding a method:
  1. Implement ``def _h_<dotted_name>(adapter, params): ...`` (returns wire dict).
  2. Register it in ``_HANDLERS`` below; declare its contract in ``method_specs``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .service import RemoteControlAdapter

from zcu_tools.gui.app.analysis_project.entry import target_payload
from zcu_tools.gui.remote.method_spec import BoundMethod, build_method_registry
from zcu_tools.gui.remote.readonly_handlers import (
    h_resources_versions,
)

from .method_specs import METHOD_SPECS

logger = logging.getLogger(__name__)

# Precise per-app handler alias (assignable to the shared, unconstrained
# ``method_spec.Handler``): every handler takes this app's RemoteControlAdapter.
Handler = Callable[["RemoteControlAdapter", Mapping[str, object]], Mapping[str, object]]


# ---------------------------------------------------------------------------
# Read-only handlers — the agent observes, the user drives.
# ---------------------------------------------------------------------------


def _h_project_info(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    state = adapter.ctrl.state
    target = state.project
    return {
        **target_payload(target),
        "dispersive_path": str(target.resource.dispersive_path) if target else None,
        "predictor": state.predictor,
    }


def _h_fit_inputs_info(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    inputs = adapter.ctrl.state.fit_inputs
    if inputs is None:
        return {
            "has_inputs": False,
            "params": None,
            "flux": None,
            "bare_rf_seed": None,
        }
    ej, ec, el = inputs.params
    return {
        "has_inputs": True,
        "params": {"EJ": ej, "EC": ec, "EL": el},
        "flux": inputs.flux.to_dict(),
        "bare_rf_seed": inputs.bare_rf_seed,
    }


def _h_preprocess_status(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    pp = adapter.ctrl.state.preprocess
    if pp is None:
        return {"has_preprocess": False, "n_flux": 0, "n_freq": 0, "edelay": None}
    return {
        "has_preprocess": True,
        "n_flux": int(pp.norm_phases.shape[0]),
        "n_freq": int(pp.norm_phases.shape[1]),
        "edelay": float(pp.edelay),
    }


def _h_fit_result(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    fit = adapter.ctrl.state.disp_fit
    return {
        "has_result": fit.has_result,
        "g": fit.g,
        "bare_rf": fit.bare_rf,
        "res_dim": fit.res_dim,
    }


def _h_state_check(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    state = adapter.ctrl.state
    return {
        "has_project": state.project is not None,
        "has_fit_inputs": state.fit_inputs is not None,
        "has_onetone": state.onetone is not None,
        "has_preprocess": state.preprocess is not None,
        "has_result": state.disp_fit.has_result,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_HANDLERS: dict[str, Handler] = {
    "project.info": _h_project_info,
    "fit_inputs.info": _h_fit_inputs_info,
    "preprocess.status": _h_preprocess_status,
    "fit.result": _h_fit_result,
    "resources.versions": h_resources_versions,
    "state.check": _h_state_check,
}

METHOD_REGISTRY: dict[str, BoundMethod] = build_method_registry(_HANDLERS, METHOD_SPECS)
