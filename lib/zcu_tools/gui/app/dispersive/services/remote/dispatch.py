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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping

if TYPE_CHECKING:
    from .service import RemoteControlAdapter

from zcu_tools.gui.remote.param_spec import ParamSpec

from .method_specs import METHOD_SPECS, MethodSpec

logger = logging.getLogger(__name__)

Handler = Callable[["RemoteControlAdapter", Mapping[str, object]], Mapping[str, object]]


@dataclass(frozen=True)
class BoundMethod:
    handler: Handler
    spec: MethodSpec

    @property
    def timeout_seconds(self) -> float:
        return self.spec.timeout_seconds

    @property
    def params(self) -> tuple[ParamSpec, ...]:
        return self.spec.params

    @property
    def off_main_thread(self) -> bool:
        return self.spec.off_main_thread


# ---------------------------------------------------------------------------
# Read-only handlers — the agent observes, the user drives.
# ---------------------------------------------------------------------------


def _h_project_info(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    project = adapter.ctrl.state.project
    return {
        "chip_name": project.chip_name,
        "qub_name": project.qub_name,
        "result_dir": project.result_dir,
        "database_path": project.database_path,
    }


def _h_fit_inputs_info(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    inputs = adapter.ctrl.state.fit_inputs
    if inputs is None:
        return {
            "has_inputs": False,
            "params": None,
            "flux_half": None,
            "flux_int": None,
            "flux_period": None,
            "bare_rf_seed": None,
        }
    ej, ec, el = inputs.params
    return {
        "has_inputs": True,
        "params": {"EJ": ej, "EC": ec, "EL": el},
        "flux_half": inputs.flux_half,
        "flux_int": inputs.flux_int,
        "flux_period": inputs.flux_period,
        "bare_rf_seed": inputs.bare_rf_seed,
    }


def _h_preprocess_status(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
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
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    fit = adapter.ctrl.state.disp_fit
    return {
        "has_result": fit.has_result,
        "g": fit.g,
        "bare_rf": fit.bare_rf,
        "g_bound": list(fit.g_bound),
        "fit_bare_rf": fit.fit_bare_rf,
        "qub_dim": fit.qub_dim,
        "qub_cutoff": fit.qub_cutoff,
        "res_dim": fit.res_dim,
        "auto_fit_done": fit.auto_fit_done,
    }


def _h_resources_versions(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"versions": adapter.ctrl.state.version.snapshot()}


def _h_state_check(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    from zcu_tools.gui.app.dispersive.state import DEFAULT_CHIP, DEFAULT_QUBIT

    state = adapter.ctrl.state
    project = state.project
    # "has_project" means the user set a real chip/qubit — not the unknown_*
    # placeholders the project defaults to.
    has_project = bool(
        project.chip_name
        and project.qub_name
        and (project.chip_name, project.qub_name) != (DEFAULT_CHIP, DEFAULT_QUBIT)
    )
    return {
        "has_project": has_project,
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
    "resources.versions": _h_resources_versions,
    "state.check": _h_state_check,
}

# Every spec must have a handler and vice versa — fail fast on drift.
if set(_HANDLERS) != set(METHOD_SPECS):
    missing_spec = sorted(set(_HANDLERS) - set(METHOD_SPECS))
    missing_handler = sorted(set(METHOD_SPECS) - set(_HANDLERS))
    raise RuntimeError(
        "dispatch/method_specs drift — "
        f"handlers without spec: {missing_spec}; specs without handler: {missing_handler}"
    )

METHOD_REGISTRY: dict[str, BoundMethod] = {
    method: BoundMethod(handler=_HANDLERS[method], spec=METHOD_SPECS[method])
    for method in METHOD_SPECS
}
