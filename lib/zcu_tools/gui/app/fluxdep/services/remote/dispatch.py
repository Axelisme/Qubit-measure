"""Method dispatcher for the fluxdep RemoteControlAdapter.

Every handler is a pure synchronous function ``(adapter, params) -> dict`` that
runs on the Qt main thread. The adapter layer is responsible for marshalling —
handlers must not touch threading or Qt directly. Handlers reach the fluxdep
command façade via ``adapter.ctrl`` (a ``Controller``).

Adding a method:
  1. Implement ``def _h_<dotted_name>(adapter, params): ...`` (returns wire dict).
  2. Register it in ``_HANDLERS`` below; declare its contract in ``method_specs``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only: a runtime import of the adapter would cycle (service.py imports
    # this module). String annotations keep pyright checking the call sites.
    from .service import RemoteControlAdapter

from zcu_tools.gui.project import is_real_project, project_info_payload
from zcu_tools.gui.remote.method_spec import BoundMethod, build_method_registry

from .method_specs import METHOD_SPECS

logger = logging.getLogger(__name__)

# Precise per-app handler alias (assignable to the shared, unconstrained
# ``method_spec.Handler``): every handler takes this app's RemoteControlAdapter.
Handler = Callable[["RemoteControlAdapter", Mapping[str, object]], Mapping[str, object]]


# ---------------------------------------------------------------------------
# Read-only handlers — the agent observes, the user drives.
#
# Every handler here is a pure query that runs on the Qt main thread and returns
# a wire dict. There are deliberately NO mutating handlers (load / align / pick
# points / select / fit / export): those are user actions in the GUI. Point
# picking and axis-orientation judgement need the human's eye on the preview,
# which the agent does not have, so driving them over RPC was removed.
# ---------------------------------------------------------------------------


def _h_project_info(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return project_info_payload(adapter.ctrl.state.project)


def _h_spectrum_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    spectrums = adapter.ctrl.state.spectrums
    return {
        "spectrums": [
            {
                "name": entry.name,
                "spec_type": entry.spec_type,
                "aligned": bool(entry.aligned),
                "points_selected": bool(entry.points_selected),
            }
            for entry in spectrums.values()
        ]
    }


def _h_selection_pointcloud(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    fluxs, freqs = adapter.ctrl.derive_pointcloud()
    return {"fluxs": fluxs.tolist(), "freqs": freqs.tolist()}


def _h_fit_result(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    fit = adapter.ctrl.state.fit
    params_payload = (
        {"EJ": fit.params[0], "EC": fit.params[1], "EL": fit.params[2]}
        if fit.params is not None
        else None
    )
    # transitions is a TypedDict with tuple values; lists serialise over JSON.
    transitions_payload = {
        key: [list(p) for p in value] if isinstance(value, list) else value
        for key, value in fit.transitions.items()
    }
    return {
        "has_result": fit.has_result,
        "params": params_payload,
        "database_path": fit.database_path,
        "EJb": list(fit.EJb),
        "ECb": list(fit.ECb),
        "ELb": list(fit.ELb),
        "transitions": transitions_payload,
        "r_f": fit.r_f,
        "sample_f": fit.sample_f,
    }


# ---------------------------------------------------------------------------
# Resource versions / state handlers
# ---------------------------------------------------------------------------


def _h_resources_versions(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"versions": adapter.ctrl.state.version.snapshot()}


def _h_state_check(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    state = adapter.ctrl.state
    return {
        "has_project": is_real_project(state.project),
        "spectrum_count": len(state.spectrums),
        "has_active": state.active_spectrum is not None,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_HANDLERS: dict[str, Handler] = {
    "project.info": _h_project_info,
    "spectrum.list": _h_spectrum_list,
    "selection.pointcloud": _h_selection_pointcloud,
    "fit.result": _h_fit_result,
    "resources.versions": _h_resources_versions,
    "state.check": _h_state_check,
}

METHOD_REGISTRY: dict[str, BoundMethod] = build_method_registry(_HANDLERS, METHOD_SPECS)
