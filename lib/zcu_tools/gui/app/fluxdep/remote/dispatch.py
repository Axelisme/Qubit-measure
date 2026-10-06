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
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    # Type-only: a runtime import of the adapter would cycle (service.py imports
    # this module). String annotations keep pyright checking the call sites.
    from .service import RemoteControlAdapter

from zcu_tools.gui.project import ProjectInfo, is_real_project, project_info_payload
from zcu_tools.gui.remote.method_spec import BoundMethod, build_method_registry
from zcu_tools.gui.remote.readonly_handlers import (
    h_project_info,
    h_resources_versions,
)

from .dto import (
    ActiveSpectrumReply,
    AxisSnapshot,
    FitParametersReply,
    FitResultReply,
    NameReply,
    PointcloudReply,
    ProjectSetupReply,
    SelectionSnapshotReply,
    SpectrumListItem,
    SpectrumListReply,
    SpectrumRemovedReply,
    SpectrumSnapshotReply,
    StateCheckReply,
    TransitionWire,
)
from .method_specs import METHOD_SPECS

logger = logging.getLogger(__name__)

# Precise per-app handler alias (assignable to the shared, unconstrained
# ``method_spec.Handler``): every handler takes this app's RemoteControlAdapter.
Handler = Callable[["RemoteControlAdapter", Mapping[str, object]], Mapping[str, object]]


# ---------------------------------------------------------------------------
# Owner-thread handlers. Shared dispatch validates ParamSpec inputs and
# applies the declared observation policy before invoking these functions.
# ---------------------------------------------------------------------------


def _h_project_setup(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> ProjectSetupReply:
    chip_name = params["chip_name"]
    qub_name = params["qub_name"]
    result_dir = params["result_dir"]
    database_path = params["database_path"]
    # ParamSpec already validated these strings, including optional defaults.
    assert isinstance(chip_name, str)
    assert isinstance(qub_name, str)
    assert isinstance(result_dir, str)
    assert isinstance(database_path, str)
    adapter.ctrl.setup_project(
        ProjectInfo(
            chip_name=chip_name,
            qub_name=qub_name,
            result_dir=result_dir,
            database_path=database_path,
            root_dir=adapter.ctrl.get_project_root(),
        )
    )
    return {"project": project_info_payload(adapter.ctrl.state.project)}


def _h_spectrum_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> SpectrumListReply:
    del params
    spectrums = adapter.ctrl.state.spectrums
    return {
        "spectrums": [
            SpectrumListItem(
                name=entry.name,
                spec_type=entry.spec_type,
                aligned=entry.aligned,
                points_completed=entry.points_completed,
                point_count=entry.point_count,
            )
            for entry in spectrums.values()
        ]
    }


def _axis_snapshot(
    axis: NDArray[np.float64], unit: Literal["native", "Phi_0", "GHz"]
) -> AxisSnapshot:
    return {
        "count": int(axis.size),
        "minimum": float(axis.min()) if axis.size else None,
        "maximum": float(axis.max()) if axis.size else None,
        "unit": unit,
    }


def _h_spectrum_snapshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> SpectrumSnapshotReply:
    name = params["name"]
    assert isinstance(name, str)  # ParamSpec validated the nonempty literal name.
    entry = adapter.ctrl.state.spectrums.get(name)
    if entry is None:
        return {"name": name, "exists": False}
    return {
        "name": entry.name,
        "exists": True,
        "spec_type": entry.spec_type,
        "aligned": entry.aligned,
        "points_completed": entry.points_completed,
        "alignment_seeded": entry.alignment_seeded,
        "flux_half": entry.flux_half,
        "flux_int": entry.flux_int,
        "flux_period": entry.flux_period,
        "raw_axes": {
            "dev_values": _axis_snapshot(entry.raw["dev_values"], "native"),
            "fluxs": _axis_snapshot(entry.raw["fluxs"], "Phi_0"),
            "freqs": _axis_snapshot(entry.raw["freqs"], "GHz"),
            "signals_shape": list(entry.raw["signals"].shape),
        },
        "points": {
            "dev_values": entry.points["dev_values"].tolist(),
            "fluxs": entry.points["fluxs"].tolist(),
            "freqs": entry.points["freqs"].tolist(),
        },
    }


def _h_spectrum_remove(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> SpectrumRemovedReply:
    name = params["name"]
    assert isinstance(name, str)  # ParamSpec validated the nonempty literal name.
    adapter.ctrl.remove_spectrum(name)
    return {"name": name, "removed": True}


def _h_spectrum_set_active(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> ActiveSpectrumReply:
    name = params["name"]
    assert name is None or isinstance(name, str)  # ParamSpec applied null/default.
    adapter.ctrl.set_active_spectrum(name)
    return {"active_spectrum": adapter.ctrl.state.active_spectrum}


def _h_spectrum_reset_alignment(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> NameReply:
    name = params["name"]
    assert isinstance(name, str)  # ParamSpec validated the nonempty literal name.
    adapter.ctrl.reset_alignment(name)
    return {"name": name}


def _h_spectrum_reset_points(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> NameReply:
    name = params["name"]
    assert isinstance(name, str)  # ParamSpec validated the nonempty literal name.
    adapter.ctrl.reset_points(name)
    return {"name": name}


def _h_selection_snapshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> SelectionSnapshotReply:
    del params
    selection = adapter.ctrl.state.selection
    return {
        "selected": selection.selected.tolist()
        if selection.selected is not None
        else None,
        "min_distance": selection.min_distance,
    }


def _h_selection_pointcloud(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> PointcloudReply:
    del params
    fluxs, freqs = adapter.ctrl.derive_pointcloud()
    return {"fluxs": fluxs.tolist(), "freqs": freqs.tolist()}


def _h_fit_result(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> FitResultReply:
    del params
    fit = adapter.ctrl.state.fit
    params_payload: FitParametersReply | None = (
        {"EJ": fit.params[0], "EC": fit.params[1], "EL": fit.params[2]}
        if fit.params is not None
        else None
    )
    # transitions is a TypedDict with tuple values; lists serialise over JSON.
    transitions_payload: TransitionWire = {}
    for key, value in fit.transitions.items():
        if isinstance(value, list):
            transitions_payload[key] = [list(pair) for pair in value]
        elif key == "r_f":
            transitions_payload["r_f"] = value
        elif key == "sample_f":
            transitions_payload["sample_f"] = value
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
# State handler (app-specific; project.info + resources.versions are shared, see
# zcu_tools.gui.remote.readonly_handlers).
# ---------------------------------------------------------------------------


def _h_state_check(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> StateCheckReply:
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
    "project.info": h_project_info,
    "project.setup": _h_project_setup,
    "spectrum.list": _h_spectrum_list,
    "spectrum.snapshot": _h_spectrum_snapshot,
    "spectrum.remove": _h_spectrum_remove,
    "spectrum.set_active": _h_spectrum_set_active,
    "spectrum.reset_alignment": _h_spectrum_reset_alignment,
    "spectrum.reset_points": _h_spectrum_reset_points,
    "selection.snapshot": _h_selection_snapshot,
    "selection.pointcloud": _h_selection_pointcloud,
    "fit.result": _h_fit_result,
    "resources.versions": h_resources_versions,
    "state.check": _h_state_check,
}

METHOD_REGISTRY: dict[str, BoundMethod] = build_method_registry(_HANDLERS, METHOD_SPECS)
