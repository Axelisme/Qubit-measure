"""Method dispatcher for the fluxdep RemoteControlAdapter.

Every handler is a pure synchronous function ``(adapter, params) -> dict`` that
runs on the State owner except operation.await, which only waits on a
thread-safe SearchOwner handle off-owner. Shared dispatch owns marshalling;
handlers must not touch threading or Qt directly. Handlers reach the fluxdep
command façade via ``adapter.ctrl`` (a ``Controller``).

Adding a method:
  1. Implement ``def _h_<dotted_name>(adapter, params): ...`` (returns wire dict).
  2. Register it in ``_HANDLERS`` below; declare its contract in ``method_specs``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    # Type-only: a runtime import of the adapter would cycle (service.py imports
    # this module). String annotations keep pyright checking the call sites.
    from .service import RemoteControlAdapter

from zcu_tools.analysis.fluxdep.models import TransitionDict
from zcu_tools.gui.project import ProjectInfo, is_real_project, project_info_payload
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
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
    FitUpdatedReply,
    NameReply,
    NamesReply,
    OperationAwaitReply,
    OperationCancelledReply,
    OperationOutcomeReply,
    OperationStartedReply,
    OperationStatusReply,
    ParamsExportedReply,
    PointcloudReply,
    ProjectSetupReply,
    SelectionSnapshotReply,
    SpectrumListItem,
    SpectrumListReply,
    SpectrumRemovedReply,
    SpectrumsExportedReply,
    SpectrumSnapshotReply,
    StateCheckReply,
    TransitionWire,
)
from .interactive import (
    h_interactive_read,
    h_selection_interactive_command,
    h_selection_interactive_open,
    h_spectrum_interactive_command,
    h_spectrum_interactive_open,
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


def _h_spectrum_load(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> NameReply:
    filepath = params["filepath"]
    spec_type = params["spec_type"]
    inherit_from = params["inherit_from"]
    transpose_axes = params["transpose_axes"]
    # ParamSpec validated types, enum and optional defaults.
    assert isinstance(filepath, str)
    assert isinstance(spec_type, str)
    assert spec_type == "OneTone" or spec_type == "TwoTone"
    assert inherit_from is None or isinstance(inherit_from, str)
    assert isinstance(transpose_axes, bool)
    name = adapter.ctrl.load_spectrum(filepath, spec_type, inherit_from, transpose_axes)
    return {"name": name}


def _h_spectrum_load_processed(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> NamesReply:
    filepath = params["filepath"]
    assert isinstance(filepath, str)  # ParamSpec validated the nonempty path.
    return {"names": adapter.ctrl.load_processed_spectrums(filepath)}


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


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"'{label}' must be a finite number"
        )
    try:
        number = float(value)
    except OverflowError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"'{label}' must be representable as a float"
        ) from exc
    if not math.isfinite(number):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"'{label}' must be a finite number"
        )
    return number


def _fit_bounds(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"'{label}' must be a two-element number list"
        )
    return _finite_number(value[0], label), _finite_number(value[1], label)


def _fit_transitions(value: object) -> TransitionDict:
    if not isinstance(value, dict):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'transitions' must be an object")
    transitions: TransitionDict = {}
    for key, group in value.items():
        if not isinstance(key, str):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "'transitions' keys must be strings"
            )
        if key == "r_f":
            transitions["r_f"] = _finite_number(group, "transitions.r_f")
        elif key == "sample_f":
            transitions["sample_f"] = _finite_number(group, "transitions.sample_f")
        else:
            if not isinstance(group, list):
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS, f"'transitions.{key}' must be a pair list"
                )
            pairs: list[tuple[int, int]] = []
            for pair in group:
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or isinstance(pair[0], bool)
                    or not isinstance(pair[0], int)
                    or isinstance(pair[1], bool)
                    or not isinstance(pair[1], int)
                ):
                    raise RemoteError(
                        ErrorCode.INVALID_PARAMS,
                        f"'transitions.{key}' must contain two-element integer lists",
                    )
                pairs.append((pair[0], pair[1]))
            transitions[key] = pairs
    return transitions


def _h_fit_set_params(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> FitUpdatedReply:
    database_path = params["database_path"]
    assert isinstance(database_path, str)  # ParamSpec validated the nonempty path.
    # Parse the complete request before publishing. Domain interpretation of
    # bounds/categories and separate frequency precedence stays in the kernel.
    EJb = _fit_bounds(params["EJb"], "EJb")
    ECb = _fit_bounds(params["ECb"], "ECb")
    ELb = _fit_bounds(params["ELb"], "ELb")
    transitions = _fit_transitions(params["transitions"])
    r_f = _finite_number(params["r_f"], "r_f") if params["r_f"] is not None else None
    sample_f = (
        _finite_number(params["sample_f"], "sample_f")
        if params["sample_f"] is not None
        else None
    )
    adapter.ctrl.set_fit_params(
        database_path, EJb, ECb, ELb, transitions, r_f, sample_f
    )
    return {"fit": _h_fit_result(adapter, {})}


def _h_fit_search(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> OperationStartedReply:
    del params
    return {"token": adapter.ctrl.search.start()}


def _h_operation_status(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> OperationStatusReply:
    token = params["token"]
    search = adapter.ctrl.search
    if token is None:
        activity = search.current
        return {
            "activity": (
                {
                    "token": activity.token,
                    "status": activity.status,
                    "error": activity.error,
                }
                if activity is not None
                else None
            )
        }
    assert isinstance(token, int)  # ParamSpec rejects booleans/non-integers.
    outcome = search.outcome(token)
    return {
        "activity": {
            "token": token,
            "status": outcome.status if outcome is not None else "pending",
            "error": outcome.error if outcome is not None else None,
        }
    }


def _h_operation_cancel(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> OperationCancelledReply:
    token = params["token"]
    assert isinstance(token, int)
    adapter.ctrl.search.cancel(token)
    return {"token": token, "cancel_requested": True}


def _h_operation_await(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> OperationAwaitReply:
    # This is the only off-owner handler. Do not read State, current activity,
    # versions or observations; SearchOwner owns the thread-safe wait.
    token = params["token"]
    assert isinstance(token, int)
    timeout = _finite_number(params["timeout"], "timeout")
    if not 0.0 <= timeout <= 30.0:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "'timeout' must be between 0 and 30 seconds"
        )
    result = adapter.ctrl.search.await_outcome(token, timeout)
    outcome: OperationOutcomeReply | None = None
    if result.outcome is not None:
        # The native wait returns only terminal outcomes; the shared outcome
        # type also represents pending for other users, so narrow that union.
        status = result.outcome.status
        assert status != "pending"
        outcome = {"status": status, "error": result.outcome.error}
    return {
        "token": token,
        "reason": result.reason,
        "outcome": outcome,
        "feedback": result.feedback,
    }


def _h_export_spectrums(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> SpectrumsExportedReply:
    filepath = params["filepath"]
    overwrite = params["overwrite"]
    # ParamSpec supplied optional defaults; only the native create/replace
    # modes are exposed, not arbitrary h5py modes.
    assert filepath is None or isinstance(filepath, str)
    assert isinstance(overwrite, bool)
    return {
        "filepath": adapter.ctrl.export_spectrums(
            filepath, mode="w" if overwrite else "x"
        )
    }


def _h_fit_export_params(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> ParamsExportedReply:
    savepath = params["savepath"]
    assert savepath is None or isinstance(savepath, str)
    return {"savepath": adapter.ctrl.export_params(savepath)}


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
    "spectrum.load": _h_spectrum_load,
    "spectrum.load_processed": _h_spectrum_load_processed,
    "spectrum.remove": _h_spectrum_remove,
    "spectrum.set_active": _h_spectrum_set_active,
    "spectrum.reset_alignment": _h_spectrum_reset_alignment,
    "spectrum.reset_points": _h_spectrum_reset_points,
    "selection.snapshot": _h_selection_snapshot,
    "selection.pointcloud": _h_selection_pointcloud,
    "interactive.read": h_interactive_read,
    "spectrum.interactive.open": h_spectrum_interactive_open,
    "spectrum.interactive.command": h_spectrum_interactive_command,
    "selection.interactive.open": h_selection_interactive_open,
    "selection.interactive.command": h_selection_interactive_command,
    "fit.result": _h_fit_result,
    "fit.set_params": _h_fit_set_params,
    "fit.search": _h_fit_search,
    "operation.status": _h_operation_status,
    "operation.cancel": _h_operation_cancel,
    "operation.await": _h_operation_await,
    "export.spectrums": _h_export_spectrums,
    "fit.export_params": _h_fit_export_params,
    "resources.versions": h_resources_versions,
    "state.check": _h_state_check,
}

METHOD_REGISTRY: dict[str, BoundMethod] = build_method_registry(_HANDLERS, METHOD_SPECS)
