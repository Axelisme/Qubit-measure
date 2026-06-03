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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, cast

import numpy as np
from numpy.typing import NDArray

from zcu_tools.fluxdep_gui.state import ProjectInfo, SpecType

if TYPE_CHECKING:
    # Type-only: a runtime import of the adapter would cycle (service.py imports
    # this module). String annotations keep pyright checking the call sites.
    from .service import RemoteControlAdapter

from .errors import ErrorCode, RemoteError
from .method_specs import METHOD_SPECS, MethodSpec
from .param_spec import ParamSpec

logger = logging.getLogger(__name__)

Handler = Callable[["RemoteControlAdapter", Mapping[str, object]], Mapping[str, object]]

_SPEC_TYPES: frozenset[str] = frozenset({"OneTone", "TwoTone"})


# ---------------------------------------------------------------------------
# Runtime registry entry — binds a synchronous handler to a Qt-free MethodSpec.
# ---------------------------------------------------------------------------


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
# Coercion helpers (raw wire JSON → typed fluxdep values)
# ---------------------------------------------------------------------------


def _coerce_spec_type(value: object) -> SpecType:
    if value not in _SPEC_TYPES:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"spec_type must be one of {sorted(_SPEC_TYPES)}, got {value!r}",
        )
    return cast(SpecType, value)


def _coerce_float_array(value: object, field: str) -> NDArray[np.float64]:
    if not isinstance(value, list):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'{field}' must be a JSON array, got {type(value).__name__}",
        )
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"'{field}' must be an array of numbers: {exc}"
        ) from exc


def _coerce_bool_array(value: object, field: str) -> NDArray[np.bool_]:
    if not isinstance(value, list):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'{field}' must be a JSON array, got {type(value).__name__}",
        )
    if not all(isinstance(v, bool) for v in value):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"'{field}' must be an array of booleans"
        )
    return np.asarray(value, dtype=np.bool_)


# ---------------------------------------------------------------------------
# Project handlers
# ---------------------------------------------------------------------------


def _h_project_setup(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    chip = str(params["chip_name"])
    qub = str(params["qub_name"])
    result_dir = params["result_dir"]
    database_path = params["database_path"]
    adapter.ctrl.setup_project(
        ProjectInfo(
            chip_name=chip,
            qub_name=qub,
            result_dir=str(result_dir) if result_dir is not None else "",
            database_path=str(database_path) if database_path is not None else "",
        )
    )
    return {"ok": True}


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


# ---------------------------------------------------------------------------
# Spectrum collection handlers
# ---------------------------------------------------------------------------


def _h_spectrum_load(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    filepath = str(params["filepath"])
    spec_type = _coerce_spec_type(params["spec_type"])
    inherit_raw = params["inherit_from"]
    inherit_from = str(inherit_raw) if inherit_raw is not None else None
    try:
        name = adapter.ctrl.load_spectrum(filepath, spec_type, inherit_from)
    except (OSError, KeyError, ValueError) as exc:
        raise RemoteError(ErrorCode.CONTROLLER_ERROR, str(exc)) from exc
    return {"name": name}


def _h_spectrum_list(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
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


def _h_spectrum_remove(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        adapter.ctrl.remove_spectrum(name)
    except KeyError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"unknown spectrum: {name!r}"
        ) from exc
    return {"ok": True}


def _h_spectrum_set_active(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        adapter.ctrl.set_active_spectrum(name)
    except KeyError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"unknown spectrum: {name!r}"
        ) from exc
    return {"ok": True}


# ---------------------------------------------------------------------------
# Alignment / points handlers
# ---------------------------------------------------------------------------


def _h_alignment_set(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    flux_half = float(cast(float, params["flux_half"]))
    flux_int = float(cast(float, params["flux_int"]))
    try:
        adapter.ctrl.set_alignment(name, flux_half, flux_int)
    except KeyError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"unknown spectrum: {name!r}"
        ) from exc
    return {"ok": True}


def _h_points_set(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    dev_values = _coerce_float_array(params["dev_values"], "dev_values")
    freqs = _coerce_float_array(params["freqs"], "freqs")
    try:
        adapter.ctrl.set_points(name, dev_values, freqs)
    except KeyError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"unknown spectrum: {name!r}"
        ) from exc
    except ValueError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {"ok": True}


# ---------------------------------------------------------------------------
# Cross-spectrum selection handlers
# ---------------------------------------------------------------------------


def _h_selection_pointcloud(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    fluxs, freqs = adapter.ctrl.derive_pointcloud()
    return {"fluxs": fluxs.tolist(), "freqs": freqs.tolist()}


def _h_selection_set(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    selected = _coerce_bool_array(params["selected"], "selected")
    try:
        adapter.ctrl.set_selection(selected)
    except ValueError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {"ok": True}


# ---------------------------------------------------------------------------
# Export handler
# ---------------------------------------------------------------------------


def _h_export_spectrums(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    filepath_raw = params["filepath"]
    filepath = str(filepath_raw) if filepath_raw is not None else None
    try:
        path = adapter.ctrl.export_spectrums(filepath)
    except ValueError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    except (OSError, FileExistsError) as exc:
        raise RemoteError(ErrorCode.CONTROLLER_ERROR, str(exc)) from exc
    return {"path": path}


# ---------------------------------------------------------------------------
# Resource versions / state handlers
# ---------------------------------------------------------------------------


def _h_resources_versions(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"versions": adapter.ctrl.state.version.snapshot()}


def _h_state_check(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    state = adapter.ctrl.state
    project = state.project
    has_project = bool(project.chip_name and project.qub_name)
    return {
        "has_project": has_project,
        "spectrum_count": len(state.spectrums),
        "has_active": state.active_spectrum is not None,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_HANDLERS: dict[str, Handler] = {
    "project.setup": _h_project_setup,
    "project.info": _h_project_info,
    "spectrum.load": _h_spectrum_load,
    "spectrum.list": _h_spectrum_list,
    "spectrum.remove": _h_spectrum_remove,
    "spectrum.set_active": _h_spectrum_set_active,
    "alignment.set": _h_alignment_set,
    "points.set": _h_points_set,
    "selection.pointcloud": _h_selection_pointcloud,
    "selection.set": _h_selection_set,
    "export.spectrums": _h_export_spectrums,
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

# `auth` is a sentinel handled by the service before the registry — left out here.
METHOD_REGISTRY: dict[str, BoundMethod] = {
    method: BoundMethod(handler=_HANDLERS[method], spec=METHOD_SPECS[method])
    for method in METHOD_SPECS
}
