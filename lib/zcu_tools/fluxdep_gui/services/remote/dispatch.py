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
from zcu_tools.notebook.persistance import TransitionDict

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


def _coerce_opt_float(value: object, field: str) -> "float | None":
    """Coerce an optional number: None / missing → None, else float."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"'{field}' must be a number or null"
        )
    return float(value)


def _coerce_bound(value: object, field: str) -> tuple[float, float]:
    """Coerce a JSON ``[min, max]`` array to a float bound pair."""
    if not isinstance(value, list) or len(value) != 2:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"'{field}' must be a [min, max] pair"
        )
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"'{field}' bounds must be numbers: {exc}"
        ) from exc


def _coerce_transitions(value: object) -> TransitionDict:
    """Coerce a JSON object (category -> list of [i, j]) to a TransitionDict.

    Each value must be a list of 2-int pairs; an empty/malformed entry fails
    fast. r_f / sample_f are NOT carried here (they are separate params on
    fit.set_params and injected by the handler).
    """
    if not isinstance(value, dict):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'transitions' must be a JSON object, got {type(value).__name__}",
        )
    result: dict[str, list[tuple[int, int]]] = {}
    for key, pairs in value.items():
        if not isinstance(pairs, list):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"transitions[{key!r}] must be a list of [i, j] pairs",
            )
        coerced: list[tuple[int, int]] = []
        for pair in pairs:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or not all(isinstance(v, int) for v in pair)
            ):
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    f"transitions[{key!r}] entries must be [i, j] integer pairs, "
                    f"got {pair!r}",
                )
            coerced.append((int(pair[0]), int(pair[1])))
        result[str(key)] = coerced
    return cast(TransitionDict, result)


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
    # Omitted / None result_dir / database_path → empty, which ProjectInfo's
    # __post_init__ derives from chip/qubit (the single derivation point); a value
    # overrides it.
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
    transpose_axes = bool(params.get("transpose_axes", False))
    try:
        name = adapter.ctrl.load_spectrum(
            filepath, spec_type, inherit_from, transpose_axes
        )
    except (OSError, KeyError, ValueError) as exc:
        raise RemoteError(ErrorCode.CONTROLLER_ERROR, str(exc)) from exc
    return {"name": name}


def _h_spectrum_load_processed(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    filepath = str(params["filepath"])
    try:
        names = adapter.ctrl.load_processed_spectrums(filepath)
    except (OSError, KeyError, ValueError) as exc:
        raise RemoteError(ErrorCode.CONTROLLER_ERROR, str(exc)) from exc
    return {"names": names}


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
    min_distance = float(params.get("min_distance", 0.0))  # type: ignore[arg-type]
    try:
        adapter.ctrl.set_selection(selected, min_distance)
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
# Database-search fit handlers (v2)
# ---------------------------------------------------------------------------


def _h_fit_set_params(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    database_path = str(params["database_path"])
    EJb = _coerce_bound(params["EJb"], "EJb")
    ECb = _coerce_bound(params["ECb"], "ECb")
    ELb = _coerce_bound(params["ELb"], "ELb")
    transitions = _coerce_transitions(params["transitions"])
    # r_f / sample_f are optional: omitted or null → None (not provided). The
    # service injects the keys into the transitions when set.
    r_f = _coerce_opt_float(params.get("r_f"), "r_f")
    sample_f = _coerce_opt_float(params.get("sample_f"), "sample_f")
    adapter.ctrl.set_fit_params(
        database_path, EJb, ECb, ELb, transitions, r_f, sample_f
    )
    return {"ok": True}


def _h_fit_search(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    try:
        result = adapter.ctrl.search_database(plot=False)
    except ValueError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    except (OSError, KeyError, RuntimeError) as exc:
        raise RemoteError(ErrorCode.CONTROLLER_ERROR, str(exc)) from exc
    EJ, EC, EL = result.params
    return {"EJ": EJ, "EC": EC, "EL": EL}


def _h_fit_result(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
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


def _h_fit_export_params(
    adapter: "RemoteControlAdapter", params: Mapping[str, object]
) -> Mapping[str, object]:
    savepath_raw = params.get("savepath")
    savepath = str(savepath_raw) if savepath_raw is not None else None
    try:
        path = adapter.ctrl.export_params(savepath)
    except ValueError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    except OSError as exc:
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
    from zcu_tools.fluxdep_gui.state import DEFAULT_CHIP, DEFAULT_QUBIT

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
    "spectrum.load_processed": _h_spectrum_load_processed,
    "spectrum.list": _h_spectrum_list,
    "spectrum.remove": _h_spectrum_remove,
    "spectrum.set_active": _h_spectrum_set_active,
    "alignment.set": _h_alignment_set,
    "points.set": _h_points_set,
    "selection.pointcloud": _h_selection_pointcloud,
    "selection.set": _h_selection_set,
    "fit.set_params": _h_fit_set_params,
    "fit.search": _h_fit_search,
    "fit.result": _h_fit_result,
    "fit.export_params": _h_fit_export_params,
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
