from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

from .syncfile import SyncFile

PARAMS_SCHEMA_VERSION = 1
UNKNOWN_PROJECT_NAME = "unknown"
UNKNOWN_RESONATOR_NAME = "unknown"
_T1_CURVE_FIT_PARAM_NAMES = ("Q_cap", "x_qp", "Q_ind", "Temp")


class QubitParamsError(ValueError):
    """Expected params.json validation/storage failure."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class ParamsProject:
    chip_name: str
    qub_name: str
    resonator_name: str = UNKNOWN_RESONATOR_NAME

    def __post_init__(self) -> None:
        if not self.chip_name:
            raise QubitParamsError(
                "project chip_name must not be empty",
                reason_code="project_invalid",
            )
        if not self.qub_name:
            raise QubitParamsError(
                "project qub_name must not be empty",
                reason_code="project_invalid",
            )

    @property
    def name(self) -> str:
        return f"{self.chip_name}/{self.qub_name}"

    def to_json_project(self) -> dict[str, str]:
        return {
            "chip_name": self.chip_name,
            "qubit_name": self.qub_name,
            "resonator_name": self.resonator_name,
        }


@dataclass(frozen=True)
class FluxoniumModelParams:
    EJ: float
    EC: float
    EL: float
    flux_half: float
    flux_period: float
    flux_bias: float = 0.0

    @property
    def params(self) -> tuple[float, float, float]:
        return (self.EJ, self.EC, self.EL)


@dataclass(frozen=True)
class FluxDepFit:
    EJ: float
    EC: float
    EL: float
    flux_half: float
    flux_int: float
    flux_period: float
    plot_transitions: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str | None = None

    @property
    def params(self) -> tuple[float, float, float]:
        return (self.EJ, self.EC, self.EL)

    def to_json_section(self) -> dict[str, Any]:
        section = {
            "params": {"EJ": self.EJ, "EC": self.EC, "EL": self.EL},
            "flux_half": self.flux_half,
            "flux_int": self.flux_int,
            "flux_period": self.flux_period,
            "plot_transitions": deepcopy(dict(self.plot_transitions)),
        }
        if self.timestamp is not None:
            section["timestamp"] = self.timestamp
        return section


@dataclass(frozen=True)
class DispersiveFit:
    g: float
    bare_rf: float
    timestamp: str | None = None

    def to_json_section(self) -> dict[str, Any]:
        section: dict[str, Any] = {"g": self.g, "bare_rf": self.bare_rf}
        if self.timestamp is not None:
            section["timestamp"] = self.timestamp
        return section


@dataclass(frozen=True)
class DispersiveFitInputs:
    params: tuple[float, float, float]
    flux_half: float
    flux_int: float
    flux_period: float
    bare_rf_seed: float


@dataclass(frozen=True)
class T1CurveFitParams:
    Q_cap: float
    x_qp: float
    Q_ind: float
    Temp: float

    def to_json_params(self) -> dict[str, float]:
        return {
            "Q_cap": _json_required_float(self.Q_cap, path="t1_curve_fit.params.Q_cap"),
            "x_qp": _json_required_float(self.x_qp, path="t1_curve_fit.params.x_qp"),
            "Q_ind": _json_required_float(self.Q_ind, path="t1_curve_fit.params.Q_ind"),
            "Temp": _json_required_float(self.Temp, path="t1_curve_fit.params.Temp"),
        }


@dataclass(frozen=True)
class T1CurveFitUncertainty:
    Q_cap: float | None = None
    x_qp: float | None = None
    Q_ind: float | None = None
    Temp: float | None = None

    def to_json_params(self) -> dict[str, float | None]:
        return {
            "Q_cap": _json_nullable_float(self.Q_cap),
            "x_qp": _json_nullable_float(self.x_qp),
            "Q_ind": _json_nullable_float(self.Q_ind),
            "Temp": _json_nullable_float(self.Temp),
        }


@dataclass(frozen=True)
class T1CurveFit:
    params: T1CurveFitParams
    stderr: T1CurveFitUncertainty | None = None
    fixed: tuple[str, ...] = ()
    free: tuple[str, ...] = ()
    cost: float | None = None
    reduced_chi2: float | None = None
    success: bool | None = None
    message: str | None = None
    residual_mode: str | None = None
    loss: str | None = None
    max_nfev: int | None = None
    init: T1CurveFitParams | None = None
    bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    timestamp: str | None = None

    def to_json_section(self) -> dict[str, Any]:
        section: dict[str, Any] = {
            "params": self.params.to_json_params(),
            "fixed": list(self.fixed),
            "free": list(self.free),
        }
        if self.stderr is not None:
            section["stderr"] = self.stderr.to_json_params()
        if self.cost is not None:
            section["cost"] = _json_nullable_float(self.cost)
        if self.reduced_chi2 is not None:
            section["reduced_chi2"] = _json_nullable_float(self.reduced_chi2)
        if self.success is not None:
            section["success"] = bool(self.success)
        if self.message is not None:
            section["message"] = str(self.message)
        if self.residual_mode is not None:
            section["residual_mode"] = str(self.residual_mode)
        if self.loss is not None:
            section["loss"] = str(self.loss)
        if self.max_nfev is not None:
            section["max_nfev"] = int(self.max_nfev)
        if self.init is not None:
            section["init"] = self.init.to_json_params()
        if self.bounds:
            bounds: dict[str, list[float]] = {}
            for name, bound in self.bounds.items():
                if name not in _T1_CURVE_FIT_PARAM_NAMES:
                    raise QubitParamsError(
                        f"t1_curve_fit.bounds.{name} is not a known T1 fit parameter",
                        reason_code="params_value_invalid",
                    )
                bounds[name] = [
                    _json_required_float(
                        bound[0], path=f"t1_curve_fit.bounds.{name}[0]"
                    ),
                    _json_required_float(
                        bound[1], path=f"t1_curve_fit.bounds.{name}[1]"
                    ),
                ]
            section["bounds"] = bounds
        if self.timestamp is not None:
            section["timestamp"] = self.timestamp
        return section


def params_path_for_result_dir(result_dir: str | Path) -> str:
    return str(Path(result_dir) / "params.json")


def _now_timestamp() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _with_current_timestamp(section: dict[str, Any]) -> dict[str, Any]:
    section["timestamp"] = _now_timestamp()
    return section


def _clean_name(value: object) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _mapping(raw: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise QubitParamsError(
            f"{path} must be an object",
            reason_code="params_section_invalid",
        )
    return raw


def _coerce_float(raw: object, *, path: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, str | int | float):
        raise QubitParamsError(
            f"{path} must be a number",
            reason_code="params_value_invalid",
        )
    try:
        value = float(raw)
    except ValueError as exc:
        raise QubitParamsError(
            f"{path} must be a number",
            reason_code="params_value_invalid",
        ) from exc
    if not math.isfinite(value):
        raise QubitParamsError(
            f"{path} must be finite",
            reason_code="params_value_invalid",
        )
    return value


def _json_required_float(value: object, *, path: str) -> float:
    return _coerce_float(value, path=path)


def _json_nullable_float(value: object) -> float | None:
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, str | int | float)
    ):
        return None
    try:
        converted = float(value)
    except ValueError:
        return None
    return converted if math.isfinite(converted) else None


def _optional_timestamp(raw: Mapping[str, Any], *, path: str) -> str | None:
    if "timestamp" not in raw:
        return None
    timestamp = raw["timestamp"]
    if isinstance(timestamp, str) and timestamp:
        return timestamp
    raise QubitParamsError(
        f"{path}.timestamp must be a non-empty string",
        reason_code="params_value_invalid",
    )


def _require_key(raw: Mapping[str, Any], key: str, *, path: str) -> object:
    try:
        return raw[key]
    except KeyError as exc:
        raise QubitParamsError(
            f"{path}.{key} is required",
            reason_code="params_key_missing",
        ) from exc


def _parse_nullable_float(raw: object, *, path: str) -> float | None:
    if raw is None:
        return None
    return _coerce_float(raw, path=path)


def _parse_optional_bool(raw: object, *, path: str) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    raise QubitParamsError(
        f"{path} must be a boolean or null",
        reason_code="params_value_invalid",
    )


def _parse_optional_int(raw: object, *, path: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise QubitParamsError(
            f"{path} must be an integer or null",
            reason_code="params_value_invalid",
        )
    return raw


def _parse_optional_string(raw: object, *, path: str) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    raise QubitParamsError(
        f"{path} must be a string or null",
        reason_code="params_value_invalid",
    )


def _parse_string_tuple(raw: object, *, path: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list | tuple):
        raise QubitParamsError(
            f"{path} must be an array of strings",
            reason_code="params_value_invalid",
        )
    values: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, str):
            raise QubitParamsError(
                f"{path}[{index}] must be a string",
                reason_code="params_value_invalid",
            )
        values.append(value)
    return tuple(values)


def _parse_t1_curve_fit_params(raw: object, *, path: str) -> T1CurveFitParams:
    section = _mapping(raw, path=path)
    return T1CurveFitParams(
        Q_cap=_coerce_float(
            _require_key(section, "Q_cap", path=path),
            path=f"{path}.Q_cap",
        ),
        x_qp=_coerce_float(
            _require_key(section, "x_qp", path=path),
            path=f"{path}.x_qp",
        ),
        Q_ind=_coerce_float(
            _require_key(section, "Q_ind", path=path),
            path=f"{path}.Q_ind",
        ),
        Temp=_coerce_float(
            _require_key(section, "Temp", path=path),
            path=f"{path}.Temp",
        ),
    )


def _parse_t1_curve_fit_uncertainty(
    raw: object,
    *,
    path: str,
) -> T1CurveFitUncertainty:
    section = _mapping(raw, path=path)
    return T1CurveFitUncertainty(
        Q_cap=_parse_nullable_float(section.get("Q_cap"), path=f"{path}.Q_cap"),
        x_qp=_parse_nullable_float(section.get("x_qp"), path=f"{path}.x_qp"),
        Q_ind=_parse_nullable_float(section.get("Q_ind"), path=f"{path}.Q_ind"),
        Temp=_parse_nullable_float(section.get("Temp"), path=f"{path}.Temp"),
    )


def _parse_t1_curve_bounds(
    raw: object,
    *,
    path: str,
) -> dict[str, tuple[float, float]]:
    section = _mapping(raw, path=path)
    bounds: dict[str, tuple[float, float]] = {}
    for name, raw_bound in section.items():
        if name not in _T1_CURVE_FIT_PARAM_NAMES:
            raise QubitParamsError(
                f"{path}.{name} is not a known T1 fit parameter",
                reason_code="params_value_invalid",
            )
        if not isinstance(raw_bound, list | tuple) or len(raw_bound) != 2:
            raise QubitParamsError(
                f"{path}.{name} must be a two-number array",
                reason_code="params_value_invalid",
            )
        bounds[name] = (
            _coerce_float(raw_bound[0], path=f"{path}.{name}[0]"),
            _coerce_float(raw_bound[1], path=f"{path}.{name}[1]"),
        )
    return bounds


def _canonical_project_from_raw(raw: Mapping[str, Any]) -> ParamsProject | None:
    project = raw.get("project")
    if isinstance(project, Mapping):
        chip = _clean_name(project.get("chip_name"))
        qub = _clean_name(project.get("qubit_name")) or _clean_name(
            project.get("qub_name")
        )
        resonator = _clean_name(project.get("resonator_name"))
        if chip and qub:
            return ParamsProject(
                chip_name=chip,
                qub_name=qub,
                resonator_name=resonator or UNKNOWN_RESONATOR_NAME,
            )

    return None


def _legacy_project_from_raw(raw: Mapping[str, Any]) -> ParamsProject | None:
    name = _clean_name(raw.get("name"))
    parts = [part.strip() for part in name.split("/")]
    if len(parts) == 2 and all(parts):
        return ParamsProject(chip_name=parts[0], qub_name=parts[1])
    return None


def _project_from_raw(raw: Mapping[str, Any]) -> ParamsProject | None:
    return _canonical_project_from_raw(raw) or _legacy_project_from_raw(raw)


def _project_from_result_path(
    params_path: str | Path, result_root: str | Path
) -> ParamsProject:
    try:
        rel_dir = (
            Path(params_path).resolve().parent.relative_to(Path(result_root).resolve())
        )
    except ValueError:
        return ParamsProject(UNKNOWN_PROJECT_NAME, UNKNOWN_PROJECT_NAME)

    parts = rel_dir.parts
    if len(parts) == 2 and all(parts):
        return ParamsProject(chip_name=parts[0], qub_name=parts[1])
    if len(parts) == 1 and parts[0]:
        return ParamsProject(chip_name=parts[0], qub_name=parts[0])
    return ParamsProject(UNKNOWN_PROJECT_NAME, UNKNOWN_PROJECT_NAME)


class QubitParams(SyncFile):
    """Typed owner of one result-scope ``params.json`` file.

    Known sections are exposed through typed methods; unknown sections are
    preserved during typed writes so future sections can coexist with current
    callers.  The raw helpers exist for transitional legacy wrappers only.
    """

    def __init__(
        self, params_path: str | Path | None = None, readonly: bool = False
    ) -> None:
        self._data: dict[str, Any] = {}
        super().__init__(params_path, readonly=readonly)

    @classmethod
    def for_result_dir(cls, result_dir: str | Path, *, readonly: bool = False) -> Self:
        return cls(params_path_for_result_dir(result_dir), readonly=readonly)

    def _load(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except json.JSONDecodeError as exc:
            raise QubitParamsError(
                f"Failed to read params.json at {path}: {exc}",
                reason_code="params_read_failed",
            ) from exc
        if not isinstance(loaded, dict):
            raise QubitParamsError(
                f"params.json at {path} must contain a JSON object",
                reason_code="params_not_object",
            )
        self._data.clear()
        self._data.update(loaded)

    def _dump(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=4)
            f.write("\n")

    def _sync_existing(self) -> None:
        if self._path is not None and not self._path.exists():
            raise FileNotFoundError(f"params.json not found at {self._path!s}")
        self.sync()

    def _sync_before_write(self, *, require_existing: bool) -> None:
        self._check_can_write()
        if require_existing:
            self._sync_existing()
        elif self._path is None or self._path.exists():
            self.sync()

    def _store(self) -> None:
        self._dirty = True
        self.sync()

    def to_raw(self) -> dict[str, Any]:
        self._sync_existing()
        return deepcopy(self._data)

    def replace_raw(self, raw: Mapping[str, Any]) -> None:
        self._sync_before_write(require_existing=False)
        self._data.clear()
        self._data.update(deepcopy(dict(raw)))
        self._store()

    def update_raw(self, update: Mapping[str, Any]) -> None:
        self._sync_before_write(require_existing=True)
        self._data.update(deepcopy(dict(update)))
        self._store()

    def get_project(self) -> ParamsProject | None:
        self._sync_existing()
        return _project_from_raw(self._data)

    def require_project(self) -> ParamsProject:
        project = self.get_project()
        if project is None:
            raise QubitParamsError(
                "params.json has no project identity",
                reason_code="params_missing_identity",
            )
        return project

    def ensure_project(self, project: ParamsProject) -> None:
        self._sync_before_write(require_existing=False)
        self._data["schema_version"] = PARAMS_SCHEMA_VERSION
        self._data["project"] = project.to_json_project()
        self._data["name"] = project.name
        self._store()

    def migrate_project_from_path(self, *, result_root: str | Path) -> ParamsProject:
        self._sync_existing()
        project = _canonical_project_from_raw(self._data)
        if project is None:
            if self._path is None:
                raise QubitParamsError(
                    "cannot infer project identity without a params path",
                    reason_code="params_missing_identity",
                )
            project = _project_from_result_path(self._path, result_root)
        self.ensure_project(project)
        return project

    def get_fluxdep_fit(self) -> FluxDepFit | None:
        self._sync_existing()
        raw = self._data.get("fluxdep_fit")
        if raw is None:
            return None
        return self._parse_fluxdep_fit(raw)

    def require_fluxdep_fit(self) -> FluxDepFit:
        fit = self.get_fluxdep_fit()
        if fit is None:
            raise QubitParamsError(
                "params.json has no 'fluxdep_fit' section",
                reason_code="fluxdep_fit_missing",
            )
        return fit

    def set_fluxdep_fit(self, fit: FluxDepFit) -> None:
        self._sync_before_write(require_existing=False)
        self._data["fluxdep_fit"] = _with_current_timestamp(fit.to_json_section())
        self._store()

    def get_dispersive_fit(self) -> DispersiveFit | None:
        self._sync_existing()
        raw = self._data.get("dispersive")
        if raw is None:
            return None
        return self._parse_dispersive_fit(raw)

    def set_dispersive_fit(self, fit: DispersiveFit) -> None:
        self._sync_before_write(require_existing=True)
        self.require_fluxdep_fit()
        self._data["dispersive"] = _with_current_timestamp(fit.to_json_section())
        self._store()

    def get_t1_curve_fit(self) -> T1CurveFit | None:
        self._sync_existing()
        raw = self._data.get("t1_curve_fit")
        if raw is None:
            return None
        return self._parse_t1_curve_fit(raw)

    def require_t1_curve_fit(self) -> T1CurveFit:
        fit = self.get_t1_curve_fit()
        if fit is None:
            raise QubitParamsError(
                "params.json has no 't1_curve_fit' section",
                reason_code="t1_curve_fit_missing",
            )
        return fit

    def set_t1_curve_fit(self, fit: T1CurveFit) -> None:
        self._sync_before_write(require_existing=True)
        self.require_fluxdep_fit()
        self._data["t1_curve_fit"] = _with_current_timestamp(fit.to_json_section())
        self._store()

    def require_fluxonium_model(
        self, *, flux_bias: float = 0.0
    ) -> FluxoniumModelParams:
        fit = self.require_fluxdep_fit()
        return FluxoniumModelParams(
            EJ=fit.EJ,
            EC=fit.EC,
            EL=fit.EL,
            flux_half=fit.flux_half,
            flux_period=fit.flux_period,
            flux_bias=flux_bias,
        )

    def require_dispersive_inputs(
        self, *, default_bare_rf: float
    ) -> DispersiveFitInputs:
        fit = self.require_fluxdep_fit()
        dispersive = self.get_dispersive_fit()
        if dispersive is not None:
            bare_rf_seed = dispersive.bare_rf
        else:
            r_f = fit.plot_transitions.get("r_f")
            bare_rf_seed = (
                _coerce_float(r_f, path="fluxdep_fit.plot_transitions.r_f")
                if r_f is not None
                else default_bare_rf
            )
        return DispersiveFitInputs(
            params=fit.params,
            flux_half=fit.flux_half,
            flux_int=fit.flux_int,
            flux_period=fit.flux_period,
            bare_rf_seed=bare_rf_seed,
        )

    @staticmethod
    def _parse_fluxdep_fit(raw: object) -> FluxDepFit:
        section = _mapping(raw, path="fluxdep_fit")
        params = _mapping(
            _require_key(section, "params", path="fluxdep_fit"),
            path="fluxdep_fit.params",
        )
        transitions = section.get("plot_transitions") or {}
        return FluxDepFit(
            EJ=_coerce_float(
                _require_key(params, "EJ", path="fluxdep_fit.params"),
                path="fluxdep_fit.params.EJ",
            ),
            EC=_coerce_float(
                _require_key(params, "EC", path="fluxdep_fit.params"),
                path="fluxdep_fit.params.EC",
            ),
            EL=_coerce_float(
                _require_key(params, "EL", path="fluxdep_fit.params"),
                path="fluxdep_fit.params.EL",
            ),
            flux_half=_coerce_float(
                _require_key(section, "flux_half", path="fluxdep_fit"),
                path="fluxdep_fit.flux_half",
            ),
            flux_int=_coerce_float(
                _require_key(section, "flux_int", path="fluxdep_fit"),
                path="fluxdep_fit.flux_int",
            ),
            flux_period=_coerce_float(
                _require_key(section, "flux_period", path="fluxdep_fit"),
                path="fluxdep_fit.flux_period",
            ),
            plot_transitions=deepcopy(
                dict(_mapping(transitions, path="plot_transitions"))
            ),
            timestamp=_optional_timestamp(section, path="fluxdep_fit"),
        )

    @staticmethod
    def _parse_dispersive_fit(raw: object) -> DispersiveFit:
        section = _mapping(raw, path="dispersive")
        return DispersiveFit(
            g=_coerce_float(
                _require_key(section, "g", path="dispersive"),
                path="dispersive.g",
            ),
            bare_rf=_coerce_float(
                _require_key(section, "bare_rf", path="dispersive"),
                path="dispersive.bare_rf",
            ),
            timestamp=_optional_timestamp(section, path="dispersive"),
        )

    @staticmethod
    def _parse_t1_curve_fit(raw: object) -> T1CurveFit:
        section = _mapping(raw, path="t1_curve_fit")
        return T1CurveFit(
            params=_parse_t1_curve_fit_params(
                _require_key(section, "params", path="t1_curve_fit"),
                path="t1_curve_fit.params",
            ),
            stderr=(
                _parse_t1_curve_fit_uncertainty(
                    section["stderr"],
                    path="t1_curve_fit.stderr",
                )
                if "stderr" in section
                else None
            ),
            fixed=_parse_string_tuple(
                section.get("fixed", ()), path="t1_curve_fit.fixed"
            ),
            free=_parse_string_tuple(section.get("free", ()), path="t1_curve_fit.free"),
            cost=_parse_nullable_float(section.get("cost"), path="t1_curve_fit.cost"),
            reduced_chi2=_parse_nullable_float(
                section.get("reduced_chi2"),
                path="t1_curve_fit.reduced_chi2",
            ),
            success=_parse_optional_bool(
                section.get("success"),
                path="t1_curve_fit.success",
            ),
            message=_parse_optional_string(
                section.get("message"),
                path="t1_curve_fit.message",
            ),
            residual_mode=_parse_optional_string(
                section.get("residual_mode"),
                path="t1_curve_fit.residual_mode",
            ),
            loss=_parse_optional_string(section.get("loss"), path="t1_curve_fit.loss"),
            max_nfev=_parse_optional_int(
                section.get("max_nfev"),
                path="t1_curve_fit.max_nfev",
            ),
            init=(
                _parse_t1_curve_fit_params(
                    section["init"],
                    path="t1_curve_fit.init",
                )
                if "init" in section
                else None
            ),
            bounds=(
                _parse_t1_curve_bounds(
                    section["bounds"],
                    path="t1_curve_fit.bounds",
                )
                if "bounds" in section
                else {}
            ),
            timestamp=_optional_timestamp(section, path="t1_curve_fit"),
        )
