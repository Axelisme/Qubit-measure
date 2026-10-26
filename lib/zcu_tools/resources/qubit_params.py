"""Typed owner of the result-scope ``params.json`` handoff file. See qubit_params.md."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Self

from zcu_tools.utils import atomic_replace

PARAMS_SCHEMA_VERSION = 1
UNKNOWN_PROJECT_NAME = "unknown"
UNKNOWN_RESONATOR_NAME = "unknown"
_T1_CURVE_FIT_NOISE_PARAM_NAMES = ("Q_cap", "x_qp", "Q_ind")
_T1_CURVE_FIT_PARAM_NAMES = (*_T1_CURVE_FIT_NOISE_PARAM_NAMES, "Temp")


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
        section: dict[str, Any] = {
            "params": {"EJ": self.EJ, "EC": self.EC, "EL": self.EL},
            "flux_half": self.flux_half,
            "flux_int": self.flux_int,
            "flux_period": self.flux_period,
            "plot_transitions": deepcopy(dict(self.plot_transitions)),
        }
        if self.timestamp is not None:
            section["timestamp"] = self.timestamp
        return section


class FluxDepFitView:
    """Detached read view of a fluxdep_fit JSON section.

    section must contain params.EJ, EC and EL in GHz as finite numbers.
    EJ, EC and EL expose Josephson, charging and inductive energies respectively.
    Unknown fields are ignored. Calibration properties validate only when
    accessed and raise QubitParamsError naming a missing or invalid field.
    This view does not write files; FluxDepFit remains the write payload.
    """

    def __init__(self, section: Mapping[str, object]) -> None:
        self._section = deepcopy(dict(section))
        params = _mapping(
            _require_key(section, "params", path="fluxdep_fit"),
            path="fluxdep_fit.params",
        )
        self.EJ = _coerce_float(
            _require_key(params, "EJ", path="fluxdep_fit.params"),
            path="fluxdep_fit.params.EJ",
        )
        self.EC = _coerce_float(
            _require_key(params, "EC", path="fluxdep_fit.params"),
            path="fluxdep_fit.params.EC",
        )
        self.EL = _coerce_float(
            _require_key(params, "EL", path="fluxdep_fit.params"),
            path="fluxdep_fit.params.EL",
        )

    @property
    def params(self) -> tuple[float, float, float]:
        """Return (EJ, EC, EL) in GHz."""
        return (self.EJ, self.EC, self.EL)

    def _calibration_value(self, name: str) -> float:
        return _coerce_float(
            _require_key(self._section, name, path="fluxdep_fit"),
            path=f"fluxdep_fit.{name}",
        )

    @property
    def flux_half(self) -> float:
        """Return the half-flux device value; fail if missing or non-finite."""
        return self._calibration_value("flux_half")

    @property
    def flux_int(self) -> float:
        """Return the integer-flux device value; fail if missing or non-finite."""
        return self._calibration_value("flux_int")

    @property
    def flux_period(self) -> float:
        """Return the device span per flux quantum; fail if missing or non-finite."""
        return self._calibration_value("flux_period")

    @property
    def plot_transitions(self) -> Mapping[str, object]:
        """Return transition settings; fail if missing or not a JSON object."""
        return deepcopy(
            dict(
                _mapping(
                    _require_key(self._section, "plot_transitions", path="fluxdep_fit"),
                    path="fluxdep_fit.plot_transitions",
                )
            )
        )

    @property
    def timestamp(self) -> str | None:
        """Return the optional timestamp; fail if present but not a nonempty string."""
        return _optional_timestamp(self._section, path="fluxdep_fit")


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


@dataclass(frozen=True, kw_only=True)
class T1CurveFitParams:
    Temp: float
    Q_cap: float | None = None
    x_qp: float | None = None
    Q_ind: float | None = None

    @property
    def active_noise_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.Q_cap is not None:
            names.append("Q_cap")
        if self.x_qp is not None:
            names.append("x_qp")
        if self.Q_ind is not None:
            names.append("Q_ind")
        return tuple(names)

    @property
    def active_param_names(self) -> tuple[str, ...]:
        return (*self.active_noise_names, "Temp")

    def to_json_params(self) -> dict[str, float]:
        if not self.active_noise_names:
            raise QubitParamsError(
                "t1_curve_fit.params must include at least one noise parameter",
                reason_code="params_value_invalid",
            )
        params: dict[str, float] = {}
        if self.Q_cap is not None:
            params["Q_cap"] = _json_required_float(
                self.Q_cap, path="t1_curve_fit.params.Q_cap"
            )
        if self.x_qp is not None:
            params["x_qp"] = _json_required_float(
                self.x_qp, path="t1_curve_fit.params.x_qp"
            )
        if self.Q_ind is not None:
            params["Q_ind"] = _json_required_float(
                self.Q_ind, path="t1_curve_fit.params.Q_ind"
            )
        params["Temp"] = _json_required_float(
            self.Temp, path="t1_curve_fit.params.Temp"
        )
        return params


@dataclass(frozen=True)
class T1CurveFitUncertainty:
    Q_cap: float | None = None
    x_qp: float | None = None
    Q_ind: float | None = None
    Temp: float | None = None

    def to_json_params(
        self, *, active_param_names: tuple[str, ...] = _T1_CURVE_FIT_PARAM_NAMES
    ) -> dict[str, float | None]:
        values = {
            "Q_cap": _json_nullable_float(self.Q_cap),
            "x_qp": _json_nullable_float(self.x_qp),
            "Q_ind": _json_nullable_float(self.Q_ind),
            "Temp": _json_nullable_float(self.Temp),
        }
        return {name: values[name] for name in active_param_names}


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
        active_param_names = self.params.active_param_names
        _validate_t1_curve_fit_names(
            self.fixed, path="t1_curve_fit.fixed", active_param_names=active_param_names
        )
        _validate_t1_curve_fit_names(
            self.free, path="t1_curve_fit.free", active_param_names=active_param_names
        )
        if self.init is not None:
            _validate_t1_curve_fit_init(
                self.init,
                active_noise_names=self.params.active_noise_names,
                path="t1_curve_fit.init",
            )
        section: dict[str, Any] = {
            "params": self.params.to_json_params(),
            "fixed": list(self.fixed),
            "free": list(self.free),
        }
        if self.stderr is not None:
            _validate_t1_curve_fit_uncertainty(
                self.stderr, active_param_names=active_param_names
            )
            section["stderr"] = self.stderr.to_json_params(
                active_param_names=active_param_names
            )
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
        section.update(
            _t1_curve_fit_bounds_section(
                self.bounds, active_param_names=active_param_names
            )
        )
        if self.timestamp is not None:
            section["timestamp"] = self.timestamp
        return section


def _t1_curve_fit_bounds_section(
    bounds: Mapping[str, tuple[float, float]], *, active_param_names: tuple[str, ...]
) -> dict[str, dict[str, list[float]]]:
    """Serialize bounds for active T1 parameters into a partial JSON section.

    Empty bounds omit the key; nonempty bounds retain the input Mapping order.
    Each name must be known, then active, before its lower and upper endpoints
    are converted to finite floats. Unknown or inactive names, invalid endpoint
    types, and nonfinite values raise QubitParamsError with params_value_invalid.
    Numeric conversion ValueError is retained as the QubitParamsError cause;
    OverflowError propagates unchanged.
    """
    section: dict[str, dict[str, list[float]]] = {}
    if bounds:
        json_bounds: dict[str, list[float]] = {}
        for name, bound in bounds.items():
            if name not in _T1_CURVE_FIT_PARAM_NAMES:
                raise QubitParamsError(
                    f"t1_curve_fit.bounds.{name} is not a known T1 fit parameter",
                    reason_code="params_value_invalid",
                )
            if name not in active_param_names:
                raise QubitParamsError(
                    f"t1_curve_fit.bounds.{name} is not an active T1 fit parameter",
                    reason_code="params_value_invalid",
                )
            json_bounds[name] = [
                _json_required_float(bound[0], path=f"t1_curve_fit.bounds.{name}[0]"),
                _json_required_float(bound[1], path=f"t1_curve_fit.bounds.{name}[1]"),
            ]
        section["bounds"] = json_bounds
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


def _validate_t1_curve_fit_names(
    names: tuple[str, ...], *, path: str, active_param_names: tuple[str, ...]
) -> None:
    unknown = set(names) - set(_T1_CURVE_FIT_PARAM_NAMES)
    if unknown:
        raise QubitParamsError(
            f"{path} contains unknown T1 fit parameter(s): {sorted(unknown)}",
            reason_code="params_value_invalid",
        )
    inactive = set(names) - set(active_param_names)
    if inactive:
        raise QubitParamsError(
            f"{path} contains inactive T1 fit parameter(s): {sorted(inactive)}",
            reason_code="params_value_invalid",
        )


def _validate_t1_curve_fit_init(
    init: T1CurveFitParams, *, active_noise_names: tuple[str, ...], path: str
) -> None:
    if init.active_noise_names != active_noise_names:
        raise QubitParamsError(
            f"{path} active noise parameters must match t1_curve_fit.params",
            reason_code="params_value_invalid",
        )


def _validate_t1_curve_fit_uncertainty(
    uncertainty: T1CurveFitUncertainty, *, active_param_names: tuple[str, ...]
) -> None:
    values = {
        "Q_cap": uncertainty.Q_cap,
        "x_qp": uncertainty.x_qp,
        "Q_ind": uncertainty.Q_ind,
        "Temp": uncertainty.Temp,
    }
    inactive_with_values = [
        name
        for name, value in values.items()
        if name not in active_param_names and value is not None
    ]
    if inactive_with_values:
        raise QubitParamsError(
            "t1_curve_fit.stderr contains inactive T1 fit parameter(s): "
            f"{inactive_with_values}",
            reason_code="params_value_invalid",
        )


def _optional_t1_curve_fit_param(
    section: Mapping[str, Any], name: str, *, path: str
) -> float | None:
    if name not in section:
        return None
    return _coerce_float(section[name], path=f"{path}.{name}")


def _parse_t1_curve_fit_params(raw: object, *, path: str) -> T1CurveFitParams:
    section = _mapping(raw, path=path)
    params = T1CurveFitParams(
        Temp=_coerce_float(
            _require_key(section, "Temp", path=path),
            path=f"{path}.Temp",
        ),
        Q_cap=_optional_t1_curve_fit_param(section, "Q_cap", path=path),
        x_qp=_optional_t1_curve_fit_param(section, "x_qp", path=path),
        Q_ind=_optional_t1_curve_fit_param(section, "Q_ind", path=path),
    )
    if not params.active_noise_names:
        raise QubitParamsError(
            f"{path} must include at least one noise parameter",
            reason_code="params_value_invalid",
        )
    return params


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


class QubitParams:
    """Typed owner of one result-scope ``params.json`` file.

    Known sections are exposed through typed methods; unknown sections are
    preserved during typed writes so future sections can coexist with current
    callers. The raw helpers exist for transitional legacy wrappers only.

    A path binds the owner to a JSON file, refreshed before public reads and
    writes. Writes serialize fully before atomically replacing that file.
    With no path, operations use memory only. Readonly owners reject writes
    with RuntimeError before reading or changing data. See qubit_params.md
    for method prerequisites and error classifications.
    """

    def __init__(
        self, params_path: str | Path | None = None, *, readonly: bool = False
    ) -> None:
        self._data: dict[str, Any] = {}
        self._path = Path(params_path) if params_path is not None else None
        self._readonly = readonly
        if self._path is not None and self._path.exists():
            self._load(str(self._path))

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

    def _refresh_existing(self) -> None:
        if self._path is not None:
            if not self._path.exists():
                raise FileNotFoundError(f"params.json not found at {self._path!s}")
            self._load(str(self._path))

    def _refresh_before_write(self, *, require_existing: bool) -> None:
        if self._readonly:
            raise RuntimeError(f"{self.__class__.__name__} is read-only")
        if require_existing:
            self._refresh_existing()
        elif self._path is not None and self._path.exists():
            self._load(str(self._path))

    def _store(self) -> None:
        if self._path is None:
            return
        content = json.dumps(self._data, ensure_ascii=False, indent=4) + "\n"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            atomic_replace(temporary, self._path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def to_raw(self) -> dict[str, Any]:
        self._refresh_existing()
        return deepcopy(self._data)

    def replace_raw(self, raw: Mapping[str, Any]) -> None:
        self._refresh_before_write(require_existing=False)
        self._data.clear()
        self._data.update(deepcopy(dict(raw)))
        self._store()

    def update_raw(self, update: Mapping[str, Any]) -> None:
        self._refresh_before_write(require_existing=True)
        self._data.update(deepcopy(dict(update)))
        self._store()

    def get_project(self) -> ParamsProject | None:
        self._refresh_existing()
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
        self._refresh_before_write(require_existing=False)
        self._data["schema_version"] = PARAMS_SCHEMA_VERSION
        self._data["project"] = project.to_json_project()
        self._data["name"] = project.name
        self._store()

    def migrate_project_from_path(self, *, result_root: str | Path) -> ParamsProject:
        self._refresh_before_write(require_existing=True)
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

    def get_fluxdep_fit(self) -> FluxDepFitView | None:
        """Refresh and read model energies, or return None for an absent section.

        Only params.EJ/EC/EL are required here. Calibration is checked by the
        returned view on access. File and model-value errors propagate.
        """
        self._refresh_existing()
        raw = self._data.get("fluxdep_fit")
        if raw is None:
            return None
        return self._parse_fluxdep_fit(raw)

    def require_fluxdep_fit(self) -> FluxDepFitView:
        """Read model energies; raise QubitParamsError if section or energies are missing.

        Calibration fields remain unchecked until accessed on the returned view.
        File errors propagate as for get_fluxdep_fit.
        """
        fit = self.get_fluxdep_fit()
        if fit is None:
            raise QubitParamsError(
                "params.json has no 'fluxdep_fit' section",
                reason_code="fluxdep_fit_missing",
            )
        return fit

    def set_fluxdep_fit(self, fit: FluxDepFit) -> None:
        self._refresh_before_write(require_existing=False)
        self._data["fluxdep_fit"] = _with_current_timestamp(fit.to_json_section())
        self._store()

    def get_dispersive_fit(self) -> DispersiveFit | None:
        self._refresh_existing()
        raw = self._data.get("dispersive")
        if raw is None:
            return None
        return self._parse_dispersive_fit(raw)

    def set_dispersive_fit(self, fit: DispersiveFit) -> None:
        self._refresh_before_write(require_existing=True)
        self._validate_fluxdep_for_write()
        self._data["dispersive"] = _with_current_timestamp(fit.to_json_section())
        self._store()

    def get_t1_curve_fit(self) -> T1CurveFit | None:
        """Refresh and read T1 parameter values, or None for an absent section.

        Ignore all fit metadata, including stderr, bounds and fixed/free.
        The returned payload contains only params; metadata fields keep their
        dataclass defaults. Missing/invalid params raise QubitParamsError;
        file errors propagate.
        """
        self._refresh_existing()
        raw = self._data.get("t1_curve_fit")
        if raw is None:
            return None
        return self._parse_t1_curve_fit(raw)

    def require_t1_curve_fit(self) -> T1CurveFit:
        """Read T1 parameter values, ignoring metadata as in get_t1_curve_fit.

        Raise QubitParamsError if the section or required params are missing
        or invalid. File errors propagate.
        """
        fit = self.get_t1_curve_fit()
        if fit is None:
            raise QubitParamsError(
                "params.json has no 't1_curve_fit' section",
                reason_code="t1_curve_fit_missing",
            )
        return fit

    def set_t1_curve_fit(self, fit: T1CurveFit) -> None:
        self._refresh_before_write(require_existing=True)
        self._validate_fluxdep_for_write()
        self._data["t1_curve_fit"] = _with_current_timestamp(fit.to_json_section())
        self._store()

    def require_fluxonium_model(
        self, *, flux_bias: float = 0.0
    ) -> FluxoniumModelParams:
        """Read energies and device calibration for a model at flux_bias.

        flux_bias is an additive device-value offset in the same unit as
        flux_half and flux_period. Require both calibration fields in addition
        to model energies; missing or invalid
        fields raise QubitParamsError naming the field. File errors propagate.
        """
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
        """Read energies, device calibration and a bare resonator seed in GHz.

        Require flux_half, flux_int and flux_period. Prefer dispersive.bare_rf;
        otherwise require plot_transitions and use its r_f, or default_bare_rf
        when that key is absent. Missing/invalid used fields raise
        QubitParamsError naming the field. File errors propagate.
        """
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

    def _validate_fluxdep_for_write(self) -> None:
        # Writes retain the complete-section prerequisites used before reads
        # became field-local. Missing plot settings still mean an empty mapping.
        self.require_fluxdep_fit()
        section = _mapping(self._data["fluxdep_fit"], path="fluxdep_fit")
        for name in ("flux_half", "flux_int", "flux_period"):
            _coerce_float(
                _require_key(section, name, path="fluxdep_fit"),
                path=f"fluxdep_fit.{name}",
            )
        _mapping(section.get("plot_transitions") or {}, path="plot_transitions")
        _optional_timestamp(section, path="fluxdep_fit")

    @staticmethod
    def _parse_fluxdep_fit(raw: object) -> FluxDepFitView:
        return FluxDepFitView(_mapping(raw, path="fluxdep_fit"))

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
            )
        )


__all__ = [
    "PARAMS_SCHEMA_VERSION",
    "UNKNOWN_PROJECT_NAME",
    "UNKNOWN_RESONATOR_NAME",
    "DispersiveFit",
    "DispersiveFitInputs",
    "FluxDepFit",
    "FluxDepFitView",
    "FluxoniumModelParams",
    "ParamsProject",
    "QubitParams",
    "QubitParamsError",
    "T1CurveFit",
    "T1CurveFitParams",
    "T1CurveFitUncertainty",
    "params_path_for_result_dir",
]
