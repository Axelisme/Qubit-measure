"""Frequency and native-flux policy composed from generic recipe cfg operations."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal, TypeGuard

from pydantic import BaseModel
from zcu_tools.format_version import YamlValue
from zcu_tools.gui.app.measure.remote.form_description import RecipeSelectionResult
from zcu_tools.gui.form_adapter.sweep import LinRange
from zcu_tools.mcp.measure.recipe_api import (
    Expr,
    FieldEdit,
    FormPath,
    RecipeTab,
)
from zcu_tools.mcp.measure.recipe_contracts import MissingParameter
from zcu_tools.mcp.measure.session import GuiRpcError


def report_number(value: float | None) -> float | None:
    """Keep a finite fit scalar, replacing absent or nonfinite values with null.

    Units are unchanged. This matches the native summary's nonfinite-to-null
    presentation and does not select fits, change quality or authorize writeback.
    """
    return value if value is not None and isfinite(value) else None


def _finite(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


def _points(expts: object) -> None:
    if expts is not None and (
        isinstance(expts, bool) or not isinstance(expts, int) or expts < 2
    ):
        raise ValueError("expts must be an integer of at least two")


def retain_scalar(tab: RecipeTab, field: FormPath) -> None:
    """Retain one available logical scalar and its captured source label.

    tab is opened; field uses str names or int original list indexes. Retain
    authored input without copying its captured reference. Unavailable values or
    field errors raise GuiRpcError(invalid_cfg); path/source/connection errors
    propagate. No selector is copied or changed.
    """
    state = tab.read(field)
    if state.value_status != "available" or state.errors:
        raise GuiRpcError(f"Invalid retained {field}", reason="invalid_cfg")
    tab.set(field, None)


class _ParameterQuery(BaseModel):
    """Requested parameter paths and module slots in the tab capture."""

    paths: tuple[str, ...]
    module_refs: tuple[str, ...] = ()


class _ParameterValues(BaseModel):
    """Parameter projection; absent leaves are omitted, explicit null is retained."""

    values: dict[str, YamlValue]


def _captured_parameter(tab: RecipeTab, path: str) -> YamlValue:
    values = tab.call(
        "tab.parameter_read", _ParameterQuery(paths=(path,)), result=_ParameterValues
    )
    return values.values.get(path)


@dataclass(frozen=True)
class FrequencyChoice:
    """A frequency policy decision, before the recipe copies or edits selectors.

    edit is the logical scalar edit for an explicit or calibrated MHz value, or
    None when retaining a captured library value or reporting a missing input.
    missing is empty for a usable source, otherwise contains the required tool
    parameter and its reason. A retained library value keeps its captured Form
    source; no selector changes.
    """

    edit: FieldEdit | None
    missing: tuple[MissingParameter, ...] = ()


@dataclass(frozen=True)
class FrequencyCalibration:
    """Fallback policy for one fixed pulse frequency.

    path is the captured source scalar in MHz, such as $resonator.freq. required
    names the recipe input reported when no source is usable. prefer_library
    retains a finite captured library frequency before trying path; false skips
    that source. An unregistered path is treated as missing calibration.
    Explicit frequency always wins.
    """

    path: str
    required: str
    prefer_library: bool = True


def choose_frequency(
    tab: RecipeTab,
    field: FormPath,
    frequency_mhz: float | None = None,
    *,
    calibration: FrequencyCalibration,
) -> FrequencyChoice:
    """Choose an explicit, captured-library, or calibrated scalar frequency in MHz.

    tab is opened and supplies its fixed parameter calibration capture.
    field is a scalar path of str names or int original list indexes, such as
    ("modules", "readout", "freq"). calibration declares the fallback source
    path, missing-input name and library preference.

    A finite nonboolean frequency_mhz wins. Otherwise, when prefer_library is
    true, retain a finite error-free captured library scalar. Otherwise use a
    finite calibration as an expression edit labeled with its path. No usable
    source returns missing without edits or provenance changes. Invalid explicit
    values raise ValueError; an unregistered calibration is missing, while other
    source/connection errors propagate. Call
    tab.select(inline=True) on selector ancestors before applying a returned
    edit; this function copies no selectors.
    """
    path = calibration.path
    required = calibration.required
    prefer_library = calibration.prefer_library
    if frequency_mhz is not None:
        if not _finite(frequency_mhz):
            raise ValueError("frequency_mhz must be finite and not boolean")
        return FrequencyChoice(
            FieldEdit(path=field, value=frequency_mhz, source="explicit")
        )
    if prefer_library:
        state = tab.read(field)
        if (
            state.value_status == "available"
            and not state.errors
            and state.source is not None
            and _finite(state.value)
        ):
            tab.set(field, None)
            return FrequencyChoice(None)
    calibrated = _captured_parameter(tab, path)
    if _finite(calibrated):
        return FrequencyChoice(
            FieldEdit(path=field, value=Expr(text=path), source=path)
        )
    return FrequencyChoice(
        None,
        (
            MissingParameter(
                required,
                f"Provide an explicit frequency, valid library, or calibrated {path}",
            ),
        ),
    )


def apply_frequency(tab: RecipeTab, edit: FieldEdit) -> None:
    """Apply a choose_frequency edit and require a valid MHz scalar.

    tab is opened; edit carries the scalar path, value or calibration
    expression, and source label. Call tab.select(inline=True) on selector
    ancestors first. Write the edit, then require an available finite nonboolean
    value without field errors; otherwise raise GuiRpcError(invalid_cfg).
    Admission, validation and connection failures propagate; no selector is
    copied here.
    """
    tab.apply([edit])
    state = tab.read(edit.path)
    if state.value_status != "available" or state.errors or not _finite(state.value):
        raise GuiRpcError(f"Invalid {edit.path} frequency", reason="invalid_cfg")


@dataclass(frozen=True)
class FrequencySweepCalibration:
    """Caller-owned numeric frequency-range calibration policy.

    frequency_path and linewidth_path are nonblank logical scalar paths, such as
    $resonator.freq and $resonator.kappa, not arithmetic formulas. Both values
    use MHz. span_linewidths is a finite positive full-span multiplier: 3 means
    start=center-1.5*linewidth and stop=center+1.5*linewidth. It is not an
    endpoint coefficient. The helper validates the policy before any read or
    sweep edit.
    """

    frequency_path: str
    linewidth_path: str
    span_linewidths: float


def _validate_frequency_sweep_inputs(
    calibration: FrequencySweepCalibration,
    center_mhz: float | None,
    span_mhz: float | None,
    expts: int | None,
) -> None:
    """Reject invalid explicit values or calibration policy before any GUI read."""
    _points(expts)
    if not calibration.frequency_path.strip() or not calibration.linewidth_path.strip():
        raise ValueError("calibration paths must be non-empty logical scalar paths")
    if not _finite(calibration.span_linewidths) or calibration.span_linewidths <= 0:
        raise ValueError("span_linewidths must be finite and positive")
    if center_mhz is not None and not _finite(center_mhz):
        raise ValueError("center_mhz must be finite and not boolean")
    if span_mhz is not None and (not _finite(span_mhz) or span_mhz <= 0):
        raise ValueError("span_mhz must be finite and positive")


def _frequency_endpoints(center: object, span: object) -> tuple[float, float]:
    """Compute finite increasing MHz edges from finite center and positive span."""
    if not _finite(center) or not _finite(span):
        raise ValueError("center and span must be finite")
    start, stop = center - span / 2, center + span / 2
    if (
        not _finite(start)
        or not _finite(stop)
        or stop <= start
        or not _finite(stop - start)
    ):
        raise ValueError("frequency range must be finite and increasing")
    return float(start), float(stop)


@dataclass(frozen=True)
class FrequencySweepRange:
    """Requested frequency range and sampling, before calibration fallback.

    center_mhz is the center in MHz, or None to read the calibration frequency.
    span_mhz is the positive full span in MHz, or None to derive it from the
    calibrated linewidth and caller multiplier. expts is the count, at least
    two, or None to keep the GUI count. This carrier invents no defaults.
    """

    center_mhz: float | None = None
    span_mhz: float | None = None
    expts: int | None = None


def set_frequency_sweep(
    tab: RecipeTab,
    field: FormPath,
    *,
    calibration: FrequencySweepCalibration,
    scan: FrequencySweepRange,
) -> tuple[MissingParameter, ...]:
    """Write numeric MHz endpoints using explicit values or current calibration.

    tab is opened and supplies its fixed parameter calibration capture.
    field uses str names or int original list indexes. calibration supplies
    frequency/linewidth paths and a full-span multiplier. scan supplies the
    requested center/span/count. Explicit center/span win; omitted values read
    calibration now. Write center +/- span/2. expts is an integer >=2 or None to
    retain the GUI count. Explicit edges carry explicit provenance; derived
    edges retain their calibration/linewidth labels.

    Return all missing center_mhz/span_mhz inputs as MissingParameter values
    without sweep edits when center is unavailable/nonfinite or the required
    linewidth is boolean, nonfinite or nonpositive. Return () after success.
    Invalid explicit inputs, policy, range or sweep path raise ValueError;
    explicit inputs and policy are checked before reads/edits. Source and native
    cfg failures propagate.
    """
    center_mhz, span_mhz, expts = scan.center_mhz, scan.span_mhz, scan.expts
    _validate_frequency_sweep_inputs(calibration, center_mhz, span_mhz, expts)
    state = tab.read(field)
    if not {"start", "stop", "n"}.issubset(state.fields):
        raise ValueError(f"Not a sweep cfg field: {field}")
    missing: list[MissingParameter] = []
    center = (
        _captured_parameter(tab, calibration.frequency_path)
        if center_mhz is None
        else center_mhz
    )
    if not _finite(center):
        missing.append(
            MissingParameter(
                "center_mhz", f"No finite {calibration.frequency_path} calibration"
            )
        )
    span = span_mhz
    if span is None:
        width = _captured_parameter(tab, calibration.linewidth_path)
        if _finite(width) and width > 0:
            span = calibration.span_linewidths * width
        if span is None or not _finite(span) or span <= 0:
            missing.append(
                MissingParameter(
                    "span_mhz",
                    f"No finite positive {calibration.linewidth_path} linewidth",
                )
            )
    if missing:
        return tuple(missing)
    start, stop = _frequency_endpoints(center, span)
    edge_source = (
        "explicit"
        if center_mhz is not None and span_mhz is not None
        else "gui_calibration"
        if center_mhz is None and span_mhz is None
        else calibration.frequency_path
        if center_mhz is None
        else "gui_linewidth"
    )
    if expts is not None:
        tab.apply(
            [
                FieldEdit(
                    path=field,
                    value=LinRange(start=start, stop=stop, n=expts),
                    source=edge_source,
                ),
            ]
        )
    else:
        tab.apply(
            [
                FieldEdit(path=(*field, "start"), value=start, source=edge_source),
                FieldEdit(path=(*field, "stop"), value=stop, source=edge_source),
            ]
        )
        tab.set((*field, "n"), None)
    return ()


def set_flux_sweep(
    tab: RecipeTab,
    field: FormPath,
    *,
    start: float | None,
    stop: float | None,
    expts: int | None = None,
) -> None:
    """Write explicit distinct native flux endpoints, without unit conversion.

    tab is opened; field uses str names or int original list indexes. start/stop
    must be finite nonboolean native coordinates in either direction. expts is
    an integer >=2 or None to retain the GUI count. Invalid endpoints or count
    raise ValueError before edits; native cfg/connection errors propagate.
    Supplied values have explicit provenance; an omitted count has gui_default.
    """
    _points(expts)
    if not _finite(start) or not _finite(stop) or start == stop:
        raise ValueError("flux endpoints must both be finite and distinct")
    if expts is not None:
        tab.set(field, LinRange(start=float(start), stop=float(stop), n=expts))
    else:
        tab.set((*field, "start"), float(start))
        tab.set((*field, "stop"), float(stop))
        tab.set((*field, "n"), None)


def use_flux_device(tab: RecipeTab, name: str, *, unit: str) -> None:
    """Select a registered flux device after exact cached-snapshot unit validation.

    tab is opened; name/unit must be nonblank strings or raise ValueError. unit
    must exactly match the cached snapshot unit, including native.
    Missing/blank/none snapshot units or a mismatch raise
    GuiRpcError(invalid_device); unknown-device and field errors propagate.
    Validate before editing flux_dev/flux_unit. No device connection or
    coordinate conversion occurs.
    """
    if not name or not name.strip():
        raise ValueError("flux device name must be non-empty")
    if not unit or not unit.strip():
        raise ValueError("flux unit must be non-empty")
    tab.read(("flux_dev",))
    observed = tab.call(
        "device.snapshot", DeviceParams(name=name), result=DeviceReply
    ).snapshot.unit
    if not isinstance(observed, str) or not observed.strip() or observed == "none":
        raise GuiRpcError(
            "Flux device has no usable flux coordinate", reason="invalid_device"
        )
    if unit != observed:
        raise GuiRpcError(
            "Flux unit does not match the device coordinate", reason="invalid_device"
        )
    tab.set(("flux_dev",), name)
    tab.set(("flux_unit",), observed)


class DeviceParams(BaseModel):
    """Read one registered device's cached snapshot, without connecting.

    name is the exact registered device name, not a qubit alias.
    """

    name: str


class CachedDevice(BaseModel):
    """Cached device coordinate metadata.

    unit is the device's exact native coordinate label, or None when absent.
    """

    unit: str | None


class DeviceReply(BaseModel):
    """Reply to device.snapshot.

    snapshot contains cached coordinate metadata; this reply does not poll.
    """

    snapshot: CachedDevice


class DisableParams(BaseModel):
    """Disable one nullable Form selector on the host-bound tab.

    path is the logical selector tuple with original list indexes. action is the
    existing RPC's disable command; it does not mean set(null).
    """

    path: FormPath
    action: Literal["disable"] = "disable"


def select_or_disable(
    tab: RecipeTab, path: FormPath, ref: str | Literal["off"]
) -> None:
    """Select a preparation module, or disable it explicitly with off.

    tab is open and path is a nullable module selector. ref is an actual library
    key or off; None is not accepted. The GUI checks locks, sources and
    ancestry. A failed disable observation raises invalid_cfg. This does not
    connect hardware or create a default pulse.
    """
    if ref == "off":
        reply = tab.call(
            "tab.select_recipe_module",
            DisableParams(path=path),
            result=RecipeSelectionResult,
        )
        field = reply["field"]
        selector = field["selector"]
        if selector is None or selector["mode"] != "disabled" or field["errors"]:
            raise GuiRpcError(f"Cannot disable {path}", reason="invalid_cfg")
    else:
        tab.select(path, ref=ref)


def require_pi(
    tab: RecipeTab, path: FormPath, ref: str | None, *, required: str
) -> MissingParameter | None:
    """Select or retain a calibrated pi or pi/2 pulse reference.

    tab is open; path names the pulse selector and required names its recipe
    input. ref=None keeps the GUI selection. Return MissingParameter for a
    retained selection without an available, error-free reference key; inline
    pulses do not count as calibration. Explicit invalid selections raise
    invalid_cfg. Owner errors propagate without copying or changing defaults.
    """
    tab.select(path, ref=ref)
    state = tab.read(path)
    selector = state.selector
    if selector is None:
        raise ValueError(f"Not a module reference field: {path}")
    valid = (
        selector.mode == "ref"
        and selector.key is not None
        and state.value_status == "available"
        and not state.errors
    )
    if ref is not None and (not valid or selector.key != ref):
        raise GuiRpcError(f"Invalid {path} reference", reason="invalid_cfg")
    if ref is None and not valid:
        return MissingParameter(required, f"Provide a valid {path} library key")
    return None
