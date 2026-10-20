"""Frequency and native-flux policy composed from generic recipe cfg operations."""

from dataclasses import dataclass
from math import isfinite
from typing import TypeGuard

from zcu_tools.mcp.measure.recipe import (
    MissingParameter,
    RecipeExpression,
    RecipeFieldEdit,
    RecipeFieldPath,
    RecipeTab,
)
from zcu_tools.mcp.measure.recipe_capture import SweepSources
from zcu_tools.mcp.measure.session import GuiRpcError


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


def retain_scalar(tab: RecipeTab, field: RecipeFieldPath) -> None:
    """Retain one available logical scalar and its captured source label.

    tab is opened; field uses str names or int original list indexes.
    Record library:<actual-key> for a captured source; otherwise retain authored
    input with gui_default provenance. Unavailable values or field errors raise
    GuiRpcError(invalid_cfg); path/source/connection errors propagate.
    No selector is copied or changed.
    """
    state = tab.read_field(field)
    if state.value_status != "available" or state.errors:
        raise GuiRpcError(f"Invalid retained {field}", reason="invalid_cfg")
    if state.source is not None:
        tab.retain_source(field, f"library:{state.source['key']}")
    else:
        tab.set(field, None)


@dataclass(frozen=True)
class FrequencyChoice:
    """A frequency policy decision, before the recipe copies or edits selectors.

    edit is the logical scalar edit for an explicit or calibrated MHz value,
    or None when retaining a captured library value or reporting a missing input.
    missing is empty for a usable source, otherwise contains the required tool
    parameter and its reason. A retained library value already has its provenance
    recorded by choose_frequency, but no Form input or selector changes.
    """

    edit: RecipeFieldEdit | None
    missing: tuple[MissingParameter, ...] = ()


def choose_frequency(
    tab: RecipeTab,
    field: RecipeFieldPath,
    frequency_mhz: float | None = None,
    *,
    calibration: str,
    prefer_library: bool = True,
    required: str,
) -> FrequencyChoice:
    """Choose an explicit, captured-library, or calibrated scalar frequency in MHz.

    tab is opened; field is a scalar path of str names or int original list
    indexes, such as ("modules", "readout", "freq"). calibration is a logical
    scalar path such as $resonator.freq, not a formula; required names the tool
    input to report when no source is usable.

    A finite nonboolean frequency_mhz wins. Otherwise, when prefer_library is
    true, retain a finite error-free captured library scalar and record
    library:<actual-key>. Otherwise use a finite calibration as an expression
    edit labeled with its path. No usable source returns missing without edits
    or provenance changes. Invalid explicit values raise ValueError;
    source/connection errors propagate. Call use_inline on selector ancestors
    before applying a returned edit; this function copies no selectors.
    """
    if frequency_mhz is not None:
        if not _finite(frequency_mhz):
            raise ValueError("frequency_mhz must be finite and not boolean")
        return FrequencyChoice(RecipeFieldEdit(field, frequency_mhz, "explicit"))
    if prefer_library:
        state = tab.read_field(field)
        if (
            state.value_status == "available"
            and not state.errors
            and state.source is not None
            and _finite(state.value)
        ):
            tab.retain_source(field, f"library:{state.source['key']}")
            return FrequencyChoice(None)
    calibrated = tab.read_value(calibration)
    if _finite(calibrated):
        return FrequencyChoice(
            RecipeFieldEdit(field, RecipeExpression(expr=calibration), calibration)
        )
    return FrequencyChoice(
        None,
        (
            MissingParameter(
                required,
                f"Provide an explicit frequency, valid library, or calibrated {calibration}",
            ),
        ),
    )


def apply_frequency(tab: RecipeTab, edit: RecipeFieldEdit) -> None:
    """Apply a choose_frequency edit and require a valid MHz scalar.

    tab is opened; edit carries the scalar path, value or calibration expression,
    and source label. Call use_inline on selector ancestors first.
    Write the edit, then require an available finite nonboolean value without
    field errors; otherwise raise GuiRpcError(invalid_cfg). Admission,
    validation and connection failures propagate; no selector is copied here.
    """
    tab.set_fields([edit])
    state = tab.read_field(edit.field)
    if state.value_status != "available" or state.errors or not _finite(state.value):
        raise GuiRpcError(f"Invalid {edit.field} frequency", reason="invalid_cfg")


@dataclass(frozen=True)
class FrequencySweepCalibration:
    """Caller-owned numeric frequency-range calibration policy.

    frequency_path and linewidth_path are nonblank logical scalar paths, such as
    $resonator.freq and $resonator.kappa, not arithmetic formulas. Both values
    use MHz. span_linewidths is a finite positive full-span multiplier: 3 means
    start=center-1.5*linewidth and stop=center+1.5*linewidth. It is not an endpoint
    coefficient. The helper validates the policy before any read or sweep edit.
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


def set_frequency_sweep(
    tab: RecipeTab,
    field: RecipeFieldPath,
    *,
    calibration: FrequencySweepCalibration,
    center_mhz: float | None = None,
    span_mhz: float | None = None,
    expts: int | None = None,
) -> tuple[MissingParameter, ...]:
    """Write numeric MHz endpoints using explicit values or current calibration.

    tab is opened; field uses str names or int original list indexes.
    calibration supplies frequency/linewidth paths and a full-span multiplier.
    Explicit center/span win; omitted values read calibration now. Write
    center +/- span/2. expts is an integer >=2 or None to retain the GUI count.
    Explicit edges carry explicit provenance; derived edges retain their
    calibration/linewidth labels.

    Return all missing center_mhz/span_mhz inputs as MissingParameter values
    without sweep edits when center is unavailable/nonfinite or the required
    linewidth is boolean, nonfinite or nonpositive. Return () after success.
    Invalid explicit inputs, policy, range or sweep path raise ValueError;
    explicit inputs and policy are checked before reads/edits.
    Source and native cfg failures propagate.
    """
    _validate_frequency_sweep_inputs(calibration, center_mhz, span_mhz, expts)
    state = tab.read_field(field)
    if not {"start", "stop", "n"}.issubset(state.fields):
        raise ValueError(f"Not a sweep cfg field: {field}")
    missing: list[MissingParameter] = []
    center = (
        tab.read_value(calibration.frequency_path) if center_mhz is None else center_mhz
    )
    if not _finite(center):
        missing.append(
            MissingParameter(
                "center_mhz", f"No finite {calibration.frequency_path} calibration"
            )
        )
    span = span_mhz
    if span is None:
        width = tab.read_value(calibration.linewidth_path)
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
    sources: SweepSources = {
        "start": edge_source,
        "stop": edge_source,
        "expts": "explicit",
    }
    tab.set_sweep(field, start=start, stop=stop, expts=expts, sources=sources)
    return ()


def set_flux_sweep(
    tab: RecipeTab,
    field: RecipeFieldPath,
    *,
    start: float | None,
    stop: float | None,
    expts: int | None = None,
) -> None:
    """Write explicit distinct native flux endpoints, without unit conversion.

    tab is opened; field uses str names or int original list indexes.
    start/stop must be finite nonboolean native coordinates in either direction.
    expts is an integer >=2 or None to retain the GUI count. Invalid endpoints
    or count raise ValueError before edits; native cfg/connection errors propagate.
    Supplied values have explicit provenance; an omitted count has gui_default.
    """
    _points(expts)
    if not _finite(start) or not _finite(stop) or start == stop:
        raise ValueError("flux endpoints must both be finite and distinct")
    tab.set_sweep(
        field,
        start=start,
        stop=stop,
        expts=expts,
        sources={"start": "explicit", "stop": "explicit", "expts": "explicit"},
    )


def use_flux_device(tab: RecipeTab, name: str, *, unit: str) -> None:
    """Select a registered flux device after exact cached-snapshot unit validation.

    tab is opened; name/unit must be nonblank strings or raise ValueError.
    unit must exactly match the cached snapshot unit, including native.
    Missing/blank/none snapshot units or a mismatch raise
    GuiRpcError(invalid_device); unknown-device and field errors propagate.
    Validate before editing flux_dev/flux_unit. No device connection or
    coordinate conversion occurs.
    """
    if not name or not name.strip():
        raise ValueError("flux device name must be non-empty")
    if not unit or not unit.strip():
        raise ValueError("flux unit must be non-empty")
    tab.read_field(("flux_dev",))
    observed = tab.read_device(name).unit
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
