"""Frequency and native-flux policy composed from generic recipe cfg operations."""

from dataclasses import dataclass
from math import isfinite
from typing import TypeGuard

from zcu_tools.mcp.measure.recipe import (
    MissingParameter,
    RecipeExpression,
    RecipeFieldEdit,
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


# A frequency leaf carries its dotted Form path, source module, and value projection.
# The Form path includes S2 inline envelopes (module); read_module values do not.
type _FrequencyLeaf = tuple[str, str | None, tuple[str, ...]]


def _frequency_fields(
    tab: RecipeTab, field: str
) -> tuple[list[_FrequencyLeaf], str | None]:
    """Return scalar paths with their source module/projection and outer edit root."""
    parts = field.split(".")
    # Find the module before reading a leaf that does not exist in key input.
    for length in range(1, len(parts)):
        parent_path = ".".join(parts[:length])
        parent = tab.read_field(parent_path)
        if parent.mode in {"key", "inline"}:
            return [
                (field, parent_path, _module_value_path(parent_path, field))
            ], parent_path
    node = tab.read_field(field)
    if node.mode in {"key", "inline"}:
        fields = (
            tab.read_module(field).value
            if node.mode == "key"
            else tab.read_field(f"{field}.module").fields
        )
        if "freq" in fields:
            leaves: list[_FrequencyLeaf] = [(f"{field}.module.freq", field, ("freq",))]
        else:
            raise GuiRpcError("Unsupported readout shape", reason="invalid_cfg")
        return leaves, field
    if node.fields:
        raise ValueError(f"Not a frequency field: {field}")
    return [(field, None, ())], None


def _module_value_path(module_field: str, field: str) -> tuple[str, ...]:
    prefix = f"{module_field}.module."
    if not field.startswith(prefix):
        raise ValueError(f"Not an inline module frequency field: {field}")
    return tuple(field.removeprefix(prefix).split("."))


def set_module_scalar(tab: RecipeTab, module_field: str, edit: RecipeFieldEdit) -> None:
    """Edit a flat module scalar, retaining key input/source when value is None.

    module_field is a complete public S2 module path; edit.field is a scalar
    directly under its inline module envelope. Explicit values convert the
    module with use_inline before writing. None retains a key's scalar value
    and provenance; inline input follows RecipeTab.set's default retention.
    Path, validation, lock and source errors propagate without fallback.
    """
    value_path = _module_value_path(module_field, edit.field)
    if edit.value is not None:
        tab.use_inline(module_field)
    elif tab.read_field(module_field).mode == "key":
        tab.retain_module_source(
            module_field, summary_field=edit.field, value_path=value_path
        )
        return
    tab.set(edit.field, edit.value, source=edit.source)


def set_nested_module_scalar(
    tab: RecipeTab,
    outer_field: str,
    child_field: str,
    edit: RecipeFieldEdit,
    outer_value_path: tuple[str, ...],
) -> None:
    """Set or retain a scalar whose nested module may still be a library key.

    tab is opened with fixed Reset sources. outer_field and child_field are
    complete public S2 module paths. edit carries the complete scalar path
    directly under child_field, its value, and its unchanged provenance label.
    outer_value_path is the core-relative projection from outer_field to that
    scalar, without inline wrappers. A supplied value converts a key outer
    module and then the child through use_inline before set. None retains an
    outer key first, otherwise a child key, with the same absolute summary
    field. If both modules are inline, set(None) preserves input and records
    gui_default. Child-relative projection follows the public scalar path.
    Path, validation, lock-admission, and source/connection errors propagate
    from RecipeTab without refresh, fallback, or rollback.
    No caller layout construction or returned-data patching occurs.
    """
    child_value_path = _module_value_path(child_field, edit.field)
    if edit.value is not None:
        if tab.read_field(outer_field).mode == "key":
            tab.use_inline(outer_field)
        tab.use_inline(child_field)
    elif tab.read_field(outer_field).mode == "key":
        tab.retain_module_source(
            outer_field, summary_field=edit.field, value_path=outer_value_path
        )
        return
    elif tab.read_field(child_field).mode == "key":
        tab.retain_module_source(
            child_field, summary_field=edit.field, value_path=child_value_path
        )
        return
    tab.set(edit.field, edit.value, source=edit.source)


def _usable_frequency(tab: RecipeTab, leaf: _FrequencyLeaf) -> bool:
    field, module_field, value_path = leaf
    if module_field is not None and tab.read_field(module_field).mode == "key":
        value: object = tab.read_module(module_field).value
        for part in value_path:
            if not isinstance(value, dict):
                return False
            value = value.get(part)
        return _finite(value)
    state = tab.read_field(field)
    return not state.errors and _finite(state.value)


def _write_frequency_fields(
    tab: RecipeTab,
    module_field: str | None,
    leaves: list[_FrequencyLeaf],
    edits: list[RecipeFieldEdit],
) -> None:
    """Convert only for edits, apply one scalar batch, then verify all frequencies."""
    if edits and module_field is not None:
        tab.use_inline(module_field)
    tab.set_fields(edits)
    for leaf in leaves:
        if not _usable_frequency(tab, leaf):
            raise GuiRpcError(f"Invalid {leaf[0]} frequency", reason="invalid_cfg")


def set_frequency(
    tab: RecipeTab,
    field: str,
    frequency_mhz: float | None = None,
    *,
    calibration: str,
    prefer_library: bool = True,
    required: str,
) -> MissingParameter | None:
    """Select explicit, usable library, or calibrated frequency in MHz.

    tab is opened; field is a dotted scalar frequency or core readout module root.
    Inline leaf paths include module, for example modules.qub_pulse.module.freq.
    Key input stays unchanged when retaining library values; explicit/calibrated
    writes first ask the owner to convert the whole module to inline. Retained
    library values and sources use the public module read. A readout root
    writes its single frequency at <root>.module.freq for generator and ADC.
    Other readout shapes raise GuiRpcError(invalid_cfg).
    calibration is a caller-owned logical scalar path (not a formula). required
    names the missing tool parameter when no source is usable. Explicit finite
    nonboolean frequency_mhz wins; otherwise prefer_library enables current
    valid library leaf selection, followed by finite calibration. Supplied values
    carry explicit provenance; library:<root> and calibration labels persist.
    Validate all leaves before one batch edit; retained library leaves cause no
    edit. Return None on success; absent sources return MissingParameter(required,
    reason) with no edit. Invalid explicit values raise ValueError. Invalid cfg
    and source/connection errors propagate without fallback or retry. Calibration
    uses a GUI expression.
    """
    if frequency_mhz is not None and not _finite(frequency_mhz):
        raise ValueError("frequency_mhz must be finite and not boolean")
    leaves, module_field = _frequency_fields(tab, field)
    library_roots = {
        path: tab.read_library_root(tab.read_field(source_field).key)
        if frequency_mhz is None and prefer_library and source_field is not None
        else None
        for path, source_field, _ in leaves
    }
    needs_calibration = frequency_mhz is None and any(
        library_roots[leaf[0]] is None or not _usable_frequency(tab, leaf)
        for leaf in leaves
    )
    calibrated = tab.read_value(calibration) if needs_calibration else None
    edits: list[RecipeFieldEdit] = []
    retained: list[_FrequencyLeaf] = []
    for leaf in leaves:
        path = leaf[0]
        if frequency_mhz is not None:
            edits.append(RecipeFieldEdit(path, frequency_mhz, "explicit"))
        elif library_roots[path] is not None and _usable_frequency(tab, leaf):
            retained.append(leaf)
        elif _finite(calibrated):
            edits.append(
                RecipeFieldEdit(path, RecipeExpression(expr=calibration), calibration)
            )
        else:
            return MissingParameter(
                required,
                f"Provide an explicit frequency, valid library, or calibrated {calibration}",
            )
    _write_frequency_fields(tab, module_field, leaves, edits)
    for path, source_field, value_path in retained:
        if source_field is not None:
            tab.retain_module_source(
                source_field,
                summary_field=path,
                value_path=value_path,
            )
    return None


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
    field: str,
    *,
    calibration: FrequencySweepCalibration,
    center_mhz: float | None = None,
    span_mhz: float | None = None,
    expts: int | None = None,
) -> tuple[MissingParameter, ...]:
    """Write numeric MHz endpoints using explicit values or current calibration.

    tab is an opened recipe tab; field names its dotted sweep path. calibration
    is the caller-owned path/full-width policy described by FrequencySweepCalibration.
    Missing center/span reads these paths now, then computes center +/- span/2.
    expts must be an integer >=2; None retains the GUI count. Explicit values
    carry explicit provenance; derived edges retain calibration/linewidth labels.
    Missing/nonfinite center or nonpositive/nonfinite/bool linewidth reports all
    missing center_mhz/span_mhz as a tuple of MissingParameter without sweep edits.
    Return an empty tuple after a successful sweep edit.
    Invalid explicit inputs raise ValueError before reads/edits. Source errors
    propagate without fallback/retry. Native cfg failures propagate from tab.
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
    field: str,
    *,
    start: float | None,
    stop: float | None,
    expts: int | None = None,
) -> None:
    """Write explicit distinct native flux endpoints, without unit conversion.

    tab is opened; field is a dotted sweep path. start/stop are finite nonboolean
    native coordinates in either direction. expts is integer >=2 or None to
    retain GUI count. Invalid endpoints/count raise ValueError before any edit.
    Native cfg/connection errors propagate without retry. Sources are explicit
    for supplied values and gui_default for omitted count.
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

    tab is opened; name/unit are nonblank strings. unit must match snapshot unit
    exactly, including explicitly requested native. Missing/blank/none snapshot
    units or mismatch raise GuiRpcError(invalid_device); unknown-device errors
    propagate from the owner. Validate before editing flux_dev/flux_unit.
    Never connect/setup a device, convert coordinates, or retry failures.
    """
    if not name or not name.strip():
        raise ValueError("flux device name must be non-empty")
    if not unit or not unit.strip():
        raise ValueError("flux unit must be non-empty")
    tab.read_field("flux_dev")
    observed = tab.read_device(name).unit
    if not isinstance(observed, str) or not observed.strip() or observed == "none":
        raise GuiRpcError(
            "Flux device has no usable flux coordinate", reason="invalid_device"
        )
    if unit != observed:
        raise GuiRpcError(
            "Flux unit does not match the device coordinate", reason="invalid_device"
        )
    tab.set("flux_dev", name)
    tab.set("flux_unit", observed)
