"""Project evidence-backed processed collections and retain only residual HDF5 leaves."""

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fluxdep.io import dump_spectrums, load_spectrums
from zcu_tools.analysis.fluxdep.models import SpectrumResult

from .models import KeyMappingItem
from .state import MigrationSession, clear_pending, record_key_mapping, record_pending

SPECTRUM_PATH = Path("data/fluxdep/spectrums.hdf5")
LEGACY_PATH = Path("data/fluxdep/spectrums.legacy.hdf5")


@dataclass
class _Projection:
    """Canonical results, consumed source leaves and located conversion gaps."""

    results: dict[str, SpectrumResult] = field(default_factory=dict)
    datasets: dict[tuple[str, ...], tuple[str, ...]] = field(default_factory=dict)
    attributes: set[tuple[str, ...]] = field(default_factory=set)
    pending: dict[str, str] = field(default_factory=dict)


def _array(group: h5py.Group, name: str, ndim: int) -> NDArray[np.float64]:
    item = group.get(name)
    if not isinstance(item, h5py.Dataset):
        raise ValueError(f"Missing dataset {group.name}/{name}")
    value = np.asarray(item[()])
    if value.ndim != ndim or value.dtype.kind not in "iuf":
        raise ValueError(f"{item.name}: expected {ndim}-dimensional numeric values")
    return value


def _scalar(group: h5py.Group, name: str) -> float:
    value = _array(group, name, 0)
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{group.name}/{name}: calibration must be finite")
    return result


def _project_group(name: str, group: h5py.Group, plan: _Projection) -> None:
    old_axes = "mA_c" in group
    half_key = (
        "mA_c" if old_axes else "flx_half" if "flx_half" in group else "flux_half"
    )
    period_key = (
        "period"
        if old_axes
        else "flx_period"
        if "flx_period" in group
        else "flux_period"
    )
    int_key = "flx_int" if "flx_int" in group else "flux_int"
    half, period = _scalar(group, half_key), _scalar(group, period_key)
    if period <= 0:
        raise ValueError(f"{group.name}/{period_key}: positive period required")
    integer = half + 0.5 * period if old_axes else _scalar(group, int_key)
    spectrum, points = group.get("spectrum"), group.get("points")
    if not isinstance(spectrum, h5py.Group) or not isinstance(points, h5py.Group):
        raise ValueError(f"{group.name}: spectrum and points groups required")
    dev_key, freq_key, signal_key = (
        ("mAs", "fpts", "data") if old_axes else ("dev_values", "freqs", "signals")
    )
    dev, freqs = _array(spectrum, dev_key, 1), _array(spectrum, freq_key, 1)
    picked, picked_freqs = _array(points, dev_key, 1), _array(points, freq_key, 1)
    signal_data = spectrum.get(signal_key)
    if not isinstance(signal_data, h5py.Dataset):
        raise ValueError(f"{spectrum.name}/{signal_key}: signals dataset required")
    signals = np.asarray(signal_data[()])
    if signals.ndim != 2 or signals.dtype.kind not in "iufc":
        raise ValueError(f"{signal_data.name}: numeric 2D signals required")
    if old_axes and len(dev) == len(freqs):
        raise ValueError(
            "Square axes lack historical version evidence for signal axis order"
        )
    if old_axes and signals.shape == (len(freqs), len(dev)):
        # Earlier processing reverses spectrum[:, ::-1] for mAs: frequency-major.
        signals = signals.T
    if signals.shape != (len(dev), len(freqs)):
        raise ValueError(
            "Signal axis shape does not match the historical device/frequency axes"
        )
    if picked.shape != picked_freqs.shape:
        raise ValueError("Picked device/frequency axes must have matching lengths")
    result = SpectrumResult(
        flux_half=half,
        flux_int=integer,
        flux_period=period,
        spectrum={
            "dev_values": dev,
            "freqs": freqs,
            "signals": signals,
            "fluxs": (dev - half) / period + 0.5
            if old_axes
            else _array(spectrum, "fluxs", 1),
        },
        points={
            "dev_values": picked,
            "freqs": picked_freqs,
            "fluxs": (picked - half) / period + 0.5
            if old_axes
            else _array(points, "fluxs", 1),
        },
    )
    metadata_type = group.attrs.get("type")
    if isinstance(metadata_type, str):
        result["type"] = metadata_type
        plan.attributes.add((name, "type"))
    consumed: dict[tuple[str, ...], tuple[str, ...]] = {
        (name, half_key): (name, "flux_half"),
        (name, period_key): (name, "flux_period"),
        (name, "spectrum", dev_key): (name, "spectrum", "dev_values"),
        (name, "spectrum", freq_key): (name, "spectrum", "freqs"),
        (name, "spectrum", signal_key): (name, "spectrum", "signals"),
        (name, "points", dev_key): (name, "points", "dev_values"),
        (name, "points", freq_key): (name, "points", "freqs"),
    }
    if not old_axes:
        consumed[(name, int_key)] = (name, "flux_int")
        for section in ("spectrum", "points"):
            consumed[(name, section, "fluxs")] = (name, section, "fluxs")
    plan.results[name] = result
    plan.datasets.update(consumed)


def _plan(source: Path) -> _Projection:
    plan = _Projection()
    with h5py.File(source, "r") as handle:
        for name, item in handle.items():
            if not isinstance(item, h5py.Group):
                continue
            # Unknown groups are unsupported content, not missing conversion evidence.
            if not any(
                key in item
                for key in ("mA_c", "flx_half", "flux_half", "spectrum", "points")
            ):
                continue
            try:
                _project_group(name, item, plan)
            except ValueError as error:
                plan.pending[name] = str(error)
    return plan


def _residual_rows(
    source: Path, plan: _Projection
) -> list[tuple[tuple[str, ...], bool]]:
    rows: list[tuple[tuple[str, ...], bool]] = []

    def visit(group: h5py.Group, keys: tuple[str, ...]) -> None:
        for attr in group.attrs:
            attr_keys = (*keys, str(attr))
            if attr_keys not in plan.attributes:
                rows.append((attr_keys, True))
        for name, item in group.items():
            path = (*keys, name)
            if isinstance(item, h5py.Group):
                if not len(item) and not len(item.attrs):
                    rows.append((path, False))
                visit(item, path)
            elif isinstance(item, h5py.Dataset):
                if path not in plan.datasets:
                    rows.append((path, False))
                for attr in item.attrs:
                    rows.append(((*path, str(attr)), True))

    with h5py.File(source, "r") as handle:
        visit(handle, ())
    return rows


def _write_residual(source: Path, target: Path, plan: _Projection) -> None:
    def copy(group: h5py.Group, output: h5py.Group, keys: tuple[str, ...]) -> None:
        for attr in group.attrs:
            if (*keys, str(attr)) not in plan.attributes:
                output.attrs.create(
                    attr, group.attrs[attr], dtype=group.attrs.get_id(attr).dtype
                )
        for name, item in group.items():
            path = (*keys, name)
            if isinstance(item, h5py.Group):
                child = output.create_group(name)
                copy(item, child, path)
                if (
                    not len(child)
                    and not len(child.attrs)
                    and (len(item) or len(item.attrs))
                ):
                    del output[name]
            elif isinstance(item, h5py.Dataset):
                if path not in plan.datasets:
                    group.copy(item, output, name=name)
                elif len(item.attrs):
                    # A metadata-only group keeps the original path, without dataset data.
                    child = output.create_group(name)
                    for attr in item.attrs:
                        child.attrs.create(
                            attr, item.attrs[attr], dtype=item.attrs.get_id(attr).dtype
                        )

    with h5py.File(source, "r") as handle, h5py.File(target, "x") as output:
        copy(handle, output, ())


def convert_processed_spectrums(session: MigrationSession) -> None:
    """Publish supported processed spectra and residual HDF5 content in the entry.

    session supplies verified legacy roots and publication ownership. Only the
    explicit data/fluxdep/spectrums.hdf5 route is recognized. Historical mA_c,
    period, mAs/fpts/data map without unit scaling; non-square shape uniquely
    selects the two evidenced signal orders. Old fluxs and integer-flux values
    use the documented historical formulas. Square old axes stay located pending.
    Formal output is written/read by fluxdep.io. Unmapped datasets and attributes
    retain hierarchy/values/dtypes in spectrums.legacy.hdf5; consumed datasets with
    attrs become metadata-only groups there. Runtime does not read the companion.
    Dry-run only plans; owned prepared/published hashes recover on resume. Any
    unowned collision or source/destination hash change raises before overwrite.
    """
    report = session.manifest.report
    source = report.source.result_path / SPECTRUM_PATH
    if not source.is_file():
        return
    plan = _plan(source)
    target = report.destination.result_path / SPECTRUM_PATH
    legacy = report.destination.result_path / LEGACY_PATH
    clear_pending(session, source, "processed spectrums")
    if plan.pending:
        reasons = "; ".join(
            f"{name}: {reason}" for name, reason in plan.pending.items()
        )
        record_pending(
            session,
            source,
            "processed spectrums",
            f"{reasons}; original retained at {legacy}",
        )
    rows = _residual_rows(source, plan)
    _report_projection(session, source, target, legacy, plan, rows)
    if plan.results:

        def build(path: Path) -> None:
            dump_spectrums(str(path), plan.results)
            load_spectrums(str(path))

        state = session.publish(source, target, operation="spectrums", build=build)
        if not session.dry_run:
            load_spectrums(str(target))
        session.completed(state)
    if rows:
        state = session.publish(
            source,
            legacy,
            operation="spectrums_legacy",
            build=lambda path: _write_residual(source, path, plan),
        )
        session.completed(state)


def _report_projection(
    session: MigrationSession,
    source: Path,
    target: Path,
    legacy: Path,
    plan: _Projection,
    rows: list[tuple[tuple[str, ...], bool]],
) -> None:
    for keys, destination in plan.datasets.items():
        record_key_mapping(
            session,
            KeyMappingItem(
                old_file=source,
                old_key=f"dataset:{keys!r}",
                source_keys=keys,
                new_file=target,
                new_path="/".join(destination),
                destination_keys=destination,
                action="value",
                reason="Historical processed-spectrum reader/writer mapping; signal order follows unique shape and historical axis rules; no numeric unit conversion",
            ),
        )
    for keys in plan.attributes:
        record_key_mapping(
            session,
            KeyMappingItem(
                old_file=source,
                old_key=f"attribute:{keys!r}",
                source_keys=keys,
                new_file=target,
                new_path="/".join(keys),
                destination_keys=keys,
                action="value",
                reason="Supported type attribute retained unchanged",
            ),
        )
    for name in plan.results:
        if (name, "mA_c") not in plan.datasets:
            continue
        for destination in (
            (name, "flux_int"),
            (name, "spectrum", "fluxs"),
            (name, "points", "fluxs"),
        ):
            record_key_mapping(
                session,
                KeyMappingItem(
                    old_file=source,
                    old_key=f"derived:{destination!r}",
                    source_keys=(name, "mA_c"),
                    new_file=target,
                    new_path="/".join(destination),
                    destination_keys=destination,
                    action="value",
                    reason="Historical derivation: flux_int=mA_c+0.5*period (187e2d75a); fluxs=(mAs-mA_c)/period+0.5 (8d08978c2); source mAs and period in the same group, no unit scaling",
                ),
            )
    for keys, attribute in rows:
        record_key_mapping(
            session,
            KeyMappingItem(
                old_file=source,
                old_key=f"{'attribute' if attribute else 'dataset'}:{keys!r}",
                source_keys=keys,
                new_file=legacy,
                new_path=None,
                destination_keys=keys,
                action="preserved",
                reason="Unconverted original HDF5 attribute"
                if attribute
                else "Unconverted original HDF5 dataset/group",
            ),
        )
