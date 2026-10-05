"""Labber Browser sidecar exports derived from autofluxdep Results."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zcu_tools.datafile import (
    LabberMetadata,
    StreamingLabberVariableSpec,
    StreamingLabberWriter,
    open_streaming_labber_data,
    save_labber_data,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import PlacedNode
from zcu_tools.gui.app.autofluxdep.results import (
    FrequencySweepResult,
    SweepResult1D,
    require_workflow_result,
)
from zcu_tools.gui.app.autofluxdep.services.artifact_paths import (
    relative_to_artifact,
    safe_artifact_slug,
)
from zcu_tools.gui.app.autofluxdep.services.fluxdep_export import (
    export_qubit_freq_fluxdep_spectrum,
)

LABBER_BROWSER_ROOT_EXPORT_KEY = "labber_browser_root"
LABBER_BROWSER_SIDECARS_EXPORT_KEY = "labber_browser_sidecars"

_RUN_SLUG_RE = re.compile(r"^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})-")


@dataclass(frozen=True)
class LabberBrowserSidecar:
    """One Labber Browser single-log export entry for the manifest.

    index is the placed-node index; node/node_type identify its name and type.
    variable names the exported measured variable. path is relative to the run
    data root. Serialization retains the existing manifest ``role`` key.
    """

    index: int
    node: str
    node_type: str
    variable: str
    path: str

    def to_manifest(self) -> dict[str, int | str]:
        """Serialize this export using the existing manifest ``role`` key."""
        return {
            "index": self.index,
            "node": self.node,
            "node_type": self.node_type,
            "role": self.variable,
            "path": self.path,
        }


@dataclass(frozen=True)
class LabberBrowserExport:
    """Manifest-ready Labber Browser export collection."""

    root: str
    sidecars: tuple[LabberBrowserSidecar, ...]

    def to_manifest_exports(self) -> dict[str, Any]:
        return {
            LABBER_BROWSER_ROOT_EXPORT_KEY: self.root,
            LABBER_BROWSER_SIDECARS_EXPORT_KEY: [
                sidecar.to_manifest() for sidecar in self.sidecars
            ],
        }


@dataclass(frozen=True)
class _StreamingSidecarSpec:
    sidecar: LabberBrowserSidecar
    spec: StreamingLabberVariableSpec


@dataclass(slots=True)
class _StreamingSidecarHandle:
    sidecar: LabberBrowserSidecar
    writer: StreamingLabberWriter


class LabberBrowserSidecarWriters:
    """Open live writers for Labber Browser single-log sidecars."""

    def __init__(
        self,
        *,
        data_root: Path,
        root_path: Path,
        handles_by_node: Mapping[str, Sequence[_StreamingSidecarHandle]],
        sidecars: Sequence[LabberBrowserSidecar],
    ) -> None:
        self._data_root = data_root
        self._root_path = root_path
        self._handles_by_node = {
            str(node): tuple(handles) for node, handles in handles_by_node.items()
        }
        self._sidecars = tuple(sidecars)
        self._closed = False

    @property
    def sidecars(self) -> tuple[LabberBrowserSidecar, ...]:
        return self._sidecars

    def to_manifest_export(self) -> LabberBrowserExport:
        return LabberBrowserExport(
            root=relative_to_artifact(self._data_root, self._root_path),
            sidecars=self._sidecars,
        )

    def write_node_row(
        self,
        node_name: str,
        node_type: str,
        result: object,
        flux_idx: int,
        *,
        timestamp: float | None = None,
    ) -> None:
        self._ensure_open()
        handles = self._handles_by_node.get(node_name, ())
        if not handles:
            return
        for handle in handles:
            row = _streaming_row_value(
                node_name, node_type, result, handle.sidecar.variable, flux_idx
            )
            handle.writer.write_outer_slice(flux_idx, row, timestamp=timestamp)
        self.flush()

    def flush(self) -> None:
        self._ensure_open()
        for handles in self._handles_by_node.values():
            for handle in handles:
                handle.writer.flush()

    def finalize(self) -> None:
        self._ensure_open()
        for handles in self._handles_by_node.values():
            for handle in handles:
                handle.writer.finalize()

    def close(self) -> None:
        if self._closed:
            return
        for handles in self._handles_by_node.values():
            for handle in handles:
                handle.writer.close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Labber Browser sidecar writers are closed")


def open_streaming_labber_browser_sidecars(
    *,
    data_root: str | Path,
    nodes: Sequence[PlacedNode],
    results: Mapping[str, object],
    metadata: LabberMetadata | None = None,
) -> LabberBrowserSidecarWriters:
    """Open live Labber Browser sidecar writers for supported fixed-axis Results."""
    data_root_path = Path(data_root)
    root_path = labber_browser_root(data_root_path)
    handles_by_node: dict[str, list[_StreamingSidecarHandle]] = {}
    sidecars: list[LabberBrowserSidecar] = []
    opened: list[StreamingLabberWriter] = []
    try:
        for index, node in enumerate(nodes):
            result = results.get(node.name)
            if result is None:
                continue
            for sidecar_spec in _streaming_node_sidecar_specs(
                data_root=data_root_path,
                root_path=root_path,
                index=index,
                node_name=node.name,
                node_type=node.type_name,
                result=result,
            ):
                path = data_root_path / sidecar_spec.sidecar.path
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = open_streaming_labber_data(
                    str(path),
                    sidecar_spec.spec,
                    metadata=metadata,
                )
                opened.append(writer)
                handles_by_node.setdefault(node.name, []).append(
                    _StreamingSidecarHandle(sidecar_spec.sidecar, writer)
                )
                sidecars.append(sidecar_spec.sidecar)
    except Exception:
        for writer in opened:
            writer.close()
        raise
    return LabberBrowserSidecarWriters(
        data_root=data_root_path,
        root_path=root_path,
        handles_by_node=handles_by_node,
        sidecars=tuple(sidecars),
    )


def export_labber_browser_sidecars(
    *,
    data_root: str | Path,
    nodes: Sequence[PlacedNode],
    results: Mapping[str, object],
    committed_masks: Mapping[str, NDArray[np.bool_]],
) -> LabberBrowserExport:
    """Write Labber Browser single-log sidecars for supported node Results."""
    root_path = labber_browser_root(data_root)
    entries: list[LabberBrowserSidecar] = []
    data_root_path = Path(data_root)

    for index, node in enumerate(nodes):
        result = results.get(node.name)
        if result is None:
            continue
        entries.extend(
            _export_node_sidecars(
                data_root=data_root_path,
                root_path=root_path,
                index=index,
                node_name=node.name,
                node_type=node.type_name,
                result=result,
                committed_mask=_committed_mask_for(node.name, result, committed_masks),
            )
        )

    return LabberBrowserExport(
        root=relative_to_artifact(data_root_path, root_path),
        sidecars=tuple(entries),
    )


def export_qubit_freq_labber_browser_sidecar(
    *,
    data_root: str | Path,
    index: int,
    node_name: str,
    node_type: str,
    result: FrequencySweepResult,
    committed_mask: NDArray[np.bool_],
) -> LabberBrowserSidecar:
    """Write a frequency-spectrum sidecar and return its manifest entry.

    data_root is the run directory whose name starts with YYYYMMDD-. index is
    the zero-based workflow position; node_name and node_type identify the node
    in metadata. result follows FrequencySweepResult; committed_mask selects
    persisted rows and has shape (n_flux,). No input arrays are mutated.
    Raise TypeError for invalid result kind/fields/dtypes and ValueError for
    inconsistent shapes/mask, an invalid root or an unusable frequency grid.
    File/IO errors propagate. Existing sidecar naming and axis units are retained.
    """
    return _export_qubit_freq(
        Path(data_root),
        labber_browser_root(data_root),
        index,
        node_name,
        node_type,
        result,
        committed_mask,
    )


def labber_browser_root(data_root: str | Path) -> Path:
    """Return the Labber Browser dated folder for a run-scoped data root."""
    root = Path(data_root)
    match = _RUN_SLUG_RE.match(root.name)
    if match is None:
        raise ValueError(
            "autofluxdep data_root name must start with YYYYMMDD- to derive "
            f"Labber Browser date, got {root.name!r}"
        )
    year = match.group("year")
    month = match.group("month")
    day = match.group("day")
    return root / "labber" / year / month / f"Data_{month}{day}"


def _export_node_sidecars(
    *,
    data_root: Path,
    root_path: Path,
    index: int,
    node_name: str,
    node_type: str,
    result: object,
    committed_mask: NDArray[np.bool_],
) -> tuple[LabberBrowserSidecar, ...]:
    if not committed_mask.any():
        return ()
    checked = require_workflow_result(result)
    if (
        isinstance(checked, FrequencySweepResult)
        and checked.result_kind == "qubit_freq"
    ):
        return (
            _export_qubit_freq(
                data_root,
                root_path,
                index,
                node_name,
                node_type,
                checked,
                committed_mask,
            ),
        )
    if checked.result_kind == "sweep1d" and node_type in {"lenrabi", "mist"}:
        return (
            _export_sweep1d_signal(
                data_root,
                root_path,
                index,
                node_name,
                node_type,
                _require_sweep1d(node_name, result),
                committed_mask,
            ),
        )
    if checked.result_kind == "sweep1d" and node_type in _SWEEP1D_SCALAR_VARIABLES:
        sweep = _require_sweep1d(node_name, result)
        return (
            _export_sweep1d_signal(
                data_root,
                root_path,
                index,
                node_name,
                node_type,
                sweep,
                committed_mask,
            ),
            _export_sweep1d_scalar(
                data_root,
                root_path,
                index,
                node_name,
                node_type,
                sweep,
                committed_mask,
            ),
        )
    return ()


def _streaming_node_sidecar_specs(
    *,
    data_root: Path,
    root_path: Path,
    index: int,
    node_name: str,
    node_type: str,
    result: object,
) -> tuple[_StreamingSidecarSpec, ...]:
    checked = require_workflow_result(result)
    if checked.result_kind != "sweep1d":
        return ()
    if checked.result_kind == "sweep1d" and node_type in {"lenrabi", "mist"}:
        sweep = _require_sweep1d(node_name, result)
        return (
            _streaming_sweep1d_signal_spec(
                data_root, root_path, index, node_name, node_type, sweep
            ),
        )
    if checked.result_kind == "sweep1d" and node_type in _SWEEP1D_SCALAR_VARIABLES:
        sweep = _require_sweep1d(node_name, result)
        return (
            _streaming_sweep1d_signal_spec(
                data_root, root_path, index, node_name, node_type, sweep
            ),
            _streaming_sweep1d_scalar_spec(
                data_root, root_path, index, node_name, node_type, sweep
            ),
        )
    return ()


def _streaming_sweep1d_signal_spec(
    data_root: Path,
    root_path: Path,
    index: int,
    node_name: str,
    node_type: str,
    result: SweepResult1D,
) -> _StreamingSidecarSpec:
    variable = "signal"
    sidecar = _sidecar(
        index,
        node_name,
        node_type,
        variable,
        data_root,
        root_path / _filename(index, node_type, variable),
    )
    spec = StreamingLabberVariableSpec(
        variable,
        "Signal",
        "a.u.",
        axes=[
            (result.x_label, "", result.x),
            ("Flux device value", "", result.flux),
        ],
        shape=result.signal.shape,
        attrs=_streaming_attrs(node_name, node_type, variable),
    )
    return _StreamingSidecarSpec(sidecar, spec)


def _streaming_sweep1d_scalar_spec(
    data_root: Path,
    root_path: Path,
    index: int,
    node_name: str,
    node_type: str,
    result: SweepResult1D,
) -> _StreamingSidecarSpec:
    variable, label, unit = _SWEEP1D_SCALAR_VARIABLES[node_type]
    sidecar = _sidecar(
        index,
        node_name,
        node_type,
        variable,
        data_root,
        root_path / _filename(index, node_type, variable),
    )
    spec = StreamingLabberVariableSpec(
        variable,
        label,
        unit,
        axes=[("Flux device value", "", result.flux)],
        shape=result.fit_value.shape,
        attrs=_streaming_attrs(node_name, node_type, variable),
    )
    return _StreamingSidecarSpec(sidecar, spec)


def _streaming_row_value(
    node_name: str,
    node_type: str,
    result: object,
    variable: str,
    flux_idx: int,
) -> NDArray[np.float64] | float:
    sweep = _require_sweep1d(node_name, result)
    if flux_idx < 0 or flux_idx >= sweep.flux.size:
        raise IndexError(
            f"flux row index {flux_idx} out of range for {sweep.flux.size} rows"
        )
    if variable == "signal":
        return sweep.signal[int(flux_idx)]
    if (
        node_type in _SWEEP1D_SCALAR_VARIABLES
        and variable == _SWEEP1D_SCALAR_VARIABLES[node_type][0]
    ):
        return float(sweep.fit_value[int(flux_idx)])
    raise ValueError(
        f"unsupported Labber Browser streaming variable {variable!r} for node "
        f"{node_name!r} ({node_type})"
    )


def _export_qubit_freq(
    data_root: Path,
    root_path: Path,
    index: int,
    node_name: str,
    node_type: str,
    result: FrequencySweepResult,
    committed_mask: NDArray[np.bool_],
) -> LabberBrowserSidecar:
    variable = "qubit_freq"
    path = root_path / _filename(index, node_type, variable)
    written = export_qubit_freq_fluxdep_spectrum(
        result,
        path,
        committed_mask=committed_mask,
    )
    return _sidecar(index, node_name, node_type, variable, data_root, Path(written))


def _export_sweep1d_signal(
    data_root: Path,
    root_path: Path,
    index: int,
    node_name: str,
    node_type: str,
    result: SweepResult1D,
    committed_mask: NDArray[np.bool_],
) -> LabberBrowserSidecar:
    variable = "signal"
    values = _masked_rows(result.signal, committed_mask)
    path = root_path / _filename(index, node_type, variable)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = save_labber_data(
        str(path),
        z=("Signal", "a.u.", values),
        axes=[
            (result.x_label, "", result.x),
            ("Flux device value", "", result.flux),
        ],
    )
    return _sidecar(index, node_name, node_type, variable, data_root, Path(written))


_SWEEP1D_SCALAR_VARIABLES: dict[str, tuple[str, str, str]] = {
    "t1": ("t1", "T1", "us"),
    "t2ramsey": ("t2r", "T2 Ramsey", "us"),
    "t2echo": ("t2e", "T2 Echo", "us"),
}


def _export_sweep1d_scalar(
    data_root: Path,
    root_path: Path,
    index: int,
    node_name: str,
    node_type: str,
    result: SweepResult1D,
    committed_mask: NDArray[np.bool_],
) -> LabberBrowserSidecar:
    variable, label, unit = _SWEEP1D_SCALAR_VARIABLES[node_type]
    values = _masked_vector(result.fit_value, committed_mask)
    path = root_path / _filename(index, node_type, variable)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = save_labber_data(
        str(path),
        z=(label, unit, values),
        axes=[("Flux device value", "", result.flux)],
    )
    return _sidecar(index, node_name, node_type, variable, data_root, Path(written))


def _committed_mask_for(
    node_name: str,
    result: object,
    committed_masks: Mapping[str, NDArray[np.bool_]],
) -> NDArray[np.bool_]:
    try:
        raw_mask = committed_masks[node_name]
    except KeyError as exc:
        raise KeyError(f"missing committed row mask for node {node_name!r}") from exc
    mask = np.asarray(raw_mask, dtype=np.bool_)
    expected = (_result_n_flux(result),)
    if mask.shape != expected:
        raise ValueError(
            f"committed mask for node {node_name!r} has shape {mask.shape}, "
            f"expected {expected}"
        )
    return mask


def _result_n_flux(result: object) -> int:
    kind: object = getattr(type(result), "result_kind", None)
    if kind not in {"qubit_freq", "sweep1d", "sweep2d"}:
        raise TypeError(
            "Labber Browser export cannot derive n_flux from unsupported result type "
            f"{type(result).__name__}"
        )
    return require_workflow_result(result).flux.size


def _masked_rows(
    values: NDArray[np.float64],
    committed_mask: NDArray[np.bool_],
) -> NDArray[np.complex128]:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim < 1:
        raise ValueError(f"row-masked values must be at least 1D, got {array.shape}")
    if array.shape[0] != committed_mask.shape[0]:
        raise ValueError(
            f"row-masked values first dimension {array.shape[0]} does not match "
            f"committed mask length {committed_mask.shape[0]}"
        )
    exported = np.full(array.shape, np.nan + 0j, dtype=np.complex128)
    exported[committed_mask] = array[committed_mask]
    return exported


def _masked_vector(
    values: NDArray[np.float64],
    committed_mask: NDArray[np.bool_],
) -> NDArray[np.complex128]:
    array = np.asarray(values, dtype=np.complex128)
    if array.shape != committed_mask.shape:
        raise ValueError(
            f"vector values shape {array.shape} does not match committed mask "
            f"shape {committed_mask.shape}"
        )
    exported = np.full(array.shape, np.nan + 0j, dtype=np.complex128)
    exported[committed_mask] = array[committed_mask]
    return exported


def _require_sweep1d(node_name: str, result: object) -> SweepResult1D:
    checked = require_workflow_result(result)
    if isinstance(checked, SweepResult1D) and checked.result_kind == "sweep1d":
        return checked
    raise TypeError(f"node {node_name!r} requires sweep1d for Labber Browser export")


def _filename(index: int, node_type: str, variable: str) -> str:
    node_slug = safe_artifact_slug(node_type)
    variable_slug = safe_artifact_slug(variable)
    return f"{index:03d}-{node_slug}_{variable_slug}.hdf5"


def _sidecar(
    index: int,
    node_name: str,
    node_type: str,
    variable: str,
    data_root: Path,
    path: Path,
) -> LabberBrowserSidecar:
    return LabberBrowserSidecar(
        index=index,
        node=node_name,
        node_type=node_type,
        variable=variable,
        path=relative_to_artifact(data_root, path),
    )


def _streaming_attrs(node_name: str, node_type: str, variable: str) -> dict[str, str]:
    return {
        "zcu_tools.autofluxdep.node_name": node_name,
        "zcu_tools.autofluxdep.node_type": node_type,
        "zcu_tools.autofluxdep.result_role": variable,
        "zcu_tools.autofluxdep.sidecar_kind": "labber_browser",
    }


__all__ = [
    "LABBER_BROWSER_ROOT_EXPORT_KEY",
    "LABBER_BROWSER_SIDECARS_EXPORT_KEY",
    "LabberBrowserExport",
    "LabberBrowserSidecar",
    "LabberBrowserSidecarWriters",
    "export_labber_browser_sidecars",
    "export_qubit_freq_labber_browser_sidecar",
    "labber_browser_root",
    "open_streaming_labber_browser_sidecars",
]
