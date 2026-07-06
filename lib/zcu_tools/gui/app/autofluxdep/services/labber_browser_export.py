"""Labber Browser sidecar exports derived from autofluxdep Results."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zcu_tools.gui.app.autofluxdep.nodes.builder import PlacedNode
from zcu_tools.gui.app.autofluxdep.nodes.result import (
    QubitFreqResult,
    Sweep1DResult,
)
from zcu_tools.gui.app.autofluxdep.services.artifact_paths import (
    relative_to_artifact,
    safe_artifact_slug,
)
from zcu_tools.gui.app.autofluxdep.services.fluxdep_export import (
    export_qubit_freq_fluxdep_spectrum,
)
from zcu_tools.utils.datasaver import save_labber_data

LABBER_BROWSER_ROOT_EXPORT_KEY = "labber_browser_root"
LABBER_BROWSER_SIDECARS_EXPORT_KEY = "labber_browser_sidecars"

_RUN_SLUG_RE = re.compile(r"^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})-")


@dataclass(frozen=True)
class LabberBrowserSidecar:
    """One Labber Browser single-log export entry for the manifest."""

    index: int
    node: str
    node_type: str
    role: str
    path: str

    def to_manifest(self) -> dict[str, int | str]:
        return {
            "index": self.index,
            "node": self.node,
            "node_type": self.node_type,
            "role": self.role,
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
    if node_type == "ro_optimize":
        return ()
    if node_type == "qubit_freq":
        return (
            _export_qubit_freq(
                data_root,
                root_path,
                index,
                node_name,
                node_type,
                _require_result(node_name, result, QubitFreqResult),
                committed_mask,
            ),
        )
    if node_type in {"lenrabi", "mist"}:
        return (
            _export_sweep1d_signal(
                data_root,
                root_path,
                index,
                node_name,
                node_type,
                _require_result(node_name, result, Sweep1DResult),
                committed_mask,
            ),
        )
    if node_type in _SWEEP1D_SCALAR_ROLES:
        sweep = _require_result(node_name, result, Sweep1DResult)
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


def _export_qubit_freq(
    data_root: Path,
    root_path: Path,
    index: int,
    node_name: str,
    node_type: str,
    result: QubitFreqResult,
    committed_mask: NDArray[np.bool_],
) -> LabberBrowserSidecar:
    role = "qubit_freq"
    path = root_path / _filename(index, node_type, role)
    written = export_qubit_freq_fluxdep_spectrum(
        result,
        path,
        committed_mask=committed_mask,
    )
    return _sidecar(index, node_name, node_type, role, data_root, Path(written))


def _export_sweep1d_signal(
    data_root: Path,
    root_path: Path,
    index: int,
    node_name: str,
    node_type: str,
    result: Sweep1DResult,
    committed_mask: NDArray[np.bool_],
) -> LabberBrowserSidecar:
    role = "signal"
    values = _masked_rows(result.signal, committed_mask)
    path = root_path / _filename(index, node_type, role)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = save_labber_data(
        str(path),
        z=("Signal", "a.u.", values),
        axes=[
            (result.x_label, "", result.x),
            ("Flux device value", "", result.flux),
        ],
    )
    return _sidecar(index, node_name, node_type, role, data_root, Path(written))


_SWEEP1D_SCALAR_ROLES: dict[str, tuple[str, str, str]] = {
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
    result: Sweep1DResult,
    committed_mask: NDArray[np.bool_],
) -> LabberBrowserSidecar:
    role, label, unit = _SWEEP1D_SCALAR_ROLES[node_type]
    values = _masked_vector(result.fit_value, committed_mask)
    path = root_path / _filename(index, node_type, role)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = save_labber_data(
        str(path),
        z=(label, unit, values),
        axes=[("Flux device value", "", result.flux)],
    )
    return _sidecar(index, node_name, node_type, role, data_root, Path(written))


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
    if isinstance(result, (QubitFreqResult, Sweep1DResult)):
        return int(result.n_flux)
    return int(getattr(result, "n_flux"))


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


def _require_result(
    node_name: str,
    result: object,
    expected: type[QubitFreqResult] | type[Sweep1DResult],
) -> Any:
    if not isinstance(result, expected):
        raise TypeError(
            f"node {node_name!r} expected {expected.__name__} for Labber Browser "
            f"export, got {type(result).__name__}"
        )
    return result


def _filename(index: int, node_type: str, role: str) -> str:
    node_slug = safe_artifact_slug(node_type)
    role_slug = safe_artifact_slug(role)
    return f"{index:03d}-{node_slug}_{role_slug}.hdf5"


def _sidecar(
    index: int,
    node_name: str,
    node_type: str,
    role: str,
    data_root: Path,
    path: Path,
) -> LabberBrowserSidecar:
    return LabberBrowserSidecar(
        index=index,
        node=node_name,
        node_type=node_type,
        role=role,
        path=relative_to_artifact(data_root, path),
    )


__all__ = [
    "LABBER_BROWSER_ROOT_EXPORT_KEY",
    "LABBER_BROWSER_SIDECARS_EXPORT_KEY",
    "LabberBrowserExport",
    "LabberBrowserSidecar",
    "export_labber_browser_sidecars",
    "labber_browser_root",
]
