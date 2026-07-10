"""Labber Browser sidecar export tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from zcu_tools.gui.app.autofluxdep.experiments._support.result import (
    QubitFreqResult,
    Sweep1DResult,
    Sweep2DResult,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.orchestrator import InfoStore
from zcu_tools.gui.app.autofluxdep.services import run_store as run_store_module
from zcu_tools.gui.app.autofluxdep.services.labber_browser_export import (
    export_labber_browser_sidecars,
)
from zcu_tools.gui.app.autofluxdep.services.run_store import (
    RunStore,
    load_journal_events,
    load_manifest,
)
from zcu_tools.gui.app.autofluxdep.state import ProjectInfo
from zcu_tools.utils.datasaver import load_labber_data

from ._helpers import make_builder, place


def _project(tmp_path: Path, *, database_path: Path | None = None) -> ProjectInfo:
    return ProjectInfo(
        chip_name="chip",
        qub_name="q1",
        result_dir=str(tmp_path / "results"),
        database_path=str(database_path or tmp_path / "Database" / "chip" / "q1"),
        params_path=str(tmp_path / "params.json"),
    )


def _fixed_run_datetime(monkeypatch: Any) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            base = cls(2026, 7, 5, 22, 39, 8)
            if tz is None:
                return base
            return base.replace(tzinfo=tz)

    monkeypatch.setattr(run_store_module, "datetime", FixedDateTime)


def _qubit_freq_result(flux: np.ndarray) -> QubitFreqResult:
    result = QubitFreqResult.allocate(flux, np.array([-1.0, 0.0, 1.0], dtype=float))
    result.predict_freq[:] = [5001.0, 6001.0, 7001.0]
    result.fit_freq[:] = result.predict_freq + 0.25
    result.snr[:] = [10.0, 20.0, 30.0]
    result.signal[:] = np.array(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    result.fit_curve[:] = result.signal + 0.5
    return result


def _sweep1d_result(
    flux: np.ndarray,
    *,
    x_label: str,
    offset: float,
) -> Sweep1DResult:
    result = Sweep1DResult.allocate(
        flux,
        np.array([1.0, 2.0, 3.0], dtype=float),
        x_label=x_label,
    )
    result.signal[:] = offset + np.array(
        [
            [0.1, 0.2, 0.3],
            [1.1, 1.2, 1.3],
            [2.1, 2.2, 2.3],
        ],
        dtype=float,
    )
    result.fit_curve[:] = result.signal + 0.05
    result.fit_value[:] = offset + np.array([10.0, 20.0, 30.0], dtype=float)
    result.snr[:] = offset + np.array([100.0, 200.0, 300.0], dtype=float)
    return result


def _ro_optimize_result(flux: np.ndarray) -> Sweep2DResult:
    result = Sweep2DResult.allocate(
        flux,
        np.array([6000.0, 6001.0], dtype=float),
        np.array([0.1, 0.2], dtype=float),
    )
    result.signal[:] = np.arange(12, dtype=float).reshape(3, 2, 2)
    result.best_freq[:] = [6000.0, 6001.0, 6000.5]
    result.best_gain[:] = [0.1, 0.2, 0.15]
    return result


def _entry_by_node_role(
    sidecars: list[dict[str, Any]], node_type: str, role: str
) -> dict[str, Any]:
    matches = [
        entry
        for entry in sidecars
        if entry["node_type"] == node_type and entry["role"] == role
    ]
    assert len(matches) == 1
    return matches[0]


def _assert_single_log_readable(path: Path) -> None:
    load_labber_data(str(path))
    with h5py.File(path, "r") as handle:
        assert "Data" in handle
        assert "zcu_tools.grouped_dataset_version" not in handle.attrs
        assert not any(str(key).startswith("Log_") for key in handle)


def test_run_store_finalizes_labber_browser_sidecars_from_committed_rows(
    tmp_path, monkeypatch
):
    _fixed_run_datetime(monkeypatch)
    flux = np.array([0.0, 0.5, 1.0], dtype=float)
    nodes = [
        place(make_builder("qubit_freq")),
        place(make_builder("lenrabi")),
        place(make_builder("ro_optimize")),
        place(make_builder("t1")),
        place(make_builder("t2ramsey")),
        place(make_builder("t2echo")),
        place(make_builder("mist")),
    ]
    results = {
        "qubit_freq": _qubit_freq_result(flux),
        "lenrabi": _sweep1d_result(flux, x_label="pulse length (us)", offset=0.0),
        "ro_optimize": _ro_optimize_result(flux),
        "t1": _sweep1d_result(flux, x_label="relax time (us)", offset=100.0),
        "t2ramsey": _sweep1d_result(flux, x_label="delay time (us)", offset=200.0),
        "t2echo": _sweep1d_result(flux, x_label="delay time (us)", offset=300.0),
        "mist": _sweep1d_result(flux, x_label="gain", offset=400.0),
    }
    store = RunStore.create(
        project=_project(tmp_path),
        flux_values=flux.tolist(),
        flux_device_name="fake_flux",
        nodes=nodes,
        results=results,
    )
    for node in nodes:
        store.write_node_row(node.name, 0, Patch(), InfoStore(point={"flux_idx": 0}))

    store.finalize("finished")

    manifest = load_manifest(store.manifest_path)
    exports = manifest["exports"]
    assert exports["fluxdep_spectrum"] == "exports/fluxdep/qubit_freq.hdf5"
    assert exports["labber_browser_root"] == "labber/2026/07/Data_0705"
    sidecars = exports["labber_browser_sidecars"]
    assert isinstance(sidecars, list)
    assert len(sidecars) == 9
    assert not any(entry["node_type"] == "ro_optimize" for entry in sidecars)

    expected_paths = {
        "labber/2026/07/Data_0705/000-qubit_freq_qubit_freq.hdf5",
        "labber/2026/07/Data_0705/001-lenrabi_signal.hdf5",
        "labber/2026/07/Data_0705/003-t1_signal.hdf5",
        "labber/2026/07/Data_0705/003-t1_t1.hdf5",
        "labber/2026/07/Data_0705/004-t2ramsey_signal.hdf5",
        "labber/2026/07/Data_0705/004-t2ramsey_t2r.hdf5",
        "labber/2026/07/Data_0705/005-t2echo_signal.hdf5",
        "labber/2026/07/Data_0705/005-t2echo_t2e.hdf5",
        "labber/2026/07/Data_0705/006-mist_signal.hdf5",
    }
    assert {entry["path"] for entry in sidecars} == expected_paths

    for relpath in expected_paths:
        _assert_single_log_readable(store.data_dir / relpath)

    for node_type in ("lenrabi", "t1", "t2ramsey", "t2echo", "mist"):
        signal_entry = _entry_by_node_role(sidecars, node_type, "signal")
        signal = load_labber_data(str(store.data_dir / signal_entry["path"]))
        result = results[node_type]
        assert isinstance(result, Sweep1DResult)
        np.testing.assert_allclose(signal.axes[0].values, result.x)
        np.testing.assert_allclose(signal.axes[1].values, flux)
        np.testing.assert_allclose(signal.z.real[0], result.signal[0])
        assert np.isnan(signal.z.real[1:]).all()

    for node_type, role in (
        ("t1", "t1"),
        ("t2ramsey", "t2r"),
        ("t2echo", "t2e"),
    ):
        scalar_entry = _entry_by_node_role(sidecars, node_type, role)
        scalar = load_labber_data(str(store.data_dir / scalar_entry["path"]))
        result = results[node_type]
        assert isinstance(result, Sweep1DResult)
        np.testing.assert_allclose(scalar.axes[0].values, flux)
        assert scalar.z.real[0] == result.fit_value[0]
        assert np.isnan(scalar.z.real[1:]).all()

    qubit_entry = _entry_by_node_role(sidecars, "qubit_freq", "qubit_freq")
    qubit = load_labber_data(str(store.data_dir / qubit_entry["path"]))
    fluxdep = load_labber_data(str(store.data_dir / exports["fluxdep_spectrum"]))
    np.testing.assert_allclose(qubit.axes[0].values, flux)
    np.testing.assert_allclose(qubit.z.real[:, 0], results["qubit_freq"].signal[0])
    assert np.isnan(qubit.z.real[:, 1:]).all()
    assert len(qubit.axes) == len(fluxdep.axes)
    for qubit_axis, fluxdep_axis in zip(qubit.axes, fluxdep.axes):
        assert qubit_axis.name == fluxdep_axis.name
        assert qubit_axis.unit == fluxdep_axis.unit
        np.testing.assert_allclose(qubit_axis.values, fluxdep_axis.values)
    np.testing.assert_allclose(qubit.z, fluxdep.z, equal_nan=True)

    report = (store.run_dir / "report.md").read_text(encoding="utf-8")
    assert "- Export labber_browser_root: labber/2026/07/Data_0705" in report
    assert (
        "- Export labber_browser_sidecar t1/t1: labber/2026/07/Data_0705/003-t1_t1.hdf5"
    ) in report


def test_run_store_streams_labber_browser_sidecar_rows_before_finalize(
    tmp_path, monkeypatch
):
    _fixed_run_datetime(monkeypatch)
    flux = np.array([0.0, 0.5, 1.0], dtype=float)
    node = place(make_builder("t1"))
    result = _sweep1d_result(flux, x_label="relax time (us)", offset=100.0)
    store = RunStore.create(
        project=_project(tmp_path),
        flux_values=flux.tolist(),
        flux_device_name="fake_flux",
        nodes=[node],
        results={"t1": result},
    )

    manifest = load_manifest(store.manifest_path)
    exports = manifest["exports"]
    assert exports["labber_browser_root"] == "labber/2026/07/Data_0705"
    sidecars = exports["labber_browser_sidecars"]
    assert {entry["role"] for entry in sidecars} == {"signal", "t1"}

    store.write_node_row(node.name, 1, Patch(), InfoStore(point={"flux_idx": 1}))

    signal_entry = _entry_by_node_role(sidecars, "t1", "signal")
    signal_path = store.data_dir / signal_entry["path"]
    signal = load_labber_data(str(signal_path))
    np.testing.assert_allclose(signal.z.real[1], result.signal[1])
    assert np.isnan(signal.z.real[0]).all()
    assert np.isnan(signal.z.real[2]).all()

    scalar_entry = _entry_by_node_role(sidecars, "t1", "t1")
    scalar_path = store.data_dir / scalar_entry["path"]
    scalar = load_labber_data(str(scalar_path))
    assert scalar.z.real[1] == result.fit_value[1]
    assert np.isnan(scalar.z.real[0])
    assert np.isnan(scalar.z.real[2])

    with h5py.File(signal_path, "r") as handle:
        assert bool(handle.attrs["zcu_tools.streaming_finalized"]) is False

    store.finalize("stopped", next_flux_idx=2)

    with h5py.File(signal_path, "r") as handle:
        assert bool(handle.attrs["zcu_tools.streaming_finalized"]) is True


def test_run_store_sidecar_write_failure_does_not_append_row_journal(
    tmp_path, monkeypatch
):
    _fixed_run_datetime(monkeypatch)
    flux = np.array([0.0, 0.5, 1.0], dtype=float)
    node = place(make_builder("t1"))
    result = _sweep1d_result(flux, x_label="relax time (us)", offset=100.0)
    store = RunStore.create(
        project=_project(tmp_path),
        flux_values=flux.tolist(),
        flux_device_name="fake_flux",
        nodes=[node],
        results={"t1": result},
    )

    def fail_sidecar(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("sidecar boom")

    monkeypatch.setattr(store, "_write_labber_browser_row", fail_sidecar)

    with pytest.raises(RuntimeError, match="sidecar boom"):
        store.write_node_row(node.name, 0, Patch(), InfoStore(point={"flux_idx": 0}))

    assert load_journal_events(store.run_dir / "journal.jsonl") == []
    assert store._row_counts[node.name] == 0

    manifest = load_manifest(store.manifest_path)
    signal_entry = _entry_by_node_role(
        manifest["exports"]["labber_browser_sidecars"], "t1", "signal"
    )
    signal = load_labber_data(str(store.data_dir / signal_entry["path"]))
    assert np.isnan(signal.z.real).all()

    store.close_writers(finalize=True)


def test_run_store_places_autofluxdep_run_under_qubit_database_root_for_dated_database_path(
    tmp_path, monkeypatch
):
    _fixed_run_datetime(monkeypatch)
    flux = np.array([0.0, 0.5, 1.0], dtype=float)
    nodes = [place(make_builder("qubit_freq"))]
    dated_database_path = (
        tmp_path / "Database" / "chip" / "q1" / "2026" / "07" / "Data_0706"
    )
    store = RunStore.create(
        project=_project(tmp_path, database_path=dated_database_path),
        flux_values=flux.tolist(),
        flux_device_name="fake_flux",
        nodes=nodes,
        results={"qubit_freq": _qubit_freq_result(flux)},
    )

    store.finalize("stopped")

    expected_parent = tmp_path / "Database" / "chip" / "q1" / "autofluxdep_runs"
    assert store.data_dir.parent == expected_parent
    manifest = load_manifest(store.manifest_path)
    assert manifest["project"]["database_path"] == str(dated_database_path)
    assert manifest["paths"]["data_root"] == str(store.data_dir)


def test_labber_browser_export_rejects_unknown_result_type(tmp_path):
    node = place(make_builder("lenrabi"))

    with pytest.raises(TypeError, match="unsupported result type object"):
        export_labber_browser_sidecars(
            data_root=tmp_path / "20260705-223908_flux-sweep-test",
            nodes=[node],
            results={node.name: object()},
            committed_masks={node.name: np.array([True], dtype=np.bool_)},
        )
