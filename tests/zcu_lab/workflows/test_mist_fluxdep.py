"""MIST gain sweeps through real Engine/native simulator and typed storage."""

import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib.collections import QuadMesh
from zcu_tools.datafile import SoftwareProvenance, load_run_data
from zcu_tools.experiment.native_persistence import load_run_with_metadata
from zcu_tools.experiment.workflows import Engine, EnginePorts, RunIdentity, RunPaths
from zcu_tools.plotting.plots import NonPresentingHost, Plots

import zcu_lab.workflows as workflows
from tests.experiment.workflows._engine_fakes import (
    ManualClock,
    RecordingBar,
    RecordingDevices,
)
from tests.zcu_lab.workflows._point_files import point_files
from zcu_lab.v2.mist.power_dep.single_trace.core import PowerDepExp
from zcu_lab.workflows import analysis
from zcu_lab.workflows.calibration import Calibrated, CalibrationTunables

from ._sim_inputs import Inputs, make_plan, make_snapshot, seed_from_inputs


def start_mist(
    inputs: Inputs, root: Path, *, gain_points: int = 3
) -> tuple[Engine[Calibrated], RecordingDevices, Plots]:
    """Start fixed gain scans at two setpoints with isolated ports."""
    base = make_plan(inputs)
    plan = workflows.MISTPlan(
        flux_dev=base.flux_dev,
        flux_unit=base.flux_unit,
        fluxes=base.fluxes,
        qub_pulse=base.qub_pulse,
        mist_pulse=base.readout.pulse_cfg.model_copy(deep=True),
        readout=base.readout,
        reset=None,
        gain_start=0.0,
        gain_stop=1.0,
        gain_points=gain_points,
        snapshot=base.snapshot,
    )
    devices, plots = RecordingDevices(), Plots(NonPresentingHost())
    engine = Engine(
        EnginePorts(
            plots,
            RecordingBar,
            ManualClock(),
            context=seed_from_inputs(inputs),
            soc=inputs.soc,
            soccfg=inputs.soccfg,
            devices=devices,
        )
    )
    engine.start(
        workflows.mist_fluxdep,
        plan=plan,
        tunables=workflows.MISTTunables(
            cal=CalibrationTunables(span_mhz=2.0, rabi_start_factor=0.125), rounds=2
        ),
        paths=RunPaths(root / "metadata", root / "data"),
        identity=RunIdentity(
            "workflow",
            SoftwareProvenance(
                software_versions={},
                git_commit=None,
                git_dirty=None,
                qick_version=None,
                soc_fingerprint=None,
                hostname="offline",
            ),
        ),
    )
    return engine, devices, plots


def test_mist_projection_uses_centered_magnitude_spread_without_threshold() -> None:
    signal = np.array([1 + 1j, 2 + 1j, 3 + 1j], dtype=np.complex128)
    result = analysis.mist_signal2real(signal)
    np.testing.assert_allclose(result, [2.1213203435596424, 0.0, 2.1213203435596424])
    np.testing.assert_array_equal(signal, [1 + 1j, 2 + 1j, 3 + 1j])


def test_mist_commits_two_gain_traces_with_calibrated_preparation(
    inputs: Inputs, tmp_path: Path
) -> None:
    engine, devices, plots = start_mist(inputs, tmp_path)
    try:
        status = engine.execute("workflow")
        assert status.lifecycle == "done", status.reason
        records = engine.records("workflow")
        assert len(records) == 2
        assert devices.values == [("flux", 1.0), ("flux", 1.1)]
        for index, record in enumerate(records):
            point = record.encoded_record.value
            assert isinstance(point, dict)
            assert point["index"] == index and point["flux"] == devices.values[index][1]
            assert point["success"] is True and point["reason"] is None
            cal = point["rabi"]
            assert isinstance(cal, dict)
            final = tmp_path / "data" / record.iteration_dir / point_files(record)[-1]
            stored = load_run_data(final)
            assert stored.metadata.experiment == "mist/gain"
            assert stored.metadata.snapshot == make_snapshot()
            assert stored.metadata.device_snapshot is not None
            source, metadata = load_run_with_metadata(
                final, spec=PowerDepExp().persistence_spec()
            )
            assert source.cfg is not None
            init = source.cfg.modules.init_pulse
            assert init is not None
            assert init.waveform.model_dump()["length"] == pytest.approx(
                cal["pi_length_us"]
            )
            assert init.gain == make_plan(inputs).qub_pulse.gain
            assert (
                source.cfg.modules.probe_pulse.ch
                == make_plan(inputs).readout.pulse_cfg.ch
            )
            assert (
                source.cfg.sweep.gain.start == 0.0 and source.cfg.sweep.gain.stop == 1.0
            )
            np.testing.assert_array_equal(source.result.gains, [0.0, 0.5, 1.0])
            assert (
                source.result.signals.shape == (3,)
                and np.isfinite(source.result.signals).all()
            )
            assert (
                metadata.run_id != "workflow"
                and metadata.provenance.hostname is not None
            )
        events = [
            json.loads(line)
            for line in (tmp_path / "metadata" / "journal.jsonl")
            .read_text()
            .splitlines()
        ]
        assert [
            e["encoded_record"]["value"]
            for e in events
            if e["kind"] == "step_committed"
        ] == [r.encoded_record.value for r in records]
        assert "mist/gain" in plots and "mist/vs_flux" in plots
        last = records[-1]
        final = tmp_path / "data" / last.iteration_dir / point_files(last)[-1]
        source, _ = load_run_with_metadata(final, spec=PowerDepExp().persistence_spec())
        np.testing.assert_allclose(
            np.asarray(plots["mist/gain"].axes[0].lines[0].get_ydata()),
            analysis.mist_signal2real(source.result.signals),
        )
    finally:
        plots.finish()
        plots.release()


def test_mist_map_uses_actual_native_gain_and_flux_coordinates(
    inputs: Inputs, tmp_path: Path
) -> None:
    engine, devices, plots = start_mist(inputs, tmp_path, gain_points=101)
    try:
        status = engine.execute("workflow")
        assert status.lifecycle == "done", status.reason
        records = engine.records("workflow")
        assert len(records) == 2
        sources = [
            load_run_with_metadata(
                tmp_path / "data" / record.iteration_dir / point_files(record)[-1],
                spec=PowerDepExp().persistence_spec(),
            )[0]
            for record in records
        ]
        expected_gains = sources[0].result.gains
        assert not np.array_equal(expected_gains, np.linspace(0.0, 1.0, 101))
        for source in sources:
            np.testing.assert_array_equal(source.result.gains, expected_gains)
        expected_fluxes = np.array([value for _, value in devices.values])
        expected_signals = np.stack(
            [analysis.mist_signal2real(source.result.signals) for source in sources]
        )
        axes = plots["mist/vs_flux"].axes[0]
        if axes.collections:
            mesh = axes.collections[0]
            assert isinstance(mesh, QuadMesh)
            coordinates = np.asarray(mesh.get_coordinates(), dtype=np.float64)
            gains = (coordinates[0, :-1, 0] + coordinates[0, 1:, 0]) / 2
            fluxes = (coordinates[:-1, 0, 1] + coordinates[1:, 0, 1]) / 2
            signals = np.asarray(mesh.get_array())
        else:
            # A raster's extent describes pixel edges, not sample centres.
            raster = axes.images[0]
            signals = np.asarray(raster.get_array())
            left, right, bottom, top = raster.get_extent()
            rows, columns = signals.shape
            half_gain = (right - left) / (2 * columns)
            half_flux = (top - bottom) / (2 * rows)
            gains = np.linspace(left + half_gain, right - half_gain, columns)
            fluxes = np.linspace(bottom + half_flux, top - half_flux, rows)
        np.testing.assert_allclose(gains, expected_gains, rtol=0, atol=1e-15)
        np.testing.assert_allclose(fluxes, expected_fluxes, rtol=0, atol=1e-15)
        np.testing.assert_allclose(signals, expected_signals, rtol=0, atol=1e-15)
    finally:
        plots.finish()
        plots.release()


def test_point_names_native_gain_trace(inputs: Inputs, tmp_path: Path) -> None:
    engine, _, plots = start_mist(inputs, tmp_path)
    try:
        status = engine.execute("workflow")
        assert status.lifecycle == "done", status.reason
        record = engine.records("workflow")[0]
        point = record.encoded_record.value
        assert isinstance(point, dict)
        assert point["gain_file"] == "mist.h5"
        native, metadata = load_run_with_metadata(
            tmp_path / "data" / record.iteration_dir / "mist.h5",
            spec=PowerDepExp().persistence_spec(),
        )
        assert native.cfg is not None
        assert metadata.snapshot == make_snapshot()
        assert metadata.device_snapshot is not None
    finally:
        plots.finish()
        plots.release()
