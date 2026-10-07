"""Native T1 declaration through Engine, simulator, publication and typed loading."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.datafile import load_run_data
from zcu_tools.experiment.native_persistence import load_run_with_metadata
from zcu_tools.experiment.workflows import (
    Actor,
    DeviceSnapshot,
    Engine,
    EnginePorts,
    RunIdentity,
    RunPaths,
    TunableChange,
)
from zcu_tools.program.v2 import SweepCfg

from tests.experiment.workflows._engine_fakes import (
    ManualClock,
    RecordingBar,
    RecordingDevices,
    RecordingPlots,
)
from zcu_lab.v2.twotone.time_domain.t1.core import T1Exp
from zcu_lab.workflows.calibration import Calibrated, CalibrationTunables
from zcu_lab.workflows.t1 import T1Tunables, t1_fluxdep

from ._sim_inputs import Inputs, make_plan, make_snapshot, seed_from_inputs


def start_t1(
    inputs: Inputs,
    root: Path,
    tun: T1Tunables,
    devices: RecordingDevices,
    plots: RecordingPlots,
) -> Engine[Calibrated]:
    """Start the shipped declaration with real acquisitions and isolated offline ports."""
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
        t1_fluxdep,
        plan=make_plan(inputs),
        tunables=tun,
        paths=RunPaths(root / "metadata", root / "data"),
        identity=RunIdentity("workflow", DeviceSnapshot({}), "offline"),
    )
    return engine


def tunables() -> T1Tunables:
    """Use stable simulator averages and the accepted calibration range."""
    return T1Tunables(
        cal=CalibrationTunables(span_mhz=2.0, rabi_start_factor=0.125), rounds=2
    )


def test_native_t1_multi_point_run_commits_scalars_and_loadable_acquisitions(
    inputs: Inputs, tmp_path: Path
) -> None:
    devices, plots = RecordingDevices(), RecordingPlots()
    before = inputs.t1.model_dump(mode="python")
    engine = start_t1(inputs, tmp_path, tunables(), devices, plots)
    status = engine.execute("workflow")
    assert status.lifecycle == "done", status.reason
    assert devices.values == [("flux", 1.0), ("flux", 1.1)]
    records = engine.records("workflow")
    assert len(records) == 2
    assert [len(record.run_files) for record in records] == [4, 3]
    run_ids: set[str] = set()
    for index, record in enumerate(records):
        assert record.encoded_record.mode == "json"
        point = record.encoded_record.value
        assert isinstance(point, dict)
        assert point["index"] == index
        assert point["flux"] == devices.values[index][1]
        assert point["reason"] is None
        lifetime = point["lifetime_us"]
        assert isinstance(lifetime, float) and lifetime == pytest.approx(20.0, rel=0.2)
        for filename in record.run_files:
            path = tmp_path / "data" / record.iteration_dir / filename
            saved = load_run_data(path)
            assert saved.metadata.completion == "complete"
            assert saved.metadata.snapshot == make_snapshot()
            run_ids.add(saved.metadata.run_id)
        path = tmp_path / "data" / record.iteration_dir / record.run_files[-1]
        source, metadata = load_run_with_metadata(path, spec=T1Exp().persistence_spec())
        assert source.cfg is not None and source.cfg.rounds == 2
        assert source.result.times.shape == source.result.signals.shape == (21,)
        assert np.isfinite(source.result.signals).all()
        assert metadata.run_id != "workflow"
    assert len(run_ids) == 7
    assert inputs.t1.model_dump(mode="python") == before


@pytest.mark.filterwarnings(
    "ignore:fit_func failed; returning init_p fallback:RuntimeWarning"
)
def test_failed_calibration_commits_reason_and_never_runs_t1(
    inputs: Inputs, tmp_path: Path
) -> None:
    tun = tunables()
    tun.cal.max_frequency_rel_err = 1e-8
    engine = start_t1(inputs, tmp_path, tun, RecordingDevices(), RecordingPlots())
    status = engine.execute("workflow")
    assert status.lifecycle == "done", status.reason
    records = engine.records("workflow")
    assert len(records) == 2
    for record in records:
        point = record.encoded_record.value
        assert isinstance(point, dict)
        assert point["lifetime_us"] is None and point["error_us"] is None
        assert point["reason"] == "No accepted frequency fit"
        assert len(record.run_files) == len(tun.cal.qf_widen)
        for filename in record.run_files:
            path = tmp_path / "data" / record.iteration_dir / filename
            saved = load_run_data(path)
            assert saved.metadata.experiment != T1Exp().persistence_spec().tag


def test_rejected_t1_fit_keeps_raw_data_but_not_lifetime_history(
    inputs: Inputs, tmp_path: Path
) -> None:
    tun = tunables()
    tun.max_rel_err = 1e-8
    engine = start_t1(inputs, tmp_path, tun, RecordingDevices(), RecordingPlots())
    status = engine.execute("workflow")
    assert status.lifecycle == "done", status.reason
    records = engine.records("workflow")
    assert len(records) == 2
    for record in records:
        point = record.encoded_record.value
        assert isinstance(point, dict)
        assert point["lifetime_us"] is None and point["error_us"] is None
        assert point["reason"] == "T1 fit rejected"
        path = tmp_path / "data" / record.iteration_dir / record.run_files[-1]
        source, _metadata = load_run_with_metadata(
            path, spec=T1Exp().persistence_spec()
        )
        assert source.cfg is not None
        # No accepted history means the second point still uses the original guess.
        assert isinstance(source.cfg.sweep.length, SweepCfg)
        assert source.cfg.sweep.length.stop == tun.guess_us * tun.sweep_factor
        assert np.isfinite(source.result.signals).all()


def test_pause_during_native_acquisition_discards_step_then_uses_new_revision(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plots, devices = RecordingPlots(), RecordingDevices()
    engine = start_t1(inputs, tmp_path, tunables(), devices, plots)
    refresh = plots.refresh
    requested_at: float | None = None

    def request_pause(figure: Figure) -> None:
        nonlocal requested_at
        refresh(figure)
        if requested_at is None and any(
            np.asarray(line.get_xdata()).size > 2
            for ax in figure.axes
            for line in ax.lines
        ):
            requested_at = perf_counter()
            engine.pause("workflow")

    monkeypatch.setattr(plots, "refresh", request_pause)
    status = engine.execute("workflow")
    returned_at = perf_counter()
    assert requested_at is not None
    latency = returned_at - requested_at
    print(json.dumps({"scenario": "pause-native-frequency", "latency_s": latency}))
    assert latency < 1.0
    assert status.lifecycle == "paused", status.reason
    assert not engine.records("workflow")
    engine.update_tunables(
        "workflow",
        (TunableChange("reps", 200),),
        expected_revision=0,
        actor=Actor("user", "operator"),
    )
    engine.resume("workflow", devices=DeviceSnapshot({}))
    status = engine.execute("workflow")
    assert status.lifecycle == "done", status.reason
    records = engine.records("workflow")
    assert len(records) == 2 and all(record.revision == 1 for record in records)
    assert devices.values == [("flux", 1.0), ("flux", 1.0), ("flux", 1.1)]
    for record in records:
        path = tmp_path / "data" / record.iteration_dir / record.run_files[-1]
        source, _metadata = load_run_with_metadata(
            path, spec=T1Exp().persistence_spec()
        )
        assert source.cfg is not None and source.cfg.reps == 200


def test_stop_during_native_acquisition_preserves_only_prior_committed_point(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plots, devices = RecordingPlots(), RecordingDevices()
    engine = start_t1(inputs, tmp_path, tunables(), devices, plots)
    refresh = plots.refresh
    requested_at: float | None = None

    def request_stop(figure: Figure) -> None:
        nonlocal requested_at
        refresh(figure)
        if (
            requested_at is None
            and len(devices.values) == 2
            and figure is plots.figures["calibration/qubit_freq"]
            and any(
                np.asarray(line.get_xdata()).size > 2
                for ax in figure.axes
                for line in ax.lines
            )
        ):
            requested_at = perf_counter()
            engine.stop("workflow")

    monkeypatch.setattr(plots, "refresh", request_stop)
    status = engine.execute("workflow")
    returned_at = perf_counter()
    assert requested_at is not None
    latency = returned_at - requested_at
    print(json.dumps({"scenario": "stop-native-frequency", "latency_s": latency}))
    assert latency < 1.0
    assert status.lifecycle == "stopped", status.reason
    records = engine.records("workflow")
    assert len(records) == 1
    point = records[0].encoded_record.value
    assert isinstance(point, dict) and point["flux"] == 1.0
    path = tmp_path / "data" / records[0].iteration_dir / records[0].run_files[-1]
    source, metadata = load_run_with_metadata(path, spec=T1Exp().persistence_spec())
    assert source.cfg is not None and np.isfinite(source.result.signals).all()
    assert metadata.completion == "complete"
