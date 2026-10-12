"""Native T2 fluxdep science through Engine, simulator and typed readers."""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from zcu_tools.datafile import SoftwareProvenance, load_run_data
from zcu_tools.experiment.native_persistence import load_run_with_metadata
from zcu_tools.experiment.workflows import Engine, EnginePorts, RunIdentity, RunPaths
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

import zcu_lab.workflows as workflows
from tests.experiment.workflows._engine_fakes import (
    ManualClock,
    RecordingBar,
    RecordingDevices,
)
from tests.zcu_lab.workflows._point_files import point_files
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoExp
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyExp
from zcu_lab.workflows.calibration import Calibrated, CalibrationTunables

from ._sim_inputs import Inputs, make_plan, make_snapshot, seed_from_inputs


def start_t2(
    inputs: Inputs,
    root: Path,
    kind: Literal["echo", "ramsey"],
    *,
    sweep_factor: float = 5.0,
    fine_fringe: bool = False,
) -> tuple[Engine[Calibrated], RecordingDevices, Plots]:
    """Run each shipped T2 declaration on independent simulator ports."""
    plan = workflows.T2Plan.model_validate(make_plan(inputs).model_dump(mode="python"))
    if fine_fringe:
        plan = plan.model_copy(update={"points": 101})
    cal = CalibrationTunables(span_mhz=2.0, rabi_start_factor=0.125)
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
    if kind == "echo":
        engine.start(
            workflows.t2echo_fluxdep,
            plan=plan,
            tunables=workflows.T2EchoTunables(
                cal=cal, rounds=2, sweep_factor=sweep_factor
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
    else:
        engine.start(
            workflows.t2ramsey_fluxdep,
            plan=plan,
            tunables=workflows.T2RamseyTunables(
                cal=cal,
                rounds=10 if fine_fringe else 2,
                sweep_factor=sweep_factor,
                detune_ratio=0.25 if fine_fringe else 0.1,
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


@pytest.mark.parametrize("kind", ["echo", "ramsey"])
def test_t2_fluxdep_commits_two_native_points_with_fitted_pi2(
    inputs: Inputs, tmp_path: Path, kind: Literal["echo", "ramsey"]
) -> None:
    engine, devices, plots = start_t2(
        inputs, tmp_path, kind, fine_fringe=kind == "ramsey"
    )
    status = engine.execute("workflow")
    assert status.lifecycle == "done", status.reason
    records = engine.records("workflow")
    assert len(records) == 2
    assert devices.values == [("flux", 1.0), ("flux", 1.1)]
    for index, record in enumerate(records):
        point = record.encoded_record.value
        assert isinstance(point, dict)
        assert point["index"] == index and point["flux"] == devices.values[index][1]
        assert point["reason"] is None
        assert point["lifetime_us"] == pytest.approx(10.0, rel=0.3)
        calibration = point["calibration"]
        assert isinstance(calibration, dict)
        final_path = tmp_path / "data" / record.iteration_dir / point_files(record)[-1]
        stored = load_run_data(final_path)
        assert stored.metadata.experiment == f"twotone/ge/t2{kind}"
        assert stored.metadata.snapshot == make_snapshot()
        assert stored.metadata.device_snapshot is not None
        assert stored.metadata.provenance.hostname is not None
        if kind == "echo":
            source, metadata = load_run_with_metadata(
                final_path, spec=T2EchoExp().persistence_spec()
            )
            assert source.cfg is not None
            drive = source.cfg.modules.pi2_pulse
            assert source.cfg.detune == 0.0
            assert source.cfg.modules.pi_pulse.waveform.style == drive.waveform.style
        else:
            ramsey, metadata = load_run_with_metadata(
                final_path, spec=T2RamseyExp().persistence_spec()
            )
            assert ramsey.cfg is not None
            drive = ramsey.cfg.modules.pi2_pulse
            assert ramsey.cfg.detune == pytest.approx(
                0.25 / ramsey.cfg.sweep.length.step
            )
        assert isinstance(drive.waveform, ConstWaveformCfg)
        assert drive.waveform.length == pytest.approx(calibration["pi2_length_us"])
        assert drive.waveform.style == make_plan(inputs).qub_pulse.waveform.style
        assert drive.gain == make_plan(inputs).qub_pulse.gain
        assert metadata.run_id != "workflow"
    events = [
        json.loads(line)
        for line in (tmp_path / "metadata" / "journal.jsonl").read_text().splitlines()
    ]
    assert [
        e["encoded_record"]["value"] for e in events if e["kind"] == "step_committed"
    ] == [r.encoded_record.value for r in records]
    assert f"t2{kind}/decay" in plots and f"t2{kind}/vs_flux" in plots
    assert plots[f"t2{kind}/decay"].axes[0].lines
    assert np.asarray(
        plots[f"t2{kind}/vs_flux"].axes[0].collections[0].get_offsets()
    ).shape == (2, 2)


@pytest.mark.parametrize("kind", ["echo", "ramsey"])
def test_t2_fluxdep_rejects_overlong_fit_but_preserves_native_trace(
    inputs: Inputs, tmp_path: Path, kind: Literal["echo", "ramsey"]
) -> None:
    engine, _, _ = start_t2(inputs, tmp_path, kind, sweep_factor=0.1)
    status = engine.execute("workflow")
    assert status.lifecycle == "done", status.reason
    records = engine.records("workflow")
    assert len(records) == 2
    for record in records:
        point = record.encoded_record.value
        assert isinstance(point, dict)
        assert point["lifetime_us"] is None and point["error_us"] is None
        assert "exceeds" in str(point["reason"])
        stored = load_run_data(
            tmp_path / "data" / record.iteration_dir / point_files(record)[-1]
        )
        assert stored.metadata.experiment == f"twotone/ge/t2{kind}"
        trace = next(iter(stored.payload.variables.values()))
        assert np.isfinite(trace.z).all() and trace.z.shape == (21,)


@pytest.mark.parametrize("kind", ["echo", "ramsey"])
def test_point_names_native_t2_decay(
    inputs: Inputs, tmp_path: Path, kind: Literal["echo", "ramsey"]
) -> None:
    engine, _, plots = start_t2(inputs, tmp_path, kind)
    try:
        status = engine.execute("workflow")
        assert status.lifecycle == "done", status.reason
        record = engine.records("workflow")[0]
        point = record.encoded_record.value
        assert isinstance(point, dict)
        assert point["decay_file"] == f"t2{kind}.h5"
        path = tmp_path / "data" / record.iteration_dir / f"t2{kind}.h5"
        if kind == "echo":
            native, metadata = load_run_with_metadata(
                path, spec=T2EchoExp().persistence_spec()
            )
        else:
            native, metadata = load_run_with_metadata(
                path, spec=T2RamseyExp().persistence_spec()
            )
        assert native.cfg is not None
        assert metadata.snapshot == make_snapshot()
        assert metadata.device_snapshot is not None
    finally:
        plots.finish()
        plots.release()
