"""Independent calibration evidence through MIST's public Engine/native seam."""

from pathlib import Path

import pytest
from zcu_tools.datafile import load_run_data
from zcu_tools.experiment.native_persistence import load_run_with_metadata
from zcu_tools.experiment.workflows import Engine, EnginePorts, RunIdentity, RunPaths
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.experiment.workflows._engine_fakes import (
    ManualClock,
    RecordingBar,
    RecordingDevices,
)
from zcu_lab.v2.mist.power_dep.single_trace.core import PowerDepExp
from zcu_lab.workflows.calibration import Calibrated, CalibrationTunables
from zcu_lab.workflows.mist import MISTPlan, MISTTunables, mist_fluxdep

from ._sim_inputs import Inputs, make_plan, seed_from_inputs


def start_scan(
    inputs: Inputs, root: Path, tun: CalibrationTunables
) -> tuple[Engine[Calibrated], Plots]:
    """Run the real MIST caller with isolated simulator and native writers."""
    base = make_plan(inputs)
    plan = MISTPlan(
        flux_dev=base.flux_dev,
        flux_unit=base.flux_unit,
        fluxes=base.fluxes,
        qub_pulse=base.qub_pulse,
        readout=base.readout,
        snapshot=base.snapshot,
        mist_pulse=base.readout.pulse_cfg.model_copy(deep=True),
        reset=None,
        gain_start=0.0,
        gain_stop=1.0,
        gain_points=3,
    )
    plots = Plots(NonPresentingHost())
    engine = Engine(
        EnginePorts(
            plots,
            RecordingBar,
            ManualClock(),
            context=seed_from_inputs(inputs),
            soc=inputs.soc,
            soccfg=inputs.soccfg,
            devices=RecordingDevices(),
        )
    )
    engine.start(
        mist_fluxdep,
        plan=plan,
        tunables=MISTTunables(cal=tun, rounds=2),
        paths=RunPaths(root / "metadata", root / "data"),
        identity=RunIdentity("stages", load_provenance()),
    )
    return engine, plots


def load_provenance():
    """Explicit offline provenance, with no user or hardware reads."""
    from zcu_tools.datafile import SoftwareProvenance

    return SoftwareProvenance(
        software_versions={},
        git_commit=None,
        git_dirty=None,
        qick_version=None,
        soc_fingerprint=None,
        hostname="offline",
    )


def test_rabi_schedule_failure_keeps_accepted_frequency_and_native_evidence(
    inputs: Inputs, tmp_path: Path
) -> None:
    engine, plots = start_scan(
        inputs, tmp_path, CalibrationTunables(span_mhz=2.0, rabi_start_factor=0.001)
    )
    try:
        status = engine.execute("stages")
        assert status.lifecycle == "done", status.reason
        point = engine.records("stages")[0].encoded_record.value
        assert isinstance(point, dict)
        frequency = point.get("frequency")
        assert isinstance(frequency, dict), (
            "Accepted frequency must survive this point's Rabi failure"
        )
        assert frequency["status"] == "accepted"
        assert frequency["frequency_mhz"] == pytest.approx(
            seed_from_inputs(inputs).f01_mhz, abs=10.0
        )
        files = frequency["files"]
        assert files == ["frequency-01.h5"]
        record = engine.records("stages")[0]
        native = load_run_data(
            tmp_path / "data" / record.iteration_dir / "frequency-01.h5"
        )
        assert native.metadata.snapshot == make_plan(inputs).snapshot
        rabi = point["rabi"]
        assert isinstance(rabi, dict)
        assert rabi["status"] == "failed" and rabi["file"] is None
        assert point["success"] is False and point["gain_file"] is None
    finally:
        plots.finish()
        plots.release()


def test_rejected_readout_retains_seed_and_does_not_block_mist(
    inputs: Inputs, tmp_path: Path
) -> None:
    engine, plots = start_scan(
        inputs,
        tmp_path,
        CalibrationTunables(
            span_mhz=2.0, rabi_start_factor=0.125, min_readout_snr=1e12, ro_every=2
        ),
    )
    try:
        status = engine.execute("stages")
        assert status.lifecycle == "done", status.reason
        records = engine.records("stages")
        assert len(records) == 2
        for record in records:
            point = record.encoded_record.value
            assert isinstance(point, dict)
            assert point["success"] is True, (
                "Readout rejection must retain a legal prior setting and continue MIST"
            )
            source, metadata = load_run_with_metadata(
                tmp_path / "data" / record.iteration_dir / "mist.h5",
                spec=PowerDepExp().persistence_spec(),
            )
            assert source.cfg is not None
            assert source.cfg.modules.readout == make_plan(inputs).readout.model_copy(
                update={
                    "pulse_cfg": make_plan(inputs).readout.pulse_cfg.model_copy(
                        update={"gain": seed_from_inputs(inputs).readout_gain}
                    )
                },
                deep=True,
            )
            assert metadata.snapshot == make_plan(inputs).snapshot
        first = records[0].encoded_record.value
        second = records[1].encoded_record.value
        assert isinstance(first, dict) and isinstance(second, dict)
        readout = first["readout"]
        skipped = second["readout"]
        assert isinstance(readout, dict) and isinstance(skipped, dict)
        assert readout["status"] == "rejected" and readout["file"] == "readout.h5"
        assert skipped["status"] == "skipped" and skipped["file"] is None
    finally:
        plots.finish()
        plots.release()
