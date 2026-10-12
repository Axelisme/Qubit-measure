"""Real flux-bound native science and stopped-run evidence on Coherence."""

from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.datafile import JsonObject, SoftwareProvenance
from zcu_tools.device import FakeDevice
from zcu_tools.experiment.native_persistence import load_run_with_metadata
from zcu_tools.experiment.workflows import Engine, EnginePorts, RunIdentity, RunPaths
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseReadoutCfg, ReadoutCfg

import zcu_lab.workflows as workflows
from tests.experiment.workflows._engine_fakes import (
    ManualClock,
    RecordingBar,
    RecordingDevices,
)
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiExp
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import FreqGainExp
from zcu_lab.v2.twotone.time_domain.t1.core import T1Exp
from zcu_lab.workflows.calibration import Calibrated

from ._sim_inputs import Inputs, make_plan, seed_from_inputs


class StopSecondFrequency(NonPresentingHost):
    """Stop during second-point native live delivery, without replacing callbacks."""

    engine: Engine[Calibrated] | None = None
    requested: bool = False

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        super().refresh(figure, final=final)
        if self.requested or self.engine is None:
            return
        if len(self.engine.records("flux")) == 1 and any(
            axis.get_xlabel() == "Frequency (MHz)"
            and any(np.asarray(line.get_xdata()).size > 2 for line in axis.lines)
            for axis in figure.axes
        ):
            self.requested = True
            self.engine.stop("flux")


def readout_preparation(
    root: Path, point: JsonObject, native_readout: ReadoutCfg
) -> None:
    """Check real calibration preparation and the accepted native readout setting."""
    rabi, _ = load_run_with_metadata(
        root / "rabi.h5", spec=LenRabiExp().persistence_spec()
    )
    readout, _ = load_run_with_metadata(
        root / "readout.h5", spec=FreqGainExp().persistence_spec()
    )
    assert rabi.cfg is not None and readout.cfg is not None
    assert readout.cfg.modules.qub_pulse.freq == rabi.cfg.modules.qub_pulse.freq
    fitted = point["rabi"]
    assert isinstance(fitted, dict)
    assert readout.cfg.modules.qub_pulse.waveform.length == pytest.approx(
        fitted["pi_length_us"]
    )
    accepted_ro = point["readout"]
    assert isinstance(accepted_ro, dict) and accepted_ro["status"] == "accepted"
    assert isinstance(native_readout, PulseReadoutCfg)
    assert native_readout.ro_cfg.ro_freq == accepted_ro["frequency_mhz"]


def test_flux_bound_native_coherence_retains_calibrated_drive_readout_and_snapshot(
    inputs: Inputs, tmp_path: Path
) -> None:
    plan = workflows.CoherencePlan(
        **make_plan(inputs).model_dump(mode="python"),
        measure_t2ramsey=False,
        measure_t2echo=False,
    ).model_copy(
        update={
            "fluxes": tuple(inputs.flux_value + offset for offset in (0.0, 0.01, 0.02))
        }
    )
    tun = workflows.CoherenceTunables.model_validate(
        {
            "cal": {"span_mhz": 10.0, "rabi_start_factor": 0.125, "ro_every": 2},
            "t1": {"rounds": 2},
        }
    )
    devices = RecordingDevices()
    driver = FakeDevice(fast_mode=True)
    devices.drivers["flux"] = driver
    plots = Plots(NonPresentingHost())
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
    inputs.soc.set_flux_source(driver.get_value)
    try:
        engine.start(
            workflows.coherence_fluxdep,
            plan=plan,
            tunables=tun,
            paths=RunPaths(tmp_path / "metadata", tmp_path / "data"),
            identity=RunIdentity(
                "flux",
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
        status = engine.execute("flux")
        assert status.lifecycle == "done", status.reason
        assert devices.values == [("flux", flux) for flux in plan.fluxes]
        frequencies = []
        prior_readout = None
        for record in engine.records("flux"):
            point = record.encoded_record.value
            assert isinstance(point, dict)
            members = point["measurements"]
            assert isinstance(members, list)
            t1 = members[0]
            assert isinstance(t1, dict) and t1["status"] == "accepted"
            assert t1["lifetime_us"] == pytest.approx(20.0, rel=0.2)
            root = tmp_path / "data" / record.iteration_dir
            native, metadata = load_run_with_metadata(
                root / "t1.h5", spec=T1Exp().persistence_spec()
            )
            assert native.cfg is not None and metadata.device_snapshot is not None
            assert metadata.device_snapshot["flux"]["value"] == point["flux"]
            assert metadata.device_snapshot["flux"]["value_unit"] == "native"
            assert isinstance(native.cfg.modules.pi_pulse.freq, float)
            frequencies.append(native.cfg.modules.pi_pulse.freq)
            if point["index"] == 1:
                assert native.cfg.modules.readout == prior_readout
            if point["index"] == 0:
                readout_preparation(root, point, native.cfg.modules.readout)
                prior_readout = native.cfg.modules.readout.model_copy(deep=True)
                assert isinstance(native.cfg.sweep.length, SweepCfg)
        assert frequencies[0] > frequencies[1] > frequencies[2]
    finally:
        inputs.soc.set_flux_source(None)
        plots.finish()
        plots.release()
        driver.close()


def test_stop_during_second_acquisition_retains_only_first_committed_point(
    inputs: Inputs, tmp_path: Path
) -> None:
    plan = workflows.CoherencePlan(
        **make_plan(inputs).model_dump(mode="python"),
        measure_t2ramsey=False,
        measure_t2echo=False,
    )
    tun = workflows.CoherenceTunables.model_validate(
        {"cal": {"span_mhz": 2.0, "rabi_start_factor": 0.125}}
    )
    host = StopSecondFrequency()
    plots = Plots(host)
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
    host.engine = engine
    try:
        engine.start(
            workflows.coherence_fluxdep,
            plan=plan,
            tunables=tun,
            paths=RunPaths(tmp_path / "metadata", tmp_path / "data"),
            identity=RunIdentity(
                "flux",
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
        status = engine.execute("flux")
        assert host.requested and status.lifecycle == "stopped", status.reason
        records = engine.records("flux")
        assert len(records) == 1
        point = records[0].encoded_record.value
        assert isinstance(point, dict) and point["flux"] == 1.0
        native, metadata = load_run_with_metadata(
            tmp_path / "data" / records[0].iteration_dir / "t1.h5",
            spec=T1Exp().persistence_spec(),
        )
        assert native.cfg is not None and metadata.completion == "complete"
    finally:
        plots.finish()
        plots.release()
