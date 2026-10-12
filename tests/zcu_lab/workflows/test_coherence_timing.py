"""Preserved calibration timing and independent finite Ramsey admission."""

from pathlib import Path

import pytest
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.native_persistence import load_run_with_metadata
from zcu_tools.experiment.workflows import Engine, EnginePorts, RunIdentity, RunPaths
from zcu_tools.plotting.plots import NonPresentingHost, Plots

import zcu_lab.workflows as workflows
from tests.experiment.workflows._engine_fakes import (
    ManualClock,
    RecordingBar,
    RecordingDevices,
)
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiExp
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import FreqGainExp

from ._sim_inputs import Inputs, make_plan, seed_from_inputs


def test_readout_and_rabi_use_latest_accepted_t1_not_its_delay_ewma(
    inputs: Inputs, tmp_path: Path
) -> None:
    plan = workflows.CoherencePlan(
        **make_plan(inputs).model_dump(mode="python"),
        measure_t2ramsey=False,
        measure_t2echo=False,
    )
    tun = workflows.CoherenceTunables.model_validate(
        {
            "cal": {"span_mhz": 2.0, "rabi_start_factor": 0.125, "ro_every": 1},
            "t1": {"guess_us": 50.0, "rounds": 2},
        }
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
    try:
        engine.start(
            workflows.coherence_fluxdep,
            plan=plan,
            tunables=tun,
            paths=RunPaths(tmp_path / "metadata", tmp_path / "data"),
            identity=RunIdentity(
                "timing",
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
        status = engine.execute("timing")
        assert status.lifecycle == "done", status.reason
        expected_relax_us = 250.0
        for record in engine.records("timing"):
            root = tmp_path / "data" / record.iteration_dir
            rabi, _ = load_run_with_metadata(
                root / "rabi.h5", spec=LenRabiExp().persistence_spec()
            )
            readout, _ = load_run_with_metadata(
                root / "readout.h5", spec=FreqGainExp().persistence_spec()
            )
            assert rabi.cfg is not None and readout.cfg is not None
            assert rabi.cfg.relax_delay == pytest.approx(expected_relax_us)
            assert readout.cfg.relax_delay == pytest.approx(expected_relax_us)
            point = record.encoded_record.value
            assert isinstance(point, dict)
            members = point["measurements"]
            assert isinstance(members, list)
            member = members[0]
            assert isinstance(member, dict) and member["status"] == "accepted"
            lifetime = member["lifetime_us"]
            assert isinstance(lifetime, float)
            expected_relax_us = 5 * lifetime
    finally:
        plots.finish()
        plots.release()


@pytest.mark.parametrize("ratio", [float("nan"), float("inf"), -float("inf")])
def test_enabled_ramsey_nonfinite_ratio_is_invalid(ratio: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        workflows.CoherenceTunables.model_validate(
            {"t2ramsey": {"detune_ratio": ratio}}
        )
