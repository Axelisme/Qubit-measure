"""Coherence cancellation and lifetime isolation via public Engine/plot ports."""

from pathlib import Path

import pytest
from matplotlib.figure import Figure
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.native_persistence import load_run_with_metadata
from zcu_tools.experiment.workflows import (
    Actor,
    Engine,
    EnginePorts,
    RunIdentity,
    RunPaths,
    TunableChange,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots

import zcu_lab.workflows as workflows
from tests.experiment.workflows._engine_fakes import (
    ManualClock,
    RecordingBar,
    RecordingDevices,
)
from zcu_lab.v2.twotone.freq.core import FreqExp
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoExp
from zcu_lab.workflows.calibration import Calibrated

from ._sim_inputs import Inputs, make_plan, seed_from_inputs


class PauseAfterRabi(NonPresentingHost):
    """Request cancellation at the public plot-host refresh after a fitted Rabi."""

    engine: Engine[Calibrated] | None = None
    requested: bool = False

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        super().refresh(figure, final=final)
        if not self.requested and any(
            line.get_label() == "observed"
            for axis in figure.axes
            for line in axis.lines
        ):
            assert self.engine is not None
            self.requested = True
            self.engine.pause("lifecycle")


def test_pause_discards_partial_stage_carry_and_resume_captures_new_knobs(
    inputs: Inputs, tmp_path: Path
) -> None:
    plan = workflows.CoherencePlan(**make_plan(inputs).model_dump(mode="python"))
    tun = workflows.CoherenceTunables.model_validate(
        {"cal": {"span_mhz": 2.0, "rabi_start_factor": 0.125}}
    )
    host = PauseAfterRabi()
    plots = Plots(host)
    devices = RecordingDevices()
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
    host.engine = engine
    try:
        engine.start(
            workflows.coherence_fluxdep,
            plan=plan,
            tunables=tun,
            paths=RunPaths(tmp_path / "metadata", tmp_path / "data"),
            identity=RunIdentity(
                "lifecycle",
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
        paused = engine.execute("lifecycle")
        assert host.requested and paused.lifecycle == "paused", paused.reason
        assert engine.records("lifecycle") == ()
        engine.update_tunables(
            "lifecycle",
            (TunableChange("cal.max_frequency_rel_err", 1e-8),),
            expected_revision=0,
            actor=Actor("user", "operator"),
        )
        engine.resume("lifecycle")
        status = engine.execute("lifecycle")
        assert status.lifecycle == "done", status.reason
        records = engine.records("lifecycle")
        assert len(records) == 2 and all(record.revision == 1 for record in records)
        assert devices.values == [("flux", 1.0), ("flux", 1.0), ("flux", 1.1)]
        first = records[0]
        assert first.call_seq > 0
        frequency, _ = load_run_with_metadata(
            tmp_path / "data" / first.iteration_dir / "frequency-01.h5",
            spec=FreqExp().persistence_spec(),
        )
        assert frequency.cfg is not None
        grid = frequency.cfg.sweep.freq
        assert (grid.start + grid.stop) / 2 == pytest.approx(
            seed_from_inputs(inputs).f01_mhz
        )
        point = first.encoded_record.value
        assert isinstance(point, dict)
        freq = point["frequency"]
        assert isinstance(freq, dict) and freq["status"] == "rejected"
    finally:
        plots.finish()
        plots.release()


def test_echo_rejection_isolated_from_t1_and_ramsey_and_does_not_change_echo_grid(
    inputs: Inputs, tmp_path: Path
) -> None:
    # The public declarations are resolved in the body before creating a producer.
    plan = workflows.CoherencePlan(
        **make_plan(inputs).model_dump(mode="python"),
        measure_t1=True,
        measure_t2ramsey=True,
        measure_t2echo=True,
    ).model_copy(update={"points": 101})
    tun = workflows.CoherenceTunables.model_validate(
        {
            "cal": {
                "span_mhz": 2.0,
                "rabi_start_factor": 0.125,
                "min_readout_snr": 1e12,
            },
            "t1": {"rounds": 2},
            "t2ramsey": {"rounds": 10, "guess_us": 10.0, "detune_ratio": 0.25},
            "t2echo": {"rounds": 2, "sweep_factor": 0.1},
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
                "lifecycle",
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
        status = engine.execute("lifecycle")
        assert status.lifecycle == "done", status.reason
        for record in engine.records("lifecycle"):
            point = record.encoded_record.value
            assert isinstance(point, dict)
            members = point["measurements"]
            assert isinstance(members, list)
            assert [m["status"] for m in members if isinstance(m, dict)] == [
                "accepted",
                "accepted",
                "rejected",
            ]
            echo = members[2]
            assert isinstance(echo, dict)
            assert "exceeds" in str(echo["reason"]) and echo["file"] == "t2echo.h5"
            assert echo["lifetime_us"] is None
            native, _ = load_run_with_metadata(
                tmp_path / "data" / record.iteration_dir / "t2echo.h5",
                spec=T2EchoExp().persistence_spec(),
            )
            assert native.cfg is not None and native.cfg.sweep.length.stop == 2.0
    finally:
        plots.finish()
        plots.release()


@pytest.mark.parametrize("ratio", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_ramsey_ratio_rejected_even_when_ramsey_disabled(
    inputs: Inputs, ratio: float
) -> None:
    plan = workflows.CoherencePlan(
        **make_plan(inputs).model_dump(mode="python"), measure_t2ramsey=False
    )
    assert not plan.measure_t2ramsey
    with pytest.raises(ValueError, match="finite"):
        workflows.CoherenceTunables.model_validate(
            {"t2ramsey": {"detune_ratio": ratio}}
        )
