"""Calibration policy through the public Engine/calibrate/native simulator seam.

Recording writers exercise Engine publication, not native storage acceptance.
"""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.experiment.workflows import (
    Actor,
    Completed,
    DeviceSnapshot,
    Done,
    Engine,
    EnginePorts,
    InitEnv,
    Next,
    RunIdentity,
    RunPaths,
    Step,
    TunableChange,
    WorkflowEnv,
    workflow,
)
from zcu_tools.program.v2.modules import PulseReadoutCfg

from tests.experiment.workflows._engine_fakes import (
    ManualClock,
    RecordingBar,
    RecordingDevices,
    RecordingPlots,
)
from zcu_lab.v2.twotone.freq.core import FreqCfg
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiCfg
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import FreqGainCfg
from zcu_lab.workflows import calibration
from zcu_lab.workflows.acquire import (
    FrequencyAcquisition,
    RabiAcquisition,
    ReadoutAcquisition,
)
from zcu_lab.workflows.analysis import FrequencyFit, analyze_qubit_freq
from zcu_lab.workflows.calibration import (
    CalCarry,
    Calibrated,
    CalibrationFailed,
    CalibrationSavers,
    CalibrationTunables,
    calibrate,
)
from zcu_lab.workflows.t1 import T1Plan

from ._sim_inputs import Inputs, make_plan


@dataclass
class State:
    carry: CalCarry
    index: int = 0


@dataclass(frozen=True)
class Record:
    result: Calibrated | CalibrationFailed
    carry: CalCarry


@dataclass
class Rig:
    inputs: Inputs
    root: Path
    plan: T1Plan = field(init=False)
    seed: Calibrated = field(init=False)
    engine: Engine[Calibrated] = field(init=False)
    frequencies: list[Completed[FreqCfg, FrequencyAcquisition]] = field(
        default_factory=list
    )
    rabis: list[Completed[LenRabiCfg, RabiAcquisition]] = field(default_factory=list)
    readouts: list[Completed[FreqGainCfg, ReadoutAcquisition]] = field(
        default_factory=list
    )
    results: list[Calibrated | CalibrationFailed] = field(default_factory=list)
    carries: list[CalCarry] = field(default_factory=list)
    pause_frequency: bool = False

    def __post_init__(self) -> None:
        pi_pulse = self.inputs.rabi.modules.qub_pulse
        readout = self.inputs.readout.modules.readout
        assert isinstance(pi_pulse.freq, float)
        assert isinstance(readout.pulse_cfg.freq, float)
        assert isinstance(readout.pulse_cfg.gain, float)
        self.seed = Calibrated(
            pi_pulse.freq, 0.4, readout.pulse_cfg.freq, readout.pulse_cfg.gain
        )
        self.plan = make_plan(self.inputs)
        self.engine = Engine(
            EnginePorts(
                RecordingPlots(),
                RecordingBar,
                ManualClock(),
                context=self.seed,
                soc=self.inputs.soc,
                soccfg=self.inputs.soccfg,
                devices=RecordingDevices(),
            )
        )

    def save_frequency(
        self, source: Completed[FreqCfg, FrequencyAcquisition], path: Path
    ) -> None:
        self.frequencies.append(source)
        path.write_text(source.result.evidence.run_id, encoding="utf-8")
        if self.pause_frequency:
            self.pause_frequency = False
            self.engine.pause("calibration")

    def save_rabi(
        self, source: Completed[LenRabiCfg, RabiAcquisition], path: Path
    ) -> None:
        self.rabis.append(source)
        path.write_text(source.result.evidence.run_id, encoding="utf-8")

    def save_readout(
        self, source: Completed[FreqGainCfg, ReadoutAcquisition], path: Path
    ) -> None:
        self.readouts.append(source)
        path.write_text(source.result.evidence.run_id, encoding="utf-8")

    def start(
        self,
        tun: CalibrationTunables,
        *,
        count: int = 1,
        t1_estimate_us: float = 20.0,
    ) -> None:
        def init(env: InitEnv[Calibrated], _plan: T1Plan) -> State:
            return State(CalCarry(env.context))

        @workflow(
            "calibration",
            plan=T1Plan,
            tunables=CalibrationTunables,
            state=State,
            record=Record,
            init=init,
            requires=("soc", "context"),
        )
        def step(
            env: WorkflowEnv[Calibrated],
            plan: T1Plan,
            tun: CalibrationTunables,
            state: State,
        ) -> Step[State, Record]:
            if state.index == count:
                return Done()
            result = yield from calibrate(
                env,
                plan.qub_pulse,
                plan.readout,
                tun,
                float(state.index),
                state.carry,
                t1_estimate_us=t1_estimate_us,
                savers=CalibrationSavers(
                    self.save_frequency, self.save_rabi, self.save_readout
                ),
            )
            self.results.append(result)
            self.carries.append(deepcopy(state.carry))
            state.index += 1
            return Next(Record(result, state.carry), state)

        self.engine.start(
            step,
            plan=self.plan,
            tunables=tun,
            paths=RunPaths(self.root / "metadata", self.root / "data"),
            identity=RunIdentity("calibration", DeviceSnapshot({}), "offline"),
        )


@pytest.mark.parametrize(
    "estimate", [0.0, -1.0, float("nan"), float("inf"), -float("inf")]
)
def test_invalid_t1_estimate_fails_before_any_acquisition(
    inputs: Inputs, tmp_path: Path, estimate: float
) -> None:
    rig = Rig(inputs, tmp_path)
    rig.start(CalibrationTunables(), t1_estimate_us=estimate)
    status = rig.engine.execute("calibration")
    assert status.lifecycle == "failed"
    assert status.reason is not None and "T1 estimate" in status.reason
    assert not rig.frequencies and not rig.rabis and not rig.readouts
    assert not rig.engine.records("calibration")


@pytest.mark.parametrize("estimate", [12.0, 35.0])
def test_native_calibration_timing_uses_workflow_estimate(
    inputs: Inputs, tmp_path: Path, estimate: float
) -> None:
    rig = Rig(inputs, tmp_path)
    rig.start(
        CalibrationTunables(span_mhz=2.0, rabi_start_factor=0.125),
        t1_estimate_us=estimate,
    )
    status = rig.engine.execute("calibration")
    assert status.lifecycle == "done", status.reason
    frequency, rabi, readout = (
        rig.frequencies[0].cfg,
        rig.rabis[0].cfg,
        rig.readouts[0].cfg,
    )
    assert frequency.relax_delay == 1.0
    assert rabi.relax_delay == readout.relax_delay == 5 * estimate
    native_defaults = (
        FreqCfg(reps=frequency.reps, modules=frequency.modules, sweep=frequency.sweep),
        LenRabiCfg(reps=rabi.reps, modules=rabi.modules, sweep=rabi.sweep),
        FreqGainCfg(reps=readout.reps, modules=readout.modules, sweep=readout.sweep),
    )
    for cfg, default in zip((frequency, rabi, readout), native_defaults, strict=True):
        assert cfg.initial_delay == default.initial_delay
        assert cfg.dev == default.dev


def test_calibration_uses_native_fits_and_commits_complete_scalar_carry(
    inputs: Inputs,
    tmp_path: Path,
) -> None:
    rig = Rig(inputs, tmp_path)
    original = rig.plan.model_copy(deep=True)
    rig.start(CalibrationTunables(span_mhz=2.0, rabi_start_factor=0.125))
    status = rig.engine.execute("calibration")
    assert status.lifecycle == "done", status.reason
    assert len(rig.results) == 1
    result = rig.results[0]
    assert isinstance(result, Calibrated), result
    assert result.f01_mhz == pytest.approx(rig.seed.f01_mhz, abs=10.0)
    assert result.pi_length_us == pytest.approx(rig.seed.pi_length_us, rel=0.05)
    readout = rig.readouts[0].result
    row, col = np.unravel_index(np.argmax(readout.signals), readout.signals.shape)
    assert result.readout_freq_mhz == readout.freqs[row]
    assert result.readout_gain == readout.gains[col]
    assert rig.rabis[0].cfg.modules.qub_pulse.freq == result.f01_mhz
    rabi_readout = rig.rabis[0].cfg.modules.readout
    assert isinstance(rabi_readout, PulseReadoutCfg)
    assert rabi_readout.ro_cfg.ro_freq == rig.seed.readout_freq_mhz
    assert rig.readouts[0].cfg.modules.qub_pulse.freq == result.f01_mhz
    assert rig.carries[0].last == result
    assert rig.carries[0].f01_by_flux == [(0.0, result.f01_mhz)]
    assert rig.carries[0].n_calibrated == 1
    assert rig.plan == original
    records = rig.engine.records("calibration")
    assert len(records) == 1
    assert len(records[0].run_files) == 3
    for run_file in records[0].run_files:
        assert (tmp_path / "data" / records[0].iteration_dir / run_file).is_file()


def test_readout_scan_uses_the_calibrated_rabi_drive(
    inputs: Inputs, tmp_path: Path
) -> None:
    rig = Rig(inputs, tmp_path)
    original = rig.plan.model_copy(deep=True)
    rig.start(CalibrationTunables(span_mhz=2.0, rabi_start_factor=0.125))

    status = rig.engine.execute("calibration")

    assert status.lifecycle == "done", status.reason
    assert len(rig.readouts) == 1
    readout_cfg = rig.readouts[0].cfg
    rabi_drive = rig.rabis[0].cfg.modules.qub_pulse
    assert readout_cfg.modules.qub_pulse.gain == rabi_drive.gain
    assert readout_cfg.modules.qub_pulse.phase == rabi_drive.phase
    result = rig.results[0]
    assert isinstance(result, Calibrated), result
    expected_drive = original.qub_pulse.model_copy(deep=True)
    expected_drive.set_param("freq", result.f01_mhz)
    expected_drive.set_param("length", result.pi_length_us)
    assert readout_cfg.modules.qub_pulse == expected_drive
    assert isinstance(readout_cfg.modules.readout, PulseReadoutCfg)
    assert readout_cfg.modules.readout.pulse_cfg.ch == original.readout.pulse_cfg.ch
    assert readout_cfg.modules.readout.ro_cfg.ro_ch == original.readout.ro_cfg.ro_ch
    assert readout_cfg.sweep.freq.expts == 51
    assert readout_cfg.sweep.freq.stop - readout_cfg.sweep.freq.start == pytest.approx(
        10.0
    )
    assert (
        readout_cfg.sweep.freq.start + readout_cfg.sweep.freq.stop
    ) / 2 == pytest.approx(rig.seed.readout_freq_mhz)
    assert rig.plan == original


def test_readout_cadence_counts_only_successes_and_preserves_prior_readout(
    inputs: Inputs,
    tmp_path: Path,
) -> None:
    rig = Rig(inputs, tmp_path)
    rig.start(CalibrationTunables(span_mhz=2.0, ro_every=2), count=3)
    status = rig.engine.execute("calibration")
    assert status.lifecycle == "done", status.reason
    assert all(isinstance(result, Calibrated) for result in rig.results)
    assert len(rig.frequencies) == len(rig.rabis) == 3
    assert len(rig.readouts) == 2
    first, second = rig.results[:2]
    assert isinstance(first, Calibrated) and isinstance(second, Calibrated)
    assert second.readout_freq_mhz == first.readout_freq_mhz
    assert second.readout_gain == first.readout_gain
    assert rig.carries[-1].n_calibrated == 3
    assert len(rig.carries[-1].f01_by_flux) == 3
    assert tuple(
        len(record.run_files) for record in rig.engine.records("calibration")
    ) == (3, 2, 3)


def test_unavailable_frequency_diagnostics_reject_every_span_without_carry_update(
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(result: FrequencyAcquisition) -> FrequencyFit:
        fit = analyze_qubit_freq(result)
        return replace(fit, quality=replace(fit.quality, normalized_residual_rms=None))

    monkeypatch.setattr(calibration, "analyze_qubit_freq", unavailable)
    rig = Rig(inputs, tmp_path)
    rig.start(
        CalibrationTunables(span_mhz=2.0, rabi_start_factor=0.125, qf_widen=(1.0, 2.0))
    )
    status = rig.engine.execute("calibration")
    assert status.lifecycle == "done", status.reason
    assert rig.results == [CalibrationFailed("frequency", "No accepted frequency fit")]
    assert rig.carries == [CalCarry(rig.seed)]
    assert len(rig.frequencies) == 2
    assert not rig.rabis and not rig.readouts
    assert rig.frequencies[1].cfg.sweep.freq.stop - rig.frequencies[
        1
    ].cfg.sweep.freq.start == pytest.approx(4.0)
    assert len(rig.engine.records("calibration")[0].run_files) == 2


def test_rabi_schedule_failure_retains_frequency_file_but_not_calibration(
    inputs: Inputs,
    tmp_path: Path,
) -> None:
    rig = Rig(inputs, tmp_path)
    # The initial pulse rounds to zero; valid grid spacing still reaches Schedule.
    rig.start(CalibrationTunables(span_mhz=2.0, rabi_start_factor=0.001))
    status = rig.engine.execute("calibration")
    assert status.lifecycle == "done", status.reason
    result = rig.results[0]
    assert isinstance(result, CalibrationFailed) and result.stage == "rabi"
    assert "length" in result.reason
    assert rig.carries == [CalCarry(rig.seed)]
    assert not rig.rabis and not rig.readouts
    assert len(rig.engine.records("calibration")[0].run_files) == 1


def test_readout_rejection_does_not_commit_partial_frequency_or_pi(
    inputs: Inputs,
    tmp_path: Path,
) -> None:
    rig = Rig(inputs, tmp_path)
    rig.start(
        CalibrationTunables(
            span_mhz=2.0, rabi_start_factor=0.125, min_readout_snr=1e12
        ),
        count=2,
    )
    status = rig.engine.execute("calibration")
    assert status.lifecycle == "done", status.reason
    assert (
        rig.results
        == [CalibrationFailed("readout", "No readout SNR meets threshold")] * 2
    )
    assert rig.carries == [CalCarry(rig.seed), CalCarry(rig.seed)]
    assert len(rig.frequencies) == len(rig.rabis) == len(rig.readouts) == 2
    assert len(rig.engine.records("calibration")) == 2


def test_pause_discards_calibration_and_resume_uses_new_tunables(
    inputs: Inputs,
    tmp_path: Path,
) -> None:
    rig = Rig(inputs, tmp_path)
    rig.pause_frequency = True
    rig.start(CalibrationTunables(span_mhz=2.0, rabi_start_factor=0.125))
    assert rig.engine.execute("calibration").lifecycle == "paused"
    assert not rig.engine.records("calibration")
    assert not rig.results and not rig.carries
    assert len(tuple((tmp_path / "data/iter").glob("*/runs/*.h5"))) == 1
    rig.engine.update_tunables(
        "calibration",
        (TunableChange("qf_reps", 200),),
        expected_revision=0,
        actor=Actor("user", "operator"),
    )
    rig.engine.resume("calibration", devices=DeviceSnapshot({}))
    status = rig.engine.execute("calibration")
    assert status.lifecycle == "done", status.reason
    assert tuple(source.cfg.reps for source in rig.frequencies) == (100, 200)
    assert rig.carries[0].n_calibrated == 1
    last = rig.carries[0].last
    assert last is not None
    assert rig.carries[0].f01_by_flux == [(0.0, last.f01_mhz)]
    assert len(rig.engine.records("calibration")) == 1
