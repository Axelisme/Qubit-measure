"""Each independently callable calibration stage owns its result and carry."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.workflows import (
    Done,
    Engine,
    EnginePorts,
    InitEnv,
    Next,
    RunIdentity,
    RunPaths,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.experiment.workflows._engine_fakes import (
    ManualClock,
    RecordingBar,
    RecordingDevices,
)
from zcu_lab.workflows import calibration
from zcu_lab.workflows.saving import save_frequency, save_rabi, save_readout

from ._sim_inputs import Inputs, make_plan, seed_from_inputs


class StagePlan(BaseModel):
    """Test invocation count; real module inputs remain in the closure."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    count: int = 2


@dataclass
class StageState:
    """Carry copied by the Engine at each invocation."""

    carry: calibration.CalCarry
    index: int = 0


def test_public_frequency_rabi_readout_results_preserve_own_carry_after_rejection(
    inputs: Inputs, tmp_path: Path
) -> None:
    # Resolve at the test body, not collection or fixture setup.
    frequency_stage = calibration.calibrate_frequency
    rabi_stage = calibration.calibrate_rabi
    readout_stage = calibration.calibrate_readout
    seed = seed_from_inputs(inputs)
    modules = make_plan(inputs)
    tun = calibration.CalibrationTunables(
        span_mhz=2.0, rabi_start_factor=0.125, min_readout_snr=1e12
    )

    def init(env: InitEnv[calibration.Calibrated], _plan: StagePlan) -> StageState:
        return StageState(calibration.CalCarry(env.context))

    @workflow(
        "stage-contract",
        plan=StagePlan,
        tunables=calibration.CalibrationTunables,
        init=init,
    )
    def stages(
        env: WorkflowEnv[calibration.Calibrated],
        plan: StagePlan,
        tun: calibration.CalibrationTunables,
        state: StageState,
    ) -> Step[
        StageState,
        tuple[
            calibration.FrequencyResult,
            calibration.RabiResult,
            calibration.ReadoutResult,
            calibration.Calibrated,
        ],
    ]:
        if state.index == plan.count:
            return Done()
        captured = calibration.CalibrationInputs(
            modules.qub_pulse, modules.readout, tun
        )
        frequency = yield from frequency_stage(
            env,
            inputs=captured,
            flux=float(state.index),
            carry=state.carry,
            save=partial(save_frequency, snapshot=modules.snapshot),
        )
        rabi = yield from rabi_stage(
            env,
            inputs=captured,
            frequency=frequency,
            carry=state.carry,
            t1_estimate_us=50.0,
            save=partial(save_rabi, snapshot=modules.snapshot),
        )
        if rabi.status != "accepted" or calibration.readout_due(state.index, tun):
            readout = yield from readout_stage(
                env,
                inputs=captured,
                rabi=rabi,
                carry=state.carry,
                t1_estimate_us=50.0,
                save=partial(save_readout, snapshot=modules.snapshot),
            )
        else:
            readout = calibration.ReadoutResult(
                "skipped", "Readout flux-point cadence", None, None, None
            )
        assert len(state.carry.f01_by_flux) == state.index + 1
        assert state.carry.last is not None
        state.index += 1
        return Next((frequency, rabi, readout, state.carry.last), state)

    plots = Plots(NonPresentingHost())
    engine = Engine(
        EnginePorts(
            plots,
            RecordingBar,
            ManualClock(),
            context=seed,
            soc=inputs.soc,
            soccfg=inputs.soccfg,
            devices=RecordingDevices(),
        )
    )
    try:
        engine.start(
            stages,
            plan=StagePlan(),
            tunables=tun,
            paths=RunPaths(tmp_path / "metadata", tmp_path / "data"),
            identity=RunIdentity(
                "parts",
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
        status = engine.execute("parts")
        assert status.lifecycle == "done", status.reason
        for record in engine.records("parts"):
            values = record.encoded_record.value
            assert isinstance(values, list)
            frequency, rabi, readout, carry = values
            assert (
                isinstance(frequency, dict)
                and isinstance(rabi, dict)
                and isinstance(readout, dict)
                and isinstance(carry, dict)
            )
            assert frequency["status"] == rabi["status"] == "accepted"
            assert readout["status"] in ("rejected", "skipped")
            assert carry["f01_mhz"] == frequency["frequency_mhz"]
            assert carry["pi_length_us"] == rabi["pi_length_us"]
            assert carry["readout_freq_mhz"] == seed.readout_freq_mhz
            assert carry["readout_gain"] == seed.readout_gain
            assert (
                frequency["files"] == ["frequency-01.h5"] and rabi["file"] == "rabi.h5"
            )
    finally:
        plots.finish()
        plots.release()
