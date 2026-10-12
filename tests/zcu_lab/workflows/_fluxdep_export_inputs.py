"""Synthetic committed fluxdep evidence for exact public export contracts."""

from pathlib import Path
from typing import Literal

import numpy as np
from zcu_tools.datafile import (
    CfgSnapshot,
    DataVariable,
    ExperimentPayload,
    LabberMetadata,
    LabberPayload,
    save_run_data,
)
from zcu_tools.experiment.workflows import (
    Completed,
    Done,
    Engine,
    EnginePorts,
    InitEnv,
    Next,
    Run,
    RunIdentity,
    RunPaths,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests._native_support import native_metadata
from tests.experiment.workflows._engine_fakes import (
    ManualClock,
    RecordingBar,
    RecordingDevices,
)
from zcu_lab.workflows.calibration import (
    Calibrated,
    CalibratedPi2,
    CalibrationFailed,
)
from zcu_lab.workflows.mist import MISTPlan, MISTPoint, MISTTunables
from zcu_lab.workflows.t2 import T2EchoTunables, T2Plan, T2Point

from ._export_inputs import RUN_ID, Trace, TraceCfg, acquire, save_trace
from ._sim_inputs import Inputs, make_plan, seed_from_inputs


def synthetic_t2_run(
    inputs: Inputs, root: Path, kind: Literal["echo", "ramsey"]
) -> tuple[Engine[Calibrated], T2Plan, RunPaths]:
    """Commit unequal-delay raw traces, rejected fit and a missing setpoint."""
    plan = T2Plan.model_validate(
        make_plan(inputs).model_dump(mode="python")
    ).model_copy(update={"fluxes": (1.0, 1.1, 1.2, 1.3)})
    seed = seed_from_inputs(inputs)
    cal = CalibratedPi2(
        seed.f01_mhz, seed.pi_length_us, seed.readout_freq_mhz, seed.readout_gain, 0.2
    )
    paths = RunPaths(root / "metadata", root / "data")
    engine = Engine(
        EnginePorts(
            Plots(NonPresentingHost()),
            RecordingBar,
            ManualClock(),
            context=seed,
            devices=RecordingDevices(),
        )
    )

    def init(_env: InitEnv[Calibrated], _plan: T2Plan) -> int:
        return 0

    @workflow(f"T2{kind}_fluxdep", plan=T2Plan, tunables=T2EchoTunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: T2Plan, tun: T2EchoTunables, state: int
    ) -> Step[int, T2Point]:
        del tun
        if state == 3:
            return Done()
        if state == 2:
            return Next(
                T2Point(
                    state,
                    plan.fluxes[state],
                    CalibrationFailed("frequency", "missing"),
                    None,
                    None,
                    "missing",
                    None,
                    None,
                ),
                state + 1,
            )
        calibration = yield from env.run(
            acquire, TraceCfg(state, "synthetic/calibration")
        )
        assert isinstance(calibration, Completed)
        save_trace(calibration, env.iteration_dir / "calibration.h5")
        final = yield from env.run(acquire, TraceCfg(state, f"twotone/ge/t2{kind}"))
        assert isinstance(final, Completed)
        save_trace(final, env.iteration_dir / "decay.h5")
        return Next(
            T2Point(
                state,
                plan.fluxes[state],
                cal,
                20.0 if state == 0 else None,
                1.0 if state == 0 else None,
                None if state == 0 else "T2 fit rejected",
                None,
                "decay.h5",
            ),
            state + 1,
        )

    engine.start(
        step,
        plan=plan,
        tunables=T2EchoTunables(),
        paths=paths,
        identity=RunIdentity(RUN_ID, native_metadata().provenance),
    )
    return engine, plan, paths


def synthetic_mist_run(
    inputs: Inputs, root: Path, *, mismatch: bool = False
) -> tuple[Engine[Calibrated], MISTPlan, RunPaths]:
    """Commit MIST actual gains and one missing point, optionally a bad grid."""
    base = make_plan(inputs)
    plan = MISTPlan(
        flux_dev=base.flux_dev,
        flux_unit=base.flux_unit,
        fluxes=(1.0, 1.1, 1.2),
        qub_pulse=base.qub_pulse,
        readout=base.readout,
        mist_pulse=base.readout.pulse_cfg,
        reset=None,
        gain_start=0.0,
        gain_stop=1.0,
        gain_points=3,
        snapshot=base.snapshot,
    )
    seed = seed_from_inputs(inputs)
    cal = CalibratedPi2(
        seed.f01_mhz, seed.pi_length_us, seed.readout_freq_mhz, seed.readout_gain, 0.2
    )
    paths = RunPaths(root / "metadata", root / "data")
    engine = Engine(
        EnginePorts(
            Plots(NonPresentingHost()),
            RecordingBar,
            ManualClock(),
            context=seed,
            devices=RecordingDevices(),
        )
    )

    def acquire_gain(run: Run[TraceCfg]) -> Trace:
        return Trace(
            np.array([0, 0.4 if mismatch and run.cfg.index == 1 else 0.5, 1.0]),
            np.array([1 + 2j, 2 + 3j, 3 + 4j]) + run.cfg.index,
        )

    def save_gain(source: Completed[TraceCfg, Trace], destination: Path) -> None:
        save_run_data(
            destination,
            ExperimentPayload(
                variables={
                    DataVariable("data"): LabberPayload(
                        ("Signal", "a.u.", source.result.signal),
                        [("Drive Power", "a.u.", source.result.times)],
                    )
                },
                metadata=LabberMetadata(),
                representation="single",
            ),
            native_metadata("mist/gain"),
            cfg=CfgSnapshot(
                values={"index": source.cfg.index},
                cfg_type="SyntheticGain",
                schema_version="1.0",
            ),
        )

    def init(_env: InitEnv[Calibrated], _plan: MISTPlan) -> int:
        return 0

    @workflow("MIST_fluxdep", plan=MISTPlan, tunables=MISTTunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: MISTPlan, tun: MISTTunables, state: int
    ) -> Step[int, MISTPoint]:
        del tun
        if state == 3:
            return Done()
        if state == 2:
            return Next(
                MISTPoint(
                    state,
                    plan.fluxes[state],
                    CalibrationFailed("rabi", "missing"),
                    False,
                    "missing",
                    None,
                    None,
                ),
                state + 1,
            )
        final = yield from env.run(acquire_gain, TraceCfg(state, "mist/gain"))
        assert isinstance(final, Completed)
        save_gain(final, env.iteration_dir / "mist.h5")
        return Next(
            MISTPoint(
                state,
                plan.fluxes[state],
                cal,
                True,
                None,
                None,
                "mist.h5",
            ),
            state + 1,
        )

    engine.start(
        step,
        plan=plan,
        tunables=MISTTunables(),
        paths=paths,
        identity=RunIdentity(RUN_ID, native_metadata().provenance),
    )
    return engine, plan, paths
