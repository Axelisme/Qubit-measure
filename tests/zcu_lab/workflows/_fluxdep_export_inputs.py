"""Synthetic committed MIST evidence for exact public conversion contracts."""

from pathlib import Path

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
    FrequencyResult,
    RabiResult,
    ReadoutResult,
)
from zcu_lab.workflows.mist import MISTPlan, MISTPoint, MISTTunables

from ._export_inputs import RUN_ID, Trace, TraceCfg
from ._sim_inputs import Inputs, make_plan, seed_from_inputs


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
    paths = RunPaths(root / "metadata", root / "data")
    engine = Engine(
        EnginePorts(
            Plots(NonPresentingHost()),
            RecordingBar,
            ManualClock(),
            context=seed_from_inputs(inputs),
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

    @workflow("mist_fluxdep", plan=MISTPlan, tunables=MISTTunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: MISTPlan, tun: MISTTunables, state: int
    ) -> Step[int, MISTPoint]:
        del tun
        if state == 3:
            return Done()
        file = None
        if state < 2:
            final = yield from env.run(acquire_gain, TraceCfg(state, "mist/gain"))
            assert isinstance(final, Completed)
            file = "mist.h5"
            save_gain(final, env.iteration_dir / file)
        return Next(
            MISTPoint(
                state,
                plan.fluxes[state],
                FrequencyResult(
                    "accepted" if state < 2 else "failed",
                    None if state < 2 else "missing",
                    5000.0 if state < 2 else None,
                    (),
                ),
                RabiResult(
                    "accepted" if state < 2 else "skipped",
                    None if state < 2 else "dependency",
                    5000.0 if state < 2 else None,
                    0.4 if state < 2 else None,
                    0.2 if state < 2 else None,
                    None,
                ),
                ReadoutResult("skipped", "cadence", None, None, None),
                state < 2,
                None if state < 2 else "missing",
                file,
            ),
            state + 1,
        )

    engine.start(
        step,
        plan=plan,
        tunables=MISTTunables(),
        paths=paths,
        identity=RunIdentity(
            RUN_ID,
            native_metadata().provenance,
            str(plan.snapshot.entry_id),
            plan.snapshot.point,
        ),
    )
    return engine, plan, paths
