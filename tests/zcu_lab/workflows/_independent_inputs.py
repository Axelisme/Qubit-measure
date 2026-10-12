"""New Engine runs with named native acquisitions and independently chosen scalars."""

from dataclasses import dataclass, replace
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
from zcu_lab.workflows.coherence import CoherencePlan, CoherencePoint, CoherenceTunables
from zcu_lab.workflows.coherence_measurements import LifetimeKind, LifetimeResult
from zcu_lab.workflows.mist import MISTPlan, MISTPoint, MISTTunables

from ._sim_inputs import Inputs, make_plan, seed_from_inputs

RUN_ID = "20261006T010000Z-a1b2c3"


@dataclass(frozen=True)
class SavedRun:
    """Fixed paths/plan and finished Engine for corruption/lifecycle observations."""

    paths: RunPaths
    plan: CoherencePlan | MISTPlan
    engine: Engine[Calibrated]


def save_input(
    path: Path, tag: str, values: np.ndarray, *, coordinate: float, gain: bool = False
) -> None:
    """Use the native writer, never hand-encode native files."""
    save_run_data(
        path,
        ExperimentPayload(
            variables={
                DataVariable("data"): LabberPayload(
                    (
                        "Signal",
                        "a.u.",
                        np.arange(len(values))
                        + 1j * (coordinate + np.arange(len(values))),
                    ),
                    [
                        (
                            "Drive Power" if gain else "Time",
                            "a.u." if gain else "s",
                            values,
                        )
                    ],
                )
            },
            metadata=LabberMetadata(),
            representation="single",
        ),
        replace(
            native_metadata(tag),
            device_snapshot={"flux": {"value": coordinate, "value_unit": "native"}},
        ),
        cfg=CfgSnapshot(values={}, cfg_type="SyntheticTrace", schema_version="1.0"),
    )


def coherence_run(
    inputs: Inputs,
    root: Path,
    *,
    t1: bool = True,
    ramsey: bool = True,
    echo: bool = True,
) -> SavedRun:
    """Four points: accepted lifetimes, rejected Echo, frequency-only, no accepted value."""
    plan = CoherencePlan.model_validate(make_plan(inputs).model_dump()).model_copy(
        update={
            "fluxes": (1.0, 1.1, 1.2, 1.3),
            "measure_t1": t1,
            "measure_t2ramsey": ramsey,
            "measure_t2echo": echo,
        }
    )
    seed = seed_from_inputs(inputs)
    paths = RunPaths(root / "metadata", root / "data")
    plots = Plots(NonPresentingHost())
    engine = Engine(
        EnginePorts(
            plots, RecordingBar, ManualClock(), context=seed, devices=RecordingDevices()
        )
    )

    def init(_env: InitEnv[Calibrated], _plan: CoherencePlan) -> int:
        return 0

    @workflow(
        "coherence_fluxdep", plan=CoherencePlan, tunables=CoherenceTunables, init=init
    )
    def step(
        env: WorkflowEnv[Calibrated],
        plan: CoherencePlan,
        tun: CoherenceTunables,
        state: int,
    ) -> Step[int, CoherencePoint]:
        del tun
        yield from env.wait_until(env.started_at)
        if state == 4:
            return Done()
        file = "frequency-final.h5"
        save_input(
            env.iteration_dir / file,
            "twotone/ge/freq",
            np.array([1.0, 2.0, 3.0]),
            coordinate=10 + state,
        )
        frequency = FrequencyResult(
            "accepted" if state != 3 else "rejected",
            None if state != 3 else "fit rejected",
            5000.0 + state if state != 3 else None,
            (file,),
        )
        rabi = RabiResult("skipped", "dependency", None, None, None, None)
        readout = ReadoutResult("skipped", "cadence", None, None, None)
        measurements: list[LifetimeResult] = []
        for kind, enabled, count, coordinate in (
            ("t1", t1, 3, 40),
            ("t2ramsey", ramsey, 5, 50),
            ("t2echo", echo, 2, 60),
        ):
            if not enabled:
                continue
            typed_kind: LifetimeKind = kind
            if state >= 2:
                measurements.append(
                    LifetimeResult(
                        typed_kind, "skipped", "dependency", None, None, None
                    )
                )
                continue
            filename = kind + ".h5"
            save_input(
                env.iteration_dir / filename,
                "twotone/ge/" + kind,
                np.arange(count) * (state + 1) * 1e-6,
                coordinate=coordinate + state,
            )
            rejected = state == 1 and kind == "t2echo"
            measurements.append(
                LifetimeResult(
                    typed_kind,
                    "rejected" if rejected else "accepted",
                    "fit rejected" if rejected else None,
                    None if rejected else 20.0 + count,
                    None if rejected else 1.0,
                    filename,
                )
            )
        return Next(
            CoherencePoint(
                state, plan.fluxes[state], frequency, rabi, readout, tuple(measurements)
            ),
            state + 1,
        )

    engine.start(
        step,
        plan=plan,
        tunables=CoherenceTunables(),
        paths=paths,
        identity=RunIdentity(
            RUN_ID,
            native_metadata().provenance,
            str(plan.snapshot.entry_id),
            plan.snapshot.entry_name,
        ),
    )
    status = engine.execute(RUN_ID)
    assert status.lifecycle == "done", status.reason
    plots.finish()
    plots.release()
    return SavedRun(paths, plan, engine)


def mist_run(inputs: Inputs, root: Path, *, mismatch: bool = False) -> SavedRun:
    """Exact actual gain grid and one failed point, generated by a new Engine."""
    base = make_plan(inputs)
    plan = MISTPlan(
        flux_dev=base.flux_dev,
        flux_unit=base.flux_unit,
        fluxes=(1.0, 1.1),
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
    paths = RunPaths(root / "metadata", root / "data")
    plots = Plots(NonPresentingHost())
    engine = Engine(
        EnginePorts(
            plots, RecordingBar, ManualClock(), context=seed, devices=RecordingDevices()
        )
    )

    def init(_env: InitEnv[Calibrated], _plan: MISTPlan) -> int:
        return 0

    @workflow("mist_fluxdep", plan=MISTPlan, tunables=MISTTunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: MISTPlan, tun: MISTTunables, state: int
    ) -> Step[int, MISTPoint]:
        del tun
        yield from env.wait_until(env.started_at)
        if state == 2:
            return Done()
        frequency = FrequencyResult("failed", "schedule", None, ())
        rabi = RabiResult("skipped", "dependency", None, None, None, None)
        readout = ReadoutResult("skipped", "dependency", None, None, None)
        filename = None
        if state == 0:
            filename = "gain.h5"
            save_input(
                env.iteration_dir / filename,
                "mist/gain",
                np.array([0.0, 0.4 if mismatch else 0.5, 1.0]),
                coordinate=40,
                gain=True,
            )
        return Next(
            MISTPoint(
                state,
                plan.fluxes[state],
                frequency,
                rabi,
                readout,
                state == 0,
                None if state == 0 else "schedule",
                filename,
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
            plan.snapshot.entry_name,
        ),
    )
    status = engine.execute(RUN_ID)
    assert status.lifecycle == "done", status.reason
    plots.finish()
    plots.release()
    return SavedRun(paths, plan, engine)
