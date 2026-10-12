"""Synthetic committed T1 acquisitions through the public Engine seam."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.datafile import (
    CfgSnapshot,
    DataVariable,
    ExperimentPayload,
    LabberMetadata,
    LabberPayload,
    save_run_data,
)
from zcu_tools.experiment.workflows import (
    Aborted,
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
from zcu_lab.workflows.coherence import CoherencePlan, CoherencePoint, CoherenceTunables
from zcu_lab.workflows.coherence_measurements import LifetimeKind, LifetimeResult

from ._sim_inputs import Inputs, make_plan, seed_from_inputs

RUN_ID = "20261006T010000Z-a1b2c3"
SAVED_AT = datetime(2026, 10, 6, 2, tzinfo=UTC)


@dataclass(frozen=True)
class TraceCfg:
    index: int
    tag: str


@dataclass(frozen=True)
class Trace:
    times: NDArray[np.float64]
    signal: NDArray[np.complex128]


def acquire(run: Run[TraceCfg]) -> Trace:
    grids = ([0, 1e-6, 2e-6], [0, 3e-6], [0], [0, 4e-6, 8e-6, 12e-6])
    values = (
        [1 + 2j, 2 + 3j, complex(np.nan, np.nan)],
        [4 + 5j, 6 + 7j],
        [0],
        [8, 9, 10, 11],
    )
    signal = np.array(values[run.cfg.index], dtype=complex)
    if run.cfg.tag == "synthetic/calibration":
        signal[:] = 999
    return Trace(np.array(grids[run.cfg.index], dtype=float), signal)


def save_trace(source: Completed[TraceCfg, Trace], destination: Path) -> None:
    save_run_data(
        destination,
        ExperimentPayload(
            variables={
                DataVariable("data"): LabberPayload(
                    ("Signal", "a.u.", source.result.signal),
                    [("Time", "s", source.result.times)],
                )
            },
            metadata=LabberMetadata(),
            representation="single",
        ),
        native_metadata(source.cfg.tag),
        cfg=CfgSnapshot(
            values={"index": source.cfg.index},
            cfg_type="SyntheticTrace",
            schema_version="1.0",
        ),
    )


def synthetic_run(
    inputs: Inputs,
    root: Path,
    terminal: Literal["done", "stopped", "failed", "aborted"] = "done",
    *,
    run_id: str = RUN_ID,
    kind: LifetimeKind = "t1",
) -> tuple[Engine[Calibrated], CoherencePlan, RunIdentity, RunPaths]:
    plan = CoherencePlan.model_validate(make_plan(inputs).model_dump()).model_copy(
        update={
            "fluxes": (1.0, 1.1, 1.2, 1.3),
            "measure_t1": kind == "t1",
            "measure_t2ramsey": kind == "t2ramsey",
            "measure_t2echo": kind == "t2echo",
        }
    )
    metadata = native_metadata()
    identity = RunIdentity(
        run_id, metadata.provenance, str(plan.snapshot.entry_id), plan.snapshot.point
    )
    paths = RunPaths(root / "metadata", root / "data")
    clock = ManualClock()
    engine = Engine(
        EnginePorts(
            Plots(NonPresentingHost()),
            RecordingBar,
            clock,
            context=seed_from_inputs(inputs),
            devices=RecordingDevices(),
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
        if state == 3 and terminal == "done":
            return Done()
        if state == 3 and terminal == "failed":
            raise RuntimeError("synthetic failure")
        if state == 3 and terminal == "aborted":
            return Aborted("synthetic abort")
        if state == 2:
            return Next(
                CoherencePoint(
                    state,
                    plan.fluxes[state],
                    FrequencyResult("failed", "missing calibration", None, ()),
                    RabiResult("skipped", "dependency", None, None, None, None),
                    ReadoutResult("skipped", "dependency", None, None, None),
                    (LifetimeResult(kind, "skipped", "dependency", None, None, None),),
                ),
                state + 1,
            )
        calibration = yield from env.run(
            acquire, TraceCfg(state, "synthetic/calibration")
        )
        assert isinstance(calibration, Completed)
        save_trace(calibration, env.iteration_dir / "calibration.h5")
        final = yield from env.run(acquire, TraceCfg(state, "twotone/ge/" + kind))
        assert isinstance(final, Completed)
        save_trace(final, env.iteration_dir / "decay.h5")
        if state == 3:
            clock.on_wait = lambda _signal: engine.stop(run_id)
            yield from env.wait_until(clock.now() + timedelta(seconds=1))
        return Next(
            CoherencePoint(
                state,
                plan.fluxes[state],
                FrequencyResult("accepted", None, 5000.0, ()),
                RabiResult("accepted", None, 5000.0, 0.4, 0.2, None),
                ReadoutResult("skipped", "cadence", None, None, None),
                (
                    LifetimeResult(
                        kind,
                        "accepted" if state == 0 else "rejected",
                        None if state == 0 else "fit rejected",
                        20.0 if state == 0 else None,
                        1.0 if state == 0 else None,
                        "decay.h5",
                    ),
                ),
            ),
            state + 1,
        )

    engine.start(
        step, plan=plan, tunables=CoherenceTunables(), paths=paths, identity=identity
    )
    return engine, plan, identity, paths
