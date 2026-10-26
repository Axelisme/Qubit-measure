"""Completed exposes the attempt facts shared by journal and commit readers."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import TypeAdapter
from zcu_tools.datafile import JsonObject
from zcu_tools.device import FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.acquisition_facts import AcquisitionFacts, StopDecision
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.script_host import CallContext
from zcu_tools.experiment.v2.runtime import AcquireOptions, Schedule, SignalBuffer
from zcu_tools.experiment.workflows import (
    Completed,
    Failed,
    Live,
    WorkflowEnv,
    saved_run,
)
from zcu_tools.program.acquisition import CancelFlagProtocol

from ._engine_fakes import Cfg, Plan, Point, Rig, Tunables
from ._engine_ports import START


class RoundProgram:
    def __init__(self, *, always_fail: bool) -> None:
        self.attempts = 0
        self.always_fail = always_fail

    def acquire(
        self,
        soc: object,
        *,
        rounds: int,
        progress: bool,
        round_hook: Callable[[int, NDArray[np.float64], CancelFlagProtocol], object],
        cancel_flag: CancelFlagProtocol,
        **kwargs: object,
    ) -> NDArray[np.float64]:
        self.attempts += 1
        if self.always_fail or self.attempts == 1:
            raise RuntimeError("injected acquisition failure")
        raw = np.array([1.0])
        for ir in range(1, rounds + 1):
            if cancel_flag.is_set():
                break
            round_hook(ir, raw, cancel_flag)
        return raw

    acquire_decimated = acquire


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    value = Rig(tmp_path, native=True)
    value.devices.drivers["flux"] = FakeDevice(address="memory", fast_mode=True)
    try:
        yield value
    finally:
        value.close()


def test_acquisition_facts_reach_each_named_stage_including_failures(rig: Rig) -> None:
    delivered = []

    def acquire(cfg: Cfg, *, context: RunContext) -> int:
        buffer = SignalBuffer((1,), dtype=np.float64)
        with Schedule(cfg, buffer, scope=context.scope) as schedule:
            schedule.prog_builder(context.soc, context.soccfg).run_program(
                RoundProgram(always_fail=cfg.value == 0),
                rounds=4,
                raw2signal_fn=lambda raw: raw,
                options=AcquireOptions(
                    retry=1, stop_condition=lambda: StopDecision(True, 12.5)
                ),
            )
        return 1

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        for iteration in env.iterations():
            delivered.append(await iteration.run(acquire, Cfg(value=1), name="T1"))
            delivered.append(await iteration.run(acquire, Cfg(value=0), name="Echo"))
            iteration.commit(Point(value=1))
            break

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    first, failed = delivered
    assert isinstance(first, Completed)
    assert isinstance(failed, Failed)
    operations = saved_run.read_records(rig.paths.metadata_root, Point)[0].operations
    assert operations[0] == first.facts
    assert [op.name for op in operations] == ["T1", "Echo"]
    success, failure = [op.acquisitions[0] for op in operations]
    assert success.requested_rounds == 4
    assert success.completed_rounds == 1
    assert success.attempts == 2
    assert success.early_stopped
    assert success.stop_value == 12.5
    assert operations[1].outcome == "failed"
    assert failure.requested_rounds == 4
    assert failure.completed_rounds == 0
    assert failure.attempts == 2
    assert not failure.early_stopped
    assert failure.stop_value is None


@pytest.mark.parametrize("seconds", [None, 0.0, 94.0])
def test_shot_time_serializes_in_journal_and_committed_records(
    rig: Rig, seconds: float | None
) -> None:
    facts = AcquisitionFacts(10, 10, 1, False, simulated_physical_s=seconds)

    def observe(ctx: CallContext) -> None:
        ctx.scope.stop.record_acquisition(facts)

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        for iteration in env.iterations():
            await iteration.call(observe)
            iteration.commit(Point(value=1))
            break

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    record = saved_run.read_records(rig.paths.metadata_root, Point)[0]
    assert record.operations[0].acquisitions == (facts,)
    envelope = json.loads(record.record_path.read_text())
    events = [event for event in rig.events() if event["kind"] == "operation_finished"]
    for payload in (envelope["operations"][0], events[0]["operation"]):
        operation = TypeAdapter(JsonObject).validate_python(payload)
        acquisition = TypeAdapter(list[JsonObject]).validate_python(
            operation["acquisitions"]
        )[0]
        if seconds is None:
            assert "simulated_physical_s" not in acquisition
        else:
            assert acquisition["simulated_physical_s"] == seconds

    # Reading an older record does not infer shot time from its other facts.
    envelope["operations"][0]["acquisitions"][0].pop("simulated_physical_s", None)
    record.record_path.write_text(json.dumps(envelope))
    historical = saved_run.read_records(rig.paths.metadata_root, Point)[0]
    assert historical.operations[0].acquisitions[0].simulated_physical_s is None


@pytest.mark.parametrize("kind", ["call", "run", "setup_device", "wait_until"])
def test_completed_facts_match_journal_and_commit_scope(
    rig: Rig, kind: Literal["call", "run", "setup_device", "wait_until"]
) -> None:
    delivered: list[object] = []

    def acquire(cfg: Cfg, *, context: RunContext) -> int:
        # The facts must retain the pre-call snapshot, not these later settings.
        rig.devices.drivers["flux"].setup(
            FakeDeviceInfo(address="memory", value=7), progress=False
        )
        return cfg.value

    async def operation(env: Scope) -> None:
        if kind == "call":
            delivered.append(await env.call(lambda ctx: 5))
        elif kind == "run":
            delivered.append(await env.run(acquire, Cfg()))
        elif kind == "setup_device":
            delivered.append(
                await env.setup_device("flux", FakeDeviceInfo(address="memory"))
            )
        else:
            delivered.append(await env.wait_until(START))

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        for value, iteration in zip((1, 2), env.iterations(), strict=False):
            await operation(iteration)
            if value == 1:
                await operation(iteration)
            iteration.commit(Point(value=value))

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    operations = []
    for outcome in delivered:
        assert isinstance(outcome, Completed)
        assert hasattr(outcome, "facts"), "Completed must expose its operation facts"
        assert outcome.facts.kind == kind
        assert outcome.facts.stopped is outcome.partial_result is False
        assert outcome.facts.outcome == "completed"
        assert outcome.facts.reason is None
        assert outcome.facts.started_at == outcome.facts.finished_at == START
        operations.append(outcome.facts)
    assert [(op.iteration, op.attempt) for op in operations] == [(1, 1), (1, 2), (2, 1)]
    if kind == "run":
        assert operations[0].device_snapshot == {
            "flux": FakeDeviceInfo(address="memory").model_dump(mode="json")
        }
        assert operations[1].device_snapshot is not None
        assert operations[1].device_snapshot["flux"]["value"] == 7
    else:
        assert all(op.device_snapshot is None for op in operations)
        assert rig.devices.captures == 0
    records = rig.engine.records("run")
    saved = saved_run.read_records(rig.paths.metadata_root, Point)
    assert records[0].operations == saved[0].operations == tuple(operations[:2])
    assert records[1].operations == saved[1].operations == tuple(operations[2:])
    journal = [
        event["operation"]
        for event in rig.events()
        if event["kind"] == "operation_finished"
    ]
    assert journal == [
        TypeAdapter(type(op)).dump_python(op, mode="json") for op in operations
    ]
