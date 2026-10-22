"""Completed exposes the attempt facts shared by journal and commit readers."""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest
from pydantic import TypeAdapter
from zcu_tools.device import FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Completed, Live, WorkflowEnv, saved_run

from ._engine_fakes import Cfg, Plan, Point, Rig, Tunables
from ._engine_ports import START


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    value = Rig(tmp_path, native=True)
    value.devices.drivers["flux"] = FakeDevice(address="memory", fast_mode=True)
    try:
        yield value
    finally:
        value.close()


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
