"""Explicit author scope lifetimes and nested record publication through Engine."""

import json
from collections.abc import Iterator
from pathlib import Path
from datetime import timedelta

import pytest
from pydantic import BaseModel
from zcu_tools.device import FakeDeviceInfo
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.scope import Scope, ScriptCancelled
from zcu_tools.experiment.workflows import Live, WorkflowEnv, saved_run

from ._engine_fakes import Cfg, Plan, Rig, Tunables, completed


class Collection(BaseModel):
    value: int | None = None
    status: str = "collecting"


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    instance = Rig(tmp_path)
    yield instance
    instance.close()


def test_nested_scopes_create_directories_lazily_and_commit_snapshots_once(rig: Rig) -> None:
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        assert callable(getattr(env, "scope", None)), "env must expose explicit scopes"
        with env.scope("point") as point:
            assert isinstance(point, Scope)
            assert list(rig.paths.data_root.iterdir()) == []
            with point.child("calibration") as calibration:
                assert list(rig.paths.data_root.iterdir()) == []
                assert calibration.dir == rig.paths.data_root / "point-000001/calibration-000001"
                assert calibration.dir.is_dir()
                record = Collection(value=7)
                await calibration.commit(record)
                record.value = 99
                with pytest.raises(ValueError, match="committed"):
                    await calibration.commit(record)
            await point.commit(Collection(value=8))
        await env.commit(Collection(value=9))

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    rows = saved_run.read_records(rig.paths.metadata_root, Collection)
    assert [(row.commit_seq, row.record.value) for row in rows] == [(1, 7), (2, 8), (3, 9)]
    assert [row.directory.relative_to(rig.paths.data_root).as_posix() for row in rows] == [
        "point-000001/calibration-000001", "point-000001", "."
    ]
    envelopes = [json.loads(row.record_path.read_text()) for row in rows]
    assert [envelope["scope_path"] for envelope in envelopes] == [
        "workflow:example/point#1/calibration#1", "workflow:example/point#1", "workflow:example"
    ]
    assert [envelope["name"] for envelope in envelopes] == [None, None, "example"]


@pytest.mark.parametrize("exit_kind", ["break", "exception"])
def test_iterations_release_on_next_and_close_even_without_commit(rig: Rig, exit_kind: str) -> None:
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        assert callable(getattr(env, "iterations", None)), "env must expose iteration lifetimes"
        iterations = env.iterations()
        first = next(iterations)
        assert first.sequence == 1
        second = next(iterations)
        assert second.sequence == 2
        assert env.live_paths == [second.path]
        iterations.close()
        assert env.live_paths == [env.path]
        with pytest.raises(RuntimeError, match="released"):
            first.child("late")
        try:
            for item, iteration in zip(range(2), env.iterations(), strict=False):
                assert iteration.sequence == 3
                if exit_kind == "exception":
                    raise LookupError("author failure")
                break
        except LookupError:
            pass
        assert env.live_paths == [env.path]
        assert list(rig.paths.data_root.iterdir()) == []

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason


def test_child_scope_runs_core_device_and_wait_conveniences(tmp_path: Path) -> None:
    rig = Rig(tmp_path, native=True)
    paths: list[str] = []

    def acquire(cfg: Cfg, *, context: RunContext) -> int:
        paths.append(context.scope.path)
        cfg.value += 1
        return cfg.value

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        assert callable(getattr(env, "scope", None)), "children need shared scope conveniences"
        with env.scope("calibration") as scope:
            cfg = Cfg(value=7)
            record = completed(await scope.run(acquire, cfg, name="T1"))
            assert record.value.result == 8
            assert cfg.value == 7
            await scope.setup_device("flux", FakeDeviceInfo(value=0.25))
            await scope.wait_until(env.started_at + timedelta(seconds=2))
            await scope.commit(Collection(value=record.value.result))

    rig.start(script)
    try:
        final = rig.engine.execute("run")
        assert final.lifecycle == "done", final.reason
        assert paths == ["workflow:example/calibration#1/operation:T1#1"]
        assert rig.devices.values == [("flux", 0.25)]
        assert rig.clock.targets == [rig.clock.value]
        assert [operation.kind for operation in rig.engine.records("run")[0].operations] == ["run", "setup_device", "wait_until"]
    finally:
        rig.close()


def test_root_cancel_raises_script_cancelled_and_runs_finally_without_error(rig: Rig) -> None:
    seen: list[str] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        try:
            def stop(ctx: object) -> None:
                rig.engine.stop("run")
            try:
                await env.call(stop)
            except ScriptCancelled:
                seen.append("cancelled")
                raise
            seen.append("delivered")
        finally:
            seen.append("finally")

    rig.start(script)
    final = rig.engine.execute("run")
    assert seen == ["cancelled", "finally"]
    assert final.lifecycle == "cancelled"
    assert final.reason is None


def test_operation_facts_use_iteration_birth_not_commit_count(rig: Rig) -> None:
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        assert callable(getattr(env, "iterations", None)), "iteration identity must be explicit"
        for value, iteration in zip(range(3), env.iterations(), strict=False):
            outcome = completed(await iteration.call(lambda ctx: ctx.scope.path, name="probe"))
            assert outcome.value == f"workflow:example/iteration#{value + 1}/operation:probe#1"
            assert outcome.facts.iteration == iteration.sequence
            if value == 1:
                continue
            await iteration.commit(Collection(value=value))

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    events = [event for event in rig.events() if event["kind"] == "operation_finished"]
    assert [event["call_seq"] for event in events] == [1, 2, 3]
    records = rig.engine.records("run")
    assert [record.call_seq for record in records] == [1, 3]
    assert [[operation.iteration for operation in record.operations] for record in records] == [[1], [3]]
