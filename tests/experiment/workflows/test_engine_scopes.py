"""Workflow scoped cancel delivers partial facts and replaces iteration at commit."""

from collections.abc import Iterator
from pathlib import Path
from threading import Event, get_ident

import pytest
from zcu_tools.core.script_protocol import Answer, Question
from zcu_tools.experiment.script_host import CallContext
from zcu_tools.experiment.workflows import Engine, EnginePorts, Live, WorkflowEnv

from ._engine_fakes import (
    Plan,
    Point,
    Rig,
    Tunables,
    completed,
    pending,
    running,
)
from ._engine_ports import RecordingBar


class Decision(Answer):
    pass


class Choose(Question[Decision]):
    pass


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    rig = Rig(tmp_path)
    yield rig
    rig.close()


@pytest.mark.parametrize("scope", ["acquire", "schedule", "run", "iteration", "script"])
def test_workflow_scope_ack_status_and_partial_facts(
    tmp_path: Path, scope: str
) -> None:
    rig = Rig(tmp_path)
    outcomes: list[tuple[bool, bool]] = []
    acquisitions: list[bool] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        def acquire(ctx: CallContext) -> int:
            schedule = ctx.stop_signal.child(label="schedule")
            first = schedule.child(label="acquire")
            try:
                status = rig.engine.status()
                assert status is not None
                assert status.active_path == (
                    "script/iteration#1/operation#1/schedule#1/acquire#1"
                )
                result = rig.engine.cancel("run", scope=scope, expected=None)
                assert result.cancelled
                assert result.reason == "scope cancellation requested"
                assert result.active_path == status.active_path
                acquisitions.append(first.is_set())
            finally:
                first.release()
            second = schedule.child(label="acquire")
            acquisitions.append(second.is_set())
            second.release()
            schedule.release()
            return 11

        first = completed(await env.call(acquire))
        outcomes.append((first.partial_result, first.facts.stopped))
        second = completed(await env.call(lambda ctx: ctx.stop_signal.is_set()))
        outcomes.append((second.partial_result, second.facts.stopped))
        await env.commit(Point(value=11))
        third = completed(await env.call(lambda ctx: ctx.stop_signal.is_set()))
        outcomes.append((third.partial_result, third.facts.stopped))
        await env.commit(Point(value=12))

    rig.start(script)
    try:
        final = rig.engine.execute("run")
        assert final.lifecycle == ("cancelled" if scope == "script" else "done"), (
            final.reason
        )
        assert acquisitions == [True, scope != "acquire"]
        assert outcomes == [
            (True, scope in {"run", "iteration", "script"}),
            (scope in {"iteration", "script"}, scope in {"iteration", "script"}),
            (scope == "script", scope == "script"),
        ]
        records = rig.engine.records("run")
        assert [
            [operation.outcome for operation in record.operations] for record in records
        ] == [
            ["partial", "partial" if scope in {"iteration", "script"} else "completed"],
            ["partial" if scope == "script" else "completed"],
        ]
    finally:
        rig.close()


def test_script_cancel_persists_stop_off_author_loop(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    author_threads: list[int] = []
    persistence_threads: dict[str, int] = {}
    original_open = Path.open

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        author_threads.append(get_ident())
        await env.ask(Choose())

    def observe_open(
        path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ):
        if path.parent == rig.paths.metadata_root:
            if path.name == "journal.jsonl" and mode == "a":
                persistence_threads.setdefault("journal", get_ident())
            elif path.name.startswith(".manifest-") and mode == "x":
                persistence_threads.setdefault("manifest", get_ident())
        return original_open(path, mode, buffering, encoding, errors, newline)

    rig.start(script)
    with running(rig) as future:
        pending(rig)
        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", observe_open)
            result = rig.engine.cancel("run", "script")
        # Stop was accepted and recorded before the author can finish.
        assert result.cancelled
        assert "stop_requested" in rig.kinds()
        assert future.result(timeout=30).lifecycle == "cancelled"
    assert persistence_threads == {"journal": get_ident(), "manifest": get_ident()}
    assert author_threads and author_threads[0] != get_ident()


@pytest.mark.parametrize("terminal", ["done", "cancelled", "failed"])
@pytest.mark.parametrize("scope", ["script", "script#1", "acquire"])
def test_terminal_cancel_returns_typed_refusal(
    rig: Rig, terminal: str, scope: str
) -> None:
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        if terminal == "failed":
            raise ValueError("author failed")

    rig.start(script)
    if terminal == "cancelled":
        rig.engine.stop("run")
    final = rig.engine.execute("run")
    assert final.lifecycle == terminal
    before = rig.events()
    try:
        result = rig.engine.cancel("run", scope)
    except RuntimeError as error:
        pytest.fail(f"Terminal cancel must return a typed refusal, not {error!r}")
    assert not result.cancelled
    assert result.reason == "scope not active"
    assert result.active_path == final.active_path
    assert rig.events() == before
    with pytest.raises(ValueError, match="Unknown stop scope label"):
        rig.engine.cancel("run", "unknown")
    with pytest.raises(ValueError, match="expected must be a positive integer"):
        rig.engine.cancel("run", "script", expected=True)


def test_script_cancel_accepts_after_host_drain_before_owner_finalization(
    rig: Rig,
) -> None:
    closing, release = Event(), Event()

    class FinalizationBar(RecordingBar):
        def close(self) -> None:
            closing.set()
            # execute closes the injected display after host drain; finally releases it.
            assert release.wait(30), "Test releases progress cleanup"
            super().close()

    def progress(**kwargs: object) -> RecordingBar:
        return FinalizationBar(**kwargs)

    rig.engine = Engine(
        EnginePorts(rig.plots, progress, rig.clock, context=7, devices=rig.devices)
    )

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        env.pbar("finalization", 1)

    rig.start(script)
    with running(rig) as future:
        try:
            # The submitted author returns unconditionally and execute closes its bar.
            assert closing.wait(30), "Host drains before progress cleanup"
            try:
                result = rig.engine.cancel("run", "script")
            except RuntimeError as error:
                pytest.fail(
                    f"Stop owner must accept before finalization, not {error!r}"
                )
            assert result.cancelled
            assert result.reason == "scope cancellation requested"
            assert "stop_requested" in rig.kinds()
        finally:
            release.set()
        assert future.result(timeout=30).lifecycle == "cancelled"
    assert rig.events()[-1]["lifecycle"] == "cancelled"


@pytest.mark.parametrize("scope,expected", [("script", 1), ("script#1", None)])
def test_stale_script_sequence_does_not_record_stop_intent(
    rig: Rig, scope: str, expected: int | None
) -> None:
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        await env.ask(Choose())

    rig.start(script)
    with running(rig) as future:
        request_id = pending(rig)
        before = rig.events()
        result = rig.engine.cancel("run", scope, expected=expected)
        assert not result.cancelled
        assert result.reason == "sequence mismatch"
        assert rig.events() == before
        assert "stop_requested" not in rig.kinds()
        rig.engine.answer("run", Decision(), request_id=request_id)
        assert future.result(timeout=30).lifecycle == "done"


@pytest.mark.parametrize("destination", ["journal", "manifest"])
def test_script_cancel_io_failure_preserves_failed_and_original_cause(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, destination: str
) -> None:
    original_open = Path.open
    cause = OSError(f"Stop {destination} storage failed")
    failed = False
    cleanup_threads: list[int] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        try:
            await env.ask(Choose())
        finally:
            cleanup_threads.append(get_ident())

    def fail_open(
        path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ):
        nonlocal failed
        target = (
            path.name == "journal.jsonl" and mode == "a"
            if destination == "journal"
            else path.name.startswith(".manifest-") and mode == "x"
        )
        if path.parent == rig.paths.metadata_root and target and not failed:
            failed = True
            raise cause
        return original_open(path, mode, buffering, encoding, errors, newline)

    rig.start(script)
    with running(rig) as future:
        pending(rig)
        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", fail_open)
            with pytest.raises(OSError, match="Stop .* storage failed") as caught:
                rig.engine.cancel("run", "script")
        assert caught.value is cause
        final = future.result(timeout=30)
    assert final.lifecycle == "failed"
    assert final.reason == f"OSError: {cause}"
    assert len(cleanup_threads) == 1 and cleanup_threads[0] != get_ident()
