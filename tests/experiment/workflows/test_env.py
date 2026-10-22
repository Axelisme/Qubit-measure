"""Environment capabilities and retained displays through an async author."""

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from qick import QickConfig
from zcu_tools.experiment.script_host import CallContext, CallResult, ScriptHost
from zcu_tools.experiment.workflows import (
    Engine,
    EnginePorts,
    Live,
    MissingCapability,
    OperationFacts,
    WorkflowEnv,
)
from zcu_tools.experiment.workflows.env import Displays, EnvHooks

from ._engine_fakes import START, Plan, Point, Rig, Tunables, completed


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    rig = Rig(tmp_path)
    yield rig
    rig.close()


class PauseOnCompletionHost(ScriptHost):
    """Control the real host's return boundary before env classifies completion.

    The adapter only pauses after the host has sampled the first complete result.
    It preserves that result, so the env sees a stale epoch with no stop bit.
    """

    async def call[T](self, fn: Callable[[CallContext], T]) -> CallResult[T]:
        """Delegate execution, then Pause once before returning the same result."""
        result = await super().call(fn)
        if self.epoch == 0:
            self.pause()
        return result


def test_natural_completion_waits_for_resume_without_retry_or_stopping_next_call(
    rig: Rig,
) -> None:
    host = PauseOnCompletionHost()
    at_boundary = Event()
    facts: list[OperationFacts] = []
    calls: list[CallContext] = []
    deliveries: list[int] = []

    hooks = EnvHooks(
        new_commit_dir=lambda: pytest.fail("This script does not allocate directories"),
        commit=lambda record, directory: pytest.fail("This script does not commit"),
        boundary=at_boundary.set,
        stopping=lambda: False,
        experiment_failed=lambda name, reason, error: pytest.fail(reason),
        question=lambda phase: pytest.fail("This script does not ask"),
        next_attempt=lambda: (1, len(facts) + 1),
        operation_finished=facts.append,
        device_setup=lambda name, cfg: pytest.fail(
            "This script does not set up devices"
        ),
    )
    env = WorkflowEnv(
        START,
        EnginePorts(rig.plots, rig.progress, rig.clock, context=7),
        host,
        Displays(rig.progress),
        hooks,
    )

    def acquire(ctx: CallContext) -> int:
        calls.append(ctx)
        assert not ctx.stop_signal.is_set()
        return len(calls)

    async def script() -> None:
        first = completed(await env.call(acquire))
        assert not first.partial_result
        assert first.facts is facts[0]
        assert first.facts.stopped is first.partial_result
        deliveries.append(first.value)
        second = completed(await env.call(acquire))
        assert not second.partial_result
        assert second.facts is facts[1]
        assert second.facts.stopped is second.partial_result
        deliveries.append(second.value)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(host.run, script())
        future.add_done_callback(lambda done: at_boundary.set())
        try:
            # The paused completion reaches the boundary, or worker failure wakes us.
            assert at_boundary.wait(30), (
                "The completed operation must reach its boundary"
            )
            assert not future.done(), future.result()
            assert deliveries == []
            assert host.epoch == 1
            host.resume()
            future.result(timeout=30)
        finally:
            host.cancel()
            future.result(timeout=30)
            host.close()

    assert deliveries == [1, 2]
    assert [ctx.epoch for ctx in calls] == [0, 1]
    assert [fact.outcome for fact in facts] == ["completed", "completed"]
    assert [(fact.iteration, fact.attempt) for fact in facts] == [(1, 1), (1, 2)]
    assert all(not ctx.stop_signal.is_set() for ctx in calls)


def test_original_start_context_and_injected_handles_are_shared(rig: Rig) -> None:
    soc = object()
    ports = EnginePorts(
        rig.plots,
        rig.progress,
        rig.clock,
        context=7,
        soc=soc,
        soccfg=QickConfig({}),
        devices=rig.devices,
    )
    rig.engine = Engine(ports)
    observed: list[object] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        observed.extend([env.started_at, env.context, env.soc, env.devices, env.plots])
        await env.commit(Point(value=1))
        assert env.started_at == START
        assert env.devices is rig.devices
        assert env.plots is rig.plots

    rig.start(script)
    assert rig.engine.execute("run").lifecycle == "done"
    assert observed == [START, 7, soc, rig.devices, rig.plots]
    assert rig.devices.values == []


@pytest.mark.parametrize("capability", ["context", "soc", "devices"])
def test_missing_capability_raises_at_use(rig: Rig, capability: str) -> None:
    ports = EnginePorts(
        rig.plots, rig.progress, rig.clock, context=7, devices=rig.devices
    )
    if capability == "context":
        ports = replace(ports, context=None)
    elif capability == "devices":
        ports = replace(ports, devices=None)
    rig.engine = Engine(ports)
    seen: list[str] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        with pytest.raises(MissingCapability) as caught:
            getattr(env, capability)
        seen.append(caught.value.capability)

    rig.start(script)
    assert rig.engine.execute("run").lifecycle == "done"
    assert seen == [capability]


def test_naive_wait_target_is_rejected_without_using_clock(rig: Rig) -> None:
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            await env.wait_until(datetime(2026, 10, 6))

    rig.start(script)
    assert rig.engine.execute("run").lifecycle == "done"
    assert rig.clock.targets == []


def test_plots_and_named_bar_are_retained_across_commits(rig: Rig) -> None:
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        bar = env.pbar("points", 10)
        bar.set_progress(3)
        viewer = env.plots.liveplot_1d("main", "x", "y")
        viewer.update(np.array([0.0, 1.0]), np.array([2.0, 3.0]))
        await env.commit(Point(value=3))
        assert env.pbar("points", 10) is bar
        assert bar.n == 3
        assert not rig.bars[0].closed
        np.testing.assert_array_equal(
            env.plots["main"].axes[0].lines[0].get_ydata(), [2.0, 3.0]
        )

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    assert len(rig.bars) == 1 and rig.bars[0].closed
    assert final.progress[0].completed == 3


def test_same_bar_name_cannot_change_total(rig: Rig) -> None:
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        bar = env.pbar("points", 10)
        with pytest.raises(ValueError, match="total changed"):
            env.pbar("points", 11)
        assert bar.total == 10

    rig.start(script)
    assert rig.engine.execute("run").lifecycle == "done"


@pytest.mark.parametrize(("name", "total"), [("", 1), ("points", -1), ("points", True)])
def test_invalid_bar_declaration_is_rejected(rig: Rig, name: str, total: int) -> None:
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        with pytest.raises(ValueError, match="nonnegative integer total"):
            env.pbar(name, total)

    rig.start(script)
    assert rig.engine.execute("run").lifecycle == "done"
