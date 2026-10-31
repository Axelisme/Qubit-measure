"""Environment capabilities and retained displays through an async author."""

from collections.abc import Iterator, MutableMapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from qick import QickConfig
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.run_core import run_core
from zcu_tools.experiment.workflows import (
    Engine,
    EnginePorts,
    Live,
    MissingCapability,
    RunIdentity,
    WorkflowEnv,
    saved_run,
    workflow,
)

from tests._plotting import create_trace

from ._engine_fakes import START, Cfg, Plan, Point, Rig, Tunables, completed, identity


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    rig = Rig(tmp_path)
    yield rig
    rig.close()


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
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            observed.extend(
                [env.started_at, env.context, env.soc, env.devices, env.plots]
            )
            iteration.commit(Point(value=1))
            iteration = next(iterations)
            assert env.started_at == START
            assert env.devices is rig.devices
            assert env.plots is rig.plots
        finally:
            iterations.close()

    rig.start(script)
    assert rig.engine.execute("run").lifecycle == "done"
    assert observed == [START, 7, soc, rig.devices, rig.plots]
    assert rig.devices.values == []


def test_author_identity_matches_supplied_run_and_saved_manifest(rig: Rig) -> None:
    supplied = replace(identity(), entry_name="entry", entry_label="Sample point")
    observed: list[RunIdentity] = []

    @workflow("identity", "Expose the existing workflow run identity to savers")
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            assert hasattr(env, "identity"), (
                "Authors must receive the existing RunIdentity"
            )
            observed.append(env.identity)
            author_copy = env.identity
            versions = author_copy.provenance.software_versions
            if isinstance(versions, MutableMapping):
                versions["author-only"] = "changed"
            assert env.identity == supplied
            await iteration.call(lambda ctx: None)
            iteration.commit(Point(value=1))
            iteration = next(iterations)
            observed.append(env.identity)
        finally:
            iterations.close()

    rig.engine.start(
        script,
        plan=Plan(),
        tunables=Tunables(),
        paths=rig.paths,
        identity=supplied,
    )
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    assert observed == [supplied, supplied]
    assert saved_run.read_manifest(rig.paths.metadata_root).identity == supplied


def test_run_accepts_typed_bound_callable_with_config_parameter(rig: Rig) -> None:
    class Experiment:
        def run(self, config: Cfg, *, context: RunContext) -> int:
            assert context.plots is not rig.plots
            assert context.devices == {}
            return config.value + 4

    rig.engine = Engine(
        EnginePorts(
            rig.plots,
            rig.progress,
            rig.clock,
            soc=object(),
            soccfg=QickConfig({}),
            devices=rig.devices,
        )
    )
    experiment = Experiment()
    results: list[int] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            outcome = completed(await run_core(iteration, experiment.run, Cfg(value=3)))
            assert outcome.value.cfg == Cfg(value=3)
            results.append(outcome.value.result)
        finally:
            iterations.close()

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    assert results == [7]


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
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            with pytest.raises(ValueError, match="timezone-aware"):
                await iteration.wait_until(datetime(2026, 10, 6))
        finally:
            iterations.close()

    rig.start(script)
    assert rig.engine.execute("run").lifecycle == "done"
    assert rig.clock.targets == []


def test_plots_and_named_bar_are_retained_across_commits(rig: Rig) -> None:
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            bar = env.pbar("points", 10)
            bar.set_progress(3)
            viewer = create_trace(env.plots, "main", "x", "y")
            viewer.update((np.array([0.0, 1.0]), np.array([2.0, 3.0])))
            iteration.commit(Point(value=3))
            iteration = next(iterations)
            assert env.pbar("points", 10) is bar
            assert bar.n == 3
            assert not rig.bars[0].closed
            np.testing.assert_array_equal(
                env.plots["main"].axes[0].lines[0].get_ydata(), [2.0, 3.0]
            )
        finally:
            iterations.close()

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
