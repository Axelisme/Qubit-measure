"""Invocation directories and acquisition-only effects through public Engine."""

from pathlib import Path

import pytest
from zcu_tools.experiment.workflows import Completed, Done, Next, Run, Step, WorkflowEnv

from ._engine_fakes import Cfg, Plan, Rig, State, Tunables


def test_iteration_root_exists_is_read_only_and_resume_allocates_new_call(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    directories: list[Path] = []

    def acquire(run: Run[Cfg]) -> int:
        return run.cfg.value

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        directory = env.iteration_dir
        assert directory.is_absolute() and directory.is_dir()
        assert directory == rig.paths.data_root / f"iter/{len(directories) + 1:06d}"
        with pytest.raises(AttributeError):
            setattr(env, "iteration_dir", tmp_path)
        directories.append(directory)
        if len(directories) == 1:
            rig.engine.pause("run")
        outcome = yield from env.run(acquire, Cfg())
        assert isinstance(outcome, Completed)
        assert env.iteration_dir == directory
        (directory / "author.txt").write_text("author-owned", encoding="utf-8")
        return Next(outcome.result, state) if len(directories) == 2 else Done()

    rig.start(step)
    paused = rig.engine.execute("run")
    assert paused.lifecycle == "paused", paused.reason
    rig.engine.resume("run")
    done = rig.engine.execute("run")
    assert done.lifecycle == "done", done.reason
    assert len(directories) == 3 and len(set(directories)) == 3
    record = rig.engine.records("run")[0]
    assert record.call_seq == 2
    assert record.iteration_dir == "iter/000002"
    assert (directories[1] / "author.txt").read_text(encoding="utf-8") == "author-owned"


def test_acquisition_without_a_writer_returns_completed(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    outcomes: list[Completed[Cfg, int]] = []

    def acquire(run: Run[Cfg]) -> int:
        run.cfg.value = 7
        return 42

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        result = yield from env.run(acquire, Cfg())
        assert isinstance(result, Completed)
        outcomes.append(result)
        return Done()

    rig.start(step)
    status = rig.engine.execute("run")
    assert status.lifecycle == "done", status.reason
    assert outcomes == [Completed(Cfg(value=7), 42)]
    assert rig.devices.captures == 1
