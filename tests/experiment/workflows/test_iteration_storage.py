"""Author files survive Pause and acquisition does not implicitly save."""

from pathlib import Path
from threading import Event

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.script_host import CallContext
from zcu_tools.experiment.workflows import Completed, Live, WorkflowEnv

from ._engine_fakes import Cfg, Plan, Point, Rig, Tunables, completed, paused, running


def test_allocated_directory_and_author_files_survive_pause(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    entered = Event()
    directories: list[Path] = []
    attempts: list[int] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        directory = env.new_commit_dir()
        assert directory.is_absolute() and directory.is_dir()
        directories.append(directory)
        (directory / "author.txt").write_text("author-owned", encoding="utf-8")

        def operation(ctx: CallContext) -> int:
            attempts.append(1)
            if len(attempts) == 1:
                entered.set()
                assert ctx.scope.stop.event.wait(10), "Pause stops the operation"
            return 42

        result = completed(await env.call(operation))
        assert (directory / "author.txt").read_text() == "author-owned"
        await env.commit(Point(value=result.value), directory)

    try:
        rig.start(script)
        with running(rig) as future:
            assert entered.wait(10), "Worker enters operation"
            rig.engine.pause("run")
            paused(rig)
            assert (directories[0] / "author.txt").read_text() == "author-owned"
            assert not (directories[0] / "record.json").exists()
            rig.engine.resume("run")
            final = future.result(timeout=10)
        assert final.lifecycle == "done", final.reason
        assert len(directories) == 1
        assert attempts == [1, 1]
        assert rig.engine.records("run")[0].directory == "commits/000001"
        assert (directories[0] / "record.json").is_file()
    finally:
        rig.close()


def test_acquisition_without_writer_returns_run_record_without_committing(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path, native=True)
    outcomes: list[Completed[RunRecord[Cfg, int]]] = []

    def acquire(cfg: Cfg, *, context: RunContext) -> int:
        cfg.value = 7
        return 42

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        outcomes.append(completed(await env.run(acquire, Cfg())))

    try:
        rig.start(script)
        final = rig.engine.execute("run")
        assert final.lifecycle == "done", final.reason
        assert len(outcomes) == 1
        assert outcomes[0].value.cfg == Cfg(value=7)
        assert outcomes[0].value.result == 42
        assert not outcomes[0].partial_result
        assert rig.engine.records("run") == ()
        assert list(rig.paths.data_root.rglob("*.h5")) == []
    finally:
        rig.close()
