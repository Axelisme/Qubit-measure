"""Acquisition returns records without implicitly saving author files."""

from pathlib import Path
from threading import Event

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.script_host import CallContext
from zcu_tools.experiment.workflows import Completed, Live, WorkflowEnv

from ._engine_fakes import Cfg, Plan, Point, Rig, Tunables, completed, running


def test_acquisition_without_writer_returns_run_record_without_committing(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path, native=True)
    outcomes: list[Completed[RunRecord[Cfg, int]]] = []

    def acquire(cfg: Cfg, *, context: RunContext) -> int:
        cfg.value = 7
        return 42

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            outcomes.append(completed(await iteration.run(acquire, Cfg())))
        finally:
            iterations.close()

    try:
        rig.start(script)
        final = rig.engine.execute("run")
        assert final.lifecycle == "done", final.reason
        assert len(outcomes) == 1
        assert outcomes[0].value.cfg == Cfg(value=7)
        assert outcomes[0].value.result == 42
        assert not outcomes[0].stop_requested
        assert rig.engine.records("run") == ()
        assert list(rig.paths.data_root.rglob("*.h5")) == []
    finally:
        rig.close()
