"""Progress cleanup settles before terminal publication and releases admission."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from zcu_tools.experiment.workflows import (
    Actor,
    Engine,
    EnginePorts,
    Live,
    RunPaths,
    WorkflowEnv,
    workflow,
)
from zcu_tools.experiment.workflows.saved_run import read_summary

from ._engine_fakes import Plan, Point, Rig, Tunables, identity
from ._engine_ports import RecordingBar


class ClosingBar(RecordingBar):
    """Record every cleanup attempt, with optional presentation failure."""

    def __init__(self, failures: bool, attempts: list[str], **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.failures = failures
        self.attempts = attempts

    def close(self) -> None:
        self.attempts.append(self.desc)
        super().close()
        if self.failures:
            raise OSError(f"close {self.desc} failed")


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    rig = Rig(tmp_path)
    yield rig
    rig.close()


@pytest.mark.parametrize("cleanup_fails", [False, True], ids=["clean", "broken"])
@pytest.mark.parametrize("primary", ["none", "author", "control"])
def test_progress_cleanup_settles_all_bars_before_terminal_and_releases_worker(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool, primary: str
) -> None:
    attempts: list[str] = []

    def progress(**kwargs: object) -> ClosingBar:
        return ClosingBar(cleanup_fails, attempts, **kwargs)

    rig.engine = Engine(EnginePorts(rig.plots, progress, rig.clock, context=7))
    journal = rig.paths.metadata_root / "journal.jsonl"
    original_open = Path.open

    @workflow("cleanup", "Synthetic cleanup and primary failure")
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> Point:
        env.pbar("first", 1)
        env.pbar("second", 1)
        if primary == "author":
            raise ValueError("author failed first")
        if primary == "control":

            def fail_append(
                path: Path,
                mode: str = "r",
                buffering: int = -1,
                encoding: str | None = None,
                errors: str | None = None,
                newline: str | None = None,
            ):
                if path == journal and mode == "a":
                    raise OSError("control journal failed first")
                return original_open(path, mode, buffering, encoding, errors, newline)

            with monkeypatch.context() as patch:
                patch.setattr(Path, "open", fail_append)
                rig.engine.update_tunables(
                    "run",
                    Tunables(value=2),
                    expected_revision=0,
                    actor=Actor("user", "test"),
                )
        return Point(value=1)

    rig.engine.start(
        script, plan=Plan(), tunables=Tunables(), paths=rig.paths, identity=identity()
    )
    try:
        final = rig.engine.execute("run")
    except OSError as error:
        pytest.fail(
            f"Progress cleanup must settle as failed, not escape execute: {error}"
        )
    assert attempts == ["first", "second"]
    assert final.lifecycle == (
        "done" if primary == "none" and not cleanup_fails else "failed"
    )
    expected_reason = {
        "none": "OSError: close first failed" if cleanup_fails else None,
        "author": "ValueError: author failed first",
        "control": "OSError: control journal failed first",
    }[primary]
    assert final.reason == expected_reason
    assert (
        read_summary(rig.paths.metadata_root) is not None
        if final.lifecycle == "done"
        else read_summary(rig.paths.metadata_root) is None
    )
    if primary != "control":
        endings = [event for event in rig.events() if event["kind"] == "run_ended"]
        assert len(endings) == 1
        assert endings[0]["lifecycle"] == final.lifecycle
    # A new run proves admission was released even if multiple close calls failed.
    rig.engine.start(
        script,
        plan=Plan(),
        tunables=Tunables(),
        paths=RunPaths(rig.root / "next-metadata", rig.root / "next-data"),
        identity=identity("next"),
    )
