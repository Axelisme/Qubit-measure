"""Async authoring through Engine execution and saved-run public readers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment import workflows as w
from zcu_tools.experiment.workflows import saved_run
from zcu_tools.plotting.plots import NonPresentingHost

from ._engine_ports import ManualClock, RecordingBar, RecordingDevices


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    count: int = 2


class Tunables(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int = 3


class FirstRecord(BaseModel):
    first: int
    latest: int


class SecondRecord(BaseModel):
    value: int


class Summary(BaseModel):
    points: int


def test_async_commits_heterogeneous_records_and_tracks_every_live_revision(
    tmp_path: Path,
) -> None:
    plots = w.WorkflowPlots(NonPresentingHost())
    engine = w.Engine(
        w.EnginePorts(
            plots, RecordingBar, ManualClock(), context=7, devices=RecordingDevices()
        )
    )
    paths = w.RunPaths(tmp_path / "metadata", tmp_path / "data")

    async def script(
        env: w.WorkflowEnv[int], plan: Plan, tun: w.Live[Tunables]
    ) -> Summary:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            initial = tun.current
            engine.update_tunables(
                "run",
                Tunables(value=8),
                expected_revision=0,
                actor=w.Actor("user", "test"),
            )
            latest = tun.current
            initial_value = initial.value
            initial.value = -1
            assert tun.current.value == 8
            iteration.commit(FirstRecord(first=initial_value, latest=latest.value))
            iteration = next(iterations)
            iteration.commit(SecondRecord(value=tun.current.value))
            iteration = next(iterations)
            return Summary(points=plan.count)
        finally:
            iterations.close()

    try:
        declared = w.workflow("example", "Commit two distinct record models")(script)
    except TypeError as error:
        pytest.fail(f"The async workflow declaration must be accepted: {error}")

    try:
        engine.start(
            declared,
            plan=Plan(),
            tunables=Tunables(),
            paths=paths,
            identity=w.RunIdentity(
                "run",
                SoftwareProvenance(
                    software_versions={},
                    git_commit=None,
                    git_dirty=None,
                    qick_version=None,
                    soc_fingerprint=None,
                    hostname="offline",
                ),
            ),
        )
        final = engine.execute("run")
        assert final.lifecycle == "done", final.reason
        assert final.committed_seq == 2
        records = engine.records("run")
        assert [record.encoded_record.value for record in records] == [
            {"first": 3, "latest": 8},
            {"value": 8},
        ]
        try:
            saved = saved_run.read_records_raw(paths.metadata_root)
        except AttributeError as error:
            pytest.fail(f"Saved-run reading must expose committed records: {error}")
        assert [point.record.value for point in saved] == [
            {"first": 3, "latest": 8},
            {"value": 8},
        ]
        first = json.loads(
            (paths.data_root / "iteration-000001/record.json").read_text()
        )
        second = json.loads(
            (paths.data_root / "iteration-000002/record.json").read_text()
        )
        assert first["started"]["revisions"] == [0, 1]
        assert second["started"]["revisions"] == [1]
        summary = saved_run.read_summary(paths.metadata_root)
        assert summary is not None
        assert summary.value == {"points": 2}
        events = [
            json.loads(line)
            for line in (paths.metadata_root / "journal.jsonl").read_text().splitlines()
        ]
        assert events[-1]["kind"] == "run_ended"
        assert events[-1]["summary"] is True
    finally:
        plots.finish()
        plots.release()
