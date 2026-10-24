"""Framework operation-attempt evidence grouped by explicit authored commits."""

import json
from datetime import timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.device import FakeDevice, FakeDeviceInfo
from zcu_tools.experiment import workflows as w
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.metadata import SoftwareProvenance
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.experiment.workflows import saved_run
from zcu_tools.plotting.plots import NonPresentingHost

from ._engine_ports import START, ManualClock, RecordingBar, RecordingDevices


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Tunables(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Cfg(ConfigBase):
    value: int = 3


class Record(BaseModel):
    value: int


def test_attempt_facts_are_grouped_by_commit_with_fixed_pre_run_device_state(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    devices = RecordingDevices()
    devices.drivers["flux"] = FakeDevice(address="memory", fast_mode=True)
    plots = w.WorkflowPlots(NonPresentingHost())
    board = white_noise()
    engine = w.Engine(
        w.EnginePorts(
            plots,
            RecordingBar,
            clock,
            devices=devices,
            soc=board.soc,
            soccfg=board.soccfg,
        )
    )
    paths = w.RunPaths(tmp_path / "metadata", tmp_path / "data")
    contexts: list[RunContext] = []
    cfg = Cfg()

    def acquire(cfg: Cfg, *, context: RunContext) -> int:
        contexts.append(context)
        assert context.plots is not plots
        assert context.soc is board.soc
        assert context.soccfg is board.soccfg
        assert context.devices["flux"] is devices.drivers["flux"]
        clock.value += timedelta(seconds=2)
        cfg.value = 8
        return 9

    @w.workflow("facts", "Group actual operation attempts by explicit commits")
    async def script(
        env: w.WorkflowEnv[None], plan: Plan, tun: w.Live[Tunables]
    ) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            await iteration.call(lambda ctx: 1)
            first = await iteration.run(acquire, cfg)
            assert isinstance(first, w.Completed)
            assert first.value.cfg is not None
            assert first.value.cfg.value == 8
            assert first.value.result == 9
            await iteration.wait_until(START + timedelta(seconds=4))
            iteration.commit(Record(value=1))
            iteration = next(iterations)
            await iteration.setup_device(
                "flux", FakeDeviceInfo(address="memory", value=7)
            )
            await iteration.run(acquire, cfg)
            iteration.commit(Record(value=2))
            iteration = next(iterations)
            await iteration.run(acquire, cfg)
        finally:
            iterations.close()

    try:
        engine.start(
            script,
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
        status = engine.execute("run")
        assert status.lifecycle == "done", status.reason
        assert cfg.value == 3
        assert_persisted_attempts(paths)
        assert len(contexts) == 3
    finally:
        plots.finish()
        plots.release()


def assert_persisted_attempts(paths: w.RunPaths) -> None:
    records = saved_run.read_records(paths.metadata_root, Record)
    assert [record.record.value for record in records] == [1, 2]
    envelopes = [json.loads(row.record_path.read_text()) for row in records]
    assert [
        [(op["kind"], op["attempt"], op["outcome"]) for op in row["operations"]]
        for row in envelopes
    ] == [
        [
            ("call", 1, "completed"),
            ("run", 2, "completed"),
            ("wait_until", 3, "completed"),
        ],
        [("setup_device", 1, "completed"), ("run", 2, "completed")],
    ]
    first_run = envelopes[0]["operations"][1]
    second_run = envelopes[1]["operations"][1]
    assert first_run["device_snapshot"]["flux"]["value"] == 0
    assert second_run["device_snapshot"]["flux"]["value"] == 7
    assert first_run["started_at"] == START.isoformat().replace("+00:00", "Z")
    assert first_run["finished_at"] == (
        START + timedelta(seconds=2)
    ).isoformat().replace("+00:00", "Z")
    assert first_run["reason"] is None
    assert envelopes[0]["operations"][0]["device_snapshot"] is None
    journal = [
        json.loads(line)
        for line in (paths.metadata_root / "journal.jsonl").read_text().splitlines()
    ]
    attempts = [
        event["operation"] for event in journal if event["kind"] == "operation_finished"
    ]
    assert attempts[:3] == envelopes[0]["operations"]
    assert attempts[3:5] == envelopes[1]["operations"]
    assert (attempts[5]["kind"], attempts[5]["attempt"]) == ("run", 1)
