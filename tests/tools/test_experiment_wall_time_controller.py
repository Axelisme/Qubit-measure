"""Run observations through the CLI with a headless Controller and real Qt loop."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from qtpy import QtCore
from qtpy.QtCore import QCoreApplication, QTimer
from zcu_tools import program
from zcu_tools.gui.app import measure
from zcu_tools.gui.app.measure import controller
from zcu_tools.gui.app.measure.events.run import RunFinishedPayload
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.simulators.ftf import config, environment

from tools import experiment_wall_time
from tools.experiment_wall_time import main


@dataclass
class _Clock:
    seconds: float = 0.0
    deadline: tuple[float, Callable[[], None]] | None = None

    def advance(self, seconds: float) -> None:
        self.seconds += seconds
        if self.deadline is not None and self.seconds >= self.deadline[0]:
            _, fire = self.deadline
            self.deadline = None
            fire()

    def monotonic(self) -> float:
        return self.seconds

    def now(self, tz: tzinfo) -> datetime:
        return datetime(2026, 11, 21, tzinfo=tz) + timedelta(seconds=self.seconds)


class _SyntheticController:
    def __init__(
        self,
        bus: BaseEventBus,
        project_root: str,
        on_open: Callable[[], None],
        on_run: Callable[[_SyntheticController], None],
        on_cancel: Callable[[_SyntheticController], None],
    ) -> None:
        self.bus = bus
        self.telemetry = Path(project_root).with_name("telemetry")
        self.adapter = ""
        self.idle = True
        self.on_run = on_run
        self.on_cancel = on_cancel
        self.run_analyze_control = SimpleNamespace(
            start_run=self.start_run, cancel_run=self.cancel_run
        )
        self.entry_control = SimpleNamespace(
            open_entry=lambda _request: on_open(), use_point=lambda _point: None
        )
        self.tab_control = SimpleNamespace(
            new_tab=lambda adapter: adapter,
            get_tab_snapshot=lambda _tab: SimpleNamespace(run=None),
        )
        self.cfg_resources = SimpleNamespace(
            lookup=lambda _tab: SimpleNamespace(result=object)
        )
        self.session_services = SimpleNamespace(
            soc_connection=SimpleNamespace(install_soc=lambda *_args: None)
        )

    def get_bus(self) -> BaseEventBus:
        return self.bus

    def active_operation_count(self) -> int:
        return int(not self.idle)

    def finish_close(self) -> None:
        pass

    def start_run(self, tab: str) -> int:
        self.adapter = tab
        self.idle = False
        self.on_run(self)
        return 1

    def cancel_run(self) -> None:
        self.on_cancel(self)

    def complete(self, outcome: str = "finished", *, idle: bool = True) -> None:
        self.idle = idle
        self.bus.emit(RunFinishedPayload(tab_id=self.adapter, outcome=outcome))


def _use_virtual_deadline(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    class DeadlineTimer(QTimer):
        def start(self, msec: int | None = None) -> None:
            if self.isSingleShot() and msec is not None:
                clock.deadline = (clock.seconds + msec / 1000, self.timeout.emit)
            elif msec is None:
                super().start()
            else:
                super().start(msec)

        def stop(self) -> None:
            if self.isSingleShot():
                clock.deadline = None
            super().stop()

    # Let virtual time drive the deadline signal; keep Qt's event loop and polling.
    monkeypatch.setattr(QtCore, "QTimer", DeadlineTimer)


@pytest.fixture
def core_app() -> Iterator[QCoreApplication]:
    """Keep the owner loop alive without opening windows or a Measure session."""
    yield QCoreApplication.instance() or QCoreApplication([])


def _headless_inputs(directory: Path) -> None:
    fixture = directory / "fixture"
    fixture.mkdir(parents=True)
    (fixture / "point.txt").write_text("abc")
    (directory / "board-profile.yaml").write_text("{}")
    (directory / "ftf-settings.yaml").write_text("{}")
    (directory / "case-inputs.json").write_text('{"entry":"demo","point":"initial"}')
    (directory / "overrides.json").write_text("[]")


def _install_headless_controller(
    monkeypatch: pytest.MonkeyPatch,
    clock: _Clock,
    adapters: Mapping[str, Path | None],
    on_run: Callable[[_SyntheticController], None],
    on_cancel: Callable[[_SyntheticController], None],
    *,
    on_ftf: Callable[[int], None] | None = None,
) -> None:
    def open_entry() -> None:
        clock.seconds += 40.0

    def make_controller(
        *, bus: BaseEventBus, project_root: str, **_options: object
    ) -> _SyntheticController:
        return _SyntheticController(bus, project_root, open_entry, on_run, on_cancel)

    # main's default composition has no public injection for lab, FTF or Controller.
    # Substitute only those collaborators; keep the tool, bus and Qt loop real.
    assembly = ModuleType("zcu_lab.assembly")
    monkeypatch.setattr(
        assembly,
        "lab_definitions",
        lambda: SimpleNamespace(
            templates={},
            register_components=lambda _: None,
            register_experiments=lambda _: None,
        ),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "zcu_lab.assembly", assembly)
    monkeypatch.setattr(
        measure,
        "Registry",
        lambda **_: SimpleNamespace(
            list_names=lambda: list(adapters),
            create=lambda key: SimpleNamespace(guide_path=adapters[key]),
        ),
    )
    monkeypatch.setattr(program, "build_soccfg", lambda _: object())
    monkeypatch.setattr(
        config, "load_ftf", lambda _: SimpleNamespace(with_telemetry=lambda **_: None)
    )

    def make_environment(soccfg: object) -> SimpleNamespace:
        return SimpleNamespace(
            soc=None,
            soccfg=soccfg,
            devices=None,
            create_device=None,
            close_backend=None,
        )

    def observe_environment(
        _settings: object, *, soccfg: object, workers: int = 1
    ) -> SimpleNamespace:
        if on_ftf is not None:
            on_ftf(workers)
        return make_environment(soccfg)

    # The CLI report does not expose the factory argument. Delegate to the existing
    # headless fake so observing it does not build a real FTF backend.
    monkeypatch.setattr(environment, "ftf", observe_environment)
    monkeypatch.setattr(controller, "Controller", make_controller)
    # The Run's monotonic and UTC observation clocks are not CLI arguments.
    monkeypatch.setattr(experiment_wall_time, "perf_counter", clock.monotonic)
    monkeypatch.setattr(experiment_wall_time, "datetime", clock)


@pytest.mark.parametrize(
    ("worker_args", "expected"), [([], 1), (["--workers", "3"], 3)]
)
@pytest.mark.usefixtures("core_app")
def test_main_configures_ftf_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_args: list[str],
    expected: int,
) -> None:
    data_dir, output = tmp_path / "inputs", tmp_path / "output"
    _headless_inputs(data_dir)
    workers_seen: list[int] = []
    _install_headless_controller(
        monkeypatch,
        _Clock(),
        {"fake/frequency": None},
        lambda ctrl: QTimer.singleShot(0, ctrl.complete),
        lambda _: pytest.fail("a completed Run must not be cancelled"),
        on_ftf=workers_seen.append,
    )

    assert (
        main(
            ["--output", str(output), *worker_args],
            data_dir=data_dir,
            read_commit=lambda: "abc",
        )
        == 0
    )
    assert workers_seen == [expected]


@pytest.mark.parametrize("orphan", [False, True], ids=["paired", "orphan"])
@pytest.mark.usefixtures("core_app")
def test_main_times_only_dispatch_to_terminal_and_sums_its_acquires(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    orphan: bool,
) -> None:
    clock = _Clock()
    _use_virtual_deadline(monkeypatch, clock)
    replace = Path.replace

    def publish_and_wait(path: Path, target: Path) -> Path:
        result = replace(path, target)
        if target.name == "start.json":
            clock.seconds += 12.0
        return result

    # Disk latency precedes dispatch; keep the atomic write and simulate its delay.
    monkeypatch.setattr(Path, "replace", publish_and_wait)

    def run(ctrl: _SyntheticController) -> None:
        records = [
            {
                "acquire_id": "inline",
                "wall_s": 4.5,
                "compile_wall_s": 0.5,
                "evolve_wall_s": 2.0,
                "readout_wall_s": 1.0,
                "materialize_wall_s": 0.25,
            },
            {
                "acquire_id": "deferred",
                "wall_s": 2.0,
                "compile_wall_s": 0.5,
                "evolve_wall_s": 0.75,
                "readout_wall_s": 0.5,
                "materialize_wall_s": 0.0,
            },
        ]
        for index, record in enumerate(records):
            (ctrl.telemetry / f"routing-{index}.jsonl").write_text(
                json.dumps({**record, "families": {}, "complete": True}) + "\n"
            )
        events = [
            {
                "event": "materialize",
                "acquire_id": "deferred",
                "materialize_wall_s": 0.5,
            },
        ]
        if orphan:
            events.append(
                {
                    "event": "materialize",
                    "acquire_id": "pruned",
                    "materialize_wall_s": 0.75,
                }
            )
        (ctrl.telemetry / "routing-2.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events)
        )

        def finish() -> None:
            clock.advance(8.0)
            ctrl.complete()
            # Post-terminal work is outside Run. Also fire the virtual deadline
            # so a missed event fails the test without blocking on a real timer.
            clock.advance(400.0)

        QTimer.singleShot(0, finish)

    data_dir, output = tmp_path / "inputs", tmp_path / "output"
    _headless_inputs(data_dir)
    _install_headless_controller(
        monkeypatch,
        clock,
        {"fake/frequency": None},
        run,
        lambda _: pytest.fail("a completed Run must not be cancelled"),
    )

    assert (
        main(["--output", str(output)], data_dir=data_dir, read_commit=lambda: "abc")
        == 0
    )

    report = json.loads((output / "report.json").read_text())
    [row] = report["rows"]
    assert report["commit"] == "abc"
    assert report["inputs"]["fixture/point.txt"] == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert row["wall_s"] == 8.0
    assert row["started_at"] == "2026-11-21T00:00:52+00:00"
    assert row["finished_at"] == "2026-11-21T00:01:00+00:00"
    assert row["acquires"] == 2
    summary = (output / "summary.md").read_text()
    if orphan:
        assert row["status"] == "missing_breakdown"
        assert row["core_s"] is None
        assert row["core_s_lower_bound"] == 6.0
        assert "| ≥6.000 |" in summary
        return
    assert row["status"] == "ok"
    assert [
        row[key]
        for key in (
            "compile_s",
            "evolve_s",
            "readout_s",
            "materialize_s",
            "core_s",
            "core_s_lower_bound",
            "remainder_s",
        )
    ] == [1.0, 2.75, 1.5, 0.75, 6.0, None, 1.0]
    assert (
        "| fake/frequency | 8.000 | 1.000 | 2.750 | 1.500 | 0.750 | 6.000 | 1.000 | 2 |"
        in summary
    )


@pytest.mark.parametrize(
    ("arrival_s", "idle", "expected_exit"),
    [(1.0, True, 0), (31.0, True, 3), (1.0, False, 3)],
    ids=["ready-before-deadline", "late-terminal", "terminal-but-busy"],
)
@pytest.mark.usefixtures("core_app")
def test_timeout_continues_only_when_terminal_and_idle_are_observed_in_grace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arrival_s: float,
    expected_exit: int,
    *,
    idle: bool,
) -> None:
    clock = _Clock()
    _use_virtual_deadline(monkeypatch, clock)
    dispatched: list[str] = []
    cancellations: list[str] = []

    def run(ctrl: _SyntheticController) -> None:
        dispatched.append(ctrl.adapter)
        if ctrl.adapter == "fake/frequency":
            if idle:
                (ctrl.telemetry / "routing-fake.jsonl").write_text(
                    '{"families":{},"wall_s":5,"compile_wall_s":1,'
                    '"evolve_wall_s":2,"readout_wall_s":1,"materialize_wall_s":0}\n'
                )
            QTimer.singleShot(0, lambda: clock.advance(180.0))
        else:
            QTimer.singleShot(0, ctrl.complete)

    def cancel(ctrl: _SyntheticController) -> None:
        cancellations.append(ctrl.adapter)
        stack = ctrl.telemetry.parent / "timeout-stack.txt"
        assert "(most recent call first)" in stack.read_text()

        def settle() -> None:
            clock.advance(arrival_s)
            ctrl.complete("cancelled", idle=idle)
            # The first case was already terminal and idle before this work.
            clock.advance(31.0)

        QTimer.singleShot(0, settle)

    data_dir, output = tmp_path / "inputs", tmp_path / "output"
    _headless_inputs(data_dir)
    _install_headless_controller(
        monkeypatch, clock, {"fake/frequency": None, "fake/power": None}, run, cancel
    )

    assert (
        main(
            ["--output", str(output), "--row-timeout", "180"],
            data_dir=data_dir,
            read_commit=lambda: "abc",
        )
        == expected_exit
    )
    assert cancellations == ["fake/frequency"]
    assert dispatched == (
        ["fake/frequency", "fake/power"] if expected_exit == 0 else ["fake/frequency"]
    )
    timeout, following = json.loads((output / "report.json").read_text())["rows"]
    assert timeout["status"] == "timeout"
    assert timeout["wall_s"] is None and timeout["wall_s_lower_bound"] == 180.0
    assert timeout["breakdown"] == ("partial" if idle else "missing")
    assert timeout["evolve_s"] == (2.0 if idle else None)
    assert following["status"] == (
        "missing_breakdown" if expected_exit == 0 else "not_run"
    )
    summary = (output / "summary.md").read_text()
    assert "Core >180 s: —\n" in summary
    assert ">180.0" in summary
    assert f"Runs: {int(expected_exit == 0)} timed, 1 timeout, 0 run_failed." in summary


@pytest.mark.parametrize(
    ("evolve_s", "lower_bound", "over_target"),
    [(173.5001, 180.0001, True), (173.5, 180.0, False), (None, None, False)],
    ids=["above-target", "at-target", "no-complete-acquire"],
)
@pytest.mark.usefixtures("core_app")
def test_timeout_judges_only_the_measured_core_lower_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evolve_s: float | None,
    lower_bound: float | None,
    *,
    over_target: bool,
) -> None:
    clock = _Clock()
    _use_virtual_deadline(monkeypatch, clock)

    def run(ctrl: _SyntheticController) -> None:
        legacy = {
            "acquire_id": "legacy",
            "families": {},
            "wall_s": 10,
            "compile_wall_s": 1,
            "evolve_wall_s": 8,
            "readout_wall_s": 1,
        }
        records = [
            legacy,
            {"event": "materialize", "acquire_id": "legacy", "materialize_wall_s": 0.5},
        ]
        if evolve_s is not None:
            records += [
                {
                    **legacy,
                    "acquire_id": "raw",
                    "wall_s": 185,
                    "evolve_wall_s": evolve_s,
                    "readout_wall_s": 4,
                    "materialize_wall_s": 1.25,
                },
                {
                    "event": "materialize",
                    "acquire_id": "raw",
                    "materialize_wall_s": 0.25,
                },
            ]
        (ctrl.telemetry / "routing-test.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        QTimer.singleShot(0, lambda: clock.advance(200.0))

    data_dir, output = tmp_path / "inputs", tmp_path / "output"
    _headless_inputs(data_dir)
    _install_headless_controller(
        monkeypatch,
        clock,
        {"fake/frequency": None},
        run,
        lambda ctrl: QTimer.singleShot(0, lambda: ctrl.complete("cancelled")),
    )
    assert (
        main(
            ["--output", str(output), "--row-timeout", "200"],
            data_dir=data_dir,
            read_commit=lambda: "abc",
        )
        == 0
    )

    [row] = json.loads((output / "report.json").read_text())["rows"]
    assert row["status"] == "timeout"
    assert row["wall_s_lower_bound"] == 200.0
    assert row["core_s"] is None
    assert row["core_s_lower_bound"] == lower_bound
    assert row["materialize_s"] == (1.5 if evolve_s is not None else None)
    assert row["acquires"] == (2 if evolve_s is not None else 1)
    summary = (output / "summary.md").read_text()
    assert "Core >180 s: —" in summary
    assert f"Core 已知至少超標: {'`fake/frequency`' if over_target else '—'}" in summary
    assert f"Core 未判定: {'—' if over_target else '`fake/frequency`'}" in summary
    if lower_bound is not None:
        assert "| ≥180.000 |" in summary
