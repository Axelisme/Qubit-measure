"""Run observations through the CLI with a headless Controller and real Qt loop."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qtpy import QtCore
from qtpy.QtCore import QCoreApplication, QTimer
from qtpy.QtWidgets import QApplication
from zcu_tools import program
from zcu_tools.gui.app.measure import Registry, controller
from zcu_tools.gui.app.measure.events.run import RunFinishedPayload
from zcu_tools.gui.app.measure.services.ports import RunPaneSnapshot, TabSnapshot
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_types import EntryRequest
from zcu_tools.simulators.ftf import config, environment
from zcu_tools.simulators.ftf.config.schema import FTFSettings, Numerics

from tests.gui.app.measure._synthetic import SampleAdapter, SampleResult, SampleRun
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
        on_point: Callable[[Path], None] | None,
        on_run: Callable[[_SyntheticController], None],
        on_cancel: Callable[[_SyntheticController], None],
        on_write: Callable[[object], None] | None,
    ) -> None:
        self.bus = bus
        self.telemetry = Path(project_root).with_name("telemetry")
        self.adapter = ""
        self.idle = True
        self.on_run = on_run
        self.on_cancel = on_cancel
        self.project = Path(project_root)
        self.entry: Path | None = None
        self.published_run: RunPaneSnapshot | None = None

        def open_entry(request: EntryRequest) -> None:
            self.entry = self.project / "result" / request.name
            if not self.entry.is_dir():
                raise FileNotFoundError(self.entry)
            on_open()

        def use_point(label: str) -> None:
            assert self.entry is not None
            point = self.entry / "points" / label
            if not point.is_dir():
                raise FileNotFoundError(point)
            if on_point is not None:
                on_point(point)

        self.run_analyze_control = SimpleNamespace(
            start_run=self.start_run, cancel_run=self.cancel_run
        )
        self.entry_control = SimpleNamespace(open_entry=open_entry, use_point=use_point)
        self.tab_control = SimpleNamespace(
            new_tab=lambda adapter: adapter,
            get_tab_snapshot=lambda tab: TabSnapshot(
                adapter_name=tab, cfg_input={}, tab_id=tab, run=self.published_run
            ),
        )
        self.cfg_resources = SimpleNamespace(
            lookup=lambda _tab: SimpleNamespace(write=on_write, result=object)
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


@pytest.fixture(scope="session")
def core_app() -> Iterator[QApplication]:
    """Keep one headless Qt owner loop alive without opening windows.

    Creates a QApplication rather than a bare QCoreApplication because Qt allows
    only one application instance per process: leaving a QCoreApplication alive
    in a shared xdist worker aborts any later GUI test that creates a QWidget.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("QT_QPA_PLATFORM", "offscreen")
        patch.setenv("QT_QPA_PLATFORMTHEME", "generic")
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
    if not isinstance(app, QApplication):
        raise RuntimeError(f"Qt tests require a QApplication, got {type(app).__name__}")
    yield app


def test_core_app_provides_a_widget_compatible_qapplication(
    core_app: QApplication,
) -> None:
    """Running wall-time controller tests must leave a QApplication for GUI tests."""
    assert isinstance(core_app, QApplication)
    assert QCoreApplication.instance() is core_app


def _headless_inputs(
    directory: Path, *adapters: str, metric: dict[str, object] | None = None
) -> None:
    if metric is None:
        metric = {"kind": "shots"}
    directory.mkdir(parents=True)
    (directory / "board-profile.yaml").write_text("{}")
    (directory / "ftf-settings.yaml").write_text("{}")
    (directory / "case-inputs.json").write_text(
        json.dumps(
            {
                "entry": "demo",
                "point": "initial",
                "unpinned": {"form": "current Form", "point": "fixture point"},
                "scenes": [
                    {"scene_id": key, "adapter": key, "metric": metric}
                    for key in adapters or ("fake/frequency",)
                ],
            }
        )
    )
    point = directory / "fixture/result/demo/points/initial"
    point.mkdir(parents=True)
    (point / "parameters.yaml").write_text("parameter-token")
    (point / "library.yaml").write_text("library-token")
    (point / "aliases.yaml").write_text("aliases-token")


def _install_headless_controller(
    monkeypatch: pytest.MonkeyPatch,
    clock: _Clock,
    adapters: tuple[str, ...],
    on_run: Callable[[_SyntheticController], None],
    on_cancel: Callable[[_SyntheticController], None],
    *,
    on_ftf: Callable[[int, FTFSettings, float | None], None] | None = None,
    on_point: Callable[[Path], None] | None = None,
    on_write: Callable[[object], None] | None = None,
) -> Callable[[], Registry]:
    def open_entry() -> None:
        clock.seconds += 40.0

    def make_controller(
        *, bus: BaseEventBus, project_root: str, **_options: object
    ) -> _SyntheticController:
        return _SyntheticController(
            bus, project_root, open_entry, on_point, on_run, on_cancel, on_write
        )

    def register() -> Registry:
        registry = Registry()
        for adapter in adapters:
            registry.register(adapter, SampleAdapter)
        return registry

    # Keep the registry, CLI, bus and Qt loop real. Replace FTF, the Controller
    # and the clocks so tests can exercise dispatch and cancellation without evolution.
    monkeypatch.setattr(program, "build_soccfg", lambda _: object())
    preset = FTFSettings.model_construct(
        numerics=Numerics.model_construct(
            time_scale=1.5, seed=42, projection_weight_floor=0.0002
        )
    )
    monkeypatch.setattr(config, "load_ftf", lambda _: preset)
    # Supply a synthetic dirty checkout at the subprocess boundary.
    monkeypatch.setattr(
        experiment_wall_time,
        "subprocess",
        SimpleNamespace(check_output=lambda *a, **kw: "?? example\n"),
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
        settings: FTFSettings,
        *,
        soccfg: object,
        workers: int = 1,
        time_scale: float | None = None,
    ) -> SimpleNamespace:
        if on_ftf is not None:
            on_ftf(workers, settings, time_scale)
        return make_environment(soccfg)

    # The CLI report does not expose the factory argument. Delegate to the existing
    # headless fake so observing it does not build a real FTF backend.
    monkeypatch.setattr(environment, "ftf", observe_environment)
    monkeypatch.setattr(controller, "Controller", make_controller)
    # The Run's monotonic and UTC observation clocks are not CLI arguments.
    monkeypatch.setattr(experiment_wall_time, "perf_counter", clock.monotonic)
    monkeypatch.setattr(experiment_wall_time, "datetime", clock)
    return register


@pytest.mark.usefixtures("core_app")
def test_main_prepares_run_from_a_disposable_copy_of_the_manifest_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, output = tmp_path / "inputs", tmp_path / "output"
    _headless_inputs(data_dir)
    point_path = Path("result/demo/points/initial")
    received: list[tuple[str, ...]] = []

    def inspect_point(point: Path) -> None:
        names = ("parameters.yaml", "library.yaml", "aliases.yaml")
        received.append(tuple((point / name).read_text() for name in names))
        (point / "library.yaml").write_text("run-local-change")

    register = _install_headless_controller(
        monkeypatch,
        _Clock(),
        ("fake/frequency",),
        lambda ctrl: QTimer.singleShot(0, ctrl.complete),
        lambda _: pytest.fail("a completed Run must not be cancelled"),
        on_point=inspect_point,
    )
    assert (
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            registry_factory=register,
            read_commit=lambda: "abc",
        )
        == 0
    )
    assert received == [("parameter-token", "library-token", "aliases-token")]
    assert (
        data_dir / "fixture" / point_path / "library.yaml"
    ).read_text() == "library-token"


@pytest.mark.parametrize(
    ("notes", "basis"),
    [
        pytest.param({}, {}, id="omitted"),
        pytest.param(
            {"unpinned": {"form": "", "point": " "}}, {"basis": ""}, id="blank"
        ),
        pytest.param({"unpinned": None}, {"basis": []}, id="opaque"),
    ],
)
@pytest.mark.usefixtures("core_app")
def test_main_applies_selected_overrides_without_reading_review_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    notes: dict[str, object],
    basis: dict[str, object],
) -> None:
    data_dir, output = tmp_path / "inputs", tmp_path / "output"
    _headless_inputs(data_dir)
    source = data_dir / "case-inputs.json"
    raw = json.dumps(
        {
            "entry": "demo",
            "point": "initial",
            **notes,
            "scenes": [
                {
                    "scene_id": "other",
                    "adapter": "fake/power",
                    "metric": {"kind": "shots"},
                    **basis,
                },
                {
                    "scene_id": "scout",
                    "adapter": "fake/frequency",
                    "metric": {"kind": "shots"},
                    **basis,
                    "overrides": [
                        {"path": ["delay", "stop"], "value": 3, **basis},
                        {"path": ["delay", "n"], "value": 5, **basis},
                    ],
                },
            ],
        },
        indent=2,
    ).encode()
    source.write_bytes(raw)
    writes: list[object] = []
    register = _install_headless_controller(
        monkeypatch,
        _Clock(),
        ("fake/frequency", "fake/power"),
        lambda ctrl: QTimer.singleShot(0, ctrl.complete),
        lambda _: pytest.fail("a completed Run must not be cancelled"),
        on_write=writes.append,
    )
    assert (
        main(
            ["--select", "scout", "--output", str(output)],
            data_dir=data_dir,
            registry_factory=register,
            read_commit=lambda: "abc",
        )
        == 0
    )
    assert writes == [{("delay", "stop"): 3, ("delay", "n"): 5}]
    report = json.loads((output / "report.json").read_text())
    assert [(row["scene_id"], row["adapter"]) for row in report["rows"]] == [
        ("scout", "fake/frequency")
    ]
    plan = json.loads((output / "batch.json").read_text())
    assert plan["inputs"][source.name] == sha256(raw).hexdigest()


@pytest.mark.parametrize(
    ("worker_args", "expected"), [([], 1), (["--workers", "3"], 3)]
)
@pytest.mark.usefixtures("core_app")
def test_main_configures_ftf_workers_and_stamps_batch_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_args: list[str],
    expected: int,
) -> None:
    data_dir, output = tmp_path / "inputs", tmp_path / "output"
    _headless_inputs(data_dir)
    workers_seen: list[int] = []
    settings_seen: list[tuple[FTFSettings, float | None]] = []

    def observe_ftf(
        workers: int, settings: FTFSettings, time_scale: float | None
    ) -> None:
        workers_seen.append(workers)
        settings_seen.append((settings, time_scale))

    register = _install_headless_controller(
        monkeypatch,
        _Clock(),
        ("fake/frequency",),
        lambda ctrl: QTimer.singleShot(0, ctrl.complete),
        lambda _: pytest.fail("a completed Run must not be cancelled"),
        on_ftf=observe_ftf,
    )

    assert (
        main(
            ["--output", str(output), *worker_args],
            data_dir=data_dir,
            registry_factory=register,
            read_commit=lambda: "abc",
        )
        == 0
    )
    assert workers_seen == [expected]
    ((settings, time_scale),) = settings_seen
    assert time_scale == 0.0
    assert settings.numerics.time_scale == 1.5
    assert settings.numerics.seed == 42
    assert settings.numerics.projection_weight_floor == 0.0002
    assert settings.numerics.telemetry.commit == "abc"
    assert settings.numerics.telemetry.dirty is True


@pytest.mark.parametrize(
    ("publication_id", "wrapped", "error"),
    [
        pytest.param(1, True, None, id="current-run-record"),
        pytest.param(0, True, RuntimeError, id="stale-publication"),
        pytest.param(None, True, RuntimeError, id="missing-publication"),
        pytest.param(1, False, TypeError, id="wrong-record-type"),
    ],
)
@pytest.mark.usefixtures("core_app")
def test_main_uses_only_the_current_controller_run_record_for_scan_grid_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication_id: int | None,
    *,
    wrapped: bool,
    error: type[Exception] | None,
) -> None:
    data_dir, output = tmp_path / "inputs", tmp_path / "output"
    _headless_inputs(data_dir, metric={"kind": "scan", "axes": [{"field": "data"}]})
    result = SampleResult(np.linspace(0.0, 1.0, 5))

    def run(ctrl: _SyntheticController) -> None:
        (ctrl.telemetry / "routing-scan.jsonl").write_text(
            '{"families":{},"points":99,"wall_s":1,"compile_wall_s":0,'
            '"evolve_wall_s":0.375,"readout_wall_s":0,"materialize_wall_s":0}\n'
        )

        def finish() -> None:
            if publication_id is not None:
                ctrl.published_run = RunPaneSnapshot(
                    result=SampleRun(cfg=None, result=result) if wrapped else result,
                    source_path=None,
                    source_operation_id=publication_id,
                )
            ctrl.complete()

        QTimer.singleShot(0, finish)

    register = _install_headless_controller(
        monkeypatch,
        _Clock(),
        ("fake/frequency",),
        run,
        lambda _: pytest.fail("a completed Run must not be cancelled"),
    )

    def measure() -> int:
        return main(
            ["--output", str(output)],
            data_dir=data_dir,
            registry_factory=register,
            read_commit=lambda: "abc",
        )

    if error is not None:
        with pytest.raises(error, match="Run 1"):
            measure()
        return

    assert measure() == 0
    [row] = json.loads((output / "report.json").read_text())["rows"]
    assert row["point_count"] == 5
    assert row["core_s_per_point"] == 0.075
    assert row["core_judgement"] == "pass"


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
    register = _install_headless_controller(
        monkeypatch,
        clock,
        ("fake/frequency",),
        run,
        lambda _: pytest.fail("a completed Run must not be cancelled"),
    )

    assert (
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            registry_factory=register,
            read_commit=lambda: "abc",
        )
        == 0
    )

    report = json.loads((output / "report.json").read_text())
    [row] = report["rows"]
    assert report["commit"] == "abc"
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
        "| fake/frequency | fake/frequency | 8.000 | 1.000 | 2.750 | 1.500 | 0.750 | 6.000 | 1.000 | 2 |"
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
    _headless_inputs(data_dir, "fake/frequency", "fake/power")
    register = _install_headless_controller(
        monkeypatch, clock, ("fake/frequency", "fake/power"), run, cancel
    )

    assert (
        main(
            ["--output", str(output), "--row-timeout", "180"],
            data_dir=data_dir,
            registry_factory=register,
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
    assert "Core 超標（實測）: —\n" in summary
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
        unreached = {
            "acquire_id": "unreached",
            "complete": False,
            "materialize_wall_s": None,
            "families": {},
            "wall_s": 10,
            "compile_wall_s": 1,
            "evolve_wall_s": 8,
            "readout_wall_s": 1,
        }
        records = [
            unreached,
            {
                "event": "materialize",
                "acquire_id": "unreached",
                "materialize_wall_s": 0.5,
            },
        ]
        if evolve_s is not None:
            records += [
                {
                    **unreached,
                    "acquire_id": "raw",
                    "complete": True,
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
    register = _install_headless_controller(
        monkeypatch,
        clock,
        ("fake/frequency",),
        run,
        lambda ctrl: QTimer.singleShot(0, lambda: ctrl.complete("cancelled")),
    )
    assert (
        main(
            ["--output", str(output), "--row-timeout", "200"],
            data_dir=data_dir,
            registry_factory=register,
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
    assert "Core 超標（實測）: —" in summary
    assert f"Core 已知至少超標: {'`fake/frequency`' if over_target else '—'}" in summary
    assert f"Core 未判定: {'—' if over_target else '`fake/frequency`'}" in summary
    if lower_bound is not None:
        assert "| ≥180.000 |" in summary
