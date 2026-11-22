"""Record simulated experiment Run wall times and acquire stages."""

from __future__ import annotations

import argparse
import faulthandler
import math
import os
import signal
import subprocess
import sys
import traceback
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from shutil import copytree
from time import perf_counter
from types import FrameType
from typing import TYPE_CHECKING, Protocol

import yaml
from pydantic import JsonValue
from zcu_tools.gui.form import FormErrors, FormWritePath

if TYPE_CHECKING:
    from qtpy.QtCore import QCoreApplication
    from zcu_tools.gui.app.measure import Registry
    from zcu_tools.gui.app.measure.controller import Controller
    from zcu_tools.gui.app.measure.events.run import RunFinishedPayload

from tools._experiment_wall_time_inputs import Manifest, Sources, read_manifest
from tools._experiment_wall_time_report import (
    BatchPlan,
    JournalUsageError,
    Row,
    RunStart,
    assemble_report,
    new_row,
    read_plan,
    reusable_rows,
    write_json,
)
from tools.analyze_routing_telemetry import (
    AcquireInput,
    MaterializeInput,
    decode_routing_record,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "tools" / "data" / "experiment_wall_time"
_OUTPUT_ROOT = _REPO_ROOT / ".agent_state" / "benchmark" / "wall-time"
_DEFAULT_ROW_TIMEOUT_S = 360.0
_MIN_ROW_TIMEOUT_S = 180.0
_QT_TIMER_MAX_MS = 2**31 - 1
_QT_TIMER_MAX_S = _QT_TIMER_MAX_MS / 1000
_CANCEL_GRACE_S = 30.0


@dataclass(frozen=True)
class WallTimeCase:
    """Curated scene, project labels and raw Form writes for this Run.

    overrides maps canonical field paths to JSON values. Empty keeps all Open
    defaults. Apply a nonempty mapping as one atomic Form write.
    """

    scene_id: str
    adapter: str
    entry: str
    point: str
    overrides: Mapping[FormWritePath, JsonValue]


@dataclass(frozen=True)
class RunObservation:
    """One dispatch or terminal observation, sampled on both clocks.

    monotonic_s is the batch's monotonic time in seconds. utc is its UTC
    ISO-8601 timestamp for display, not for calculating durations.
    """

    monotonic_s: float
    utc: str


def _observe_run() -> RunObservation:
    return RunObservation(perf_counter(), datetime.now(UTC).isoformat())


class Run(Protocol):
    """One prepared, viewless Run; call on its owner thread.

    Time arguments and results use the batch's monotonic clock, in seconds.
    """

    def start(self) -> RunObservation:
        """Dispatch this Run once and return the time sampled before dispatch."""
        ...

    def wait_until(
        self, deadline: float, interrupted: Callable[[], bool], /, *, idle: bool = False
    ) -> RunObservation | None:
        """Return the terminal observation, or None on deadline or interruption.

        With idle=True, also require the Controller to become idle by deadline.
        An observation delivered late may exceed deadline; it does not count as
        on time. Check interrupted while waiting so the batch can stop promptly.
        """
        ...

    def cancel(self) -> None:
        """Request Controller cancellation. This does not mean Run has stopped."""
        ...

    def failure(self) -> str | None:
        """Read the Controller's diagnostic on failure, or return None on success.

        Raise for a missing or unexpected outcome or unreadable diagnostics.
        """
        ...

    def is_idle(self) -> bool:
        """Return whether the Controller has no live operations."""
        ...


type _RecordAttempt = Callable[[Run | FormErrors], None]


@dataclass(frozen=True)
class Inventory:
    """Registered exact adapter keys and the Run boundary.

    measure(case, data_dir, telemetry, record_attempt) prepares exactly one Run
    with read-only case inputs in data_dir. It calls record_attempt
    once with the prepared Run, or with FormErrors if defaults do not resolve.
    The prepared Run writes acquires into the given existing telemetry directory.
    Keep the Run's resources alive until record_attempt returns. Setup failures
    propagate without calling record_attempt.
    """

    adapters: Collection[str]
    measure: Callable[[WallTimeCase, Path, Path, _RecordAttempt], None]


@dataclass
class AcquireCoreTotals:
    """Measured totals in seconds and coverage for one Run's telemetry.

    included and excluded count main acquire rows with and without all five
    wall measurements. uncovered_materialize indicates an event without a
    measured main row. wall_s sums the included acquire walls. The stage totals
    cover those same rows, plus matched deferred work in materialize_wall_s.
    remainder_s excludes only work within wall_s. All sums start at zero.
    """

    included: int = 0
    excluded: int = 0
    uncovered_materialize: bool = False
    wall_s: float = 0.0
    compile_wall_s: float = 0.0
    evolve_wall_s: float = 0.0
    readout_wall_s: float = 0.0
    materialize_wall_s: float = 0.0
    remainder_s: float = 0.0

    @property
    def core_s(self) -> float:
        """Sum the four measured stage totals, including matched deferred work."""
        return (
            self.compile_wall_s
            + self.evolve_wall_s
            + self.readout_wall_s
            + self.materialize_wall_s
        )

    def add_acquire(self, record: AcquireInput) -> bool:
        """Sum a main row only if all five wall times are measured.

        Count a row with any missing or null duration as excluded and return
        False. Keep numeric zero as a measurement. Return True for summed rows.
        """
        if (
            record.wall_s is None
            or record.compile_wall_s is None
            or record.evolve_wall_s is None
            or record.readout_wall_s is None
            or record.materialize_wall_s is None
        ):
            self.excluded += 1
            return False
        self.included += 1
        self.wall_s += record.wall_s
        self.compile_wall_s += record.compile_wall_s
        self.evolve_wall_s += record.evolve_wall_s
        self.readout_wall_s += record.readout_wall_s
        self.materialize_wall_s += record.materialize_wall_s
        self.remainder_s += (
            record.wall_s
            - record.compile_wall_s
            - record.evolve_wall_s
            - record.readout_wall_s
            - record.materialize_wall_s
        )
        return True


def summarize_acquires(directory: Path, *, live: bool = False) -> AcquireCoreTotals:
    """Sum measured acquire stages and pair deferred raw work by acquire ID.

    Read routing-*.jsonl from one Run directory; I/O and invalid record errors
    propagate. Pass live=True while a worker may still append or prune files:
    count committed lines only and skip files pruned between listing and open.
    """
    totals = AcquireCoreTotals()
    measured_ids: set[str] = set()
    deferred: dict[str, float] = {}
    for path in sorted(directory.iterdir()):
        if not path.match("routing-*.jsonl"):
            continue
        try:
            stream = path.open("rb")
        except FileNotFoundError:
            if not live:
                raise
            continue
        with stream:
            for line in stream:
                if not live or line.endswith(b"\n"):
                    record = decode_routing_record(line)
                    if isinstance(record, MaterializeInput):
                        acquire_id = record.acquire_id
                        deferred[acquire_id] = (
                            deferred.get(acquire_id, 0.0) + record.materialize_wall_s
                        )
                    elif totals.add_acquire(record) and record.acquire_id is not None:
                        measured_ids.add(record.acquire_id)
    for acquire_id, seconds in deferred.items():
        if acquire_id in measured_ids:
            # Deferred work contributes to core, but is outside acquire wall.
            totals.materialize_wall_s += seconds
        else:
            totals.uncovered_materialize = True
    return totals


def _current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
    ).strip()


def _add_breakdown(
    row: Row, totals: AcquireCoreTotals, *, timed_out: bool = False
) -> None:
    # Only acquires with every wall time present can support a core lower bound.
    covered = totals.included > 0
    complete = (
        covered
        and not timed_out
        and not totals.excluded
        and not totals.uncovered_materialize
    )
    available = complete or (covered and timed_out)
    row.update(
        acquires=totals.included + totals.excluded,
        compile_s=totals.compile_wall_s if available else None,
        evolve_s=totals.evolve_wall_s if available else None,
        readout_s=totals.readout_wall_s if available else None,
        materialize_s=totals.materialize_wall_s if available else None,
        core_s=totals.core_s if complete else None,
        core_s_lower_bound=totals.core_s if covered and not complete else None,
        remainder_s=totals.remainder_s if available else None,
        breakdown="complete" if complete else "partial" if available else "missing",
    )


class _ControllerRun:
    def __init__(self, ctrl: Controller, tab: str, app: QCoreApplication) -> None:
        from qtpy.QtCore import QEventLoop

        self.ctrl = ctrl
        self.tab = tab
        self.loop = QEventLoop(app)
        self.terminal: RunFinishedPayload | None = None
        self.observed: RunObservation | None = None
        self.wake: Callable[[], None] | None = None
        self.token: int | None = None

    def finished(self, event: RunFinishedPayload) -> None:
        if event.tab_id == self.tab:
            self.observed = _observe_run()
            self.terminal = event
            if self.wake is not None:
                self.wake()

    def start(self) -> RunObservation:
        dispatched = _observe_run()
        self.token = self.ctrl.run_analyze_control.start_run(self.tab)
        return dispatched

    def is_idle(self) -> bool:
        return self.ctrl.active_operation_count() == 0

    def safe_to_close(self) -> bool:
        return self.is_idle() and (self.token is None or self.terminal is not None)

    def wait_until(
        self, deadline: float, interrupted: Callable[[], bool], *, idle: bool = False
    ) -> RunObservation | None:
        from qtpy.QtCore import Qt, QTimer

        def ready_observation() -> RunObservation | None:
            if idle and (not self.is_idle() or perf_counter() > deadline):
                return None
            return self.observed

        remaining = deadline - perf_counter()
        observed = ready_observation()

        def wake_if_ready() -> None:
            nonlocal observed
            observed = observed or ready_observation()
            if observed is not None or interrupted():
                self.loop.quit()

        if remaining > 0 and observed is None and not interrupted():
            timer = QTimer()
            timer.setTimerType(Qt.TimerType.PreciseTimer)
            timer.setSingleShot(True)
            timer.timeout.connect(self.loop.quit)
            poll = QTimer()
            poll.timeout.connect(wake_if_ready)
            self.wake = wake_if_ready
            try:
                timer.start(min(_QT_TIMER_MAX_MS, math.ceil(remaining * 1000)))
                poll.start(50)
                self.loop.exec()
            finally:
                self.wake = None
                poll.stop()
                timer.stop()
        if self.terminal is not None and self.terminal.outcome not in {
            "finished",
            "failed",
            "cancelled",
        }:
            raise RuntimeError(f"Unknown Run outcome: {self.terminal.outcome!r}")
        # A callback can do more work after quit(); keep the observation that woke us.
        observed = observed or ready_observation()
        if observed is not None or interrupted() or perf_counter() >= deadline:
            return observed
        raise RuntimeError("Qt event loop stopped before Run completion")

    def cancel(self) -> None:
        self.ctrl.run_analyze_control.cancel_run()

    def failure(self) -> str | None:
        terminal = self.terminal
        if terminal is None:
            raise RuntimeError("Run did not publish a terminal event")
        if terminal.outcome == "finished":
            return None
        if terminal.outcome != "failed" or self.token is None:
            raise RuntimeError(f"Unexpected Run outcome: {terminal.outcome!r}")
        # ADR-0066: the Run settles its handle before publishing the event. The
        # off-owner read cannot wait for an owner callback or for a live Run.
        await_operation = self.ctrl.operation_control.await_operation
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(await_operation, self.token, 0).result()
        outcome = result.outcome
        if (
            result.reason != "completed"
            or outcome is None
            or outcome.status != "failed"
            or not outcome.error
        ):
            raise RuntimeError(
                f"Failed Run event disagrees with operation {self.token}"
            )
        return outcome.error


@dataclass
class _RunOwner:
    workers: int = 1
    commit: str | None = None
    dirty: bool | None = None
    app: QCoreApplication | None = None
    retained: list[ExitStack] = field(default_factory=list)

    def application(self) -> QCoreApplication:
        from qtpy.QtCore import QCoreApplication

        if self.app is None:
            self.app = QCoreApplication.instance() or QCoreApplication([])
        return self.app


def _close_controller(ctrl: Controller, cleanup: ExitStack, owner: _RunOwner) -> None:
    try:
        ctrl.finish_close()
    except TimeoutError:
        # An idle operation registry is not a successful background drain.
        pending = cleanup.pop_all()
        pending.callback(ctrl.finish_close)
        owner.retained.append(pending)
        raise


def _measure_run(
    registry: Registry,
    case: WallTimeCase,
    data_dir: Path,
    telemetry: Path,
    owner: _RunOwner,
    record_attempt: _RecordAttempt,
) -> None:
    # ADR-0078: this tool measures the original Open/Run path, not a simulator test.
    from zcu_tools.gui.app.measure import State
    from zcu_tools.gui.app.measure.adapter import SessionEnv
    from zcu_tools.gui.app.measure.controller import Controller
    from zcu_tools.gui.app.measure.events.run import RunFinishedPayload
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.context_types import EntryRequest
    from zcu_tools.program import build_soccfg
    from zcu_tools.simulators.ftf.config import load_ftf
    from zcu_tools.simulators.ftf.environment import ftf

    app = owner.application()
    sources = Sources(data_dir)
    # Run writes an acquired ledger event. Only the disposable project is writable.
    with ExitStack() as cleanup:
        project = telemetry.parent / "project"
        copytree(sources.fixture, project)
        soccfg = build_soccfg(yaml.safe_load(sources.board.read_text(encoding="utf-8")))
        settings = load_ftf(sources.settings).with_telemetry(
            enabled=True,
            directory=telemetry.resolve(),
            commit=owner.commit,
            dirty=owner.dirty,
        )
        env = ftf(settings, soccfg=soccfg, workers=owner.workers, time_scale=0.0)
        if env.close_backend is not None:
            cleanup.callback(env.close_backend)
        try:
            ctrl = Controller(
                state=State(SessionEnv(parameters=None, soc=None, soccfg=None)),
                registry=registry,
                view=None,
                bus=BaseEventBus(),
                project_root=str(project),
                device_registry=env.devices,
                driver_factory=env.create_device,
            )
        except Exception:
            env.devices.close_all_devices()
            raise
        cleanup.callback(_close_controller, ctrl, cleanup, owner)
        ctrl.session_services.soc_connection.install_soc(env.soc, env.soccfg)
        ctrl.entry_control.open_entry(EntryRequest(case.entry))
        ctrl.entry_control.use_point(case.point)
        tab = ctrl.tab_control.new_tab(case.adapter)
        form = ctrl.cfg_resources.lookup(tab)
        if case.overrides:
            form.write(case.overrides)
        prepared = form.result()
        if isinstance(prepared, FormErrors):
            record_attempt(prepared)
            return
        run = _ControllerRun(ctrl, tab, app)
        subscription = ctrl.get_bus().subscribe(RunFinishedPayload, run.finished)
        cleanup.callback(subscription.unsubscribe)
        try:
            record_attempt(run)
        finally:
            if not run.safe_to_close():
                # ADR-0066: cancellation alone doesn't authorize resource teardown.
                owner.retained.append(cleanup.pop_all())


def _register_lab() -> Registry:
    from zcu_tools.gui.app.measure import Registry
    from zcu_tools.resources.entry.registry import component_registry

    from zcu_lab.assembly import lab_definitions

    # ADR-0072: lab declarations enter only through the assembly callbacks.
    lab = lab_definitions()
    lab.register_components(component_registry)
    registry = Registry(templates=lab.templates)
    lab.register_experiments(registry)
    return registry


def _discover(
    owner: _RunOwner, registry_factory: Callable[[], Registry] = _register_lab
) -> Inventory:
    owner.dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=_REPO_ROOT,
            text=True,
        )
    )
    registry = registry_factory()

    def measure(
        case: WallTimeCase,
        data_dir: Path,
        telemetry: Path,
        record_attempt: _RecordAttempt,
    ) -> None:
        _measure_run(registry, case, data_dir, telemetry, owner, record_attempt)

    return Inventory(frozenset(registry.list_names()), measure)


class _RunCannotContinue(Exception):
    """A failed or timed-out Run was journaled but the Controller is not ready."""


class _InterruptedBatch(Exception):
    """Stop was requested; preserve committed rows and report from the journal."""


@dataclass
class _Interruption:
    stopped_s: float | None = None

    def request(
        self, _signum: int | None = None, _frame: FrameType | None = None
    ) -> None:
        if self.stopped_s is None:
            self.stopped_s = perf_counter()

    def requested(self) -> bool:
        return self.stopped_s is not None

    def before(self, moment: float) -> bool:
        return self.stopped_s is not None and self.stopped_s <= moment


def _finish_timeout(run: Run, location: Path, row: Row, stop: _Interruption) -> None:
    with (location / "timeout-stack.txt").open("w", encoding="utf-8") as stack:
        faulthandler.dump_traceback(file=stack, all_threads=True)
    run.cancel()
    deadline = perf_counter() + _CANCEL_GRACE_S
    telemetry = location / "telemetry"
    _add_breakdown(row, summarize_acquires(telemetry, live=True), timed_out=True)
    write_json(location / "row.json", row)
    settled = run.wait_until(deadline, stop.requested, idle=True)
    if stop.requested():
        raise _InterruptedBatch
    if settled is None or settled.monotonic_s > deadline:
        raise _RunCannotContinue
    _add_breakdown(row, summarize_acquires(telemetry), timed_out=True)
    write_json(location / "row.json", row)


def _finish_failed_run(run: Run, location: Path, row: Row, reason: str) -> None:
    path = location / "run-failure.json"
    write_json(
        path,
        {
            "scene_id": row["scene_id"],
            "adapter": row["adapter"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "reason": reason,
        },
    )
    row.update(status="run_failed", reason=reason, failure_record=path.name)
    write_json(location / "row.json", row)
    if not run.is_idle():
        raise _RunCannotContinue


def _case_row(
    inventory: Inventory,
    case: WallTimeCase,
    data_dir: Path,
    batch: _Batch,
    directory: str,
) -> None:
    location = batch.output / directory
    telemetry = location / "telemetry"
    telemetry.mkdir(parents=True)

    def record_attempt(prepared: Run | FormErrors) -> None:
        if batch.stop.requested():
            raise _InterruptedBatch
        if isinstance(prepared, FormErrors):
            reasons = [
                f"{'.'.join(map(str, path))}: {message}" if path else message
                for path, message in prepared.messages.items()
            ]
            row = new_row(
                case.scene_id,
                case.adapter,
                directory,
                "missing_defaults",
                "; ".join(reasons),
            )
            write_json(location / "row.json", row)
        else:
            row = new_row(
                case.scene_id, case.adapter, directory, "missing_breakdown", None
            )
            _measure_attempt(prepared, location, row, batch)

    inventory.measure(case, data_dir, telemetry, record_attempt)


def _measure_attempt(run: Run, location: Path, row: Row, batch: _Batch) -> None:
    start = RunStart(
        started_at=datetime.now(UTC).isoformat(),
        monotonic_s=perf_counter(),
        cold_start=batch.cold_next,
    )
    journal = location / "start.json"
    write_json(journal, start)
    if batch.stop.requested():
        journal.unlink()
        raise _InterruptedBatch
    row["cold_start"] = batch.cold_next
    batch.cold_next = False
    try:
        dispatched = run.start()
        deadline = dispatched.monotonic_s + batch.timeout_s
        observed = run.wait_until(deadline, batch.stop.requested)
    except KeyboardInterrupt:
        batch.stop.request()
        run.cancel()
        raise _InterruptedBatch from None
    decision_s = deadline if observed is None else min(observed.monotonic_s, deadline)
    if batch.stop.before(decision_s):
        run.cancel()
        raise _InterruptedBatch
    row["started_at"] = dispatched.utc
    if observed is None or observed.monotonic_s > deadline:
        row.update(status="timeout", wall_s_lower_bound=batch.timeout_s)
        _finish_timeout(run, location, row, batch.stop)
        return
    row["finished_at"] = observed.utc
    failure = run.failure()
    if failure is not None:
        _finish_failed_run(run, location, row, failure)
        return
    row["wall_s"] = observed.monotonic_s - dispatched.monotonic_s
    _add_breakdown(row, summarize_acquires(location / "telemetry"))
    row["status"] = "ok" if row["breakdown"] == "complete" else "missing_breakdown"
    write_json(location / "row.json", row)


def _row_timeout(raw: str) -> float:
    seconds = float(raw)
    if (
        not math.isfinite(seconds)
        or not _MIN_ROW_TIMEOUT_S <= seconds <= _QT_TIMER_MAX_S
    ):
        raise argparse.ArgumentTypeError(
            f"row deadline must be between {_MIN_ROW_TIMEOUT_S:g} and {_QT_TIMER_MAX_S:.3f} seconds"
        )
    return seconds


def _worker_count(raw: str) -> int:
    workers = int(raw)
    if workers < 1:
        raise argparse.ArgumentTypeError("worker count must be at least 1")
    return workers


def _recover_report(output: Path) -> int:
    all_resolved = assemble_report(output, read_plan(output))
    print(output)
    return 0 if all_resolved else 3


def main(
    argv: Sequence[str] | None = None,
    *,
    data_dir: Path = _DATA_DIR,
    discover: Callable[[], Inventory] | None = None,
    registry_factory: Callable[[], Registry] = _register_lab,
    read_commit: Callable[[], str] = _current_commit,
) -> int:
    """Write report.json, summary.md and isolated telemetry for selected Runs.

    None argv reads sys.argv; see --help for CLI options. data_dir holds the
    board profile, scene manifest, FTF settings and clean project fixture.
    registry_factory supplies registered adapters for the offline Controller;
    by default it uses the lab assembly. discover can replace both registration
    and the offline Controller with an Inventory. read_commit supplies the
    batch revision (Git HEAD by default); the Controller stamps it and the
    pre-dispatch dirty status on acquire telemetry.

    Return 0 if all rows are resolved (including timeout and run_failed), or 3
    for incomplete, not_run or missing_adapter rows. KeyboardInterrupt stops and reports from
    the journal under the same rule. Usage errors exit 2; setup and I/O errors
    propagate. --report-only rebuilds reports without discovery or dispatch.
    --resume-from retains resolved rows with matching commit and input hashes.
    """

    def make_inventory(owner: _RunOwner) -> Inventory:
        return (
            discover() if discover is not None else _discover(owner, registry_factory)
        )

    return _main(
        argv, data_dir, make_inventory, read_commit, _Interruption(), _RunOwner()
    )


def _main(
    argv: Sequence[str] | None,
    data_dir: Path,
    make_inventory: Callable[[_RunOwner], Inventory],
    read_commit: Callable[[], str],
    stop: _Interruption,
    owner: _RunOwner,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="new directory for the report and telemetry"
    )
    parser.add_argument(
        "--select", action="append", help="exact scene ID or adapter family, repeatable"
    )
    parser.add_argument(
        "--row-timeout",
        type=_row_timeout,
        default=_DEFAULT_ROW_TIMEOUT_S,
        metavar="SECONDS",
        help=(
            f"deadline from Run dispatch (default {_DEFAULT_ROW_TIMEOUT_S:g} s; "
            f"range {_MIN_ROW_TIMEOUT_S:g} to {_QT_TIMER_MAX_S:.3f} s)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=_worker_count,
        default=1,
        metavar="N",
        help="parallel FTF workers per experiment (default 1)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild reports from an existing --output",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="retain resolved rows from an existing batch journal",
    )
    args = parser.parse_args(argv)
    if args.report_only:
        if args.output is None or args.select or args.resume_from is not None:
            parser.error(
                "--report-only requires --output and does not accept --select or --resume-from"
            )
        return _recover_report(args.output)
    sources = Sources(data_dir)
    manifest_bytes = sources.manifest.read_bytes()
    try:
        manifest = read_manifest(manifest_bytes)
        selected = manifest.select(args.select or ())
    except ValueError as exc:
        parser.error(str(exc))
    owner.workers = args.workers
    commit = owner.commit = read_commit()
    inventory = make_inventory(owner)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output or (_OUTPUT_ROOT / f"{commit}-{timestamp}")
    by_id = {scene.scene_id: scene for scene in manifest.scenes}
    available = {
        scene.scene_id: scene.adapter
        for scene in manifest.scenes
        if scene.adapter in inventory.adapters
    }
    plan = BatchPlan(
        format_version=2,
        commit=commit,
        inputs=sources.hashes(manifest_bytes=manifest_bytes),
        manifest_scenes=len(by_id),
        scenes={},
    )
    try:
        reused = (
            {}
            if args.resume_from is None
            else reusable_rows(args.resume_from, plan, available)
        )
    except JournalUsageError as exc:
        parser.error(str(exc))
    plan["scenes"] = {
        scene_id: {
            "adapter": by_id[scene_id].adapter,
            "directory": f"cases/{index:03d}",
        }
        for index, scene_id in enumerate(sorted(selected | reused.keys()), start=1)
    }
    output.mkdir(parents=True)
    write_json(output / "batch.json", plan)
    for scene_id, planned in plan["scenes"].items():
        row = reused.get(scene_id)
        if scene_id not in available:
            adapter = planned["adapter"]
            row = new_row(
                scene_id,
                adapter,
                None,
                "missing_adapter",
                f"Exact adapter {adapter!r} is not registered. "
                "Maintainer: update or remove the stale scene in the manifest.",
            )
        if row is not None:
            location = output / planned["directory"]
            location.mkdir(parents=True)
            write_json(location / "row.json", row)
    return _run_batch(
        inventory,
        manifest,
        data_dir,
        _Batch(output, plan, args.row_timeout, stop),
    )


@dataclass
class _Batch:
    output: Path
    plan: BatchPlan
    timeout_s: float
    stop: _Interruption
    cold_next: bool = True


def _run_batch(
    inventory: Inventory,
    manifest: Manifest,
    data_dir: Path,
    batch: _Batch,
) -> int:
    output = batch.output
    by_id = {scene.scene_id: scene for scene in manifest.scenes}
    try:
        for scene_id, planned in batch.plan["scenes"].items():
            if batch.stop.requested():
                break
            directory = planned["directory"]
            if (output / directory / "row.json").is_file():
                continue
            scene = by_id[scene_id]
            if scene.ftf_exclusion is not None:
                row = new_row(
                    scene_id, scene.adapter, None, "not_runnable", scene.ftf_exclusion
                )
                (output / directory).mkdir(parents=True)
                write_json(output / directory / "row.json", row)
            else:
                case = WallTimeCase(
                    scene_id,
                    scene.adapter,
                    manifest.entry,
                    manifest.point,
                    {override.path: override.value for override in scene.overrides},
                )
                _case_row(inventory, case, data_dir, batch, directory)
    except KeyboardInterrupt:
        batch.stop.request()
    except (_InterruptedBatch, _RunCannotContinue):
        pass
    all_resolved = assemble_report(output, batch.plan)
    print(output)
    return 0 if all_resolved else 3


def _exit_with_live_workers(code: int) -> None:
    # Only the standalone offline CLI may abandon live simulator workers.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _cli() -> int:
    owner = _RunOwner()
    stop = _Interruption()
    previous = {
        sig: signal.signal(sig, stop.request) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        code = _main(None, _DATA_DIR, _discover, _current_commit, stop, owner)
    except Exception:
        if owner.retained:
            traceback.print_exc()
            _exit_with_live_workers(1)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if owner.retained:
        _exit_with_live_workers(code)
    return code


if __name__ == "__main__":
    raise SystemExit(_cli())
