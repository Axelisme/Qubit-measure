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
from functools import partial
from hashlib import sha256
from os.path import abspath
from pathlib import Path
from shutil import copytree
from time import perf_counter
from types import FrameType
from typing import TYPE_CHECKING, Annotated, Protocol

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    StrictStr,
    StringConstraints,
    TypeAdapter,
)
from zcu_tools.gui.form import FormErrors, FormWritePath

if TYPE_CHECKING:
    from qtpy.QtCore import QCoreApplication
    from zcu_tools.gui.app.measure import Registry
    from zcu_tools.gui.app.measure.controller import Controller
    from zcu_tools.gui.app.measure.events.run import RunFinishedPayload

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


class _CaseInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry: str
    point: str
    ftf_exclusions: dict[str, Annotated[str, StringConstraints(min_length=1)]] = {}


_GUIDE_ROOT = _REPO_ROOT / "zcu_lab" / "v2"


class _Override(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    path: Annotated[tuple[StrictStr | StrictInt, ...], Field(min_length=1)]
    value: JsonValue
    basis: str


def _load_overrides(
    raw: bytes, adapters: Collection[str]
) -> dict[str, dict[FormWritePath, JsonValue]]:
    by_adapter: dict[str, dict[FormWritePath, JsonValue]] = {}
    for override in TypeAdapter(list[_Override]).validate_json(raw):
        if override.adapter not in adapters:
            raise ValueError(
                f"Override references unregistered adapter {override.adapter!r}"
            )
        paths = by_adapter.setdefault(override.adapter, {})
        if override.path in paths:
            raise ValueError(
                f"Duplicate override for {override.adapter!r} at {override.path!r}"
            )
        paths[override.path] = override.value
    return by_adapter


@dataclass(frozen=True)
class WallTimeCase:
    """Registered adapter key, fixture labels and raw Form writes for this Run.

    overrides maps canonical field paths to JSON values. Empty keeps all Open
    defaults. Apply a nonempty mapping as one atomic Form write.
    """

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
    """Registered adapter guide declarations, operator guides, and the Run boundary.

    adapters maps each registered key to its declared path (None if undeclared).
    guides lists discovered operator guide paths, independent of selection. Use
    repo-relative paths for both, except external declarations stay absolute.

    measure(case, data_dir, telemetry, record_attempt) prepares exactly one Run
    with read-only FTF inputs and fixture in data_dir. It calls record_attempt
    once with the prepared Run, or with FormErrors if defaults do not resolve.
    The prepared Run writes acquires into the given existing telemetry directory.
    Keep the Run's resources alive until record_attempt returns. Setup failures
    propagate without calling record_attempt.
    """

    adapters: Mapping[str, str | None]
    guides: Sequence[str]
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
        """Count a main row and return whether all its wall times were summed."""
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


def _input_hashes(data_dir: Path) -> dict[str, str]:
    paths = [
        data_dir / "ftf-settings.yaml",
        data_dir / "board-profile.yaml",
        data_dir / "case-inputs.json",
        data_dir / "overrides.json",
        *sorted(
            parent / name
            for parent, _, files in (data_dir / "fixture").walk(on_error=_raise_io)
            for name in files
        ),
    ]
    return {
        p.relative_to(data_dir).as_posix(): sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def _current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
    ).strip()


def _add_breakdown(
    row: Row, totals: AcquireCoreTotals, *, timed_out: bool = False
) -> None:
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


def _relative_repo_path(path: Path) -> str:
    absolute = Path(abspath(path))
    return (
        absolute.relative_to(_REPO_ROOT)
        if absolute.is_relative_to(_REPO_ROOT)
        else absolute
    ).as_posix()


def _raise_io(error: OSError) -> None:
    raise error


def _operator_guides() -> tuple[str, ...]:
    guides: list[str] = []
    for parent, _, files in _GUIDE_ROOT.walk(on_error=_raise_io):
        for name in files:
            if name.startswith(("guide.", "guide_")) and name.endswith(".md"):
                path = parent / name
                if path.is_file():
                    guides.append(_relative_repo_path(path))
    return tuple(sorted(guides))


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
    # Run writes an acquired ledger event. Only the disposable project is writable.
    with ExitStack() as cleanup:
        project = telemetry.parent / "project"
        copytree(data_dir / "fixture", project)
        soccfg = build_soccfg(
            yaml.safe_load(
                (data_dir / "board-profile.yaml").read_text(encoding="utf-8")
            )
        )
        settings = load_ftf(data_dir / "ftf-settings.yaml").with_telemetry(
            enabled=True, directory=telemetry.resolve()
        )
        env = ftf(settings, soccfg=soccfg)
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


def _discover(owner: _RunOwner | None = None) -> Inventory:
    from zcu_tools.gui.app.measure import Registry
    from zcu_tools.resources.entry.registry import component_registry

    from zcu_lab.assembly import lab_definitions

    # ADR-0072: lab declarations enter only through the assembly callbacks.
    lab = lab_definitions()
    lab.register_components(component_registry)
    registry = Registry(templates=lab.templates)
    lab.register_experiments(registry)
    adapters = {
        name: _relative_repo_path(path)
        if (path := registry.create(name).guide_path) is not None
        else None
        for name in registry.list_names()
    }
    run_owner = owner if owner is not None else _RunOwner()

    def measure(
        case: WallTimeCase,
        data_dir: Path,
        telemetry: Path,
        record_attempt: _RecordAttempt,
    ) -> None:
        _measure_run(registry, case, data_dir, telemetry, run_owner, record_attempt)

    return Inventory(adapters, _operator_guides(), measure)


def _select_adapters(adapters: Collection[str], selectors: Sequence[str]) -> list[str]:
    selected: set[str] = set()
    for selector in selectors:
        matches = (
            {selector}
            if selector in adapters
            else {name for name in adapters if name.partition("/")[0] == selector}
        )
        if not matches:
            raise ValueError(f"unknown adapter or family: {selector!r}")
        selected.update(matches)
    return sorted(selected if selectors else adapters)


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
    guide = inventory.adapters[case.adapter]
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
                case.adapter, guide, directory, "missing_defaults", "; ".join(reasons)
            )
            write_json(location / "row.json", row)
        else:
            row = new_row(case.adapter, guide, directory, "missing_breakdown", None)
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


def _recover_report(output: Path) -> int:
    all_resolved = assemble_report(output, read_plan(output))
    print(output)
    return 0 if all_resolved else 3


def main(
    argv: Sequence[str] | None = None,
    *,
    data_dir: Path = _DATA_DIR,
    discover: Callable[[], Inventory] = _discover,
    read_commit: Callable[[], str] = _current_commit,
) -> int:
    """Write report.json, summary.md and isolated telemetry for selected Runs.

    argv accepts repeatable --select (exact adapter key or first-segment family)
    and --output (a new directory). None reads sys.argv. With no selectors,
    measure every registered adapter. With no output, create a timestamped path
    under the repository's wall-time output root. data_dir holds the board,
    FTF settings, fixture, case declarations and overrides. discover lists the
    adapters and guides and supplies a runner (by default the offline controller
    from a fresh CLI process). read_commit supplies the checkout revision before
    Run; by default read Git HEAD. Input, Run and output errors propagate. Bad
    CLI arguments exit with usage status. --report-only rebuilds both reports
    from the existing --output, without discovery or dispatch. A start without
    a result is incomplete and provides no duration or lower bound. --resume-from
    retains resolved rows from a journal with matching commit and input hashes,
    and runs only missing selected cases. Return 0 if all reported rows are
    resolved (including timeout and run_failed), or 3 if the report contains
    incomplete or not_run rows. KeyboardInterrupt stops the batch and reports
    from its journal under the same exit status rule.
    """
    return _main(argv, data_dir, discover, read_commit, _Interruption())


def _main(
    argv: Sequence[str] | None,
    data_dir: Path,
    discover: Callable[[], Inventory],
    read_commit: Callable[[], str],
    stop: _Interruption,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="new directory for the report and telemetry"
    )
    parser.add_argument(
        "--select", action="append", help="adapter key or family, repeatable"
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
    inventory = discover()
    try:
        selected = _select_adapters(inventory.adapters, args.select or ())
    except ValueError as exc:
        parser.error(str(exc))
    commit = read_commit()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output or (_OUTPUT_ROOT / f"{commit}-{timestamp}")
    point = _CaseInputs.model_validate_json(
        (data_dir / "case-inputs.json").read_bytes()
    )
    overrides = _load_overrides(
        (data_dir / "overrides.json").read_bytes(), inventory.adapters
    )
    plan = BatchPlan(
        commit=commit,
        inputs=_input_hashes(data_dir),
        cases={},
        adapters=dict(inventory.adapters),
        guides=list(inventory.guides),
    )
    try:
        reused = (
            {} if args.resume_from is None else reusable_rows(args.resume_from, plan)
        )
    except JournalUsageError as exc:
        parser.error(str(exc))
    adapters = sorted(set(selected) | reused.keys())
    plan["cases"] = {
        adapter: f"cases/{index:03d}" for index, adapter in enumerate(adapters, start=1)
    }
    output.mkdir(parents=True)
    write_json(output / "batch.json", plan)
    for adapter, row in reused.items():
        directory = output / plan["cases"][adapter]
        directory.mkdir(parents=True)
        write_json(directory / "row.json", row)
    return _run_batch(
        inventory,
        point,
        overrides,
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
    point: _CaseInputs,
    overrides: Mapping[str, Mapping[FormWritePath, JsonValue]],
    data_dir: Path,
    batch: _Batch,
) -> int:
    output = batch.output
    try:
        for adapter, directory in batch.plan["cases"].items():
            if batch.stop.requested():
                break
            if (output / directory / "row.json").is_file():
                continue
            if adapter in point.ftf_exclusions:
                row = new_row(
                    adapter,
                    inventory.adapters[adapter],
                    None,
                    "not_runnable",
                    point.ftf_exclusions[adapter],
                )
                (output / directory).mkdir(parents=True)
                write_json(output / directory / "row.json", row)
            else:
                case = WallTimeCase(
                    adapter, point.entry, point.point, overrides.get(adapter, {})
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
        code = _main(None, _DATA_DIR, partial(_discover, owner), _current_commit, stop)
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
