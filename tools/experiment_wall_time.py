"""Record one simulated experiment Run's wall time and acquire stages."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from shutil import copytree
from time import perf_counter
from typing import TYPE_CHECKING, TypedDict

import yaml
from pydantic import BaseModel

if TYPE_CHECKING:
    from qtpy.QtCore import QCoreApplication
    from zcu_tools.gui.app.measure.controller import Controller

from tools.analyze_routing_telemetry import AcquireInput, AcquireWallTotals

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "tools" / "data" / "experiment_wall_time"
_OUTPUT_ROOT = _REPO_ROOT / ".agent_state" / "benchmark" / "wall-time"


class TracerCase(BaseModel):
    """Adapter name, repo-relative guide, and entry/point labels for the Run."""

    adapter: str
    guide: str
    entry: str
    point: str


@dataclass(frozen=True)
class RunTiming:
    """RunRecord's UTC timestamps and monotonic seconds from admission to completion."""

    started_at: str
    finished_at: str
    wall_s: float


def summarize_acquires(directory: Path) -> AcquireWallTotals:
    """Sum all routing JSONL records from one dedicated Run telemetry directory.

    Return coverage counts even when no wall measurements are available.
    An empty directory produces zero included and excluded records. I/O and
    invalid record errors propagate; unrelated filenames are not read.
    """
    totals = AcquireWallTotals()
    for path in sorted(directory.glob("routing-*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                totals.add(AcquireInput.model_validate_json(line))
    return totals


def _input_hashes(data_dir: Path) -> dict[str, str]:
    paths = [
        data_dir / "ftf-settings.yaml",
        data_dir / "board-profile.yaml",
        data_dir / "tracer.json",
        *(p for p in sorted((data_dir / "fixture").rglob("*")) if p.is_file()),
    ]
    return {
        p.relative_to(data_dir).as_posix(): sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def _current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
    ).strip()


class _Provenance(TypedDict):
    commit: str
    inputs: dict[str, str]


def _write_report(
    output: Path,
    case: TracerCase,
    timing: RunTiming,
    totals: AcquireWallTotals,
    provenance: _Provenance,
) -> None:
    has_breakdown = totals.included > 0 and totals.excluded == 0
    stages = {
        "compile_s": totals.compile_wall_s if has_breakdown else None,
        "evolve_s": totals.evolve_wall_s if has_breakdown else None,
        "readout_s": totals.readout_wall_s if has_breakdown else None,
        "remainder_s": totals.remainder_s if has_breakdown else None,
    }
    acquires = totals.included + totals.excluded
    row = {
        "adapter": case.adapter,
        "guide": case.guide,
        "started_at": timing.started_at,
        "finished_at": timing.finished_at,
        "wall_s": timing.wall_s,
        "acquires": acquires,
        **stages,
        "status": "ok" if has_breakdown else "missing_breakdown",
    }
    (output / "report.json").write_text(
        json.dumps(
            {**provenance, "rows": [row]},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    times = " | ".join(
        "—" if seconds is None else f"{seconds:.3f}" for seconds in stages.values()
    )
    status = "OK" if has_breakdown else "分項缺"
    (output / "summary.md").write_text(
        "| Experiment | Run (s) | Compile (s) | Evolve (s) | Readout (s) | Remainder (s) | Acquires | Status | Guide |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n"
        f"| {case.adapter} | {timing.wall_s:.3f} | {times} | {acquires} | {status} | `{case.guide}` |\n",
        encoding="utf-8",
    )


def _await_run(ctrl: Controller, tab: str, app: QCoreApplication) -> float:
    from qtpy.QtCore import QEventLoop
    from zcu_tools.gui.app.measure.events.run import RunFinishedPayload

    loop = QEventLoop(app)
    outcome: RunFinishedPayload | None = None
    wall_s = 0.0

    def finished(event: RunFinishedPayload) -> None:
        nonlocal outcome, wall_s
        if event.tab_id == tab:
            wall_s = perf_counter() - start
            outcome = event
            loop.quit()

    subscription = ctrl.get_bus().subscribe(RunFinishedPayload, finished)
    try:
        start = perf_counter()
        ctrl.run_analyze_control.start_run(tab)
        if outcome is None:
            loop.exec()
    finally:
        subscription.unsubscribe()
    if outcome is None:
        raise RuntimeError("Qt event loop stopped before Run completion")
    if outcome.outcome != "finished":
        raise RuntimeError(f"Run {outcome.outcome}: {outcome.error_message}")
    return wall_s


def _read_run_timing(ctrl: Controller, tab: str, wall_s: float) -> RunTiming:
    from zcu_tools.experiment.metadata import RunMetadataContent
    from zcu_tools.experiment.records import RunRecord

    pane = ctrl.tab_control.get_tab_snapshot(tab).run
    if pane is None or not isinstance(pane.result, RunRecord):
        raise RuntimeError("Completed Run did not publish a native RunRecord")
    metadata = RunMetadataContent(pane.result.metadata)
    started_at, finished_at = metadata.started_at, metadata.finished_at
    if started_at is None or finished_at is None:
        raise RuntimeError("Completed RunRecord is missing its timestamps")
    return RunTiming(started_at, finished_at, wall_s)


def _measure_run(case: TracerCase, data_dir: Path, telemetry: Path) -> RunTiming:
    # ADR-0078: this tool measures the original Open/Run path, not a simulator test.
    # ADR-0072: lab declarations enter only through the assembly callbacks.
    from qtpy.QtCore import QCoreApplication
    from zcu_tools.gui.app.measure import Registry, State
    from zcu_tools.gui.app.measure.adapter import SessionEnv
    from zcu_tools.gui.app.measure.controller import Controller
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.context_types import EntryRequest
    from zcu_tools.program import build_soccfg
    from zcu_tools.resources.entry.registry import component_registry
    from zcu_tools.simulators.ftf.config import load_ftf
    from zcu_tools.simulators.ftf.environment import ftf

    from zcu_lab.assembly import lab_definitions

    app = QCoreApplication.instance() or QCoreApplication([])
    lab = lab_definitions()
    lab.register_components(component_registry)
    registry = Registry(templates=lab.templates)
    lab.register_experiments(registry)

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
        cleanup.callback(ctrl.finish_close)
        ctrl.session_services.soc_connection.install_soc(env.soc, env.soccfg)
        ctrl.entry_control.open_entry(EntryRequest(case.entry))
        ctrl.entry_control.use_point(case.point)
        tab = ctrl.tab_control.new_tab(case.adapter)
        wall_s = _await_run(ctrl, tab, app)
        return _read_run_timing(ctrl, tab, wall_s)


def main(
    argv: Sequence[str] | None = None,
    *,
    data_dir: Path = _DATA_DIR,
    measure: Callable[[TracerCase, Path, Path], RunTiming] = _measure_run,
    read_commit: Callable[[], str] = _current_commit,
) -> int:
    """Measure the tracer and write report.json, summary.md and raw telemetry.

    argv accepts --output (a new directory); None reads sys.argv. Omission
    creates a timestamped directory under the repository's wall-time output root.
    data_dir holds the board, FTF settings, fixture and tracer declaration.
    measure(case, data_dir, telemetry) may substitute the simulated Run. It
    returns terminal Run timing and writes acquire telemetry in the given
    already-created directory. The default runs the offline controller in a
    fresh process. read_commit supplies the full revision before Run; the default
    reads this checkout's Git HEAD. Input, Run and output failures propagate.
    Return 0 after writing both reports.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="new directory for the report and telemetry"
    )
    args = parser.parse_args(argv)
    commit = read_commit()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output or (_OUTPUT_ROOT / f"{commit}-{timestamp}")
    case = TracerCase.model_validate_json((data_dir / "tracer.json").read_bytes())
    provenance = _Provenance(commit=commit, inputs=_input_hashes(data_dir))
    telemetry = output / "telemetry"
    output.mkdir(parents=True)
    telemetry.mkdir()
    timing = measure(case, data_dir, telemetry)
    _write_report(output, case, timing, summarize_acquires(telemetry), provenance)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
