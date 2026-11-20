"""Record simulated experiment Run wall times and acquire stages."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from os.path import abspath
from pathlib import Path
from shutil import copytree
from time import perf_counter
from typing import TYPE_CHECKING, Annotated, Literal, TypedDict

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

from tools.analyze_routing_telemetry import AcquireInput, AcquireWallTotals

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "tools" / "data" / "experiment_wall_time"
_OUTPUT_ROOT = _REPO_ROOT / ".agent_state" / "benchmark" / "wall-time"


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
    basis: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


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
class RunTiming:
    """RunRecord's UTC timestamps and monotonic seconds from admission to completion."""

    started_at: str
    finished_at: str
    wall_s: float


@dataclass(frozen=True)
class Inventory:
    """Registered adapter guide declarations, operator guides, and the Run boundary.

    adapters maps each registered key to its declared path (None if undeclared).
    guides lists discovered operator guide paths, independent of selection. Use
    repo-relative paths for both, except external declarations stay absolute.

    measure(case, data_dir, telemetry) returns terminal Run timing and writes
    acquires into the given existing directory, or returns a case-specific
    FormErrors without timing for unresolved defaults. data_dir holds read-only
    FTF inputs and fixture.
    Startup, execution and output contract failures propagate.
    """

    adapters: Mapping[str, str | None]
    guides: Sequence[str]
    measure: Callable[[WallTimeCase, Path, Path], RunTiming | FormErrors]


def summarize_acquires(directory: Path) -> AcquireWallTotals:
    """Sum all routing JSONL records from one dedicated Run telemetry directory.

    Return coverage counts even when no wall measurements are available.
    An empty directory produces zero included and excluded records. I/O and
    invalid record errors propagate; unrelated filenames are not read.
    """
    totals = AcquireWallTotals()
    for path in sorted(directory.iterdir()):
        if path.match("routing-*.jsonl"):
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    totals.add(AcquireInput.model_validate_json(line))
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


class _Provenance(TypedDict):
    commit: str
    inputs: dict[str, str]


class _Row(TypedDict):
    adapter: str
    guide: str | None
    case_directory: str | None
    started_at: str | None
    finished_at: str | None
    wall_s: float | None
    acquires: int | None
    compile_s: float | None
    evolve_s: float | None
    readout_s: float | None
    remainder_s: float | None
    status: Literal["ok", "missing_breakdown", "missing_defaults", "not_runnable"]
    reason: str | None


def _blocked_row(
    adapter: str,
    guide: str | None,
    directory: str | None,
    status: Literal["missing_defaults", "not_runnable"],
    reason: str,
) -> _Row:
    return {
        "adapter": adapter,
        "guide": guide,
        "case_directory": directory,
        "started_at": None,
        "finished_at": None,
        "wall_s": None,
        "acquires": None,
        "compile_s": None,
        "evolve_s": None,
        "readout_s": None,
        "remainder_s": None,
        "status": status,
        "reason": reason,
    }


def _measured_row(
    adapter: str,
    guide: str | None,
    directory: str,
    timing: RunTiming,
    totals: AcquireWallTotals,
) -> _Row:
    has_breakdown = totals.included > 0 and totals.excluded == 0
    stages = {
        "compile_s": totals.compile_wall_s if has_breakdown else None,
        "evolve_s": totals.evolve_wall_s if has_breakdown else None,
        "readout_s": totals.readout_wall_s if has_breakdown else None,
        "remainder_s": totals.remainder_s if has_breakdown else None,
    }
    acquires = totals.included + totals.excluded
    return {
        "adapter": adapter,
        "guide": guide,
        "case_directory": directory,
        "started_at": timing.started_at,
        "finished_at": timing.finished_at,
        "wall_s": timing.wall_s,
        "acquires": acquires,
        "compile_s": stages["compile_s"],
        "evolve_s": stages["evolve_s"],
        "readout_s": stages["readout_s"],
        "remainder_s": stages["remainder_s"],
        "status": "ok" if has_breakdown else "missing_breakdown",
        "reason": None,
    }


def _seconds(seconds: float | None) -> str:
    return "—" if seconds is None else f"{seconds:.3f}"


def _summary_row(row: _Row) -> str:
    times = " | ".join(
        _seconds(seconds)
        for seconds in (
            row["compile_s"],
            row["evolve_s"],
            row["readout_s"],
            row["remainder_s"],
        )
    )
    status = {
        "ok": "OK",
        "missing_breakdown": "分項缺",
        "missing_defaults": "推不出預設",
        "not_runnable": "FTF 不能跑",
    }[row["status"]]
    if row["reason"] is not None:
        reason = " ".join(row["reason"].splitlines()).replace("|", r"\|")
        status = f"{status}: {reason}"
    guide = f"`{row['guide']}`" if row["guide"] else "—"
    acquires = "—" if row["acquires"] is None else str(row["acquires"])
    return f"| {row['adapter']} | {_seconds(row['wall_s'])} | {times} | {acquires} | {status} | {guide} |\n"


def _write_report(
    output: Path,
    rows: Sequence[_Row],
    provenance: _Provenance,
    guides_by_adapter: Mapping[str, str | None],
    guide_paths: Sequence[str],
) -> None:
    no_guide = sorted(
        adapter for adapter, guide in guides_by_adapter.items() if guide is None
    )
    inventoried = set(guide_paths)
    missing = {
        adapter: guide
        for adapter, guide in sorted(guides_by_adapter.items())
        if guide is not None and guide not in inventoried
    }
    no_adapter = sorted(inventoried.difference(guides_by_adapter.values()))
    over_target = [
        row["adapter"]
        for row in rows
        if (wall := row["wall_s"]) is not None and wall > 180.0
    ]
    (output / "report.json").write_text(
        json.dumps(
            {
                **provenance,
                "rows": rows,
                "adapters_without_guides": no_guide,
                "declared_guides_not_in_inventory": missing,
                "guides_without_adapters": no_adapter,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "summary.md").write_text(
        "| Experiment | Run (s) | Compile (s) | Evolve (s) | Readout (s) | Remainder (s) | Acquires | Status | Guide |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n"
        + "".join(map(_summary_row, rows))
        + f"\nRun >180 s: {_format_names(over_target)}\n"
        + f"\nAdapters without declared guides: {_format_names(no_guide)}\n"
        + "\nDeclared guides not in inventory: "
        + (
            ", ".join(f"`{adapter}` → `{guide}`" for adapter, guide in missing.items())
            or "—"
        )
        + f"\n\nGuides without adapters: {_format_names(no_adapter)}\n",
        encoding="utf-8",
    )


def _format_names(names: Sequence[str]) -> str:
    return ", ".join(f"`{name}`" for name in names) or "—"


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
        adapter = ctrl.tab_control.get_tab_adapter_name(tab)
        raise RuntimeError(f"{adapter}: Run {outcome.outcome}: {outcome.error_message}")
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


def _measure_run(
    registry: Registry,
    case: WallTimeCase,
    data_dir: Path,
    telemetry: Path,
    app: QCoreApplication,
) -> RunTiming | FormErrors:
    # ADR-0078: this tool measures the original Open/Run path, not a simulator test.
    from zcu_tools.gui.app.measure import State
    from zcu_tools.gui.app.measure.adapter import SessionEnv
    from zcu_tools.gui.app.measure.controller import Controller
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.context_types import EntryRequest
    from zcu_tools.program import build_soccfg
    from zcu_tools.simulators.ftf.config import load_ftf
    from zcu_tools.simulators.ftf.environment import ftf

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
        form = ctrl.cfg_resources.lookup(tab)
        if case.overrides:
            form.write(case.overrides)
        prepared = form.result()
        if isinstance(prepared, FormErrors):
            return prepared
        wall_s = _await_run(ctrl, tab, app)
        return _read_run_timing(ctrl, tab, wall_s)


def _discover() -> Inventory:
    from qtpy.QtCore import QCoreApplication
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
    app: QCoreApplication | None = None

    def measure(
        case: WallTimeCase, data_dir: Path, telemetry: Path
    ) -> RunTiming | FormErrors:
        # Retain one Qt application for the batch, but only if a case needs a tab.
        nonlocal app
        if app is None:
            app = QCoreApplication.instance() or QCoreApplication([])
        return _measure_run(registry, case, data_dir, telemetry, app)

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


def _case_row(
    inventory: Inventory,
    case: WallTimeCase,
    data_dir: Path,
    output: Path,
    directory: str,
) -> _Row:
    guide = inventory.adapters[case.adapter]
    telemetry = output / directory / "telemetry"
    telemetry.mkdir(parents=True)
    result = inventory.measure(case, data_dir, telemetry)
    if isinstance(result, FormErrors):
        reasons = [
            f"{'.'.join(map(str, path))}: {message}" if path else message
            for path, message in result.messages.items()
        ]
        return _blocked_row(
            case.adapter, guide, directory, "missing_defaults", "; ".join(reasons)
        )
    return _measured_row(
        case.adapter, guide, directory, result, summarize_acquires(telemetry)
    )


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
    CLI arguments exit with usage status. Return 0 after writing both reports.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="new directory for the report and telemetry"
    )
    parser.add_argument(
        "--select", action="append", help="adapter key or family, repeatable"
    )
    args = parser.parse_args(argv)
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
    provenance = _Provenance(commit=commit, inputs=_input_hashes(data_dir))
    output.mkdir(parents=True)
    rows: list[_Row] = []
    for index, adapter in enumerate(selected, start=1):
        if adapter in point.ftf_exclusions:
            rows.append(
                _blocked_row(
                    adapter,
                    inventory.adapters[adapter],
                    None,
                    "not_runnable",
                    point.ftf_exclusions[adapter],
                )
            )
        else:
            case = WallTimeCase(
                adapter, point.entry, point.point, overrides.get(adapter, {})
            )
            rows.append(
                _case_row(inventory, case, data_dir, output, f"cases/{index:03d}")
            )
    _write_report(output, rows, provenance, inventory.adapters, inventory.guides)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
