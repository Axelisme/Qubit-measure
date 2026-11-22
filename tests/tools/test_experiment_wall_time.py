"""The wall-time CLI reports Run facts without executing a real experiment."""

from __future__ import annotations

import json
from collections.abc import Callable, Collection
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError
from zcu_tools.gui.form import FormErrors

from tools.experiment_wall_time import (
    Inventory,
    RunObservation,
    WallTimeCase,
    main,
)

if TYPE_CHECKING:
    from tools.experiment_wall_time import Run

_FREQUENCY = {"fake/frequency"}
_ADAPTERS = _FREQUENCY | {"fake/power", "other"}


def _write_scenes(directory: Path, *scenes: str | dict[str, object]) -> Path:
    path = directory / "case-inputs.json"
    path.write_text(
        json.dumps(
            {
                "entry": "demo",
                "point": "initial",
                "unpinned": {"form": "current Form", "point": "fixture point"},
                "scenes": [
                    {"scene_id": scene, "adapter": scene, "basis": "test workload"}
                    if isinstance(scene, str)
                    else scene
                    for scene in scenes
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Supply minimal hashed inputs for tests which replace the Run edge."""
    data_dir = tmp_path / "inputs"
    documents = data_dir / "fixture/result/demo/points/initial"
    documents.mkdir(parents=True)
    (documents / "library.yaml").write_bytes(b"modules")
    (data_dir / "ftf-settings.yaml").write_bytes(b"settings")
    (data_dir / "board-profile.yaml").write_bytes(b"profile")
    _write_scenes(data_dir, *sorted(_ADAPTERS))
    return data_dir


class _CompletedRun:
    def __init__(self, terminal_s: float) -> None:
        self.terminal_s = terminal_s

    def start(self) -> RunObservation:
        return RunObservation(0.0, datetime(2026, 11, 21, tzinfo=UTC).isoformat())

    def wait_until(
        self, _deadline: float, _interrupted: Callable[[], bool], *, idle: bool = False
    ) -> RunObservation:
        finished = datetime(2026, 11, 21, tzinfo=UTC) + timedelta(
            seconds=self.terminal_s
        )
        return RunObservation(self.terminal_s, finished.isoformat())

    def cancel(self) -> None:
        pytest.fail("a completed Run should not be cancelled")

    def failure(self) -> str | None:
        return None

    def is_idle(self) -> bool:
        return True


class _FailedRun(_CompletedRun):
    def __init__(self, reason: str, idle: Callable[[], bool]) -> None:
        super().__init__(7.0)
        self.reason = reason
        self.check_idle = idle

    def failure(self) -> str:
        return self.reason

    def is_idle(self) -> bool:
        return self.check_idle()


def _inventory(
    adapters: Collection[str],
    measure: Callable[[WallTimeCase, Path, Path], float | FormErrors],
) -> Inventory:
    def prepare(
        case: WallTimeCase,
        data_dir: Path,
        telemetry: Path,
        record_attempt: Callable[[Run | FormErrors], None],
    ) -> None:
        result = measure(case, data_dir, telemetry)
        record_attempt(
            result if isinstance(result, FormErrors) else _CompletedRun(result)
        )

    return Inventory(adapters, prepare)


def test_main_keeps_committed_rows_when_interrupted_before_the_next_run(
    tmp_path: Path, data_dir: Path
) -> None:
    output = tmp_path / "output"

    def measure(case: WallTimeCase, _data_dir: Path, telemetry: Path) -> float:
        if case.adapter == "fake/power":
            committed = telemetry.parent.parent / "001" / "row.json"
            assert committed.is_file(), "previous row must already be on disk"
            raise KeyboardInterrupt
        assert case.adapter == "fake/frequency"
        return 5.0

    assert (
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            discover=lambda: _inventory(_ADAPTERS, measure),
            read_commit=lambda: "abc",
        )
        == 3
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert [(row["adapter"], row["status"]) for row in report["rows"]] == [
        ("fake/frequency", "missing_breakdown"),
        ("fake/power", "not_run"),
        ("other", "not_run"),
    ]
    assert report["coverage"]["resolved"] == 1
    assert report["coverage"]["selected"] == 3


def test_interrupted_run_has_a_persisted_start_but_no_measured_time(
    tmp_path: Path, data_dir: Path
) -> None:
    _write_scenes(data_dir, *sorted(_FREQUENCY))
    output = tmp_path / "output"
    cancelled = False

    class InterruptedRun:
        def start(self) -> RunObservation:
            start = json.loads((output / "cases/001/start.json").read_text())
            assert start["started_at"]
            return RunObservation(start["monotonic_s"], start["started_at"])

        def wait_until(
            self,
            _deadline: float,
            _interrupted: Callable[[], bool],
            *,
            idle: bool = False,
        ) -> RunObservation | None:
            raise KeyboardInterrupt

        def cancel(self) -> None:
            nonlocal cancelled
            cancelled = True

        def failure(self) -> str | None:
            pytest.fail("interruption cannot publish a terminal outcome")

        def is_idle(self) -> bool:
            return False

    def measure(
        _case: WallTimeCase,
        _data_dir: Path,
        _telemetry: Path,
        attempt: Callable[[Run | FormErrors], None] | None = None,
    ) -> None:
        if attempt is None:
            raise KeyboardInterrupt
        attempt(InterruptedRun())

    assert (
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            discover=lambda: Inventory(_FREQUENCY, measure),
            read_commit=lambda: "abc",
        )
        == 3
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    [row] = report["rows"]
    assert row["status"] == "incomplete"
    assert row["wall_s"] is None
    assert row["wall_s_lower_bound"] is None
    assert cancelled
    assert report["coverage"]["resolved"] == 0


def test_report_only_recovers_from_persisted_facts_after_a_hard_stop(
    tmp_path: Path, data_dir: Path
) -> None:
    output = tmp_path / "output"

    class LostRun(_CompletedRun):
        def wait_until(
            self,
            deadline: float,
            interrupted: Callable[[], bool],
            *,
            idle: bool = False,
        ) -> RunObservation:
            raise OSError("batch vanished after dispatch")

    def measure(
        case: WallTimeCase,
        _data_dir: Path,
        _telemetry: Path,
        attempt: Callable[[Run | FormErrors], None],
    ) -> None:
        attempt(LostRun(6.0) if case.adapter == "fake/power" else _CompletedRun(6.0))

    with pytest.raises(OSError, match="batch vanished"):
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            discover=lambda: Inventory(_ADAPTERS, measure),
            read_commit=lambda: "abc",
        )
    assert not (output / "report.json").exists()
    start = json.loads((output / "cases/002/start.json").read_text(encoding="utf-8"))

    assert (
        main(
            ["--report-only", "--output", str(output)],
            discover=lambda: pytest.fail("report-only must not discover experiments"),
            read_commit=lambda: pytest.fail(
                "report-only must use the batch's revision"
            ),
        )
        == 3
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["commit"] == "abc"
    done, incomplete, unstarted = report["rows"]
    assert done["wall_s"] == 6.0
    assert incomplete["status"] == "incomplete"
    assert incomplete["started_at"] == start["started_at"]
    assert unstarted["status"] == "not_run"
    for row in (incomplete, unstarted):
        assert row["wall_s"] is None
        assert row["wall_s_lower_bound"] is None
    assert report["coverage"] == {"resolved": 1, "selected": 3, "manifest": 3}
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "| fake/power | fake/power | — |" in summary
    assert "| other | other | — |" in summary


@pytest.mark.parametrize("recover", ["report-only", "resume-from"])
@pytest.mark.parametrize(
    ("corruption", "diagnostic"),
    [
        (None, "Invalid JSON"),
        ({"adapter": "other"}, "adapter"),
        ({"scene_id": "wrong-scout"}, "expected scene"),
        ({"status": "ok", "wall_s": None}, "wall_s"),
        ({"status": "missing_breakdown", "wall_s": None}, "wall_s"),
        ({"wall_s": -1.0}, "wall_s"),
        ({"wall_s": float("inf")}, "wall_s"),
        ({"wall_s": True}, "wall_s"),
    ]
    + [
        pytest.param({field: value}, field, id=f"{field}-{type(value).__name__}")
        for field in ("materialize_s", "core_s", "core_s_lower_bound")
        for value in ("181", True)
    ],
)
def test_invalid_row_journal_is_a_tool_error_not_a_usage_error(
    tmp_path: Path,
    data_dir: Path,
    recover: str,
    corruption: dict[str, object] | None,
    diagnostic: str,
) -> None:
    _write_scenes(data_dir, *sorted(_FREQUENCY))
    source = tmp_path / "original"

    def measure(_case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> float:
        return 2.0

    def discover() -> Inventory:
        return _inventory(_FREQUENCY, measure)

    assert (
        main(
            ["--output", str(source)],
            data_dir=data_dir,
            discover=discover,
            read_commit=lambda: "abc",
        )
        == 0
    )
    journal = source / "cases/001/row.json"
    row = json.loads(journal.read_text(encoding="utf-8"))
    journal.write_text(
        "not JSON" if corruption is None else json.dumps(row | corruption),
        encoding="utf-8",
    )
    (source / "report.json").unlink()
    continued = tmp_path / "continued"
    argv = (
        ["--report-only", "--output", str(source)]
        if recover == "report-only"
        else ["--resume-from", str(source), "--output", str(continued)]
    )
    with pytest.raises(ValueError, match=diagnostic):
        main(
            argv,
            data_dir=data_dir,
            discover=discover,
            read_commit=lambda: "abc",
        )
    assert not (source / "report.json").exists()
    assert not continued.exists()


def test_report_only_tells_legacy_batches_to_use_their_original_commit(
    tmp_path: Path,
) -> None:
    (tmp_path / "batch.json").write_text(
        json.dumps(
            {
                "commit": "prior-revision",
                "inputs": {},
                "cases": {"fake/power": "cases/001"},
                "adapters": {"fake/power": None},
                "guides": [],
            }
        )
    )
    with pytest.raises(ValueError, match=r"Legacy.*--report-only.*prior-revision"):
        main(["--report-only", "--output", str(tmp_path)], data_dir=tmp_path / "absent")
    assert not (tmp_path / "report.json").exists()


def test_report_only_rejects_a_journal_without_core_fields(
    tmp_path: Path,
    data_dir: Path,
) -> None:
    _write_scenes(data_dir, *sorted(_FREQUENCY))
    output = tmp_path / "original"
    assert (
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            discover=lambda: _inventory(_FREQUENCY, lambda *_: 2.0),
            read_commit=lambda: "abc",
        )
        == 0
    )
    journal = output / "cases/001/row.json"
    row = json.loads(journal.read_text())
    for field in ("materialize_s", "core_s", "core_s_lower_bound"):
        row.pop(field, None)
    journal.write_text(json.dumps(row))
    (output / "report.json").unlink()
    (output / "summary.md").unlink()

    with pytest.raises(ValidationError, match="core_s_lower_bound"):
        main(["--report-only", "--output", str(output)])
    assert not (output / "report.json").exists()
    assert not (output / "summary.md").exists()


@pytest.mark.parametrize(
    "change",
    [
        None,
        "commit",
        "case-inputs.json",
        "board-profile.yaml",
        "ftf-settings.yaml",
        "fixture/result/demo/points/initial/library.yaml",
    ],
)
def test_resume_reuses_only_matching_committed_rows_and_marks_the_first_new_run_cold(
    tmp_path: Path,
    data_dir: Path,
    change: str | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "original"
    output = tmp_path / "continued"

    def original(case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> float:
        if case.adapter != "fake/frequency":
            raise KeyboardInterrupt
        return 6.0

    assert (
        main(
            ["--output", str(source)],
            data_dir=data_dir,
            discover=lambda: _inventory(_ADAPTERS, original),
            read_commit=lambda: "abc",
        )
        == 3
    )
    (source / "report.json").unlink()  # Resume from rows, not the rendered report.
    ran = []

    def continued(case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> float:
        ran.append(case.adapter)
        return 2.0

    commit = "def" if change == "commit" else "abc"
    if change not in {None, "commit"}:
        assert change is not None
        path = data_dir / change
        path.write_bytes(path.read_bytes() + b"\n")
    argv = ["--output", str(output), "--resume-from", str(source), "--select", "fake"]

    def discover() -> Inventory:
        return _inventory(_ADAPTERS, continued)

    if change is not None:
        with pytest.raises(SystemExit) as raised:
            main(
                argv,
                data_dir=data_dir,
                discover=discover,
                read_commit=lambda: commit,
            )
        assert raised.value.code == 2
        assert (
            "commit differs" if change == "commit" else "input SHA-256 differs"
        ) in capsys.readouterr().err
        assert not ran
        assert not output.exists()
        return

    assert (
        main(
            argv,
            data_dir=data_dir,
            discover=discover,
            read_commit=lambda: commit,
        )
        == 0
    )
    assert ran == ["fake/power"]
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    kept, measured = report["rows"]
    assert (kept["adapter"], kept["wall_s"], kept["reused"]) == (
        "fake/frequency",
        6.0,
        True,
    )
    assert Path(kept["case_directory"]) == source / "cases/001"
    assert (measured["adapter"], measured["wall_s"], measured["reused"]) == (
        "fake/power",
        2.0,
        False,
    )
    assert measured["cold_start"] is True
    assert report["coverage"] == {"resolved": 2, "selected": 2, "manifest": 3}


@pytest.mark.parametrize(
    ("selection", "settled_idle", "expected_exit"),
    [("fake", True, 0), ("fake", False, 3), ("fake/frequency", False, 0)],
)
def test_failed_run_is_journaled_before_idle_check_and_never_timed(
    tmp_path: Path,
    data_dir: Path,
    selection: str,
    expected_exit: int,
    *,
    settled_idle: bool,
) -> None:
    output = tmp_path / "output"
    reason = "ValueError('envelope upload exceeded the buffer')"
    failed_at = "2026-11-21T00:00:07+00:00"
    dispatched: list[str] = []

    def check_idle(location: Path) -> bool:
        committed = json.loads((location / "row.json").read_text())
        assert committed["status"] == "run_failed"
        diagnostic = location / committed["failure_record"]
        assert json.loads(diagnostic.read_text())["reason"] == reason
        return settled_idle

    def measure(
        case: WallTimeCase,
        _data_dir: Path,
        telemetry: Path,
        attempt: Callable[[Run | FormErrors], None],
    ) -> None:
        dispatched.append(case.adapter)
        if case.adapter == "fake/frequency":
            attempt(_FailedRun(reason, lambda: check_idle(telemetry.parent)))
        else:
            attempt(_CompletedRun(2.0))

    code = main(
        ["--output", str(output), "--select", selection],
        data_dir=data_dir,
        discover=lambda: Inventory(_ADAPTERS, measure),
        read_commit=lambda: "abc",
    )

    assert code == expected_exit
    assert dispatched == (
        ["fake/frequency", "fake/power"] if settled_idle else ["fake/frequency"]
    )
    report = json.loads((output / "report.json").read_text())
    failed, *following = report["rows"]
    assert failed["status"] == "run_failed"
    assert failed["reason"] == reason
    assert failed["started_at"] == "2026-11-21T00:00:00+00:00"
    assert failed["finished_at"] == failed_at
    assert failed["wall_s"] is None and failed["wall_s_lower_bound"] is None
    assert [row["status"] for row in following] == (
        ["missing_breakdown" if settled_idle else "not_run"]
        if selection == "fake"
        else []
    )
    assert report["coverage"]["resolved"] == (2 if settled_idle else 1)
    summary = (output / "summary.md").read_text()
    assert f"Runs: {int(settled_idle)} timed, 0 timeout, 1 run_failed." in summary
    assert "Core >180 s: —" in summary


def test_summary_groups_identical_diagnostics_and_lists_each_failure_record(
    tmp_path: Path, data_dir: Path
) -> None:
    output = tmp_path / "output"
    shared = "ValueError('upload limit | 8')"
    distinct = "ValueError('upload limit | 16')"

    def measure(
        case: WallTimeCase,
        _data_dir: Path,
        _telemetry: Path,
        attempt: Callable[[Run | FormErrors], None],
    ) -> None:
        reason = distinct if case.adapter == "fake/power" else shared
        attempt(_FailedRun(reason, lambda: True))

    assert (
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            discover=lambda: Inventory(_ADAPTERS, measure),
            read_commit=lambda: "abc",
        )
        == 0
    )

    summary = (output / "summary.md").read_text()
    failures = summary.split("Run failures:\n", 1)[1]
    assert failures.count(shared) == 1
    assert failures.count(distinct) == 1
    groups = ("\n" + failures).split("\n- ")[1:]
    rows = json.loads((output / "report.json").read_text())["rows"]
    for row in rows:
        [group] = [group for group in groups if row["reason"] in group]
        record = Path(row["case_directory"]) / row["failure_record"]
        assert f"`{row['adapter']}`" in group
        assert f"`{record.as_posix()}`" in group
        assert json.loads((output / record).read_text())["reason"] == row["reason"]


def test_resume_keeps_a_failed_run_and_its_original_diagnostic_without_retrying(
    tmp_path: Path, data_dir: Path
) -> None:
    source, resumed = tmp_path / "source", tmp_path / "resumed"

    def measure(
        _case: WallTimeCase,
        _data_dir: Path,
        _telemetry: Path,
        attempt: Callable[[Run | FormErrors], None],
    ) -> None:
        attempt(_FailedRun("ValueError('bad pulse')", lambda: True))

    def run_at(output: Path, discover: Callable[[], Inventory], *options: str) -> int:
        return main(
            ["--output", str(output), "--select", "fake/frequency", *options],
            data_dir=data_dir,
            discover=discover,
            read_commit=lambda: "abc",
        )

    assert run_at(source, lambda: Inventory(_FREQUENCY, measure)) == 0

    def no_retry(*_args: object) -> None:
        pytest.fail("an already resolved Run must not be prepared")

    assert (
        run_at(
            resumed,
            lambda: Inventory(_FREQUENCY, no_retry),
            "--resume-from",
            str(source),
        )
        == 0
    )
    [row] = json.loads((resumed / "report.json").read_text())["rows"]
    assert row["status"] == "run_failed" and row["reused"] is True
    diagnostic = resumed / row["case_directory"] / row["failure_record"]
    assert diagnostic.is_relative_to(source)
    assert json.loads(diagnostic.read_text())["reason"] == "ValueError('bad pulse')"


@pytest.mark.parametrize("seconds", ["179.9", "nan", "inf", "-inf", "2147483.648"])
def test_main_rejects_an_invalid_row_deadline_before_discovery(seconds: str) -> None:
    def unexpected_discovery() -> Inventory:
        pytest.fail("invalid deadline must fail before experiment discovery or Run")

    with pytest.raises(SystemExit) as raised:
        main([f"--row-timeout={seconds}"], discover=unexpected_discovery)
    assert raised.value.code == 2


def test_main_rejects_zero_workers_before_discovery() -> None:
    def unexpected_discovery() -> Inventory:
        pytest.fail("invalid worker count must fail before experiment discovery or Run")

    with pytest.raises(SystemExit) as raised:
        main(["--workers", "0"], discover=unexpected_discovery)
    assert raised.value.code == 2


def test_main_selects_exact_experiment(tmp_path: Path, data_dir: Path) -> None:
    output = tmp_path / "output"

    def measure(case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> float:
        assert case.adapter == "fake/frequency"
        return 5.0

    assert (
        main(
            [
                "--select",
                "fake/frequency",
                "--row-timeout",
                "180",
                "--output",
                str(output),
            ],
            data_dir=data_dir,
            discover=lambda: _inventory(_ADAPTERS, measure),
            read_commit=lambda: "abc",
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert [row["adapter"] for row in report["rows"]] == ["fake/frequency"]


def test_main_selects_scenes_before_adapter_families_and_keeps_same_adapter_runs(
    tmp_path: Path, data_dir: Path
) -> None:
    _write_scenes(
        data_dir,
        {
            "scene_id": "fake",
            "adapter": "fake/frequency",
            "basis": "narrow scout",
            "overrides": [{"path": ["freq", "n"], "value": 3, "basis": "short scan"}],
        },
        {
            "scene_id": "wide",
            "adapter": "fake/frequency",
            "basis": "wide scout",
            "overrides": [{"path": ["freq", "n"], "value": 9, "basis": "longer scan"}],
        },
    )
    seen: list[object] = []

    def measure(case: WallTimeCase, _inputs: Path, _telemetry: Path) -> float:
        seen.append((case.adapter, case.overrides.get(("freq", "n"))))
        return 2.0

    def run(output: Path, *args: str) -> list[tuple[str, str]]:
        assert (
            main(
                ["--output", str(output), *args],
                data_dir=data_dir,
                discover=lambda: _inventory(_ADAPTERS, measure),
                read_commit=lambda: "abc",
            )
            == 0
        )
        report = json.loads((output / "report.json").read_text())
        return [(row["scene_id"], row["adapter"]) for row in report["rows"]]

    source = tmp_path / "exact"
    assert run(source, "--select", "fake") == [("fake", "fake/frequency")]
    assert run(tmp_path / "all", "--resume-from", str(source)) == [
        ("fake", "fake/frequency"),
        ("wide", "fake/frequency"),
    ]
    assert seen == [("fake/frequency", 3), ("fake/frequency", 9)]


def test_stale_scene_is_unmeasured_and_resume_retries_it_after_adapter_returns(
    tmp_path: Path, data_dir: Path
) -> None:
    stale, active = "old-scout", "second-scout"
    _write_scenes(
        data_dir,
        {"scene_id": stale, "adapter": "scan/retired", "basis": "kept workload"},
        {"scene_id": active, "adapter": "scan/current", "basis": "live workload"},
    )
    seen: list[str] = []

    def measure(case: WallTimeCase, _inputs: Path, _telemetry: Path) -> float:
        seen.append(case.adapter)
        return 2.0

    source, continued = tmp_path / "source", tmp_path / "continued"

    def run(output: Path, adapters: tuple[str, ...], *args: str) -> int:
        return main(
            ["--output", str(output), "--select", "scan", *args],
            data_dir=data_dir,
            discover=lambda: _inventory(dict.fromkeys(adapters), measure),
            read_commit=lambda: "abc",
        )

    assert run(source, ("scan/current",)) == 3
    assert seen == ["scan/current"]
    report_path = source / "report.json"
    report = json.loads(report_path.read_text())
    missing, measured = report["rows"]
    assert (missing["scene_id"], missing["adapter"], missing["status"]) == (
        stale,
        "scan/retired",
        "missing_adapter",
    )
    assert "scan/retired" in missing["reason"] and "manifest" in missing["reason"]
    assert missing["started_at"] is None and missing["wall_s"] is None
    assert missing["wall_s_lower_bound"] is None and missing["case_directory"] is None
    assert measured["wall_s"] == 2.0
    assert report["coverage"] == {"resolved": 1, "selected": 2, "manifest": 2}
    report_path.unlink()
    assert (
        main(
            ["--report-only", "--output", str(source)],
            data_dir=tmp_path / "absent",
            discover=lambda: pytest.fail("report-only must not load the registry"),
        )
        == 3
    )
    assert json.loads(report_path.read_text()) == report

    assert (
        run(continued, ("scan/current", "scan/retired"), "--resume-from", str(source))
        == 0
    )
    assert seen == ["scan/current", "scan/retired"]
    retried, reused = json.loads((continued / "report.json").read_text())["rows"]
    assert (retried["scene_id"], retried["reused"]) == (stale, False)
    assert (reused["scene_id"], reused["reused"]) == (active, True)


def test_main_flags_only_unrounded_core_seconds_above_180(
    tmp_path: Path, data_dir: Path
) -> None:
    evolve = {"fake/at": 174.0, "fake/above": 174.0001}
    _write_scenes(data_dir, *evolve)

    def measure(case: WallTimeCase, _data_dir: Path, telemetry: Path) -> float:
        record = {
            "families": {},
            "wall_s": 190,
            "compile_wall_s": 1,
            "evolve_wall_s": evolve[case.adapter],
            "readout_wall_s": 4,
            "materialize_wall_s": 1,
        }
        (telemetry / "routing-test.jsonl").write_text(json.dumps(record) + "\n")
        return 200.0

    output = tmp_path / "output"
    assert (
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            discover=lambda: _inventory(dict.fromkeys(evolve), measure),
            read_commit=lambda: "abc",
        )
        == 0
    )
    table = (output / "summary.md").read_text(encoding="utf-8")
    assert "Core >180 s: `fake/above`\n" in table
    assert "Core 已知至少超標: —\n" in table
    assert "Core 未判定: —\n" in table


def test_main_reports_known_ftf_exclusions_without_trying_run(
    tmp_path: Path, data_dir: Path
) -> None:
    _write_scenes(
        data_dir,
        "fake/frequency",
        {
            "scene_id": "fake/jpa",
            "adapter": "fake/jpa",
            "basis": "pump scan",
            "ftf_exclusion": "FTF has no JPA pump device",
        },
    )

    def measure(case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> float:
        if case.adapter == "fake/jpa":
            pytest.fail("the known FTF-excluded adapter was sent to Run")
        return 4.0

    output = tmp_path / "output"
    assert (
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            discover=lambda: _inventory(_FREQUENCY | {"fake/jpa"}, measure),
            read_commit=lambda: "abc",
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    measured, excluded = report["rows"]
    assert measured["wall_s"] == 4.0
    assert excluded["status"] == "not_runnable"
    assert excluded["reason"] == "FTF has no JPA pump device"
    assert excluded["wall_s"] is None
    assert "FTF has no JPA pump device" in (output / "summary.md").read_text(
        encoding="utf-8"
    )


def test_select_merges_family_with_overlapping_exact_scene(
    tmp_path: Path, data_dir: Path
) -> None:
    _write_scenes(
        data_dir,
        "narrow",
        {
            "scene_id": "wide",
            "adapter": "scan/frequency",
            "basis": "wide scout",
        },
        {
            "scene_id": "survey",
            "adapter": "scan/power",
            "basis": "power map",
        },
    )
    ran: list[str] = []

    def measure(case: WallTimeCase, _inputs: Path, _telemetry: Path) -> float:
        ran.append(case.scene_id)
        return 4.0

    output = tmp_path / "output"
    assert (
        main(
            ["--select", "wide", "--select", "scan", "--output", str(output)],
            data_dir=data_dir,
            discover=lambda: _inventory(
                {"scan/frequency", "scan/power", "narrow"}, measure
            ),
            read_commit=lambda: "abc",
        )
        == 0
    )
    assert ran == ["survey", "wide"]


def test_main_retains_unresolved_form_fields_and_slots_in_their_row(
    tmp_path: Path, data_dir: Path
) -> None:
    _write_scenes(data_dir, "fake/blocked", "fake/frequency")

    def measure(
        case: WallTimeCase, _data_dir: Path, _telemetry: Path
    ) -> float | FormErrors:
        if case.adapter == "fake/blocked":
            return FormErrors(
                {
                    ("drive",): "Missing slot $q.module.drive",
                    ("scan", "stop"): "Field required",
                }
            )
        return 4.0

    output = tmp_path / "output"
    assert (
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            discover=lambda: _inventory(_FREQUENCY | {"fake/blocked"}, measure),
            read_commit=lambda: "abc",
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    blocked, measured = report["rows"]
    assert blocked["status"] == "missing_defaults"
    assert (
        blocked["reason"]
        == "drive: Missing slot $q.module.drive; scan.stop: Field required"
    )
    assert blocked["wall_s"] is None and blocked["evolve_s"] is None
    assert measured["wall_s"] == 4.0


@pytest.mark.parametrize(
    ("failure", "expected"),
    [("setup", OSError), ("read", FileNotFoundError), ("telemetry", ValidationError)],
)
def test_main_propagates_non_experiment_failures(
    tmp_path: Path, data_dir: Path, failure: str, expected: type[Exception]
) -> None:
    _write_scenes(data_dir, *sorted(_FREQUENCY))

    def measure(_case: WallTimeCase, _data_dir: Path, telemetry: Path) -> float:
        if failure == "setup":
            raise OSError("simulation setup failed")
        if failure == "read":
            telemetry.rmdir()
        else:
            (telemetry / "routing-broken.jsonl").write_text(
                "not a record", encoding="utf-8"
            )
        return 4.0

    output = tmp_path / "output"
    with pytest.raises(expected):
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            discover=lambda: _inventory(_FREQUENCY, measure),
            read_commit=lambda: "abc",
        )
    assert not (output / "report.json").exists()


def test_main_keeps_run_wall_when_telemetry_is_missing(
    tmp_path: Path, data_dir: Path
) -> None:
    _write_scenes(data_dir, *sorted(_FREQUENCY))
    output = tmp_path / "output"

    def measure(_case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> float:
        return 3.0

    assert (
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            discover=lambda: _inventory(_FREQUENCY, measure),
            read_commit=lambda: "abc",
        )
        == 0
    )

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    [row] = report["rows"]
    assert row["status"] == "missing_breakdown"
    assert row["wall_s"] == 3.0
    assert row["acquires"] == 0
    assert all(
        row[key] is None
        for key in (
            "compile_s",
            "evolve_s",
            "readout_s",
            "materialize_s",
            "core_s",
            "core_s_lower_bound",
            "remainder_s",
        )
    )
    assert "| fake/frequency | 3.000 | — | — | — | — | — | — | 0 | 分項缺 |" in (
        output / "summary.md"
    ).read_text(encoding="utf-8")
