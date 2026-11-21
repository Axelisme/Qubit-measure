"""The wall-time CLI reports Run facts without executing a real experiment."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
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

_GUIDE = "zcu_lab/v2/fake/frequency/guide.md"
_FREQUENCY = {"fake/frequency": _GUIDE}
_ADAPTERS = {**_FREQUENCY, "fake/power": None, "other": None}


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Supply minimal hashed inputs for tests which replace the Run edge."""
    data_dir = tmp_path / "inputs"
    fixture = data_dir / "fixture" / "result" / "demo" / "points" / "initial"
    fixture.mkdir(parents=True)
    (fixture / "parameters.yaml").write_bytes(b"abc")
    (data_dir / "ftf-settings.yaml").write_bytes(b"settings")
    (data_dir / "board-profile.yaml").write_bytes(b"profile")
    (data_dir / "overrides.json").write_bytes(b"[]")
    (data_dir / "case-inputs.json").write_text(
        json.dumps(
            {
                "entry": "demo",
                "point": "initial",
            }
        ),
        encoding="utf-8",
    )
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
    adapters: Mapping[str, str | None],
    guides: Sequence[str],
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

    return Inventory(adapters, guides, prepare)


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
            discover=lambda: _inventory(_ADAPTERS, [_GUIDE], measure),
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
            discover=lambda: Inventory(_FREQUENCY, [_GUIDE], measure),
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
            discover=lambda: Inventory(_ADAPTERS, [_GUIDE], measure),
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
    assert report["coverage"] == {"resolved": 1, "selected": 3, "registered": 3}
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "| fake/power | — |" in summary
    assert "| other | — |" in summary


@pytest.mark.parametrize("recover", ["report-only", "resume-from"])
@pytest.mark.parametrize(
    ("corruption", "diagnostic"),
    [
        (None, "Invalid JSON"),
        ({"adapter": "other"}, "adapter"),
        ({"status": "ok", "wall_s": None}, "wall_s"),
        ({"status": "missing_breakdown", "wall_s": None}, "wall_s"),
        ({"wall_s": -1.0}, "wall_s"),
        ({"wall_s": float("inf")}, "wall_s"),
        ({"wall_s": True}, "wall_s"),
    ],
)
def test_invalid_row_journal_is_a_tool_error_not_a_usage_error(
    tmp_path: Path,
    data_dir: Path,
    recover: str,
    corruption: dict[str, object] | None,
    diagnostic: str,
) -> None:
    source = tmp_path / "original"

    def measure(_case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> float:
        return 2.0

    def discover() -> Inventory:
        return _inventory(_FREQUENCY, [_GUIDE], measure)

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
        main(argv, data_dir=data_dir, discover=discover, read_commit=lambda: "abc")
    assert not (source / "report.json").exists()
    assert not continued.exists()


@pytest.mark.parametrize("change", [None, "commit", "input"])
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
            discover=lambda: _inventory(_ADAPTERS, [_GUIDE], original),
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
    if change == "input":
        (data_dir / "board-profile.yaml").write_bytes(b"different profile")
    argv = ["--output", str(output), "--resume-from", str(source), "--select", "fake"]

    def discover() -> Inventory:
        return _inventory(_ADAPTERS, [_GUIDE], continued)

    if change is not None:
        with pytest.raises(SystemExit) as raised:
            main(argv, data_dir=data_dir, discover=discover, read_commit=lambda: commit)
        assert raised.value.code == 2
        assert (
            "commit differs" if change == "commit" else "input SHA-256 differs"
        ) in capsys.readouterr().err
        assert not ran
        assert not output.exists()
        return

    assert (
        main(argv, data_dir=data_dir, discover=discover, read_commit=lambda: commit)
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
    assert report["coverage"] == {"resolved": 2, "selected": 2, "registered": 3}


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
        discover=lambda: Inventory(_ADAPTERS, [_GUIDE], measure),
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
    assert "Run >180 s: —" in summary


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
            discover=lambda: Inventory(_ADAPTERS, [_GUIDE], measure),
            read_commit=lambda: "abc",
        )
        == 0
    )

    summary = (output / "summary.md").read_text()
    failures = summary.split("Run failures:\n", 1)[1].split("\nAdapters", 1)[0]
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

    assert run_at(source, lambda: Inventory(_FREQUENCY, [_GUIDE], measure)) == 0

    def no_retry(*_args: object) -> None:
        pytest.fail("an already resolved Run must not be prepared")

    assert (
        run_at(
            resumed,
            lambda: Inventory(_FREQUENCY, [_GUIDE], no_retry),
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
            discover=lambda: _inventory(_ADAPTERS, [_GUIDE], measure),
            read_commit=lambda: "abc",
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert [row["adapter"] for row in report["rows"]] == ["fake/frequency"]


def test_main_reports_every_adapter_and_both_guide_gaps(
    tmp_path: Path, data_dir: Path
) -> None:
    variant_guide = "zcu_lab/v2/fake/frequency/guide.variant.md"
    unmatched_guide = "zcu_lab/v2/fake/variant/guide.md"
    missing_guide = "zcu_lab/v2/fake/outside/notes.md"
    adapters = {
        "fake/no-guide": None,
        "fake/variant": variant_guide,
        "fake/outside": missing_guide,
        "fake/frequency": _GUIDE,
    }

    def measure(_case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> float:
        return 5.0

    output = tmp_path / "output"
    assert (
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            discover=lambda: _inventory(
                adapters, [_GUIDE, variant_guide, unmatched_guide], measure
            ),
            read_commit=lambda: "abc",
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert [(row["adapter"], row["guide"]) for row in report["rows"]] == [
        ("fake/frequency", _GUIDE),
        ("fake/no-guide", None),
        ("fake/outside", missing_guide),
        ("fake/variant", variant_guide),
    ]
    assert report["adapters_without_guides"] == ["fake/no-guide"]
    assert report["declared_guides_not_in_inventory"] == {"fake/outside": missing_guide}
    assert report["guides_without_adapters"] == [unmatched_guide]
    table = (output / "summary.md").read_text(encoding="utf-8")
    assert "Adapters without declared guides: `fake/no-guide`" in table
    assert (
        f"Declared guides not in inventory: `fake/outside` → `{missing_guide}`" in table
    )
    assert f"Guides without adapters: `{unmatched_guide}`" in table


def test_main_flags_only_unrounded_run_seconds_above_180(
    tmp_path: Path, data_dir: Path
) -> None:
    adapters = {"fake/at": 180.0, "fake/above": 180.0001}

    def measure(case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> float:
        return adapters[case.adapter]

    output = tmp_path / "output"
    assert (
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            discover=lambda: _inventory(dict.fromkeys(adapters), [], measure),
            read_commit=lambda: "abc",
        )
        == 0
    )
    table = (output / "summary.md").read_text(encoding="utf-8")
    assert "Run >180 s: `fake/above`" in table
    assert "Run >180 s: `fake/above`, `fake/at`" not in table


def test_main_reports_known_ftf_exclusions_without_trying_run(
    tmp_path: Path, data_dir: Path
) -> None:
    fixture = data_dir / "case-inputs.json"
    cfg = json.loads(fixture.read_text(encoding="utf-8"))
    cfg["ftf_exclusions"] = {"fake/jpa": "FTF has no JPA pump device"}
    fixture.write_text(json.dumps(cfg), encoding="utf-8")

    def measure(case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> float:
        if case.adapter == "fake/jpa":
            pytest.fail("the known FTF-excluded adapter was sent to Run")
        return 4.0

    output = tmp_path / "output"
    assert (
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            discover=lambda: _inventory(
                {**_FREQUENCY, "fake/jpa": None}, [_GUIDE], measure
            ),
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


def test_main_routes_overrides_only_to_the_named_adapter(
    tmp_path: Path, data_dir: Path
) -> None:
    path = data_dir / "overrides.json"
    path.write_text(
        '[{"adapter":"fake/frequency","path":["tones",0,"frequency"],'
        '"value":23.5,"basis":"Independently measured carrier"}]',
        encoding="utf-8",
    )

    def measure(case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> float:
        if case.adapter == "fake/frequency":
            assert case.overrides == {("tones", 0, "frequency"): 23.5}
        else:
            assert case.overrides == {}
        return 4.0

    output = tmp_path / "output"
    assert (
        main(
            ["--select", "fake", "--select", "fake/frequency", "--output", str(output)],
            data_dir=data_dir,
            discover=lambda: _inventory(_ADAPTERS, [_GUIDE], measure),
            read_commit=lambda: "abc",
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert [row["adapter"] for row in report["rows"]] == [
        "fake/frequency",
        "fake/power",
    ]
    assert (
        report["inputs"]["overrides.json"]
        == "fcf0ea2e8aae15515313d46bd3564c3d653515603f74d5adc17214f9ef58f18d"
    )


def test_main_retains_unresolved_form_fields_and_slots_in_their_row(
    tmp_path: Path, data_dir: Path
) -> None:
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
            discover=lambda: _inventory(
                {**_FREQUENCY, "fake/blocked": None}, [_GUIDE], measure
            ),
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
            discover=lambda: _inventory(_FREQUENCY, [_GUIDE], measure),
            read_commit=lambda: "abc",
        )
    assert not (output / "report.json").exists()


def test_main_keeps_run_wall_when_telemetry_is_missing(
    tmp_path: Path, data_dir: Path
) -> None:
    output = tmp_path / "output"

    def measure(_case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> float:
        return 3.0

    assert (
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            discover=lambda: _inventory(_FREQUENCY, [_GUIDE], measure),
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
        for key in ("compile_s", "evolve_s", "readout_s", "remainder_s")
    )
    assert "| fake/frequency | 3.000 | — | — | — | — | 0 | 分項缺 |" in (
        output / "summary.md"
    ).read_text(encoding="utf-8")
