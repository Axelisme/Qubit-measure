"""The wall-time CLI reports Run facts without executing a real experiment."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError
from zcu_tools.gui.form import FormErrors

from tools import experiment_wall_time
from tools.experiment_wall_time import Inventory, RunTiming, WallTimeCase, main

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
    def __init__(self, timing: RunTiming) -> None:
        self.record = timing

    def start(self) -> None:
        pass

    def wait_until(
        self, deadline: float, _interrupted: Callable[[], bool], *, idle: bool = False
    ) -> float:
        return deadline

    def cancel(self) -> None:
        pytest.fail("a completed Run should not be cancelled")

    def timing(self, _wall_s: float) -> RunTiming:
        return self.record


def _inventory(
    adapters: Mapping[str, str | None],
    guides: Sequence[str],
    measure: Callable[[WallTimeCase, Path, Path], RunTiming | FormErrors],
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


def test_main_records_run_and_sums_acquires_from_its_telemetry(
    tmp_path: Path, data_dir: Path
) -> None:
    output = tmp_path / "output"

    def measure(case: WallTimeCase, _data_dir: Path, telemetry: Path) -> RunTiming:
        assert case.adapter == "fake/frequency"
        assert (case.entry, case.point) == ("demo", "initial")
        records = [
            {
                "families": {},
                "complete": True,
                "wall_s": 4.5,
                "compile_wall_s": 0.5,
                "evolve_wall_s": 2.0,
                "readout_wall_s": 1.0,
            },
            {
                "families": {},
                "complete": True,
                "wall_s": 2.0,
                "compile_wall_s": 0.5,
                "evolve_wall_s": 0.75,
                "readout_wall_s": 0.5,
            },
        ]
        for index, record in enumerate(records):
            (telemetry / f"routing-{index}.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
        return RunTiming(
            "2026-11-21T00:00:00+00:00",
            "2026-11-21T00:00:08+00:00",
            8.0,
        )

    commit = "0123456789abcdef0123456789abcdef01234567"
    assert (
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            discover=lambda: _inventory(_FREQUENCY, [_GUIDE], measure),
            read_commit=lambda: commit,
        )
        == 0
    )

    report_path = output / "report.json"
    assert report_path.exists(), "successful Run must leave a JSON report"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["commit"] == commit
    input_hashes = report["inputs"]
    assert set(input_hashes) == {
        "ftf-settings.yaml",
        "board-profile.yaml",
        "case-inputs.json",
        "overrides.json",
        "fixture/result/demo/points/initial/parameters.yaml",
    }
    assert input_hashes["fixture/result/demo/points/initial/parameters.yaml"] == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    [row] = report["rows"]
    assert row == {
        "adapter": "fake/frequency",
        "guide": _GUIDE,
        "case_directory": "cases/001",
        "started_at": "2026-11-21T00:00:00+00:00",
        "finished_at": "2026-11-21T00:00:08+00:00",
        "wall_s": 8.0,
        "wall_s_lower_bound": None,
        "breakdown": "complete",
        "acquires": 2,
        "compile_s": 1.0,
        "evolve_s": 2.75,
        "readout_s": 1.5,
        "remainder_s": 1.25,
        "status": "ok",
        "reason": None,
        "cold_start": True,
        "reused": False,
    }
    table = (output / "summary.md").read_text(encoding="utf-8")
    assert "| fake/frequency | 8.000 | 1.000 | 2.750 | 1.500 | 1.250 | 2 |" in table
    assert f"`{_GUIDE}`" in table


def test_main_keeps_committed_rows_when_interrupted_before_the_next_run(
    tmp_path: Path, data_dir: Path
) -> None:
    output = tmp_path / "output"

    def measure(case: WallTimeCase, _data_dir: Path, telemetry: Path) -> RunTiming:
        if case.adapter == "fake/power":
            committed = telemetry.parent.parent / "001" / "row.json"
            assert committed.is_file(), "previous row must already be on disk"
            raise KeyboardInterrupt
        assert case.adapter == "fake/frequency"
        return RunTiming("start", "finish", 5.0)

    assert (
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            discover=lambda: _inventory(_ADAPTERS, [_GUIDE], measure),
            read_commit=lambda: "abc",
        )
        == 130
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert [(row["adapter"], row["status"]) for row in report["rows"]] == [
        ("fake/frequency", "missing_breakdown"),
        ("fake/power", "not_run"),
        ("other", "not_run"),
    ]
    assert report["coverage"]["resolved"] == 1
    assert report["coverage"]["selected"] == 3


def test_interrupted_run_has_a_persisted_start_and_a_monotonic_lower_bound(
    tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    now = 100.0
    # The CLI's monotonic time source is not an argv option.
    monkeypatch.setattr(experiment_wall_time, "perf_counter", lambda: now)

    class InterruptedRun:
        def start(self) -> None:
            start = json.loads((output / "cases/001/start.json").read_text())
            assert start["monotonic_s"] == 100.0
            assert start["started_at"]

        def wait_until(
            self,
            _deadline: float,
            _interrupted: Callable[[], bool],
            *,
            idle: bool = False,
        ) -> float | None:
            nonlocal now
            now = 142.5
            raise KeyboardInterrupt

        def cancel(self) -> None:
            nonlocal now
            now = 182.5

        def timing(self, _wall_s: float) -> RunTiming:
            pytest.fail("interruption cannot publish a completed RunRecord")

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
        == 130
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    [row] = report["rows"]
    assert row["status"] == "incomplete"
    assert row["wall_s"] is None
    assert row["wall_s_lower_bound"] == 42.5
    assert now == 182.5, "the interrupted Run must be cancelled"
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
        ) -> float:
            raise OSError("batch vanished after dispatch")

    def measure(
        case: WallTimeCase,
        _data_dir: Path,
        _telemetry: Path,
        attempt: Callable[[Run | FormErrors], None],
    ) -> None:
        timing = RunTiming("start", "finish", 6.0)
        attempt(
            LostRun(timing) if case.adapter == "fake/power" else _CompletedRun(timing)
        )

    with pytest.raises(OSError, match="batch vanished"):
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            discover=lambda: Inventory(_ADAPTERS, [_GUIDE], measure),
            read_commit=lambda: "abc",
        )
    assert not (output / "report.json").exists()
    start = json.loads((output / "cases/002/start.json").read_text(encoding="utf-8"))
    end = datetime.fromisoformat(start["started_at"]) + timedelta(seconds=25)

    assert (
        main(
            ["--report-only", "--output", str(output), "--ended-at", end.isoformat()],
            discover=lambda: pytest.fail("report-only must not discover experiments"),
            read_commit=lambda: pytest.fail(
                "report-only must use the batch's revision"
            ),
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["commit"] == "abc"
    done, incomplete, unstarted = report["rows"]
    assert done["wall_s"] == 6.0
    assert incomplete["status"] == "incomplete"
    assert incomplete["wall_s_lower_bound"] == 25.0
    assert unstarted["status"] == "not_run"
    assert report["coverage"] == {"resolved": 1, "selected": 3, "registered": 3}


@pytest.mark.parametrize("change", [None, "commit", "input"])
def test_resume_reuses_only_matching_committed_rows_and_marks_the_first_new_run_cold(
    tmp_path: Path,
    data_dir: Path,
    change: str | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "original"
    output = tmp_path / "continued"

    def original(case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> RunTiming:
        if case.adapter != "fake/frequency":
            raise KeyboardInterrupt
        return RunTiming("start-original", "finish-original", 6.0)

    assert (
        main(
            ["--output", str(source)],
            data_dir=data_dir,
            discover=lambda: _inventory(_ADAPTERS, [_GUIDE], original),
            read_commit=lambda: "abc",
        )
        == 130
    )
    (source / "report.json").unlink()  # Resume from rows, not the rendered report.
    ran = []

    def continued(case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> RunTiming:
        ran.append(case.adapter)
        return RunTiming("start-new", "finish-new", 2.0)

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


@pytest.mark.parametrize("settled_idle", [True, False])
def test_timeout_dumps_before_cancelling_and_requires_idle_to_continue(
    tmp_path: Path,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    settled_idle: bool,
) -> None:
    output = tmp_path / "output"
    now = 100.0
    events: list[str] = []
    # The CLI's monotonic time source is not an argv option.
    monkeypatch.setattr(experiment_wall_time, "perf_counter", lambda: now)

    class DeadlineRun:
        def __init__(self, telemetry: Path) -> None:
            self.telemetry = telemetry

        def start(self) -> None:
            events.append("dispatch")
            if settled_idle:
                (self.telemetry / "routing-fake.jsonl").write_text(
                    '{"families":{},"wall_s":5,"compile_wall_s":1,'
                    '"evolve_wall_s":2,"readout_wall_s":1}\n',
                    encoding="utf-8",
                )

        def wait_until(
            self,
            deadline: float,
            _interrupted: Callable[[], bool],
            *,
            idle: bool = False,
        ) -> float | None:
            nonlocal now
            if events[-1] == "cancel":
                assert idle, "a terminal Run alone does not permit the next dispatch"
                assert deadline == 310.0
                now = 281.0 if settled_idle else deadline
                return now if settled_idle else None
            assert deadline == 280.0
            now = deadline
            return None

        def cancel(self) -> None:
            dump = self.telemetry.parent / "timeout-stack.txt"
            assert "(most recent call first)" in dump.read_text(encoding="utf-8")
            events.append("cancel")

        def timing(self, _wall_s: float) -> RunTiming:
            pytest.fail("a timed-out Run must not report an exact duration")

    def measure(
        case: WallTimeCase,
        _data_dir: Path,
        telemetry: Path,
        attempt: Callable[[Run | FormErrors], None],
    ) -> None:
        if case.adapter == "fake/frequency":
            attempt(DeadlineRun(telemetry))
        else:
            assert settled_idle, (
                "must not prepare another case if Controller failed to settle"
            )
            events.append("next")
            attempt(_CompletedRun(RunTiming("start", "finish", 2.0)))

    assert main(
        ["--output", str(output), "--select", "fake", "--row-timeout", "180"],
        data_dir=data_dir,
        discover=lambda: Inventory(_ADAPTERS, [_GUIDE], measure),
        read_commit=lambda: "abc",
    ) == (0 if settled_idle else 124)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    timeout, following = report["rows"]
    assert timeout["status"] == "timeout"
    assert timeout["wall_s"] is None and timeout["wall_s_lower_bound"] == 180.0
    assert timeout["breakdown"] == ("partial" if settled_idle else "missing")
    assert timeout["evolve_s"] == (2.0 if settled_idle else None)
    assert following["status"] == ("missing_breakdown" if settled_idle else "not_run")
    assert events == (
        ["dispatch", "cancel", "next"] if settled_idle else ["dispatch", "cancel"]
    )
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "Run >180 s: `fake/frequency`" in summary
    assert ">180.0" in summary


@pytest.mark.parametrize("seconds", ["179.9", "nan", "inf", "-inf", "2147483.648"])
def test_main_rejects_an_invalid_row_deadline_before_discovery(seconds: str) -> None:
    def unexpected_discovery() -> Inventory:
        pytest.fail("invalid deadline must fail before experiment discovery or Run")

    with pytest.raises(SystemExit) as raised:
        main([f"--row-timeout={seconds}"], discover=unexpected_discovery)
    assert raised.value.code == 2


def test_main_selects_exact_experiment(tmp_path: Path, data_dir: Path) -> None:
    output = tmp_path / "output"

    def measure(case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> RunTiming:
        assert case.adapter == "fake/frequency"
        return RunTiming("start", "finish", 5.0)

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

    def measure(_case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> RunTiming:
        return RunTiming("start", "finish", 5.0)

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

    def measure(case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> RunTiming:
        return RunTiming("start", "finish", adapters[case.adapter])

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

    def measure(case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> RunTiming:
        if case.adapter == "fake/jpa":
            pytest.fail("the known FTF-excluded adapter was sent to Run")
        return RunTiming("start", "finish", 4.0)

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

    def measure(case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> RunTiming:
        if case.adapter == "fake/frequency":
            assert case.overrides == {("tones", 0, "frequency"): 23.5}
        else:
            assert case.overrides == {}
        return RunTiming("start", "finish", 4.0)

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
    ) -> RunTiming | FormErrors:
        if case.adapter == "fake/blocked":
            return FormErrors(
                {
                    ("drive",): "Missing slot $q.module.drive",
                    ("scan", "stop"): "Field required",
                }
            )
        return RunTiming("start", "finish", 4.0)

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
    def measure(_case: WallTimeCase, _data_dir: Path, telemetry: Path) -> RunTiming:
        if failure == "setup":
            raise OSError("simulation setup failed")
        if failure == "read":
            telemetry.rmdir()
        else:
            (telemetry / "routing-broken.jsonl").write_text(
                "not a record", encoding="utf-8"
            )
        return RunTiming("start", "finish", 4.0)

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

    def measure(_case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> RunTiming:
        return RunTiming("2026-11-21T00:00:00+00:00", "2026-11-21T00:00:03+00:00", 3.0)

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
