"""The wall-time CLI reports Run facts without executing a real experiment."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from zcu_tools.gui.form import FormErrors

from tools.experiment_wall_time import Inventory, RunTiming, WallTimeCase, main

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
            discover=lambda: Inventory(_FREQUENCY, [_GUIDE], measure),
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
        "acquires": 2,
        "compile_s": 1.0,
        "evolve_s": 2.75,
        "readout_s": 1.5,
        "remainder_s": 1.25,
        "status": "ok",
        "reason": None,
    }
    table = (output / "summary.md").read_text(encoding="utf-8")
    assert "| fake/frequency | 8.000 | 1.000 | 2.750 | 1.500 | 1.250 | 2 |" in table
    assert f"`{_GUIDE}`" in table


def test_main_selects_exact_experiment(tmp_path: Path, data_dir: Path) -> None:
    output = tmp_path / "output"

    def measure(case: WallTimeCase, _data_dir: Path, _telemetry: Path) -> RunTiming:
        assert case.adapter == "fake/frequency"
        return RunTiming("start", "finish", 5.0)

    assert (
        main(
            ["--select", "fake/frequency", "--output", str(output)],
            data_dir=data_dir,
            discover=lambda: Inventory(_ADAPTERS, [_GUIDE], measure),
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
            discover=lambda: Inventory(
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
            discover=lambda: Inventory(dict.fromkeys(adapters), [], measure),
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
            discover=lambda: Inventory(
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
            discover=lambda: Inventory(_ADAPTERS, [_GUIDE], measure),
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
            discover=lambda: Inventory(
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
            discover=lambda: Inventory(_FREQUENCY, [_GUIDE], measure),
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
            discover=lambda: Inventory(_FREQUENCY, [_GUIDE], measure),
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
