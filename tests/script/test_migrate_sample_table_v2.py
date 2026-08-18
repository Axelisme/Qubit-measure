"""Tests for the explicit SampleTable v2 migration shell script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "script" / "migrate_sample_table_v2.py"
_spec = importlib.util.spec_from_file_location("migrate_sample_table_v2", _SCRIPT)
assert _spec is not None and _spec.loader is not None
migrate_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate_script)

from zcu_tools.meta_tool import (  # noqa: E402
    DEV_UNIT_COLUMN,
    DEV_VALUE_COLUMN,
    FLUX_COLUMN,
    FLUX_INT_COLUMN,
    FLUX_PERIOD_COLUMN,
    validate_sample_table_v2,
)


def _write_source(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    source = tmp_path / "samples.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    return source


def test_dry_run_prints_plan_and_writes_nothing(tmp_path, capsys) -> None:
    source = _write_source(
        tmp_path,
        [
            {"calibrated mA": -0.007, "T1 (us)": 50.0},
            {"calibrated mA": 0.002, "T1 (us)": 60.0},
        ],
    )
    destination = tmp_path / "v2" / "samples.csv"

    rc = migrate_script.main(
        [
            str(source),
            str(destination),
            "--dev-value-column",
            "calibrated mA",
            "--dev-value-unit",
            "A",
        ]
    )

    assert rc == 0
    assert not destination.exists()
    out = capsys.readouterr().out
    assert "calibrated mA" in out
    assert "A → A (×1)" in out
    assert "total=2, convertible=2" in out
    assert "dry-run" in out
    assert "dev_value, dev_unit" in out


def test_write_creates_migrated_destination(tmp_path) -> None:
    source = _write_source(tmp_path, [{"calibrated mA": 0.0, "calibrated mA2": 1.0}])
    destination = tmp_path / "migrated.csv"

    rc = migrate_script.main(
        [
            str(source),
            str(destination),
            "--dev-value-column",
            "calibrated mA",
            "--dev-value-unit",
            "mA",
            "--write",
        ]
    )

    assert rc == 0
    migrated = pd.read_csv(destination)
    assert list(migrated.columns) == [
        DEV_VALUE_COLUMN,
        DEV_UNIT_COLUMN,
        "calibrated mA2",
    ]
    np.testing.assert_allclose(migrated[DEV_VALUE_COLUMN], [0.0])
    assert list(migrated[DEV_UNIT_COLUMN]) == ["A"]
    validate_sample_table_v2(migrated)


def test_write_with_flux_and_frame(tmp_path) -> None:
    source = _write_source(
        tmp_path,
        [{"Flux": 0.25, "calibrated mA": 1.0}, {"Flux": np.nan, "calibrated mA": 2.0}],
    )
    destination = tmp_path / "migrated.csv"

    rc = migrate_script.main(
        [
            str(source),
            str(destination),
            "--dev-value-column",
            "calibrated mA",
            "--dev-value-unit",
            "mA",
            "--flux-column",
            "Flux",
            "--flux-int",
            "0.0",
            "--flux-period",
            "10000.0",
            "--frame-unit",
            "mA",
            "--write",
        ]
    )

    assert rc == 0
    migrated = pd.read_csv(destination)
    assert list(migrated.columns) == [
        FLUX_COLUMN,
        DEV_VALUE_COLUMN,
        DEV_UNIT_COLUMN,
        FLUX_INT_COLUMN,
        FLUX_PERIOD_COLUMN,
    ]
    np.testing.assert_allclose(migrated[FLUX_COLUMN], [0.25, np.nan], equal_nan=True)
    np.testing.assert_allclose(migrated[DEV_VALUE_COLUMN], [0.001, 0.002])
    np.testing.assert_allclose(migrated[FLUX_INT_COLUMN], [0.0, 0.0])
    np.testing.assert_allclose(migrated[FLUX_PERIOD_COLUMN], [10.0, 10.0])
    validate_sample_table_v2(migrated)


def test_write_refuses_existing_destination(tmp_path, capsys) -> None:
    source = _write_source(tmp_path, [{"calibrated mA": 1.0}])
    destination = tmp_path / "existing.csv"
    original = "pre-existing content\n"
    destination.write_text(original)

    rc = migrate_script.main(
        [
            str(source),
            str(destination),
            "--dev-value-column",
            "calibrated mA",
            "--dev-value-unit",
            "A",
            "--write",
        ]
    )

    assert rc == 1
    assert "no-clobber" in capsys.readouterr().err
    assert destination.read_text() == original


def test_write_does_not_clobber_destination_created_during_publish(
    tmp_path, monkeypatch, capsys
) -> None:
    source = _write_source(tmp_path, [{"calibrated mA": 1.0}])
    destination = tmp_path / "raced.csv"
    competing_content = "created by another writer\n"
    real_link = migrate_script.os.link

    def competing_link(src: Path, dst: Path) -> None:
        destination.write_text(competing_content)
        real_link(src, dst)

    monkeypatch.setattr(migrate_script.os, "link", competing_link)

    rc = migrate_script.main(
        [
            str(source),
            str(destination),
            "--dev-value-column",
            "calibrated mA",
            "--dev-value-unit",
            "A",
            "--write",
        ]
    )

    assert rc == 1
    assert "no-clobber" in capsys.readouterr().err
    assert destination.read_text() == competing_content
    assert list(tmp_path.glob(".*.tmp")) == []
    assert pd.read_csv(source)["calibrated mA"].tolist() == [1.0]


def test_write_refuses_source_equal_destination(tmp_path, capsys) -> None:
    source = _write_source(tmp_path, [{"calibrated mA": 1.0}])

    rc = migrate_script.main(
        [
            str(source),
            str(source),
            "--dev-value-column",
            "calibrated mA",
            "--dev-value-unit",
            "A",
            "--write",
        ]
    )

    assert rc == 1
    assert "must differ from source" in capsys.readouterr().err
    assert pd.read_csv(source)["calibrated mA"].tolist() == [1.0]


def test_failed_write_preserves_source_and_leaves_no_partial_file(
    tmp_path, monkeypatch, capsys
) -> None:
    source = _write_source(tmp_path, [{"calibrated mA": 1.0}])
    destination = tmp_path / "migrated.csv"

    def failing_link(src: Path, dst: Path) -> None:
        raise OSError("injected link failure")

    monkeypatch.setattr(migrate_script.os, "link", failing_link)

    rc = migrate_script.main(
        [
            str(source),
            str(destination),
            "--dev-value-column",
            "calibrated mA",
            "--dev-value-unit",
            "A",
            "--write",
        ]
    )

    assert rc == 1
    assert "failed to write destination" in capsys.readouterr().err
    assert not destination.exists()
    assert list(tmp_path.glob(".*.tmp")) == []
    assert pd.read_csv(source)["calibrated mA"].tolist() == [1.0]


def test_invalid_output_fails_without_writing(tmp_path, capsys) -> None:
    source = _write_source(tmp_path, [{"calibrated mA": "not-a-number"}])
    destination = tmp_path / "migrated.csv"

    rc = migrate_script.main(
        [
            str(source),
            str(destination),
            "--dev-value-column",
            "calibrated mA",
            "--dev-value-unit",
            "A",
            "--write",
        ]
    )

    assert rc == 1
    assert "migration failed" in capsys.readouterr().err
    assert not destination.exists()


def test_frame_group_requires_all_three_parts(tmp_path) -> None:
    source = _write_source(tmp_path, [{"calibrated mA": 1.0}])
    with pytest.raises(SystemExit) as exc:
        migrate_script.main(
            [
                str(source),
                str(tmp_path / "out.csv"),
                "--dev-value-column",
                "calibrated mA",
                "--dev-value-unit",
                "A",
                "--flux-int",
                "0.0",
            ]
        )
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("flux_int", "flux_period"),
    [("nan", "1.0"), ("0.0", "0.0"), ("0.0", "inf")],
)
def test_invalid_frame_value_fails_cleanly(
    tmp_path, capsys, flux_int: str, flux_period: str
) -> None:
    source = _write_source(tmp_path, [{"calibrated mA": 1.0}])
    destination = tmp_path / "out.csv"

    rc = migrate_script.main(
        [
            str(source),
            str(destination),
            "--dev-value-column",
            "calibrated mA",
            "--dev-value-unit",
            "A",
            "--flux-int",
            flux_int,
            "--flux-period",
            flux_period,
            "--frame-unit",
            "A",
        ]
    )

    assert rc == 1
    assert "migration failed" in capsys.readouterr().err
    assert not destination.exists()


def test_dry_run_reports_partial_convertible_count(tmp_path, capsys) -> None:
    source = _write_source(
        tmp_path,
        [{"calibrated mA": 1.0}, {"calibrated mA": np.nan}, {"calibrated mA": 3.0}],
    )
    destination = tmp_path / "migrated.csv"

    rc = migrate_script.main(
        [
            str(source),
            str(destination),
            "--dev-value-column",
            "calibrated mA",
            "--dev-value-unit",
            "mA",
        ]
    )

    assert rc == 0
    out = capsys.readouterr()
    assert "total=3, convertible=2" in out.out
    assert "write would fail" in out.err
    assert not destination.exists()


def test_dry_run_does_not_count_boolean_coordinate(tmp_path, capsys) -> None:
    source = _write_source(tmp_path, [{"calibrated mA": True}])
    destination = tmp_path / "migrated.csv"

    rc = migrate_script.main(
        [
            str(source),
            str(destination),
            "--dev-value-column",
            "calibrated mA",
            "--dev-value-unit",
            "mA",
        ]
    )

    assert rc == 0
    out = capsys.readouterr()
    assert "total=1, convertible=0" in out.out
    assert "write would fail" in out.err
    assert not destination.exists()


def test_invalid_dev_value_unit_rejected_by_argparse(tmp_path) -> None:
    source = _write_source(tmp_path, [{"calibrated mA": 1.0}])
    with pytest.raises(SystemExit) as exc:
        migrate_script.main(
            [
                str(source),
                str(tmp_path / "out.csv"),
                "--dev-value-column",
                "calibrated mA",
                "--dev-value-unit",
                "kA",
            ]
        )
    assert exc.value.code == 2
