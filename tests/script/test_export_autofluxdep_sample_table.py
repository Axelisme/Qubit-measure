"""Tests for the autofluxdep SampleTable export script wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "script"
    / "export_autofluxdep_sample_table.py"
)
_spec = importlib.util.spec_from_file_location(
    "export_autofluxdep_sample_table", _SCRIPT
)
assert _spec is not None and _spec.loader is not None
export_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_script)


class _Result:
    def __init__(self, path: str, row_count: int) -> None:
        self.path = path
        self.row_count = row_count


def test_main_exports_run_path_and_prints_result(monkeypatch, capsys, tmp_path):
    calls: list[tuple[str, str | None, bool]] = []
    output = tmp_path / "samples.csv"

    def fake_export(
        run: str,
        filepath: str | None = None,
        *,
        append: bool = True,
    ) -> _Result:
        calls.append((run, filepath, append))
        return _Result(str(output), 2)

    monkeypatch.setattr(
        export_script,
        "export_sample_table_from_artifact",
        fake_export,
    )

    rc = export_script.main(["run-dir", "--output", str(output)])

    assert rc == 0
    assert calls == [("run-dir", str(output), True)]
    assert capsys.readouterr().out == f"Exported 2 sample row(s) to {output}\n"


def test_main_quiet_suppresses_success_message(monkeypatch, capsys):
    def fake_export(
        run: str,
        filepath: str | None = None,
        *,
        append: bool = True,
    ) -> _Result:
        assert append is True
        return _Result("samples.csv", 1)

    monkeypatch.setattr(
        export_script,
        "export_sample_table_from_artifact",
        fake_export,
    )

    rc = export_script.main(["manifest.json", "--quiet"])

    assert rc == 0
    assert capsys.readouterr().out == ""


def test_main_overwrite_disables_append(monkeypatch, tmp_path):
    calls: list[tuple[str, str | None, bool]] = []
    output = tmp_path / "samples.csv"

    def fake_export(
        run: str,
        filepath: str | None = None,
        *,
        append: bool = True,
    ) -> _Result:
        calls.append((run, filepath, append))
        return _Result(str(output), 1)

    monkeypatch.setattr(
        export_script,
        "export_sample_table_from_artifact",
        fake_export,
    )

    rc = export_script.main(["run-dir", "--output", str(output), "--overwrite"])

    assert rc == 0
    assert calls == [("run-dir", str(output), False)]
