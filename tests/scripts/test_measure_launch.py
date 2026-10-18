"""Measure launch rejects invalid data roots before assembling the GUI."""

from pathlib import Path

import pytest

from scripts.run_measure_gui import main


@pytest.mark.parametrize("kind", ["relative", "missing", "file", "empty", "nul"])
def test_invalid_project_root_exits_before_gui_assembly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    file_path = tmp_path / "plain-file"
    file_path.write_text("data", encoding="utf-8")
    paths = {
        "relative": "relative-root",
        "missing": str(tmp_path / "missing"),
        "file": str(file_path),
        "empty": "",
        "nul": str(tmp_path) + "\0",
    }
    reason = "absolute path" if kind in {"relative", "empty"} else "existing directory"
    with pytest.raises(SystemExit) as failure:
        main(["--project-root", paths[kind]])
    assert failure.value.code != 0
    error = capsys.readouterr().err.split("error:")[-1].lower()
    assert reason in error
