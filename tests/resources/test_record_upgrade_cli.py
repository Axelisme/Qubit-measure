"""Record scope of the long-lived upgrade CLI, without lab registration."""

from pathlib import Path

import pytest
from zcu_tools.resources.parameter_upgrade import main


def test_record_cli_reports_operator_attribution_lines_and_isolates_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "project" / "result"
    data_dir.mkdir(parents=True)
    good, bad = data_dir / "good.jsonl", data_dir / "bad.jsonl"
    original = b'{"run_id":"a","dev_value":0.5,"dev_unit":"mA"}\n'
    partial = b'{"run_id":"b","dev_value":1}\n'
    corrupt = b'{"run_id":"c","dev_value":false,"dev_unit":"V"}\n'
    good.write_bytes(original)
    bad.write_bytes(original + partial + corrupt)
    argv = ["--scope", "records", "--device-name", " Bias ", str(bad), str(good)]
    assert main(argv) == 1
    output = capsys.readouterr().out
    assert "operator-declared device: ' Bias '" in output
    assert f"{bad}: rejected" in output
    assert "line 1: old" in output
    assert "line 2: partial" in output and "dev_unit" in output
    assert "line 3: corrupt" in output and "dev_value" in output
    assert f"{good}: dry-run" in output
    assert good.read_bytes() == original
    assert bad.read_bytes() == original + partial + corrupt
    assert set(data_dir.iterdir()) == {good, bad}
    backup_root = tmp_path / "archive"
    assert main([*argv, "--apply", "--backup-root", str(backup_root)]) == 1
    output = capsys.readouterr().out
    assert f"{good}: applied" in output
    assert "backup:" in output
    assert bad.read_bytes() == original + partial + corrupt
    (backup,) = (backup_root / "result").glob("good.jsonl.*.bak")
    assert backup.read_bytes() == original
    assert (
        main(["--scope", "records", "--device-name", " Bias ", str(good), "--apply"])
        == 0
    )
    assert f"{good}: skipped" in capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [
        ["--scope", "records"],
        ["--scope", "records", "--device-name", ""],
        ["--scope", "records", "--device-name", " \t"],
        ["--scope", "records", "--device-name", "a.b"],
        ["--device-name", "bias"],
    ],
)
def test_invalid_record_scope_arguments_do_not_write(
    tmp_path: Path, args: list[str]
) -> None:
    path = tmp_path / "untouched"
    with pytest.raises(SystemExit) as error:
        main([*args, str(path)])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []
