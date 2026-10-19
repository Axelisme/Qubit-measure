from pathlib import Path

import pytest
from zcu_tools.utils.result_transfer.cli import build_copy_arguments


def _build(action: str, entries: list[str], *, dry_run: bool = False) -> list[str]:
    return build_copy_arguments(
        "lab:results",
        Path("/repo/result"),
        entries,
        action=action,
        jobs=8,
        dry_run=dry_run,
    )


def _filters(command: list[str]) -> list[str]:
    return [command[i + 1] for i, arg in enumerate(command) if arg == "--filter"]


def test_upload_sends_all_entries_in_one_command():
    command = _build("upload", ["Si001", "Si002"])

    assert command[:3] == ["copy", "/repo/result", "lab:results"]
    assert _filters(command) == ["- *.lock", "+ /Si001/**", "+ /Si002/**", "- **"]
    assert "--fast-list" in command
    assert "--create-empty-src-dirs" not in command
    assert command[command.index("--transfers") + 1] == "8"
    assert "--dry-run" not in command


def test_download_reverses_direction_and_honours_dry_run():
    command = _build("download", ["Si001"], dry_run=True)

    assert command[:3] == ["copy", "lab:results", "/repo/result"]
    assert command[-1] == "--dry-run"


def test_unknown_action_is_rejected():
    with pytest.raises(ValueError, match="Unknown transfer action"):
        _build("sync", ["Si001"])
