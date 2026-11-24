import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from zcu_tools.utils.data_sync import rclone, setup
from zcu_tools.utils.data_sync.cli import main

from tests._filesystem import symlink_or_skip


def test_unconfigured_push_instructs_unattended_user_to_run_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(sys, "stdin", io.StringIO())

    code = main(["--config-dir", str(tmp_path / "account"), "push"], checkout=tmp_path)

    assert code == 2
    assert "sync_data.py setup" in capsys.readouterr().err


@pytest.fixture
def configured_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    for folder in (".data-sync", "result", "Database"):
        (checkout / folder).mkdir(parents=True)
    (tmp_path / "account").mkdir()
    (tmp_path / "account" / "rclone.conf").touch()
    _save_checkout_settings(checkout)
    return checkout


def _save_checkout_settings(
    checkout: Path, *, source: str = "local/alice", labdata: Path | None = None
) -> None:
    (checkout / ".data-sync" / "sync.json").write_text(
        json.dumps(
            {
                "version": 1,
                "remote_root": "lab:measurements",
                "source": source,
                "labdata": str(labdata or (checkout.parent / "labdata")),
            }
        ),
        encoding="utf-8",
    )


def _sync(checkout: Path, *args: str) -> int:
    return main(
        [
            "--config-dir",
            str(checkout.parent / "account"),
            "--cache-dir",
            str(checkout.parent / "cache"),
            *args,
        ],
        checkout=checkout,
    )


@pytest.fixture
def rclone_listing() -> list[str]:
    return []


@pytest.fixture
def rclone_commands(
    monkeypatch: pytest.MonkeyPatch, rclone_listing: list[str]
) -> list[list[str]]:
    commands: list[list[str]] = []

    def run(
        _binary: Path, _config: Path, args: list[str], *, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        listing = json.dumps([{"Path": path, "IsDir": True} for path in rclone_listing])
        return subprocess.CompletedProcess(args, 0, listing if capture_output else None)

    # Capture the external tool invocation without downloading or starting rclone.
    monkeypatch.setattr(rclone, "prepare_rclone", lambda cache: cache / "rclone")
    monkeypatch.setattr(rclone, "run_cli", run)
    return commands


@pytest.mark.parametrize("source", ["alice", "local/alice", "alice/pc1/sim"])
def test_push_only_copies_checkout_data_to_its_configured_source(
    source: str, configured_checkout: Path, rclone_commands: list[list[str]]
):
    checkout = configured_checkout
    _save_checkout_settings(checkout, source=source)
    assert _sync(checkout, "push", "--dry-run") == 0
    assert {tuple(command[:3]) for command in rclone_commands} == {
        ("copy", str(checkout / "result"), f"lab:measurements/{source}/result"),
        ("copy", str(checkout / "Database"), f"lab:measurements/{source}/Database"),
    }
    for command in rclone_commands:
        assert "--update" in command
        assert command[command.index("--filter") + 1] == "- *.lock"
        assert "--dry-run" in command
        assert "--delete-excluded" not in command

    rclone_commands.clear()
    with pytest.raises(SystemExit) as error:
        _sync(checkout, "push", "--source", "different/bob")
    assert error.value.code == 2
    assert not rclone_commands


@pytest.mark.parametrize("source", ["alice", "alice/pc1/sim"])
def test_pull_copies_only_the_named_sources_own_data_trees(
    source: str, configured_checkout: Path, rclone_commands: list[list[str]]
) -> None:
    checkout = configured_checkout

    assert _sync(checkout, "pull", source, "--dry-run") == 0
    (command,) = rclone_commands
    assert command[:3] == [
        "copy",
        f"lab:measurements/{source}",
        str(checkout.parent / "labdata" / source),
    ]
    rules = [command[i + 1] for i, arg in enumerate(command) if arg == "--filter"]
    assert rules == ["- *.lock", "+ /result/**", "+ /Database/**", "- **"]
    assert "--update" in command and "--dry-run" in command


@pytest.mark.parametrize(
    "overlap", ["inside", "equal", "linked-checkout", "linked-data"]
)
def test_pull_refuses_checkout_and_realpaths_of_its_data(
    overlap: str,
    configured_checkout: Path,
    rclone_commands: list[list[str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkout = configured_checkout
    labdata = checkout.parent / "labdata"
    labdata.mkdir()
    prefix = "foreign"
    if overlap == "inside":
        _save_checkout_settings(checkout, labdata=checkout / "downloads")
    elif overlap == "equal":
        _save_checkout_settings(checkout, labdata=checkout.parent)
        prefix = checkout.name
    elif overlap == "linked-checkout":
        symlink_or_skip(labdata / prefix, checkout, target_is_directory=True)
    else:
        prefix = "local/alice"
        target = labdata / prefix / "Database"
        target.mkdir(parents=True)
        (checkout / "Database").rmdir()
        symlink_or_skip(checkout / "Database", target, target_is_directory=True)

    assert _sync(checkout, "pull", prefix, "--dry-run") != 0
    assert "checkout" in capsys.readouterr().err
    assert not rclone_commands


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "/alice",
        "alice/",
        "a//b",
        "a/./b",
        "a/../b",
        "a\\b",
        "a/result/b",
        "Database/b",
    ],
)
def test_pull_rejects_unsafe_or_reserved_source_segments(
    prefix: str,
    configured_checkout: Path,
    rclone_commands: list[list[str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _sync(configured_checkout, "pull", prefix, "--dry-run") != 0
    assert "Invalid source" in capsys.readouterr().err
    assert not rclone_commands


def test_pull_all_discovers_nested_sources_without_descending_into_data(
    configured_checkout: Path,
    rclone_commands: list[list[str]],
    rclone_listing: list[str],
) -> None:
    rclone_listing += [
        "a",
        "a/b",
        "a/result",
        "a/Database",
        "a/b/Database",
        "a/result/entry/Database",
        "a/Database/result/session/result",
        "local",
        "local/alice",
        "local/alice/Database",
        "Database",
        "notes",
    ]

    assert _sync(configured_checkout, "pull", "--all", "--dry-run") == 0
    listing, *copies = rclone_commands
    assert listing[:2] == ["lsjson", "lab:measurements"]
    assert {"--dirs-only", "--recursive", "--fast-list"}.issubset(listing)
    labdata = configured_checkout.parent / "labdata"
    assert {tuple(copy[:3]) for copy in copies} == {
        ("copy", "lab:measurements/a", str(labdata / "a")),
        ("copy", "lab:measurements/a/b", str(labdata / "a/b")),
    }


@pytest.mark.parametrize("path_option", ["flag", "prompt"])
def test_setup_records_labdata_relative_to_current_directory(
    path_option: str,
    configured_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    rclone_commands: list[list[str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkout = configured_checkout
    monkeypatch.chdir(checkout)
    options = ["--labdata", "../QubitData"] if path_option == "flag" else []
    confirmation = "\n" if path_option == "flag" else "../QubitData\n\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(confirmation))
    # Replace only the interactive browser operation; use real terminal and settings logic.
    monkeypatch.setattr(
        setup, "select_remote_root", lambda *args, **kwargs: "lab:QubitData"
    )

    assert _sync(checkout, "setup", "--source", "alice/pc1", *options) == 0
    expected = str(checkout.parent / "QubitData")
    assert expected in capsys.readouterr().out
    assert _sync(checkout, "status", "--json") == 0
    status = json.loads(capsys.readouterr().out)
    assert status["labdata"] == expected
    assert status["source"] == "alice/pc1"
    assert not rclone_commands
