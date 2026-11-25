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
def unbound_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (tmp_path / "account").mkdir()
    (tmp_path / "account" / "rclone.conf").touch()
    # Select a test cloud root without launching or downloading rclone.
    monkeypatch.setattr(rclone, "prepare_rclone", lambda cache: cache / "rclone")
    monkeypatch.setattr(
        setup, "select_remote_root", lambda *args, **kwargs: "lab:measurements"
    )
    return checkout


@pytest.fixture
def configured_checkout(
    unbound_checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    _bind(unbound_checkout, "local/alice", monkeypatch)
    return unbound_checkout


def _bind(checkout: Path, prefix: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))
    assert (
        _sync(
            checkout,
            "setup",
            "--prefix",
            prefix,
            "--labdata",
            str(checkout.parent / "labdata"),
        )
        == 0
    )


def _save_checkout_settings(
    checkout: Path, *, source: str = "local/alice", labdata: Path | None = None
) -> None:
    (checkout / ".data-sync" / "sync.json").write_text(
        json.dumps(
            {
                "version": 2,
                "remote_root": "lab:measurements",
                "prefix": source,
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
def test_push_copies_physical_data_to_its_bound_prefix(
    source: str,
    unbound_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    rclone_commands: list[list[str]],
):
    checkout = unbound_checkout
    _bind(checkout, source, monkeypatch)
    assert _sync(checkout, "push", "--dry-run") == 0
    root = checkout.parent / "labdata" / source
    assert {tuple(command[:3]) for command in rclone_commands} == {
        ("copy", str(root / "result"), f"lab:measurements/{source}/result"),
        ("copy", str(root / "Database"), f"lab:measurements/{source}/Database"),
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
    assert "Invalid prefix" in capsys.readouterr().err
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
