import io
import os
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


def _bind(
    checkout: Path,
    prefix: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    labdata: Path | None = None,
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))
    assert (
        _sync(
            checkout,
            "setup",
            "--prefix",
            prefix,
            "--labdata",
            str(labdata or checkout.parent / "labdata"),
        )
        == 0
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
def local_cloud(unbound_checkout: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    binary = os.environ.get("DATA_SYNC_TEST_RCLONE")
    if binary is None:
        pytest.skip("Set DATA_SYNC_TEST_RCLONE to a verified rclone executable")
    tmp_path = unbound_checkout.parent
    cloud = tmp_path / "cloud"
    cloud.mkdir()
    for name in tuple(os.environ):
        if name.upper().startswith("RCLONE_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("RCLONE_CACHE_DIR", str(tmp_path / "rclone-cache"))
    (tmp_path / "account" / "rclone.conf").write_text(
        f"[lab]\ntype = alias\nremote = {cloud.as_posix()}\n", encoding="utf-8"
    )
    # Replace the download only. The copy and listing use the real local backend.
    monkeypatch.setattr(rclone, "prepare_rclone", lambda cache: Path(binary))
    _bind(
        unbound_checkout,
        "local/alice",
        monkeypatch,
        labdata=unbound_checkout / "Labdata",
    )
    return cloud / "measurements"


def _write_file(root: Path, relative: str, content: bytes, *, mtime: int) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    os.utime(path, (mtime, mtime))


def test_default_push_and_pull_use_the_bound_labdata_prefix(
    unbound_checkout: Path, local_cloud: Path
) -> None:
    checkout = unbound_checkout
    remote = local_cloud / "local/alice"
    older, newer = 1_700_000_000, 1_700_001_000
    _write_file(checkout, "result/run/data.h5", b"run", mtime=older)
    _write_file(checkout, "Database/device.yml", b"older", mtime=older)
    _write_file(checkout, "result/run/data.lock", b"local lock", mtime=older)
    _write_file(remote, "Database/device.yml", b"newer", mtime=newer)
    _write_file(remote, "result/cloud.lock", b"remote lock", mtime=newer)
    _write_file(remote, "outside/notes.txt", b"not data", mtime=newer)

    assert _sync(checkout, "push", "--dry-run") == 0
    assert not (remote / "result/run/data.h5").exists()
    assert _sync(checkout, "push", "--jobs", "3") == 0
    assert (remote / "result/run/data.h5").read_bytes() == b"run"
    assert not (remote / "result/run/data.lock").exists()
    assert (remote / "Database/device.yml").read_bytes() == b"newer"
    assert (remote / "outside/notes.txt").read_bytes() == b"not data"

    assert _sync(checkout, "pull", "--dry-run") == 0
    assert (checkout / "Database/device.yml").read_bytes() == b"older"
    assert _sync(checkout, "pull") == 0
    assert (checkout / "Database/device.yml").read_bytes() == b"newer"
    assert (checkout / "result/run/data.lock").read_bytes() == b"local lock"
    physical = checkout / "Labdata/local/alice"
    assert not (physical / "result/cloud.lock").exists()
    assert not (physical / "outside/notes.txt").exists()


def test_named_push_and_pull_transfer_only_the_selected_prefix(
    unbound_checkout: Path, local_cloud: Path
) -> None:
    checkout = unbound_checkout
    local = checkout / "Labdata"
    prefix = "other/pc1"
    older, newer = 1_700_000_000, 1_700_001_000
    _write_file(local / prefix, "result/session.h5", b"local", mtime=older)
    _write_file(local / prefix, "other/notes.txt", b"not data", mtime=older)
    _write_file(checkout, "result/own.h5", b"bound", mtime=older)

    assert _sync(checkout, "push", prefix) == 0
    assert (local_cloud / prefix / "result/session.h5").read_bytes() == b"local"
    assert not (local_cloud / prefix / "other/notes.txt").exists()
    assert not (local_cloud / "local/alice").exists()

    _write_file(local_cloud / prefix, "result/session.h5", b"cloud", mtime=newer)
    assert _sync(checkout, "pull", prefix) == 0
    assert (local / prefix / "result/session.h5").read_bytes() == b"cloud"
    assert (checkout / "result/own.h5").read_bytes() == b"bound"


@pytest.mark.parametrize("direction", ["push", "pull"])
def test_force_overwrites_older_and_same_metadata_files_without_deleting(
    direction: str, unbound_checkout: Path, local_cloud: Path
) -> None:
    checkout = unbound_checkout
    prefix = "local/alice"
    local, remote = checkout / "Labdata" / prefix, local_cloud / prefix
    origin, destination = (local, remote) if direction == "push" else (remote, local)
    older, newer = 1_700_000_000, 1_700_001_000
    _write_file(origin, "result/run.h5", b"from source", mtime=older)
    _write_file(destination, "result/run.h5", b"keep newer!", mtime=newer)
    _write_file(origin, "Database/device.yml", b"new bytes", mtime=older)
    _write_file(destination, "Database/device.yml", b"old bytes", mtime=older)
    _write_file(origin, "Database/device.lock", b"lock", mtime=older)
    _write_file(destination, "Database/extra.yml", b"destination only", mtime=older)
    args = ("push",) if direction == "push" else ("pull", prefix)

    assert _sync(checkout, *args) == 0
    assert (destination / "result/run.h5").read_bytes() == b"keep newer!"
    assert (destination / "Database/device.yml").read_bytes() == b"old bytes"
    assert _sync(checkout, *args, "--force", "--dry-run") == 0
    assert (destination / "result/run.h5").read_bytes() == b"keep newer!"
    assert (destination / "Database/device.yml").read_bytes() == b"old bytes"
    assert _sync(checkout, *args, "--force") == 0
    assert (destination / "result/run.h5").read_bytes() == b"from source"
    assert (destination / "Database/device.yml").read_bytes() == b"new bytes"
    assert (destination / "Database/extra.yml").read_bytes() == b"destination only"
    assert not (destination / "Database/device.lock").exists()


@pytest.fixture
def rclone_commands(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    commands: list[list[str]] = []

    def run(
        _binary: Path, _config: Path, args: list[str], *, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, "[]" if capture_output else None)

    # Capture any unexpected external calls on rejected operations.
    monkeypatch.setattr(rclone, "run_cli", run)
    return commands


@pytest.mark.parametrize("selection", [("--all",), ("first", "second")])
def test_pull_force_rejects_more_than_one_prefix(
    selection: tuple[str, ...],
    configured_checkout: Path,
    rclone_commands: list[list[str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _sync(configured_checkout, "pull", *selection, "--force") != 0
    error = capsys.readouterr().err
    assert "--force" in error and "one prefix" in error
    assert not rclone_commands


@pytest.mark.parametrize(
    "overlap", ["equal", "linked-checkout-child", "linked-checkout", "linked-data"]
)
def test_pull_refuses_checkout_and_realpaths_of_its_data(
    overlap: str,
    unbound_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    rclone_commands: list[list[str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkout = unbound_checkout
    labdata = checkout.parent if overlap == "equal" else checkout.parent / "labdata"
    _bind(checkout, "local/alice", monkeypatch, labdata=labdata)
    prefix = "foreign"
    if overlap == "equal":
        prefix = checkout.name
    else:
        target = {
            "linked-checkout-child": checkout / "other",
            "linked-checkout": checkout,
            "linked-data": labdata / "local/alice",
        }[overlap]
        target.mkdir(exist_ok=True)
        symlink_or_skip(labdata / prefix, target, target_is_directory=True)

    assert _sync(checkout, "pull", prefix, "--dry-run") != 0
    assert "checkout" in capsys.readouterr().err
    assert not rclone_commands


def test_pull_refuses_a_prefix_whose_data_tree_is_the_checkout(
    unbound_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    rclone_commands: list[list[str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkout = unbound_checkout / "foreign/Database"
    checkout.mkdir(parents=True)
    _bind(checkout, "local/alice", monkeypatch, labdata=unbound_checkout)
    (checkout.parent / "account" / "rclone.conf").touch()

    assert _sync(checkout, "pull", "foreign", "--dry-run") != 0
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


@pytest.mark.parametrize("selection", [("--all",), ("a", "a/b")])
def test_pull_batch_downloads_other_prefixes_and_leaves_bound_data(
    selection: tuple[str, ...], unbound_checkout: Path, local_cloud: Path
) -> None:
    labdata = unbound_checkout / "Labdata"
    files = {
        "a/result/entry/Database/embedded.yml": b"parent prefix",
        "a/b/Database/device.yml": b"nested prefix",
    }
    for relative, content in files.items():
        _write_file(local_cloud, relative, content, mtime=1_700_001_000)
    _write_file(
        local_cloud, "local/alice/Database/own.yml", b"cloud", mtime=1_700_001_000
    )
    bound = labdata / "local/alice"
    _write_file(bound, "Database/own.yml", b"keep local", mtime=1_700_000_000)
    _write_file(local_cloud, "Database/ignored.yml", b"no prefix", mtime=1_700_001_000)
    # Selecting other prefixes must not require a valid bound checkout.
    link = unbound_checkout / "Database"
    if link.is_junction():
        link.rmdir()
    else:
        link.unlink()

    assert _sync(unbound_checkout, "pull", *selection) == 0
    for relative, content in files.items():
        assert (labdata / relative).read_bytes() == content
    assert (bound / "Database/own.yml").read_bytes() == b"keep local"
    assert not (labdata / "Database").exists()
