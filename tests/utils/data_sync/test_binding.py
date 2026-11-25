import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from zcu_tools.utils.data_sync import rclone, setup
from zcu_tools.utils.data_sync.cli import main

from tests._filesystem import symlink_or_skip


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "checkout & 100%! 測試"
    path.mkdir()
    monkeypatch.chdir(path)
    (tmp_path / "account").mkdir()
    (tmp_path / "account" / "rclone.conf").touch()
    # Replace only the external download and browser selection.
    monkeypatch.setattr(rclone, "prepare_rclone", lambda cache: cache / "rclone")
    monkeypatch.setattr(setup, "select_remote_root", lambda *args, **kwargs: "lab:Data")
    return path


def _run(checkout: Path, *args: str) -> int:
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


def _configure(
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    labdata: str = "../Labdata",
    prefix: str = "alice/pc1",
) -> int:
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))
    return _run(checkout, "setup", "--prefix", prefix, "--labdata", labdata)


def _remove_data_link(link: Path) -> None:
    if link.is_junction():
        link.rmdir()
    else:
        link.unlink()


@pytest.mark.parametrize(
    ("labdata", "use_flag"), [("../Labdata & 100%! 資料", True), ("Labdata", False)]
)
def test_setup_binds_and_retains_both_data_directories(
    labdata: str,
    use_flag: bool,
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = (checkout / labdata / "alice/pc1").resolve()
    (root / "Database").mkdir(parents=True)
    (root / "Database" / "existing").write_text("keep", encoding="utf-8")
    args = ["setup", "--prefix", "alice/pc1"]
    if use_flag:
        args += ["--labdata", labdata]
    confirmation = "\n" if use_flag else f"{labdata}\n\n"

    monkeypatch.setattr(sys, "stdin", io.StringIO(confirmation))
    assert _run(checkout, *args) == 0
    out = capsys.readouterr().out
    assert "alice/pc1" in out and "lab:Data/alice/pc1" in out and str(root) in out
    assert (checkout / "Database" / "existing").read_text(encoding="utf-8") == "keep"
    (checkout / "result" / "measurement").write_text("data", encoding="utf-8")
    assert (root / "result" / "measurement").read_text(encoding="utf-8") == "data"

    links = [checkout / "result", checkout / "Database"]
    for link in links:
        assert link.resolve(strict=True) == root / link.name
        if sys.platform == "win32":
            assert link.is_junction()
        else:
            assert link.is_symlink()
    link_ids = [link.lstat().st_ino for link in links]
    monkeypatch.setattr(sys, "stdin", io.StringIO(confirmation))
    assert _run(checkout, *args) == 0
    assert [link.lstat().st_ino for link in links] == link_ids
    capsys.readouterr()

    assert _run(checkout, "status", "--json") == 0
    status = json.loads(capsys.readouterr().out)
    assert status["configured"] and status["bound"]
    assert status["prefix"] == "alice/pc1"
    assert status["labdata"] == str(root.parent.parent)
    assert status["local_root"] == str(root)
    assert status["remote_data_root"] == "lab:Data/alice/pc1"


@pytest.mark.parametrize("occupied", ["file", "directory", "wrong link", "broken link"])
def test_setup_reports_conflict_without_creating_other_link_or_overwriting_settings(
    occupied: str,
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _configure(checkout, monkeypatch) == 0
    settings = checkout / ".data-sync" / "sync.json"
    saved = settings.read_bytes()
    result, database = checkout / "result", checkout / "Database"
    _remove_data_link(result)
    if occupied == "broken link":
        database.resolve(strict=True).rmdir()
    elif occupied != "wrong link":
        _remove_data_link(database)
        if occupied == "file":
            database.write_text("keep", encoding="utf-8")
        else:
            database.mkdir()
    conflict_id = database.lstat().st_ino
    capsys.readouterr()

    assert _configure(checkout, monkeypatch, prefix="alice/other") != 0
    error = capsys.readouterr().err
    expected = checkout.parent / "Labdata" / "alice" / "other" / "Database"
    assert str(database) in error and str(expected) in error and occupied in error
    assert not result.exists(follow_symlinks=False)
    assert database.lstat().st_ino == conflict_id
    assert settings.read_bytes() == saved


def test_setup_lists_both_conflicts_without_saving_settings(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (checkout / "result").mkdir()
    (checkout / "Database").write_text("keep", encoding="utf-8")

    assert _configure(checkout, monkeypatch) != 0
    error = capsys.readouterr().err
    assert str(checkout / "result") in error and str(checkout / "Database") in error
    assert not (checkout / ".data-sync" / "sync.json").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="Inject a symlink failure on POSIX")
@pytest.mark.parametrize("leaves_directory", [False, True])
def test_setup_reports_link_creation_failure_without_saving_settings(
    leaves_directory: bool,
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = checkout.parent / "Labdata" / "alice" / "pc1"
    failed_link = checkout / "Database"
    original_error = "could not apply directory link tag"
    symlink_to = Path.symlink_to

    def fail_on_database(
        link: Path, target: Path, target_is_directory: bool = False
    ) -> None:
        if link != failed_link:
            symlink_to(link, target, target_is_directory=target_is_directory)
            return
        if leaves_directory:
            link.mkdir()
        raise OSError(original_error)

    # The CLI cannot trigger this partial failure on demand; replace the OS effect.
    monkeypatch.setattr(Path, "symlink_to", fail_on_database)

    assert _configure(checkout, monkeypatch) == 1
    error = capsys.readouterr().err
    assert original_error in error
    assert ("before retrying" in error) == leaves_directory
    assert str(failed_link) in error and str(root / "Database") in error
    if leaves_directory:
        assert failed_link.is_dir() and not failed_link.is_symlink()
    else:
        assert not failed_link.exists(follow_symlinks=False)
    assert (checkout / "result").resolve(strict=True) == root / "result"
    assert not (checkout / ".data-sync" / "sync.json").exists()


@pytest.mark.parametrize("layout", ["under link", "same checkout", "aliased target"])
def test_setup_rejects_overlapping_link_and_target_paths(
    layout: str,
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    labdata, prefix = "result", "alice/pc1"
    if layout == "same checkout":
        labdata, prefix = "..", checkout.name
    if layout == "aliased target":
        labdata = "../Labdata"
        physical_root = checkout.parent / "Labdata" / prefix
        physical_root.mkdir(parents=True)
        symlink_or_skip(
            physical_root / "Database", checkout / "result", target_is_directory=True
        )

    assert _configure(checkout, monkeypatch, labdata, prefix) != 0
    error = capsys.readouterr().err
    assert "overlap" in error and str(checkout / "result") in error
    assert not (checkout / "result").exists(follow_symlinks=False)
    assert not (checkout / "Database").exists(follow_symlinks=False)
    assert not (checkout / ".data-sync" / "sync.json").exists()


@pytest.mark.parametrize("transfer", [("push",), ("pull", "alice/pc1")])
def test_missing_links_disable_status_and_bound_transfer(
    transfer: tuple[str, ...],
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _configure(checkout, monkeypatch) == 0
    _remove_data_link(checkout / "Database")
    _remove_data_link(checkout / "result")
    capsys.readouterr()

    assert _run(checkout, "status", "--json") == 1
    status = json.loads(capsys.readouterr().out)
    assert not status["configured"] and not status["bound"]
    assert str(checkout / "Database") in "\n".join(status["binding_errors"])

    commands: list[list[str]] = []

    def run(
        _binary: Path, _config: Path, args: list[str]
    ) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        return subprocess.CompletedProcess(args, 0)

    # Record any attempted transfer without starting an external process.
    monkeypatch.setattr(rclone, "run_cli", run)
    assert _run(checkout, *transfer, "--dry-run") != 0
    error = capsys.readouterr().err
    assert str(checkout / "Database") in error and "setup" in error
    assert not commands


def test_legacy_settings_require_explicit_setup(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = checkout / ".data-sync" / "sync.json"
    settings.parent.mkdir()
    legacy = json.dumps(
        {
            "version": 1,
            "remote_root": "lab:Data",
            "source": "alice/pc1",
            "labdata": str(checkout.parent / "Labdata"),
        }
    )
    settings.write_text(legacy, encoding="utf-8")

    assert _run(checkout, "status", "--json") == 1
    error = capsys.readouterr().err
    assert "predates checkout binding" in error and "run setup again" in error
    assert settings.read_text(encoding="utf-8") == legacy
    assert not (checkout / "result").exists()
    assert _configure(checkout, monkeypatch) == 0
    assert _run(checkout, "status") == 0
