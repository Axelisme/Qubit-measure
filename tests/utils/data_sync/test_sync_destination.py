import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from zcu_tools.utils.data_sync import rclone

from ._support import baseline_files, only_baseline, state_dir, sync, write_file

_CASE_INSENSITIVE_FEATURES = '{"Features": {"CaseInsensitive": true}}'


@pytest.fixture
def report_features(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str, int, str], None]:
    run_cli = rclone.run_cli

    def report_for(root: str, status: int, response: str) -> None:
        def run(
            binary: Path, config: Path, args: list[str], *, capture_output: bool = False
        ) -> subprocess.CompletedProcess[str]:
            if args == ["backend", "features", root]:
                return subprocess.CompletedProcess(args, status, response, "")
            return run_cli(binary, config, args, capture_output=capture_output)

        # Override the external backend report, not the test volume's behavior.
        monkeypatch.setattr(rclone, "run_cli", run)

    return report_for


@pytest.mark.parametrize("preview", [(), ("--dry-run",)])
@pytest.mark.parametrize("lower", ["result/run.h5", "result/run.h5/scan.h5"])
def test_sync_case_collision_blocks_both_batches_and_preserves_baseline(
    preview: tuple[str, ...],
    lower: str,
    unbound_checkout: Path,
    local_cloud: Path,
    report_features: Callable[[str, int, str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    labdata = unbound_checkout / "Labdata"
    local, remote = labdata / "local/alice", local_cloud / "local/alice"
    upper = "result/Run.h5"
    upload, deleted = "Database/new", "Database/deleted"
    stamp = 1_700_000_000
    write_file(remote, upper, b"upper", mtime=stamp)
    if (remote / upper.lower()).exists():
        pytest.skip("Test source volume must store case-distinct names")
    write_file(remote, lower, b"lower", mtime=stamp)
    write_file(local, deleted, b"base", mtime=stamp)
    assert sync(unbound_checkout, "push") == 0
    record = only_baseline(labdata)
    previous = record.read_bytes()
    for root in (local, remote):
        (root / deleted).unlink()
    write_file(local, upload, b"upload", mtime=stamp)
    report_features(str(local), 0, _CASE_INSENSITIVE_FEATURES)
    capsys.readouterr()

    assert sync(unbound_checkout, "sync", *preview) == 1
    plan = capsys.readouterr().out
    assert any(
        "Conflict:" in line and repr(upper) in line and repr(lower) in line
        for line in plan.splitlines()
    )
    assert not (remote / upload).exists()
    assert not (local / upper).exists() and not (local / lower).exists()
    assert record.read_bytes() == previous


@pytest.mark.parametrize(
    ("upload", "occupied", "is_directory"),
    [
        ("result/run.h5", "result/Run.h5", False),
        ("result/run.h5", "result/Run.h5", True),
        ("result/run/scan.h5", "result/Run", False),
        ("result/run.h5", "RESULT", False),
        ("result/Run.LOCK", "result/run.lock", False),
    ],
)
def test_sync_case_collision_with_existing_cloud_path(
    upload: str,
    occupied: str,
    is_directory: bool,
    unbound_checkout: Path,
    local_cloud: Path,
    report_features: Callable[[str, int, str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    labdata = unbound_checkout / "Labdata"
    local, remote = labdata / "local/alice", local_cloud / "local/alice"
    stamp = 1_700_000_000
    write_file(local, upload, b"upload", mtime=stamp)
    if is_directory:
        (remote / occupied).mkdir(parents=True)
    else:
        write_file(remote, occupied, b"existing", mtime=stamp)
    report_features("lab:measurements/local/alice", 0, _CASE_INSENSITIVE_FEATURES)

    assert sync(unbound_checkout, "sync") == 1
    plan = capsys.readouterr().out
    assert any(
        "Conflict:" in line and repr(upload) in line and repr(occupied) in line
        for line in plan.splitlines()
    )
    if is_directory:
        assert (remote / occupied).is_dir()
    else:
        assert (remote / occupied).read_bytes() == b"existing"
    assert not state_dir(labdata).exists()


def test_case_sensitive_destination_receives_both_case_distinct_names(
    unbound_checkout: Path, local_cloud: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labdata = unbound_checkout / "Labdata"
    local, remote = labdata / "local/alice", local_cloud / "local/alice"
    upper, lower = "result/Run.h5", "result/run.h5"
    stamp = 1_700_000_000
    write_file(remote, upper, b"upper", mtime=stamp)
    if (remote / lower).exists():
        pytest.skip("Test volume must store case-distinct names")
    write_file(remote, lower, b"lower", mtime=stamp)
    monkeypatch.setenv("RCLONE_LOCAL_CASE_SENSITIVE", "true")

    assert sync(unbound_checkout, "sync") == 0
    assert (local / upper).read_bytes() == b"upper"
    assert (local / lower).read_bytes() == b"lower"
    assert baseline_files(only_baseline(labdata)).keys() == {upper, lower}


@pytest.mark.parametrize(
    ("status", "report"),
    [(4, ""), (0, "not json"), (0, '{"Features": {"CaseInsensitive": "false"}}')],
)
def test_sync_rejects_unknown_destination_case_sensitivity(
    status: int,
    report: str,
    unbound_checkout: Path,
    local_cloud: Path,
    report_features: Callable[[str, int, str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    labdata = unbound_checkout / "Labdata"
    local, remote = labdata / "local/alice", local_cloud / "local/alice"
    upload, download = "Database/new", "result/other.h5"
    write_file(local, upload, b"upload", mtime=1_700_000_000)
    write_file(remote, download, b"download", mtime=1_700_000_000)
    report_features(str(local), status, report)

    assert sync(unbound_checkout, "sync") == 1
    error = capsys.readouterr().err
    assert "Could not determine case sensitivity" in error and str(local) in error
    assert not (remote / upload).exists() and not (local / download).exists()
    assert not state_dir(labdata).exists()
