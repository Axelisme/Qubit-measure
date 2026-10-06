"""Public software-only acquisition evidence contracts."""

import hashlib
import subprocess
from datetime import UTC, datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import Mock

import pytest
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.device import FakeDevice
from zcu_tools.experiment import (
    AcquisitionEvidence,
    acquisition,
    capture_software_provenance,
    new_run_id,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.plotting.plots import NonPresentingHost, Plots


@pytest.fixture(autouse=True)
def software_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate external software and filesystem services for capture tests."""
    exists = Path.exists

    def isolated_exists(path: Path) -> bool:
        if path.name == ".git" and not path.is_relative_to(tmp_path):
            return False
        return exists(path)

    monkeypatch.setattr(Path, "exists", isolated_exists)
    versions = {"zcu-tools": "1.2.3", "numpy": "2.1.0", "qick": "0.2.1"}
    monkeypatch.setattr(acquisition.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(acquisition.platform, "python_version", lambda: "3.13.0")
    monkeypatch.setattr(acquisition.socket, "gethostname", lambda: "acquisition-host")


def test_run_id_uses_supplied_utc_time_and_distinct_six_hex_suffixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        acquisition.secrets, "token_hex", Mock(side_effect=["000001", "000002"])
    )
    started = datetime(2026, 10, 6, 12, 34, 56, 123456, tzinfo=UTC)

    assert new_run_id(started) == "20261006T123456Z-000001"
    assert new_run_id(started) == "20261006T123456Z-000002"


@pytest.mark.parametrize(
    "started",
    [
        datetime(2026, 10, 6),
        datetime(2026, 10, 6, tzinfo=timezone(timedelta(hours=8))),
    ],
)
def test_run_id_rejects_naive_or_non_utc_time(started: datetime) -> None:
    with pytest.raises(ValueError, match="UTC"):
        new_run_id(started)


def test_non_git_install_captures_versions_without_starting_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git = Mock(side_effect=AssertionError("No Git checkout"))
    monkeypatch.setattr(acquisition.subprocess, "run", git)

    evidence = capture_software_provenance(source_directory=tmp_path)

    assert evidence.software_versions == {
        "python": "3.13.0",
        "zcu-tools": "1.2.3",
        "numpy": "2.1.0",
    }
    assert evidence.git_commit is None
    assert evidence.git_dirty is None
    assert evidence.qick_version == "0.2.1"
    assert evidence.soc_fingerprint is None
    assert evidence.hostname == "acquisition-host"
    git.assert_not_called()


@pytest.mark.parametrize("marker_kind", ["directory", "file"])
@pytest.mark.parametrize("status,dirty", [("", False), (" M source.py\n", True)])
def test_capture_reads_checkout_identity_and_dirty_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker_kind: str,
    status: str,
    dirty: bool,
) -> None:
    if marker_kind == "directory":
        (tmp_path / ".git").mkdir()
    else:
        (tmp_path / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
    source = tmp_path / "lib" / "experiment"
    source.mkdir(parents=True)
    commit = "1" * 40
    git = Mock(
        side_effect=[
            subprocess.CompletedProcess(["git"], 0, stdout=commit + "\n"),
            subprocess.CompletedProcess(["git"], 0, stdout=status),
        ]
    )
    monkeypatch.setattr(acquisition.subprocess, "run", git)

    evidence = capture_software_provenance(source_directory=source)

    assert evidence.git_commit == commit
    assert evidence.git_dirty is dirty
    # These are read-only external commands, never a test-side subprocess.
    assert [call.args[0][-2:] for call in git.call_args_list] == [
        ["rev-parse", "HEAD"],
        ["status", "--porcelain"],
    ]
    assert all(
        call.args[0][:4] == ["git", "--no-optional-locks", "-C", str(tmp_path)]
        for call in git.call_args_list
    )
    assert all(call.kwargs["check"] for call in git.call_args_list)


def test_missing_optional_qick_is_absence_not_missing_required_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def version(name: str) -> str:
        if name == "qick":
            raise PackageNotFoundError(name)
        return "required-version"

    monkeypatch.setattr(acquisition.metadata, "version", version)

    evidence = capture_software_provenance(source_directory=tmp_path)

    assert evidence.qick_version is None
    assert evidence.software_versions["zcu-tools"] == "required-version"


@pytest.mark.parametrize("package", ["zcu-tools", "numpy", "qick"])
def test_version_capture_failures_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    package: str,
) -> None:
    def version(name: str) -> str:
        if name == package:
            raise RuntimeError("Version service failed")
        return "available-version"

    monkeypatch.setattr(acquisition.metadata, "version", version)

    with pytest.raises(RuntimeError, match="Version service failed"):
        capture_software_provenance(source_directory=tmp_path)


def test_missing_required_package_is_not_downgraded_to_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        acquisition.metadata,
        "version",
        Mock(side_effect=PackageNotFoundError("zcu-tools")),
    )
    with pytest.raises(PackageNotFoundError):
        capture_software_provenance(source_directory=tmp_path)


@pytest.mark.parametrize("service", ["python_version", "hostname"])
def test_host_capture_failures_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    service: str,
) -> None:
    failure = Mock(side_effect=OSError("Host unavailable"))
    if service == "python_version":
        monkeypatch.setattr(acquisition.platform, "python_version", failure)
    else:
        monkeypatch.setattr(acquisition.socket, "gethostname", failure)
    with pytest.raises(OSError, match="Host unavailable"):
        capture_software_provenance(source_directory=tmp_path)


@pytest.mark.parametrize(
    "failure",
    [FileNotFoundError("Git unavailable"), subprocess.CalledProcessError(128, ["git"])],
)
def test_git_capture_failures_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(acquisition.subprocess, "run", Mock(side_effect=failure))
    with pytest.raises(type(failure)):
        capture_software_provenance(source_directory=tmp_path)


def test_capture_hashes_cached_config_without_touching_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serialized = '{"board": "zcu216", "revision": 1}'
    soc = Mock()
    soc.get_cfg.side_effect = AssertionError("Live SoC query forbidden")
    device = FakeDevice(fast_mode=True)
    monkeypatch.setattr(
        device, "get_info", Mock(side_effect=AssertionError("Live device query"))
    )
    monkeypatch.setattr(
        device, "setup", Mock(side_effect=AssertionError("Device setup"))
    )
    config = Mock()
    config.dump_cfg.return_value = serialized
    context = RunContext(
        soc=soc,
        soccfg=config,
        devices={"flux": device},
        plots=Plots(NonPresentingHost()),
        cancel_signal=StopSignal(),
    )

    evidence = capture_software_provenance(context=context, source_directory=tmp_path)

    assert (
        evidence.soc_fingerprint
        == hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    )
    assert soc.mock_calls == []


def test_opaque_cached_config_is_genuine_absence(tmp_path: Path) -> None:
    context = RunContext(
        soc=object(),
        soccfg=object(),
        devices={},
        plots=Plots(NonPresentingHost()),
        cancel_signal=StopSignal(),
    )
    assert (
        capture_software_provenance(
            context=context, source_directory=tmp_path
        ).soc_fingerprint
        is None
    )


@pytest.mark.parametrize("serialized", [None, 42, "", "{broken"])
def test_invalid_cached_serialization_is_not_absence(
    tmp_path: Path,
    serialized: object,
) -> None:
    config = Mock()
    config.dump_cfg.return_value = serialized
    context = RunContext(
        soc=object(),
        soccfg=config,
        devices={},
        plots=Plots(NonPresentingHost()),
        cancel_signal=StopSignal(),
    )
    with pytest.raises((ValueError, TypeError)):
        capture_software_provenance(context=context, source_directory=tmp_path)


@pytest.mark.parametrize(
    "failure", [OSError("Cached config failed"), AttributeError("Dump failed")]
)
def test_cached_config_failures_propagate(tmp_path: Path, failure: Exception) -> None:
    config = Mock()
    config.dump_cfg.side_effect = failure
    context = RunContext(
        soc=object(),
        soccfg=config,
        devices={},
        plots=Plots(NonPresentingHost()),
        cancel_signal=StopSignal(),
    )
    with pytest.raises(type(failure), match=str(failure)):
        capture_software_provenance(context=context, source_directory=tmp_path)


def test_declared_config_capability_lookup_failure_is_not_absence(
    tmp_path: Path,
) -> None:
    class BrokenConfig:
        @property
        def dump_cfg(self) -> str:
            raise AttributeError("Cached descriptor failed")

    context = RunContext(
        soc=object(),
        soccfg=BrokenConfig(),
        devices={},
        plots=Plots(NonPresentingHost()),
        cancel_signal=StopSignal(),
    )
    with pytest.raises(AttributeError, match="Cached descriptor failed"):
        capture_software_provenance(context=context, source_directory=tmp_path)


def test_existing_noncallable_dump_is_not_absence(tmp_path: Path) -> None:
    class InvalidConfig:
        dump_cfg = None

    context = RunContext(
        soc=object(),
        soccfg=InvalidConfig(),
        devices={},
        plots=Plots(NonPresentingHost()),
        cancel_signal=StopSignal(),
    )
    with pytest.raises(TypeError):
        capture_software_provenance(context=context, source_directory=tmp_path)


def test_acquisition_evidence_isolates_supplied_version_mapping() -> None:
    versions = {"python": "3.13"}
    provenance = SoftwareProvenance(
        software_versions=versions,
        git_commit=None,
        git_dirty=None,
        qick_version=None,
        soc_fingerprint=None,
        hostname=None,
    )
    started = datetime(2026, 10, 6, tzinfo=UTC)
    evidence = AcquisitionEvidence(
        run_id="20261006T000000Z-000001",
        started_at=started,
        finished_at=started,
        completion="complete",
        provenance=provenance,
    )
    versions["python"] = "changed"

    assert evidence.provenance.software_versions == {"python": "3.13"}
