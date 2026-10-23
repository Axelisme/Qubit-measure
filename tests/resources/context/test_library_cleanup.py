"""Disk-only cleanup through the public library handle, using synthetic YAML."""

import json
import logging
import multiprocessing
import os
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import IO, Literal, Never
from unittest.mock import patch

import pytest
from filelock import FileLock
from ruamel.yaml import YAML
from zcu_tools.cfg_model import ConstWaveformCfg
from zcu_tools.resources.context.library import ModuleLibrary
from zcu_tools.resources.entry import AcceptedPayload, AcceptedWrite, LedgerEvent
from zcu_tools.resources.entry.ledger import RecordsLedger
from zcu_tools.resources.entry.provenance import Provenance

ENTRY_ID = "78fa0941-f9ce-49d4-9665-4eb9302b8162"
HEADER = "format: zcu.module-library\nformat_version: '2.0'\n"
GOOD = {"style": "const", "length": 0.1}
MANUAL = {"source": "manual", "kind": None, "run_id": None, "at": "now", "stderr": None}


def open_library(path: Path) -> ModuleLibrary:
    (path.parent / "records").mkdir(exist_ok=True)
    return ModuleLibrary(
        path, ledger=path.parent / "records/ledger.jsonl", entry_id=ENTRY_ID
    )


def backups(path: Path) -> list[Path]:
    return sorted(path.parent.glob("*.bak*"))


def mixed(path: Path, partition: str = "waveforms", bad: object = None) -> bytes:
    partitions: dict[str, dict[str, object]] = {"waveforms": {"pi_half": GOOD}}
    partitions.setdefault(partition, {})["bad"] = bad
    path.write_text(
        HEADER
        + "# original µ comment\ncomponents: "
        + json.dumps({"Q1": partitions})
        + "\n",
        encoding="utf-8",
    )
    return path.read_bytes()


@pytest.mark.parametrize("partition", ["pulses", "readout", "reset", "waveforms"])
@pytest.mark.parametrize(
    "bad",
    [None, 12, {}, {"type": "unknown"}, {"style": "const", "length": 0.1, "mask": 3}],
)
def test_bad_specs_are_backed_up_and_removed(
    tmp_path: Path, partition: str, bad: object, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "module_cfg.yaml"
    original = mixed(path, partition, bad)
    with caplog.at_level(logging.WARNING):
        library = open_library(path)
    assert library.snapshot().waveform_keys() == ("Q1.waveforms.pi_half",)
    assert library.snapshot().module_keys() == ()
    assert len(backups(path)) == 1
    assert backups(path)[0].read_bytes() == original
    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert str(backups(path)[0]) in warnings[0].getMessage()
    assert f"Q1.{partition}.bad" in warnings[0].getMessage()
    assert not library.refresh()
    open_library(path)
    assert len(backups(path)) == 1


def test_complete_pulse_without_type_is_removed_with_its_sources(
    tmp_path: Path,
) -> None:
    path = tmp_path / "module_cfg.yaml"
    path.write_text(
        HEADER
        + "# missing discriminator, otherwise complete pulse\n"
        + "components:\n"
        + "  Q1:\n"
        + "    pulses:\n"
        + "      bad: {ch: 0, nqz: 1, freq: 5, gain: 0.1, waveform: {style: const, length: 0.1}}\n"
        + "    waveforms:\n"
        + "      pi_half: {style: const, length: 0.1}\n"
        + "provenance: "
        + json.dumps(
            {
                "Q1.pulses.bad": MANUAL,
                "Q1.pulses.bad.gain": MANUAL,
                "Q1.waveforms.pi_half.length": MANUAL,
            }
        )
        + "\n"
    )
    original = path.read_bytes()

    snapshot = open_library(path).snapshot()

    assert snapshot.module_keys() == ()
    assert snapshot.waveform_keys() == ("Q1.waveforms.pi_half",)
    assert set(snapshot.document().provenance) == {"Q1.waveforms.pi_half.length"}
    stored = YAML(typ="safe").load(path.read_text())
    assert stored["components"]["Q1"]["pulses"] == {}
    assert len(backups(path)) == 1
    assert backups(path)[0].read_bytes() == original


def test_shared_partition_removes_both_bad_roots_and_their_sources(
    tmp_path: Path,
) -> None:
    path = tmp_path / "module_cfg.yaml"
    path.write_text(
        HEADER
        + "# shared YAML partition\n"
        + "components:\n"
        + "  Q1:\n"
        + "    waveforms: &shared {bad: null, good: {style: const, length: 0.1}}\n"
        + "  Q2:\n"
        + "    waveforms: *shared\n"
        + "provenance: "
        + json.dumps(
            {
                "Q1.waveforms.bad": MANUAL,
                "Q1.waveforms.bad.length": MANUAL,
                "Q2.waveforms.bad": MANUAL,
                "Q2.waveforms.bad.length": MANUAL,
                "Q1.waveforms.good.length": MANUAL,
                "Q2.waveforms.good.length": MANUAL,
            }
        )
        + "\n"
    )
    original = path.read_bytes()

    snapshot = open_library(path).snapshot()

    assert snapshot.waveform_keys() == ("Q1.waveforms.good", "Q2.waveforms.good")
    assert snapshot.waveform_spec("Q1.waveforms.good").length == 0.1
    assert snapshot.waveform_spec("Q2.waveforms.good").length == 0.1
    assert set(snapshot.document().provenance) == {
        "Q1.waveforms.good.length",
        "Q2.waveforms.good.length",
    }
    stored = YAML(typ="safe").load(path.read_text())
    assert stored["components"]["Q1"]["waveforms"] == {"good": GOOD}
    assert stored["components"]["Q2"]["waveforms"] == {"good": GOOD}
    assert len(backups(path)) == 1
    assert backups(path)[0].read_bytes() == original


@pytest.mark.parametrize("rejected_component", ["Q1", "Q2"])
def test_shared_partition_sources_are_validated_per_root(
    tmp_path: Path, rejected_component: str
) -> None:
    path = tmp_path / "module_cfg.yaml"
    retained_component = "Q2" if rejected_component == "Q1" else "Q1"
    retained_root = f"{retained_component}.waveforms.pi"
    path.write_text(
        HEADER
        + "components:\n"
        + "  Q1:\n"
        + "    waveforms: &shared {pi: {style: const, length: 0.1}}\n"
        + "  Q2:\n"
        + "    waveforms: *shared\n"
        + "provenance: "
        + json.dumps(
            {
                f"{rejected_component}.waveforms.pi.bogus": MANUAL,
                f"{retained_root}.length": MANUAL,
            }
        )
        + "\n"
    )
    original = path.read_bytes()

    snapshot = open_library(path).snapshot()

    assert snapshot.waveform_keys() == (retained_root,)
    assert snapshot.waveform_spec(retained_root).length == 0.1
    assert set(snapshot.document().provenance) == {f"{retained_root}.length"}
    stored = YAML(typ="safe").load(path.read_text())
    assert stored["components"][rejected_component]["waveforms"] == {}
    assert stored["components"][retained_component]["waveforms"] == {"pi": GOOD}
    assert len(backups(path)) == 1
    assert backups(path)[0].read_bytes() == original


@pytest.mark.parametrize("defect", ["schema", "leaf", "missing"])
def test_entry_local_sources_remove_exact_root_only(
    tmp_path: Path, defect: str
) -> None:
    path = tmp_path / "module_cfg.yaml"
    source = dict(MANUAL)
    if defect == "schema":
        source["stderr"] = "bad"
    if defect == "missing":
        source["source"] = "absent-event"
    leaf = "bogus" if defect == "leaf" else "length"
    path.write_text(
        json.dumps(
            {
                "format": "zcu.module-library",
                "format_version": "2.0",
                "components": {"Q1": {"waveforms": {"pi": GOOD, "pi_half": GOOD}}},
                "provenance": {
                    f"Q1.waveforms.pi.{leaf}": source,
                    "Q1.waveforms.pi_half.length": MANUAL,
                },
            }
        )
    )
    original = path.read_bytes()
    library = open_library(path)
    assert library.snapshot().waveform_keys() == ("Q1.waveforms.pi_half",)
    assert library.snapshot().meta("Q1.waveforms.pi_half.length") is not None
    assert set(library.snapshot().document().provenance) == {
        "Q1.waveforms.pi_half.length"
    }
    assert backups(path)[0].read_bytes() == original


@pytest.mark.parametrize("operation", ["open", "refresh"])
@pytest.mark.parametrize("missing", [False, True], ids=["valid", "missing-roots"])
def test_disk_read_batches_sources_and_preserves_valid_roots(
    tmp_path: Path, operation: str, missing: bool
) -> None:
    path = tmp_path / "module_cfg.yaml"
    path.write_text(HEADER)
    library = open_library(path) if operation == "refresh" else None
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    event_id = "8bc26726-5e8b-4531-b14c-826ad570539c"
    ledger = RecordsLedger(records, entry_id=ENTRY_ID)
    ledger.append(
        LedgerEvent(
            id=event_id,
            kind="accepted",
            at="2026-10-23T00:00:00Z",
            entry_id=ENTRY_ID,
            origin="notebook",
            payload=AcceptedPayload(writes=(AcceptedWrite(path="Q1.rate", value=1),)),
        ),
        record={"evidence": 1},
    )
    good_source = {**MANUAL, "source": event_id}
    sources = {
        "Q1.waveforms.pi.length": good_source,
        "Q1.waveforms.pi.style": good_source,
        "Q1.waveforms.pi_half.length": good_source,
    }
    waveforms = {"pi": GOOD, "pi_half": GOOD}
    if missing:
        waveforms.update({"bad_first": GOOD, "bad_second": GOOD})
        sources.update(
            {
                "Q1.waveforms.bad_first.length": {**MANUAL, "source": "missing-a"},
                "Q1.waveforms.bad_second.length": {**MANUAL, "source": "missing-b"},
            }
        )
    path.write_text(
        json.dumps(
            {
                "format": "zcu.module-library",
                "format_version": "2.0",
                "components": {"Q1": {"waveforms": waveforms}},
                "provenance": sources,
            }
        )
    )
    original = path.read_bytes()

    with patch.object(Path, "open", autospec=True, side_effect=Path.open) as opened:
        if library is None:
            library = open_library(path)
        else:
            assert library.refresh()

    snapshot = library.snapshot()
    assert snapshot.waveform_keys() == ("Q1.waveforms.pi", "Q1.waveforms.pi_half")
    assert set(snapshot.document().provenance) == {
        "Q1.waveforms.pi.length",
        "Q1.waveforms.pi.style",
        "Q1.waveforms.pi_half.length",
    }
    assert snapshot.waveform_spec("Q1.waveforms.pi_half").length == 0.1
    if missing:
        assert len(backups(path)) == 1
        assert backups(path)[0].read_bytes() == original
    else:
        assert backups(path) == []
        assert path.read_bytes() == original
    ledger_reads = [
        call
        for call in opened.call_args_list
        if call.args[0] == records / "ledger.jsonl"
    ]
    # Cleanup and the independent full candidate validation each read once.
    assert len(ledger_reads) == 2


@pytest.mark.parametrize(
    "content",
    [
        "components: []",
        "components: {Q1: null}",
        "components: {Q1: {pulses: 3}}",
        "components: {'Q.1': {}}",
        "components: {Q1: {pulses: {'a.b': {}}}}",
        "provenance: []",
        "provenance: {orphan: {}}",
        "provenance: {Q1.waveforms.absent.length: {}}",
    ],
)
def test_container_and_orphan_errors_do_not_backup_even_with_bad_specs(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "module_cfg.yaml"
    path.write_text(HEADER + content + "\n")
    original = path.read_bytes()
    with pytest.raises(
        ValueError, match="components|provenance|[Ss]ource|[Ii]nvalid|library root"
    ):
        open_library(path)
    assert path.read_bytes() == original
    assert backups(path) == []


def test_all_invalid_and_forward_minor_container_extras_survive_cleanup(
    tmp_path: Path,
) -> None:
    path = tmp_path / "module_cfg.yaml"
    path.write_text(
        "format: zcu.module-library\nformat_version: '2.9'\nfuture: {x: 1}\ncomponents: {Q1: {future_partition: {x: 2}, waveforms: {bad: {style: const, length: 1, mask: 2}}}}\n"
    )
    library = open_library(path)
    assert library.snapshot().waveform_keys() == ()
    stored = YAML(typ="safe").load(path.read_text())
    assert stored["future"] == {"x": 1}
    assert stored["components"]["Q1"]["future_partition"] == {"x": 2}
    assert stored["components"]["Q1"]["waveforms"] == {}


def test_later_cleanup_never_overwrites_backup(tmp_path: Path) -> None:
    path = tmp_path / "module_cfg.yaml"
    first = mixed(path, bad=0)
    open_library(path)
    second = mixed(path, bad=1)
    open_library(path)
    assert len(backups(path)) == 2
    assert {backup.read_bytes() for backup in backups(path)} == {first, second}


@pytest.mark.parametrize("failure", ["backup", "replace"])
def test_cleanup_io_failure_retains_disk_and_published_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path = tmp_path / "module_cfg.yaml"
    path.write_text(HEADER)
    library = open_library(path)
    changes = []
    library.subscribe(changes.append)
    original = mixed(path)
    real_open = Path.open

    def fail_backup(
        self: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[str] | IO[bytes]:
        if "x" in mode:
            raise OSError("backup denied")
        return real_open(self, mode, buffering, encoding, errors, newline)

    def fail_replace(source: str | Path, destination: str | Path) -> Never:
        raise OSError("replace denied")

    if failure == "backup":
        monkeypatch.setattr(Path, "open", fail_backup)
    else:
        monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="denied"):
        library.refresh()
    assert path.read_bytes() == original
    assert library.snapshot().waveform_keys() == ()
    assert changes == []
    assert len(backups(path)) == (0 if failure == "backup" else 1)
    if backups(path):
        assert backups(path)[0].read_bytes() == original


@pytest.mark.parametrize(
    "failure",
    [OSError("ledger I/O"), ValueError("malformed ledger"), RuntimeError("unexpected")],
)
def test_ledger_failures_are_not_entry_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    path = tmp_path / "module_cfg.yaml"
    path.write_text(HEADER)
    library = open_library(path)
    source = {**MANUAL, "source": "some-event"}
    path.write_text(
        json.dumps(
            {
                "format": "zcu.module-library",
                "format_version": "2.0",
                "components": {"Q1": {"waveforms": {"pi": GOOD}}},
                "provenance": {"Q1.waveforms.pi.length": source},
            }
        )
    )
    original = path.read_bytes()

    def reject_get(self: RecordsLedger, event_id: str) -> Never:
        raise failure

    monkeypatch.setattr(RecordsLedger, "get", reject_get)
    with pytest.raises(type(failure)):
        library.refresh()
    assert library.snapshot().waveform_keys() == ()
    assert path.read_bytes() == original
    assert backups(path) == []


def _write_invalid_source(library: ModuleLibrary) -> None:
    with library.edit() as draft:
        draft.set_waveform("Q1.waveforms.new", ConstWaveformCfg(length=0.2))
        draft.accept_source(
            "Q1.waveforms.new", provenance=Provenance("absent", None, None, "now", None)
        )


def test_edit_cleanup_persists_even_when_new_draft_source_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "module_cfg.yaml"
    path.write_text(HEADER)
    library = open_library(path)
    original = mixed(path)
    with pytest.raises(ValueError, match="absent"):
        _write_invalid_source(library)
    assert backups(path)[0].read_bytes() == original
    assert open_library(path).snapshot().waveform_keys() == ("Q1.waveforms.pi_half",)
    assert library.snapshot().waveform_keys() == ()


def test_commit_reread_cleans_disk_without_resurrecting_deleted_root(
    tmp_path: Path,
) -> None:
    path = tmp_path / "module_cfg.yaml"
    path.write_text(
        HEADER + "components: {Q1: {waveforms: {bad: {style: const, length: 0.3}}}}\n"
    )
    library = open_library(path)
    capture = library.snapshot()
    with library.edit() as draft:
        draft.set_waveform("Q1.waveforms.new", ConstWaveformCfg(length=0.2))
        original = mixed(path)
    assert backups(path)[0].read_bytes() == original
    assert library.snapshot().waveform_keys() == (
        "Q1.waveforms.new",
        "Q1.waveforms.pi_half",
    )
    assert capture.waveform_keys() == ("Q1.waveforms.bad",)
    with pytest.raises(KeyError):
        library.snapshot().waveform_spec("Q1.waveforms.bad")


@dataclass(frozen=True)
class _ReadObservation:
    """One target-file I/O boundary in the child after its go message."""

    phase: Literal["enter", "leave"]  # Before the read or after return/raise.
    released: bool  # Parent ended its protected phase before this observation.


def _locked_reader(
    path: Path, connection: Connection, operation: str, release_intent: Event
) -> None:
    """Run the real library operation and report armed filesystem read boundaries.

    Setup reads precede ready and are excluded. After go, attempting is only a
    pre-call announcement, not lock arrival. Each observed read delegates to the
    original Path method and preserves its bytes or exception. The parent's
    release-intent event ends the protected phase before the actual unlock, so a
    correctly locked read cannot be mistaken for overlap after unlock.
    """
    observations: list[_ReadObservation] = []
    armed = False
    read_bytes = Path.read_bytes

    def observe_read(target: Path) -> bytes:
        if target != path or not armed:
            return read_bytes(target)
        observations.append(_ReadObservation("enter", release_intent.is_set()))
        try:
            return read_bytes(target)
        finally:
            observations.append(_ReadObservation("leave", release_intent.is_set()))

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Path, "read_bytes", observe_read)
            library = None if operation == "open" else open_library(path)
            if operation == "commit":
                if library is None:
                    raise RuntimeError("commit requires a bound library")
                with library.edit() as draft:
                    draft.set_waveform("Q1.waveforms.new", ConstWaveformCfg(length=0.2))
                    connection.send("ready")
                    # Parent sends go after ready while holding the sidecar lock.
                    assert connection.poll(30), "parent did not send go"
                    assert connection.recv() == "go"
                    armed = True
                    connection.send("attempting")
            else:
                connection.send("ready")
                # Parent sends go after ready while holding the sidecar lock.
                assert connection.poll(30), "parent did not send go"
                assert connection.recv() == "go"
                armed = True
                connection.send("attempting")
                if operation == "open":
                    library = open_library(path)
                elif library is None:
                    raise RuntimeError("refresh/edit requires a bound library")
                elif operation == "refresh":
                    library.refresh()
                else:
                    with library.edit():
                        pass
        connection.send(("done", observations))
    except Exception as exc:
        logging.exception("Locked reader failed")
        connection.send((repr(exc), observations))
    finally:
        connection.close()


@pytest.mark.parametrize("operation", ["open", "refresh", "enter", "commit"])
def test_cleanup_waits_for_sidecar_and_backs_up_latest_bytes(
    tmp_path: Path, operation: str
) -> None:
    path = tmp_path / "module_cfg.yaml"
    path.write_text(HEADER + "components: {Q1: {waveforms: {}}}\n")
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    release_intent = context.Event()
    process = context.Process(
        target=_locked_reader, args=(path, child, operation, release_intent)
    )
    try:
        process.start()
        child.close()
        # Started child completes its valid-file setup and then sends ready.
        assert parent.poll(30), "reader setup did not finish"
        assert parent.recv() == "ready"
        with FileLock(str(path) + ".lock"):
            parent.send("go")
            # Go releases the child's receive, which sends the pre-call message.
            assert parent.poll(30), "reader did not acknowledge go"
            assert parent.recv() == "attempting"
            # No public queue-arrival point: this only samples non-completion;
            # scheduling affects detection power, never correct-reader success.
            assert not parent.poll(0.15), "reader bypassed the sidecar lock"
            original = mixed(path)
            # End the protected phase BEFORE unlocking: no post-unlock race.
            release_intent.set()
        # Unlock lets the arranged real read/cleanup/commit finish and send done.
        assert parent.poll(30), "reader did not finish after unlock"
        status, observations = parent.recv()
        assert status == "done", status
        # Terminal was sent; only process shutdown remains, not lock acquisition.
        process.join(30)
        assert not process.is_alive(), "terminal reader did not exit"
        assert process.exitcode == 0
        assert observations, "operation did not read the target file"
        assert all(item.released for item in observations), observations
        assert [item.phase for item in observations] == ["enter", "leave"] * (
            len(observations) // 2
        )
        assert len(backups(path)) == 1
        assert backups(path)[0].read_bytes() == original
        snapshot = open_library(path).snapshot()
        expected = ("Q1.waveforms.pi_half",)
        if operation == "commit":
            expected = ("Q1.waveforms.new", *expected)
        assert snapshot.waveform_keys() == expected
        assert len(backups(path)) == 1, "cleaned file must not need another cleanup"
    finally:
        release_intent.set()
        try:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                # Already exited or terminated; watchdog only bounds OS shutdown.
                process.join(30)
                assert not process.is_alive(), "reader cleanup did not exit"
                process.close()
        finally:
            parent.close()
            child.close()
