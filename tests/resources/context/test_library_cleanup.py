"""Disk-only cleanup through the public library handle, using synthetic YAML."""

import json
import logging
import multiprocessing
import os
from multiprocessing.connection import Connection
from pathlib import Path
from typing import IO, Never

import pytest
from filelock import FileLock
from ruamel.yaml import YAML
from zcu_tools.cfg_model import ConstWaveformCfg
from zcu_tools.resources.context.library import ModuleLibrary
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
    with pytest.raises(ValueError, match="components|provenance|[Ss]ource|[Ii]nvalid"):
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
        library.snapshot().get_waveform("Q1.waveforms.bad")


def _locked_reader(path: Path, connection: Connection, operation: str) -> None:
    try:
        library = None if operation == "open" else open_library(path)
        if operation == "commit":
            if library is None:
                raise RuntimeError("commit requires a bound library")
            with library.edit() as draft:
                draft.set_waveform("Q1.waveforms.new", ConstWaveformCfg(length=0.2))
                connection.send("ready")
                connection.recv()
                connection.send("attempting")
        else:
            connection.send("ready")
            connection.recv()
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
        connection.send("done")
    except Exception as exc:
        logging.exception("Locked reader failed")
        connection.send(repr(exc))
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
    process = context.Process(target=_locked_reader, args=(path, child, operation))
    process.start()
    try:
        assert parent.poll(5)
        assert parent.recv() == "ready"
        with FileLock(str(path) + ".lock"):
            parent.send("go")
            assert parent.poll(5)
            assert parent.recv() == "attempting"
            assert not parent.poll(0.15), "reader bypassed the sidecar lock"
            original = mixed(path)
        assert parent.poll(5)
        assert parent.recv() == "done"
        process.join(5)
        assert process.exitcode == 0
        assert backups(path)[0].read_bytes() == original
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
        child.close()
