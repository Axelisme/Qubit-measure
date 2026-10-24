"""Project-wide saved results through readonly, in-process RPC dispatch."""

from base64 import b64decode
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import TypeAdapter
from zcu_tools.datafile import Container, ContainerAttrs, JsonObject, save
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request
from zcu_tools.gui.session.context_types import EntryRequest
from zcu_tools.resources.entry import AcquiredPayload, LedgerEvent, ResultEntry

from tests.gui.app.measure._load_support import historical_metadata

from ._form_rpc import ImmediateScheduler

_JSON = TypeAdapter(JsonObject)
_ROWS = TypeAdapter(list[dict[str, object]])
_PNG = b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)


@dataclass
class ResultsRpc:
    """Drive the real request/reply contract as a readonly observer, without sockets."""

    controller: Controller
    remote: RemoteControlAdapter
    link: ClientLink

    def call(self, method: str, params: dict[str, object]) -> JsonObject:
        self.remote.route(self.link, Request("results-test", method, params))
        return _JSON.validate_json(self.link.outbound.get(timeout=2.0))

    def result(self, method: str, params: dict[str, object]) -> JsonObject:
        reply = self.call(method, params)
        assert reply["ok"] is True, reply
        return _JSON.validate_python(reply["result"], strict=True)

    def runs(self) -> list[dict[str, object]]:
        return _ROWS.validate_python(
            self.result("results.list", {})["runs"], strict=True
        )

    def error(self, method: str, params: dict[str, object]) -> JsonObject:
        reply = self.call(method, params)
        assert reply["ok"] is False, reply
        return _JSON.validate_python(reply["error"], strict=True)


@pytest.fixture
def results_rpc(qapp, tmp_path: Path) -> Iterator[ResultsRpc]:
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    controller = Controller(
        state, Registry(), None, BaseEventBus(), project_root=str(tmp_path)
    )
    remote = RemoteControlAdapter(
        controller, ControlOptions(port=0, app_slug=""), ImmediateScheduler()
    )
    link = ClientLink("results-observer", token_required=False)
    remote.on_client_open(link)
    rpc = ResultsRpc(controller, remote, link)
    rpc.result("control.connect", {"label": "results-reader", "mode": "readonly"})
    try:
        yield rpc
    finally:
        remote.on_client_close(link, on_owner_thread=True)
        controller.close()
        qapp.processEvents()


def _native_run(root: Path, entry: str, point: str, run_id: str) -> Path:
    run = root / "Database" / entry / "runs" / f"mux_scan_{run_id}"
    (run / "figures").mkdir(parents=True)
    (run / "figures" / "tone%2FQ2.png").write_bytes(_PNG)
    (run / "analysis").mkdir()
    (run / "analysis" / "primary.json").write_text(
        '{"summary":{"tones":[{"name":"Q2","frequency":4.2}]}}', encoding="utf-8"
    )
    metadata = historical_metadata()
    metadata = metadata.model_copy(
        update={
            "run_id": run_id,
            "experiment": "mux/scan",
            "snapshot": replace(
                metadata.snapshot,
                point=point,
                roles={"qubit": "Q1"},
            ),
        }
    )
    save(
        run / "data.h5",
        Container(
            vars={},
            docs={
                "metadata": _JSON.validate_python(metadata.model_dump(mode="json")),
                "cfg": {
                    "values": {"tones": [{"name": "Q1"}, {"name": "Q2"}]},
                    "cfg_type": "SyntheticMuxCfg",
                    "schema_version": "1.0",
                },
            },
            groups={},
            attrs=ContainerAttrs("3.0", "results-rpc-test", None, {}),
        ),
    )
    return run


@pytest.fixture
def saved_runs(tmp_path: Path) -> tuple[Path, Path, Path]:
    entry = ResultEntry.create(
        "A", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )
    first = _native_run(tmp_path, "A", "p1", "20261008T000000Z-aaaaaa")
    second = _native_run(tmp_path, "B", "p2", "20261008T000000Z-bbbbbb")
    entry.ledger.append(
        LedgerEvent(
            id=str(uuid4()),
            kind="acquired",
            at="2026-10-08T00:02:00Z",
            entry_id=entry.entry_id,
            origin="gui",
            payload=AcquiredPayload(
                run_id="20261008T000000Z-aaaaaa",
                experiment="mux/scan",
                point="p1",
                roles={"qubit": "Q1"},
            ),
        )
    )
    orphan = tmp_path / "Database" / "B" / "runs" / "figures-only"
    (orphan / "figures").mkdir(parents=True)
    (orphan / "figures" / "overview.png").write_bytes(_PNG)
    return first, second, orphan


def test_empty_project_results_are_readable_without_entry_or_control(
    results_rpc: ResultsRpc,
) -> None:
    assert results_rpc.result("entry.info", {})["entry"] is None
    assert results_rpc.result("results.list", {}) == {"runs": []}
    assert results_rpc.result("control.status", {})["me"] is False


def test_results_list_projects_all_entries_history_and_missing_mux_roles(
    results_rpc: ResultsRpc, saved_runs: tuple[Path, Path, Path]
) -> None:
    first, second, orphan = saved_runs
    rows = results_rpc.runs()
    assert len(rows) == 3
    by_path = {row["run_path"]: row for row in rows}
    registered = by_path[str(first)]
    assert registered["entry"] == "A"
    assert registered["run_id"] == "20261008T000000Z-aaaaaa"
    assert registered["experiment"] == "mux/scan"
    assert registered["point"] == "p1"
    assert registered["roles"] == {"qubit": "Q1"}
    assert registered["at"] == "2026-10-08T00:02:00Z"
    assert registered["registered"] is True
    assert registered["data_path"] == str(first / "data.h5")
    assert registered["figure_paths"] == [str(first / "figures" / "tone%2FQ2.png")]
    assert registered["figures"] == ["tone%2FQ2.png"]
    assert registered["analysis_paths"] == [str(first / "analysis" / "primary.json")]
    assert "q2" in str(registered["search_text"])
    assert registered["issues"] == []
    assert by_path[str(second)]["entry"] == "B"
    assert by_path[str(second)]["point"] == "p2"
    assert by_path[str(second)]["registered"] is False
    unknown = by_path[str(orphan)]
    assert unknown["registered"] is False
    assert unknown["run_id"] is None
    assert unknown["experiment"] is None
    assert unknown["at"] is None
    assert unknown["data_path"] is None
    assert unknown["figures"] == ["overview.png"]


def test_results_list_refreshes_disk_without_switching_opened_entry(
    results_rpc: ResultsRpc, saved_runs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    results_rpc.controller.setup_control.open_entry(EntryRequest("A"))
    before = results_rpc.result("entry.info", {})
    assert len(results_rpc.runs()) == 3
    _native_run(tmp_path, "C", "p3", "20261008T000000Z-cccccc")
    assert len(results_rpc.runs()) == 4
    assert results_rpc.result("entry.info", {}) == before


@pytest.mark.parametrize("run_number", [0, 1, 2])
def test_results_figure_reads_exact_indexed_png_for_known_or_unknown_history(
    results_rpc: ResultsRpc, saved_runs: tuple[Path, Path, Path], run_number: int
) -> None:
    row = next(
        row
        for row in results_rpc.runs()
        if row["run_path"] == str(saved_runs[run_number])
    )
    selectors = row["figures"]
    assert isinstance(selectors, list)
    reply = results_rpc.result(
        "results.figure", {"run_key": row["key"], "figure": selectors[0]}
    )
    assert reply["mime_type"] == "image/png"
    assert isinstance(reply["png_b64"], str)
    assert b64decode(reply["png_b64"], validate=True) == _PNG


@pytest.mark.parametrize("key", ["missing", "../../outside", "/etc/passwd"])
def test_results_figure_rejects_unknown_or_path_like_run_keys(
    results_rpc: ResultsRpc, saved_runs: tuple[Path, Path, Path], key: str
) -> None:
    error = results_rpc.error(
        "results.figure", {"run_key": key, "figure": "overview.png"}
    )
    assert (error["code"], error["reason"]) == ("invalid_params", "unknown_result_run")


@pytest.mark.parametrize("figure", ["missing.png", "../data.h5", "/etc/passwd"])
def test_results_figure_rejects_nonindexed_selectors(
    results_rpc: ResultsRpc, saved_runs: tuple[Path, Path, Path], figure: str
) -> None:
    row = results_rpc.runs()[0]
    error = results_rpc.error(
        "results.figure", {"run_key": row["key"], "figure": figure}
    )
    assert (error["code"], error["reason"]) == (
        "invalid_params",
        "unknown_result_figure",
    )


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"run_key": "key"},
        {"run_key": 17, "figure": "one.png"},
        {"run_key": "key", "figure": []},
    ],
)
def test_results_figure_rejects_malformed_requests(
    results_rpc: ResultsRpc, params: dict[str, object]
) -> None:
    with pytest.raises(RemoteError) as caught:
        results_rpc.remote.route(
            results_rpc.link, Request("results-test", "results.figure", params)
        )
    assert caught.value.code == ErrorCode.INVALID_PARAMS


def test_results_figure_preserves_unexpected_file_read_failure(
    results_rpc: ResultsRpc,
    saved_runs: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = next(
        row for row in results_rpc.runs() if row["run_path"] == str(saved_runs[2])
    )
    original = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        if path.name == "overview.png":
            raise PermissionError("saved PNG cannot be read")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    error = results_rpc.error(
        "results.figure", {"run_key": row["key"], "figure": "overview.png"}
    )
    assert error["code"] == "controller_error"
    assert error["message"] == "saved PNG cannot be read"
