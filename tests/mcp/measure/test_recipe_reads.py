"""Generic author observations through public tab handles and recording wire."""

from pathlib import Path

import pytest
from zcu_tools.mcp.measure.recipe import RecipeSession
from zcu_tools.mcp.measure.session import GuiRpcError

from ._recipe_support import LookbackGui, recipe_client


def test_read_value_tracks_sources_and_keeps_values_detached(tmp_path: Path) -> None:
    gui = LookbackGui()
    gui.md["$custom.threshold"] = {"levels": [17.5]}
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = len(client.transport.sent)
        value = tab.read_value("$custom.threshold")
        assert value == {"levels": [17.5]}
        gui.md["$custom.threshold"]["levels"][0] = 19.0
        assert value == {"levels": [17.5]}
        assert tab.read_value("$custom.threshold") == {"levels": [17.5]}
        assert client.transport.sent[before:] == [
            (
                "tab.parameter_read",
                {"tab_id": "t", "paths": ["$custom.threshold"], "module_refs": []},
            ),
            (
                "tab.parameter_read",
                {"tab_id": "t", "paths": ["$custom.threshold"], "module_refs": []},
            ),
        ]

        assert not gui.ran


def test_read_value_retains_missing_and_null(tmp_path: Path) -> None:
    gui = LookbackGui()
    gui.md["Q1.ext.null"] = None
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        assert tab.read_value("Q1.ext.missing") is None
        assert tab.read_value("Q1.ext.null") is None


def test_reads_keep_alias_copy_and_propagate_owner_failure(tmp_path: Path) -> None:
    gui = LookbackGui()
    gui.md["$custom.threshold"] = 17.5
    gui.library["$custom.module.pulse"] = {}
    with recipe_client(tmp_path, gui) as client:
        aliases = {"custom": "Q2"}
        tab = RecipeSession(client.context).open_tab("lookback", aliases=aliases)
        aliases["custom"] = "Q3"
        assert tab.read_value("$custom.threshold") == 17.5
        assert tab.read_library_root("$custom.module.pulse") == "$custom.module.pulse"
        reads = [
            params
            for method, params in client.transport.sent
            if method == "tab.parameter_read"
        ]
        assert reads == [
            {"tab_id": "t", "paths": ["$custom.threshold"], "module_refs": []},
            {"tab_id": "t", "paths": [], "module_refs": ["$custom.module.pulse"]},
        ]
        resets = [
            params
            for method, params in client.transport.sent
            if method == "tab.reset_cfg"
        ]
        assert resets == [{"tab_id": "t", "aliases": {"custom": "Q2"}}]
        client.transport.replies["tab.parameter_read"] = {
            "ok": False,
            "error": {
                "code": "failed_precondition",
                "message": "unbound custom",
                "reason": "unbound_alias",
            },
        }
        before = len(client.transport.sent)
        with pytest.raises(GuiRpcError) as failure:
            tab.read_value("$custom.threshold")
        assert failure.value.reason == "unbound_alias"
        assert [method for method, _ in client.transport.sent[before:]] == [
            "tab.parameter_read"
        ]


def test_read_library_root_uses_owner_result_and_none_skips_rpc(tmp_path: Path) -> None:
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = len(client.transport.sent)
        assert tab.read_library_root(None) is None
        assert len(client.transport.sent) == before
        client.transport.replies["tab.parameter_read"] = {
            "ok": True,
            "result": {
                "values": {},
                "modules": {
                    "$custom.module.pulse": "canonical-pulse",
                    "<Custom:pulse>": None,
                },
            },
        }
        assert tab.read_library_root("$custom.module.pulse") == "canonical-pulse"
        assert tab.read_library_root("<Custom:pulse>") is None
        client.transport.replies["tab.parameter_read"] = {
            "ok": True,
            "result": {"values": {}, "modules": {"pulse": 4}},
        }
        with pytest.raises(GuiRpcError) as failure:
            tab.read_library_root("pulse")
        assert failure.value.reason == "incompatible_wire"


def test_read_field_retains_range_input_and_directed_errors(tmp_path: Path) -> None:
    with recipe_client(tmp_path, LookbackGui()) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        with pytest.raises(GuiRpcError, match="Invalid"):
            tab.set_sweep(("range",), start={"expr": "1 / 0"})
        before = len(client.transport.sent)
        state = tab.read_field(("range",))
        if not isinstance(state.input, dict) or not isinstance(state.value, dict):
            raise TypeError("Expected range input/values")
        assert state.input["start"] == {"expr": "1 / 0"}
        assert state.value["stop"] == 2.0
        assert state.errors[0].path == ("range", "start")
        assert state.errors[0].message
        assert state.fields == ("start", "stop", "n")
        assert client.transport.sent[before:] == [
            ("tab.read_recipe_field", {"tab_id": "t", "path": ["range"]})
        ]


def test_read_field_observes_current_input_and_propagates_failed_edits(
    tmp_path: Path,
) -> None:
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = len(client.transport.sent)
        old = tab.read_field(("rounds",))
        assert old.input == old.value == 3
        assert tab.read_field(("fixed",)).value == 2
        ref = tab.read_field(("modules", "readout"))
        assert ref.mode == "inline"
        assert ref.key is None
        assert "freq" in ref.fields
        assert tab.read_field(("modules",)).fields == ("reset", "init_pulse", "readout")
        assert len(client.transport.sent) == before + 4
        tab.set(("rounds",), 7)
        current = tab.read_field(("rounds",))
        assert current.input == current.value == 7
        assert old.input == old.value == 3
        client.transport.replies["tab.edit_recipe_fields"] = {
            "ok": False,
            "error": {
                "code": "failed_precondition",
                "message": "stale cfg",
                "reason": "stale_cfg",
            },
        }
        before = len(client.transport.sent)
        with pytest.raises(GuiRpcError, match="stale cfg"):
            tab.set(("rounds",), 9)
        assert tab.read_field(("rounds",)) == current
        assert [
            method
            for method, _ in client.transport.sent[before:]
            if method != "tab.read_recipe_field"
        ] == ["tab.edit_recipe_fields"]


@pytest.mark.parametrize(
    "field", [(), ("missing",), ("modules", "", "readout"), ("rounds", "child")]
)
def test_read_field_rejects_malformed_paths_locally_and_unknown_paths_at_owner(
    tmp_path: Path, field: tuple[str | int, ...]
) -> None:
    with recipe_client(tmp_path, LookbackGui()) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = len(client.transport.sent)
        if not field or "" in field:
            with pytest.raises(ValueError, match="field"):
                tab.read_field(field)
            assert len(client.transport.sent) == before
        else:
            with pytest.raises(GuiRpcError) as caught:
                tab.read_field(field)
            assert caught.value.code == "invalid_params"
            assert caught.value.reason == "form_path"
            assert caught.value.data == {"form_path": list(field)}
            assert len(client.transport.sent) == before + 1
            assert tab.read_field(("rounds",)).input_present


def test_read_field_rejects_malformed_wire_without_retry(tmp_path: Path) -> None:
    with recipe_client(tmp_path, LookbackGui()) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        reply = client.context.gui.send_gui_rpc(
            "tab.read_recipe_field", {"tab_id": "t", "path": ["rounds"]}
        )
        reply["input_present"] = "yes"
        client.transport.replies["tab.read_recipe_field"] = {
            "ok": True,
            "result": reply,
        }
        before = len(client.transport.sent)
        with pytest.raises(GuiRpcError) as failure:
            tab.read_field(("rounds",))
        assert failure.value.reason == "incompatible_wire"
        assert client.transport.sent[before:] == [
            ("tab.read_recipe_field", {"tab_id": "t", "path": ["rounds"]})
        ]


def test_read_device_preserves_cached_unit_without_setup(tmp_path: Path) -> None:
    with recipe_client(tmp_path, LookbackGui()) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        client.transport.replies["device.snapshot"] = {
            "ok": True,
            "result": {
                "snapshot": {
                    "name": "bias",
                    "unit": {"coordinate": ["native"]},
                    "connected": False,
                }
            },
        }
        before = len(client.transport.sent)
        state = tab.read_device("bias")
        assert state.unit == {"coordinate": ["native"]}
        assert client.transport.sent[before:] == [("device.snapshot", {"name": "bias"})]
        client.transport.replies["device.snapshot"] = {
            "ok": True,
            "result": {"snapshot": {"name": "bias"}},
        }
        with pytest.raises(GuiRpcError) as failure:
            tab.read_device("bias")
        assert failure.value.reason == "incompatible_wire"


@pytest.mark.parametrize("name", ["", " "])
def test_read_device_rejects_blank_name_without_rpc(tmp_path: Path, name: str) -> None:
    with recipe_client(tmp_path, LookbackGui()) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = len(client.transport.sent)
        with pytest.raises(ValueError, match="device name must be non-empty"):
            tab.read_device(name)
        assert len(client.transport.sent) == before


@pytest.mark.parametrize("method", ["read_value", "read_library_root"])
def test_source_reads_reject_blank_request_without_rpc(
    tmp_path: Path, method: str
) -> None:
    with recipe_client(tmp_path, LookbackGui()) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = len(client.transport.sent)
        read = tab.read_value if method == "read_value" else tab.read_library_root
        with pytest.raises(ValueError, match="must be a non-empty"):
            read(" ")
        assert len(client.transport.sent) == before
