from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.mcp.measure import server as mcp_server
from zcu_tools.mcp.measure.session_policy import (
    GUARD_DEPS,
    READ_REVEALS,
    describe_stale_keys,
)

from ._helpers import Fixture, dispatch_handler


@pytest.fixture(autouse=True)
def _qt(qapp):  # noqa: ARG001
    yield


@pytest.fixture()
def fixture(tmp_path: Path) -> Fixture:
    fx = Fixture()
    db_path = tmp_path / "Database" / "chip_a" / "q1" / "2026" / "06" / "Data_0626"
    fx.state.exp_context = replace(
        fx.state.exp_context,
        chip_name="chip_a",
        qub_name="q1",
        database_path=str(db_path),
    )
    return fx


def _recipe() -> dict[str, object]:
    return {
        "segments": [
            {"duration": 0.001, "formula": "sin(2*pi*t)"},
            {"duration": 0.001, "formula": "I*cos(2*pi*t)"},
        ],
        "normalize": "peak",
    }


def test_set_list_and_preview_round_trip(fixture: Fixture, tmp_path: Path) -> None:
    result = dispatch_handler(
        fixture.ctrl,
        "arb_waveform.set",
        {"name": "arb_wav1", "recipe": _recipe(), "overwrite": False},
    )

    assert result["success"] is True
    assert result["status"] == "created"
    assert Path(str(result["preview_figure"])).exists()
    assert fixture.state.version.get("arb_waveforms") == 1

    asset_path = (
        tmp_path / "Database" / "chip_a" / "q1" / "arb_waveforms" / "arb_wav1.npz"
    )
    assert asset_path.exists()

    listed = dispatch_handler(fixture.ctrl, "arb_waveform.list", {})
    assert listed == {"waveforms": ["arb_wav1"]}

    preview = dispatch_handler(
        fixture.ctrl,
        "arb_waveform.preview",
        {"name": "arb_wav1"},
    )
    assert preview["recipe"] == _recipe()
    assert Path(str(preview["preview_figure"])).exists()


def test_list_without_project_database_path_returns_empty(fixture: Fixture) -> None:
    fixture.state.exp_context = replace(fixture.state.exp_context, database_path="")

    assert fixture.ctrl.list_arb_waveforms() == []
    assert fixture.ctrl.list_arb_waveform_infos() == []
    assert dispatch_handler(fixture.ctrl, "arb_waveform.list", {}) == {"waveforms": []}


def test_set_existing_without_overwrite_reports_reason(fixture: Fixture) -> None:
    params = {"name": "arb_wav1", "recipe": _recipe(), "overwrite": False}
    dispatch_handler(fixture.ctrl, "arb_waveform.set", params)

    with pytest.raises(RemoteError) as exc:
        dispatch_handler(fixture.ctrl, "arb_waveform.set", params)

    assert exc.value.code is ErrorCode.PRECONDITION_FAILED
    assert exc.value.reason == "data_key_exists"
    assert exc.value.data == {"data_key": "arb_wav1"}
    assert fixture.state.version.get("arb_waveforms") == 1


def test_preview_missing_name_reports_available(fixture: Fixture) -> None:
    with pytest.raises(RemoteError) as exc:
        dispatch_handler(
            fixture.ctrl,
            "arb_waveform.preview",
            {"name": "missing"},
        )

    assert exc.value.code is ErrorCode.INVALID_PARAMS
    assert exc.value.reason == "data_key_not_found"
    data = exc.value.data
    assert data is not None
    assert data["available"] == []


def test_set_invalid_recipe_reports_reason(fixture: Fixture) -> None:
    with pytest.raises(RemoteError) as exc:
        dispatch_handler(
            fixture.ctrl,
            "arb_waveform.set",
            {
                "name": "arb_wav1",
                "recipe": {"segments": [{"duration": 0.001, "formula": "unknown(t)"}]},
                "overwrite": False,
            },
        )

    assert exc.value.code is ErrorCode.INVALID_PARAMS
    assert exc.value.reason == "invalid_recipe"
    assert fixture.state.version.get("arb_waveforms") == 0


def test_mcp_tools_have_requested_names_and_hidden_guard() -> None:
    for name in (
        "list_arb_waveform",
        "get_arb_waveform_preview",
        "set_arb_waveform",
    ):
        assert name in mcp_server.TOOLS

    assert "gui_arb_waveform_list" not in mcp_server.TOOLS
    assert "gui_arb_waveform_preview" not in mcp_server.TOOLS
    assert "gui_arb_waveform_set" not in mcp_server.TOOLS

    schema = mcp_server.TOOLS["set_arb_waveform"]["inputSchema"]
    assert set(schema["required"]) == {"name", "recipe"}
    assert set(schema["properties"]) == {"name", "recipe", "overwrite"}


def test_arb_waveform_version_policy() -> None:
    assert GUARD_DEPS["arb_waveform.set"] == ("arb_waveforms",)
    assert READ_REVEALS["arb_waveform.list"] == ("arb_waveforms",)
    assert READ_REVEALS["arb_waveform.preview"] == ("arb_waveforms",)
    assert describe_stale_keys(["arb_waveforms"]) == [
        "the arbitrary waveform asset store"
    ]
