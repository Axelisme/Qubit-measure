"""Writeback preview/set/apply dispatch handlers (ADR-0008 persistent draft).

Drives the handlers against pane-qualified snapshot and writeback controls
with crafted persistent items, so we can assert preview serialization and the
pane-qualified edit path (selected/target_name, the parameter
``proposed_value`` facet and the module/waveform ``edits`` facet with its
internalized editor identity), without a full run+analyze pipeline.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    ModuleWriteback,
    ParameterWriteback,
    WritebackItem,
)
from zcu_tools.gui.app.measure.services.ports import (
    AnalysisPaneSnapshot,
    PostAnalysisPaneSnapshot,
    TabSnapshot,
)
from zcu_tools.gui.app.measure.writeback import WritebackValues
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from ._helpers import dispatch_handler as _dispatch


def _items() -> list[WritebackItem]:
    md = ParameterWriteback(
        target_name="r_f", description="Resonator freq", proposed_value=6012.3
    )
    md.session_id = "param-1"
    mod = ModuleWriteback(
        target_name="readout_rf",
        description="readout module",
        proposal=PulseSpec(
            ch=0, nqz=1, freq=6000.0, gain=0.5, waveform=ConstWaveformCfg(length=1.0)
        ),
        role_id="readout",
    )
    mod.session_id = "ml-1"
    return [md, mod]


def _snapshot(
    tab_id: str, items: list[WritebackItem], *, has_writeback_draft: bool
) -> TabSnapshot:
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT,
        post_analysis=True,
        load_data=False,
        requires_soc=False,
    )
    ana = AnalysisPaneSnapshot(
        params=None,
        result=object(),
        figures=None,
        writeback_items=tuple(items),
        has_writeback_draft=has_writeback_draft,
    )
    post = PostAnalysisPaneSnapshot(
        params=None,
        result=None,
        figures=None,
        writeback_items=tuple(),
    )
    return TabSnapshot(
        adapter_name="fake",
        cfg_input={},
        tab_id=tab_id,
        interaction=MagicMock(),
        capabilities=caps,
        run=None,
        analysis=ana,
        post_analysis=post,
        save=None,
    )


def _ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.has_tab.return_value = True
    ctrl.get_context_version.return_value = 7
    # New pane-qualified controls
    ctrl.get_tab_snapshot.side_effect = lambda tab_id: _snapshot(
        tab_id, _items(), has_writeback_draft=True
    )
    ctrl.get_session_env.return_value = MagicMock(
        active_label="ctx001",
        chip_name="chip",
        qub_name="qubit",
        res_name="res",
        database_path="/tmp/db",
        result_dir="/tmp/result",
        is_active=lambda: True,
    )
    # writeback_control pane methods
    wc = MagicMock()
    wc.has_tab.return_value = True
    wc.get_context_version.return_value = 7
    wc.get_writeback_values_for_pane.return_value = {
        "param-1": WritebackValues(
            actual_path="r_f", current=6000.0, proposed=6012.3, unit=None
        ),
        "ml-1": WritebackValues(
            actual_path="readout_rf",
            current=None,
            proposed={"type": "pulse", "gain": 0.5},
            unit=None,
        ),
    }
    wc.set_writeback_item_for_pane.return_value = None
    wc.apply_writeback_for_pane.return_value = {
        "applied_ids": ["param-1"],
        "written": {"parameter": ["r_f"], "ml_modules": []},
    }
    wc.get_tab_snapshot = ctrl.get_tab_snapshot
    # attach to ctrl for dispatch helper fallback
    ctrl.tab_control = MagicMock()
    ctrl.tab_control.get_tab_snapshot.side_effect = lambda tab_id: _snapshot(
        tab_id, _items(), has_writeback_draft=True
    )
    ctrl.tab_control.has_tab.return_value = True
    ctrl.writeback_control = wc
    ctrl.writeback_control.has_tab.return_value = True
    return ctrl


def test_entry_edit_reply_is_lean_and_preserves_invalid_authored_input():
    from pydantic import TypeAdapter
    from zcu_tools.gui.app.measure.remote.form_observation import FormObservation
    from zcu_tools.gui.session.context_types import SavedPoint
    from zcu_tools.gui.session.library_entry import ModuleEntryForm
    from zcu_tools.gui.session.parameters import ParameterContext
    from zcu_tools.resources.context.library import (
        LibrarySnapshot,
        ModuleLibraryDocument,
    )
    from zcu_tools.resources.entry.schema import PointDocument, PointGeneral

    parameters = ParameterContext(
        point=PointDocument(
            format="zcu.parameter-container",
            format_version="2.0",
            general=PointGeneral(created_at="2026-10-15T00:00:00Z"),
        ),
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
        ),
        target=SavedPoint(entry_id="synthetic", point="working"),
        roles=None,
    )
    entry = ModuleEntryForm(
        PulseSpec(
            ch=0, nqz=1, freq=100.0, gain=0.2, waveform=ConstWaveformCfg(length=0.1)
        ),
        parameters,
        arb_names=(),
    )
    entry.state.write({("freq",): {"expr": "1 / 0"}}, entry.env)
    ctrl = _ctrl()
    ctrl.writeback_control.get_writeback_item_form_for_pane.return_value = entry
    reply = _dispatch(
        ctrl,
        "tab.writeback_set",
        {
            "tab_id": "t",
            "subtab_id": "analysis",
            "id": "ml-1",
            "edits": [{"path": ["gain"], "value": 0.3}],
        },
    )
    assert set(reply) == {"item", "input", "values", "errors"}
    observed = TypeAdapter(FormObservation).validate_python(reply)
    assert observed["input"]["freq"] == {"expr": "1 / 0"}
    assert observed["errors"][0]["path"] == ["freq"]
    assert entry.parameters is parameters


def test_preview_serializes_parameter_and_module():
    ctrl = _ctrl()
    res = _dispatch(
        ctrl, "tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"}
    )
    item_list = res["items"]
    assert isinstance(item_list, list)
    items = {it["id"]: it for it in item_list}

    md = items["param-1"]
    assert md["kind"] == "parameter"
    assert md["id"] == "param-1"
    assert md["target_name"] == "r_f"
    assert "key" not in md
    assert md["proposed_value"] == 6012.3

    mod = items["ml-1"]
    assert mod["kind"] == "module"
    assert mod["target_name"] == "readout_rf"
    assert mod["role_id"] == "readout"


def test_preview_serializes_full_service_values_at_the_requested_pane():
    ctrl = _ctrl()
    result = _dispatch(
        ctrl, "tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"}
    )
    assert isinstance(result["items"], list)
    items = {item["id"]: item for item in result["items"]}
    assert items["param-1"]["current"] == 6000.0
    assert items["param-1"]["proposed"] == 6012.3
    assert items["ml-1"]["current"] is None
    assert items["ml-1"]["proposed"] == {"type": "pulse", "gain": 0.5}
    ctrl.writeback_control.get_writeback_values_for_pane.assert_called_once_with(
        "t", "analysis"
    )


def test_preview_distinguishes_empty_draft_from_missing_draft():
    ctrl = _ctrl()
    snapshot_factory = ctrl.tab_control.get_tab_snapshot.side_effect
    snapshot = snapshot_factory("t")
    assert snapshot.analysis is not None
    ctrl.tab_control.get_tab_snapshot.side_effect = None
    ctrl.tab_control.get_tab_snapshot.return_value = replace(
        snapshot,
        analysis=replace(snapshot.analysis, writeback_items=()),
    )

    res = _dispatch(
        ctrl, "tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"}
    )

    assert res["has_draft"] is True
    assert res["items"] == []


def test_apply_reads_persistent_draft():
    ctrl = _ctrl()
    res = _dispatch(
        ctrl, "tab.writeback_apply", {"tab_id": "t", "subtab_id": "analysis"}
    )
    assert res["applied_ids"] == ["param-1"]
    # apply now enriches: written (by kind) + post-apply context version + destination_context.
    assert res["written"] == {
        "parameter": ["r_f"],
        "ml_modules": [],
    }
    assert res["context_version"] == 7
    assert "destination_context" in res
    ctrl.writeback_control.apply_writeback_for_pane.assert_called_once_with(
        "t", "analysis", item_ids=None
    )


@pytest.mark.parametrize("ids", [["param-1"], []])
def test_apply_passes_explicit_ids_without_editing_selection(ids):
    ctrl = _ctrl()
    _dispatch(
        ctrl,
        "tab.writeback_apply",
        {
            "tab_id": "t",
            "subtab_id": "analysis",
            "ids": ids,
        },
    )
    ctrl.writeback_control.apply_writeback_for_pane.assert_called_once_with(
        "t", "analysis", item_ids=tuple(ids)
    )
    ctrl.writeback_control.set_writeback_item_for_pane.assert_not_called()


@pytest.mark.parametrize("ids", ["param-1", [""], [1]])
def test_apply_rejects_malformed_ids_before_service(ids):
    ctrl = _ctrl()
    with pytest.raises(RemoteError):
        _dispatch(
            ctrl,
            "tab.writeback_apply",
            {
                "tab_id": "t",
                "subtab_id": "analysis",
                "ids": ids,
            },
        )
    ctrl.writeback_control.apply_writeback_for_pane.assert_not_called()


def test_destination_context_projection_does_not_hide_programmer_errors():
    ctrl = _ctrl()
    error = RuntimeError("context projection bug")
    ctrl.get_session_env.side_effect = error

    with pytest.raises(RuntimeError) as exc_info:
        _dispatch(
            ctrl, "tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"}
        )

    assert exc_info.value is error


def test_preview_delegates_to_writeback_control_without_ctrl_fallback():
    ctrl = _ctrl()
    writeback_control = _ctrl().writeback_control
    tab_control = ctrl.tab_control
    ctrl.writeback_control = writeback_control
    # preview now reads via tab_control snapshot (pane-qualified), not directly via writeback_control items
    # but still requires writeback_control.has_tab for existence check
    ctrl.has_tab = MagicMock(
        side_effect=AssertionError(
            "tab.writeback_preview must use writeback_control/tab_control"
        )
    )

    res = _dispatch(
        ctrl, "tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"}
    )

    assert res["has_draft"] is True
    # has_tab still via writeback_control
    writeback_control.has_tab.assert_called_once_with("t")
    # snapshot via tab_control
    tab_control.get_tab_snapshot.assert_called_with("t")
    ctrl.has_tab.assert_not_called()


def test_set_delegates_to_writeback_control_without_ctrl_fallback():
    ctrl = _ctrl()
    writeback_control = _ctrl().writeback_control
    ctrl.writeback_control = writeback_control
    ctrl.has_tab = MagicMock(
        side_effect=AssertionError("tab.writeback_set must use writeback_control")
    )

    res = _dispatch(
        ctrl,
        "tab.writeback_set",
        {"tab_id": "t", "subtab_id": "analysis", "id": "param-1", "selected": False},
    )

    item = res["item"]
    assert isinstance(item, dict)
    assert item["id"] == "param-1"
    writeback_control.has_tab.assert_called_once_with("t")
    writeback_control.set_writeback_item_for_pane.assert_called_once_with(
        "t", "analysis", "param-1", selected=False
    )
    ctrl.has_tab.assert_not_called()


def test_apply_delegates_to_writeback_control_without_ctrl_fallback():
    ctrl = _ctrl()
    writeback_control = _ctrl().writeback_control
    ctrl.writeback_control = writeback_control
    ctrl.has_tab = MagicMock(
        side_effect=AssertionError("tab.writeback_apply must use writeback_control")
    )

    res = _dispatch(
        ctrl, "tab.writeback_apply", {"tab_id": "t", "subtab_id": "analysis"}
    )

    assert res["context_version"] == 7
    assert "destination_context" in res
    writeback_control.has_tab.assert_called_once_with("t")
    writeback_control.apply_writeback_for_pane.assert_called_once_with(
        "t", "analysis", item_ids=None
    )
    writeback_control.get_context_version.assert_called_once_with()
    ctrl.has_tab.assert_not_called()


def test_set_edits_parameter_item():
    ctrl = _ctrl()
    res = _dispatch(
        ctrl,
        "tab.writeback_set",
        {
            "tab_id": "t",
            "subtab_id": "analysis",
            "id": "param-1",
            "selected": False,
            "proposed_value": 6015.0,
        },
    )
    ctrl.writeback_control.set_writeback_item_for_pane.assert_called_once()
    args, kwargs = ctrl.writeback_control.set_writeback_item_for_pane.call_args
    assert args[:3] == ("t", "analysis", "param-1")
    assert kwargs["selected"] is False
    assert kwargs["proposed_value"] == 6015.0
    # The reply echoes the (re-read) edited item.
    item = res["item"]
    assert isinstance(item, dict)
    assert item["id"] == "param-1"


def test_set_proposed_value_and_edits_mutually_exclusive():
    ctrl = _ctrl()
    with pytest.raises(RemoteError) as exc:
        _dispatch(
            ctrl,
            "tab.writeback_set",
            {
                "tab_id": "t",
                "subtab_id": "analysis",
                "id": "param-1",
                "proposed_value": 1.0,
                "edits": [{"path": "p", "value": 1}],
            },
        )
    assert exc.value.code is ErrorCode.INVALID_PARAMS
    ctrl.writeback_control.set_writeback_item_for_pane.assert_not_called()


def test_set_edits_malformed_entry_rejected():
    ctrl = _ctrl()
    with pytest.raises(RemoteError) as exc:
        _dispatch(
            ctrl,
            "tab.writeback_set",
            {
                "tab_id": "t",
                "subtab_id": "analysis",
                "id": "ml-1",
                "edits": [{"path": "p"}],
            },
        )
    assert exc.value.code is ErrorCode.INVALID_PARAMS


def test_set_target_name_override():
    ctrl = _ctrl()
    _dispatch(
        ctrl,
        "tab.writeback_set",
        {
            "tab_id": "t",
            "subtab_id": "analysis",
            "id": "ml-1",
            "target_name": "readout_v2",
        },
    )
    _, kwargs = ctrl.writeback_control.set_writeback_item_for_pane.call_args
    assert kwargs["target_name"] == "readout_v2"


def test_set_deselect_flows_through():
    """``selected=False`` is a legit falsy value — it must reach the service.

    The ``params.get('selected') is not None`` guard must distinguish a real
    ``False`` from an omitted/null optional.
    """
    ctrl = _ctrl()
    _dispatch(
        ctrl,
        "tab.writeback_set",
        {"tab_id": "t", "subtab_id": "analysis", "id": "param-1", "selected": False},
    )
    _, kwargs = ctrl.writeback_control.set_writeback_item_for_pane.call_args
    assert kwargs == {"selected": False}


def test_set_ignores_null_optionals():
    """The wire collapses 'omitted optional' to an explicit null-valued key.

    A retarget-only call on a module item must not forward ``selected`` or
    ``proposed_value`` just because they arrive as null — otherwise it flips the
    selection off and trips the parameter-only guard on a module item.
    """
    ctrl = _ctrl()
    _dispatch(
        ctrl,
        "tab.writeback_set",
        {
            "tab_id": "t",
            "subtab_id": "analysis",
            "id": "ml-1",
            "selected": None,
            "target_name": "readout_v2",
            "proposed_value": None,
        },
    )
    _, kwargs = ctrl.writeback_control.set_writeback_item_for_pane.call_args
    assert kwargs == {"target_name": "readout_v2"}
    assert "selected" not in kwargs
    assert "proposed_value" not in kwargs


def test_set_empty_target_name_rejected():
    ctrl = _ctrl()
    with pytest.raises(RemoteError) as exc:
        _dispatch(
            ctrl,
            "tab.writeback_set",
            {
                "tab_id": "t",
                "subtab_id": "analysis",
                "id": "param-1",
                "target_name": "",
            },
        )
    assert exc.value.code is ErrorCode.INVALID_PARAMS


def test_set_unknown_id_rejected():
    ctrl = _ctrl()
    ctrl.writeback_control.set_writeback_item_for_pane.side_effect = InvalidInputError(
        "unknown writeback session_id"
    )
    with pytest.raises(RemoteError) as exc:
        _dispatch(
            ctrl,
            "tab.writeback_set",
            {
                "tab_id": "t",
                "subtab_id": "analysis",
                "id": "param-99",
                "selected": True,
            },
        )
    assert exc.value.code is ErrorCode.INVALID_PARAMS


# ---------------------------------------------------------------------------
# Complex writeback scalar wire round-trip (WIRE 23). A complex parameter
# proposed_value serializes losslessly as {"__complex__": [re, im]} and a
# matching tag on tab.writeback_set coerces back to a Python complex.
# ---------------------------------------------------------------------------


def _complex_items() -> list[WritebackItem]:
    md = ParameterWriteback(
        target_name="g_center",
        description="|g> IQ centre",
        proposed_value=complex(1.5, -2.25),
    )
    md.session_id = "param-1"
    return [md]


def test_preview_serializes_complex_as_tag():
    ctrl = _ctrl()
    ctrl.writeback_control.get_writeback_values_for_pane.return_value = {
        "param-1": WritebackValues(
            actual_path="g_center",
            current=None,
            proposed=complex(1.5, -2.25),
            unit=None,
        ),
    }
    ctrl.tab_control.get_tab_snapshot.side_effect = lambda tab_id: _snapshot(
        tab_id, _complex_items(), has_writeback_draft=True
    )
    res = _dispatch(
        ctrl, "tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"}
    )
    # tolerated: The handler returns a list of wire items; the response annotation erases container types to object.
    item = list(res["items"])[0]  # pyright: ignore[reportArgumentType]
    assert item["proposed_value"] == {"__complex__": [1.5, -2.25]}
    assert item["proposed"] == {"__complex__": [1.5, -2.25]}


def test_set_coerces_complex_tag_back_to_complex():
    ctrl = _ctrl()
    _dispatch(
        ctrl,
        "tab.writeback_set",
        {
            "tab_id": "t",
            "subtab_id": "analysis",
            "id": "param-1",
            "proposed_value": {"__complex__": [1.5, -2.25]},
        },
    )
    _, kwargs = ctrl.writeback_control.set_writeback_item_for_pane.call_args
    assert kwargs["proposed_value"] == complex(1.5, -2.25)
    assert isinstance(kwargs["proposed_value"], complex)


def test_complex_preview_set_round_trip_is_lossless():
    """preview tag -> set -> the same complex the service would apply."""
    ctrl = _ctrl()
    ctrl.tab_control.get_tab_snapshot.side_effect = lambda tab_id: _snapshot(
        tab_id, _complex_items(), has_writeback_draft=True
    )
    preview = _dispatch(
        ctrl, "tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"}
    )
    # tolerated: The handler returns a list of wire items; the response annotation erases container types to object.
    wire_value = list(preview["items"])[0]["proposed_value"]  # pyright: ignore[reportArgumentType]

    _dispatch(
        ctrl,
        "tab.writeback_set",
        {
            "tab_id": "t",
            "subtab_id": "analysis",
            "id": "param-1",
            "proposed_value": wire_value,
        },
    )
    _, kwargs = ctrl.writeback_control.set_writeback_item_for_pane.call_args
    assert kwargs["proposed_value"] == complex(1.5, -2.25)


# ---------------------------------------------------------------------------
# Non-scalar (nested-list) writeback wire round-trip. A confusion matrix is
# already JSON-safe, so it needs no wire tag: it serializes verbatim through
# preview and passes through tab.writeback_set untouched (no _coerce_wire_value tag).
# ---------------------------------------------------------------------------


_CONFUSION = [[0.95, 0.03, 0.02], [0.03, 0.95, 0.02], [0.0, 0.0, 1.0]]


def _matrix_items() -> list[WritebackItem]:
    md = ParameterWriteback(
        target_name="confusion_matrix",
        description="3x3 confusion matrix",
        proposed_value=_CONFUSION,
    )
    md.session_id = "param-1"
    return [md]


def test_preview_serializes_nested_list_verbatim():
    ctrl = _ctrl()
    ctrl.writeback_control.get_writeback_values_for_pane.return_value = {
        "param-1": WritebackValues(
            actual_path="confusion_matrix", current=None, proposed=_CONFUSION, unit=None
        ),
    }
    ctrl.tab_control.get_tab_snapshot.side_effect = lambda tab_id: _snapshot(
        tab_id, _matrix_items(), has_writeback_draft=True
    )
    res = _dispatch(
        ctrl, "tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"}
    )
    # tolerated: The handler returns a list of wire items; the response annotation erases container types to object.
    item = list(res["items"])[0]  # pyright: ignore[reportArgumentType]
    assert item["proposed_value"] == _CONFUSION
    assert item["proposed"] == _CONFUSION


def test_set_passes_nested_list_through_untouched():
    ctrl = _ctrl()
    _dispatch(
        ctrl,
        "tab.writeback_set",
        {
            "tab_id": "t",
            "subtab_id": "analysis",
            "id": "param-1",
            "proposed_value": _CONFUSION,
        },
    )
    _, kwargs = ctrl.writeback_control.set_writeback_item_for_pane.call_args
    assert kwargs["proposed_value"] == _CONFUSION


def test_nested_list_preview_set_round_trip_is_lossless():
    """preview verbatim -> set -> the same nested list the service would apply."""
    ctrl = _ctrl()
    ctrl.tab_control.get_tab_snapshot.side_effect = lambda tab_id: _snapshot(
        tab_id, _matrix_items(), has_writeback_draft=True
    )
    preview = _dispatch(
        ctrl, "tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"}
    )
    # tolerated: The handler returns a list of wire items; the response annotation erases container types to object.
    wire_value = list(preview["items"])[0]["proposed_value"]  # pyright: ignore[reportArgumentType]

    _dispatch(
        ctrl,
        "tab.writeback_set",
        {
            "tab_id": "t",
            "subtab_id": "analysis",
            "id": "param-1",
            "proposed_value": wire_value,
        },
    )
    _, kwargs = ctrl.writeback_control.set_writeback_item_for_pane.call_args
    assert kwargs["proposed_value"] == _CONFUSION
