"""Writeback preview/set/apply dispatch handlers (ADR-0008 persistent draft).

Drives the handlers against a mock Controller whose get_tab_writeback_items
returns crafted persistent items, so we can assert preview serialization, the
tab.writeback_set edit path (the single editing surface — selected/target_name,
the metadict ``proposed_value`` facet and the module/waveform ``edits`` facet with
its internalized editor_id), and the enriched apply path, without a full
run+analyze pipeline.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import MetaDictWriteback, ModuleWriteback
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError


def _items() -> list:
    md = MetaDictWriteback(
        target_name="r_f", description="Resonator freq", proposed_value=6012.3
    )
    md.session_id = "md-1"
    mod = ModuleWriteback(
        target_name="readout_rf",
        description="readout module",
        edit_schema=MagicMock(),
        role_id="readout",
    )
    mod.session_id = "ml-1"
    mod.editor_id = "editor-9"
    return [md, mod]


def _ctrl() -> MagicMock:
    from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode
    from zcu_tools.gui.app.main.services.ports import (
        AnalysisPaneSnapshot,
        PathResourceSnapshot,
        PostAnalysisPaneSnapshot,
        TabSnapshot,
    )
    from zcu_tools.gui.cfg import CfgSchema

    def _snap(tab_id: str = "t", items=None):
        if items is None:
            items = _items()
        caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True, load_data=False, requires_soc=False)
        ana = AnalysisPaneSnapshot(params=None, result=object(), figure=None, writeback_items=tuple(items), image_path=PathResourceSnapshot(override=None, path=None))
        post = PostAnalysisPaneSnapshot(params=None, result=None, figure=None, writeback_items=tuple(), image_path=PathResourceSnapshot(override=None, path=None))
        return TabSnapshot(adapter_name="fake", cfg_schema=MagicMock(spec=CfgSchema), save_paths_override=None, tab_id=tab_id, interaction=MagicMock(), capabilities=caps, analyze_params=None, post_analyze_params=None, post_figure=None, writeback_items=tuple(items), figure=None, save_paths=None, result_source_path=None, run=None, analysis=ana, post_analysis=post, save=None, paths=None)

    ctrl = MagicMock()
    ctrl.has_tab.return_value = True
    ctrl.get_tab_writeback_items.side_effect = lambda tab_id: _items()
    ctrl.apply_writeback.return_value = {
        "applied_ids": ["md-1"],
        "written": {"md": ["r_f"], "ml_modules": [], "ml_waveforms": []},
    }
    ctrl.get_context_version.return_value = 7
    ctrl.set_writeback_item.return_value = {"valid": True, "removed": [], "added": []}
    # New pane-qualified controls
    ctrl.get_tab_snapshot.side_effect = lambda tab_id: _snap(tab_id)
    ctrl.get_exp_context.return_value = MagicMock(active_label="ctx001", chip_name="chip", qub_name="qubit", res_name="res", database_path="/tmp/db", result_dir="/tmp/result", is_active=lambda: True)
    # writeback_control pane methods
    wc = MagicMock()
    wc.has_tab.return_value = True
    wc.get_tab_writeback_items.side_effect = lambda tab_id: _items()
    wc.get_context_version.return_value = 7
    wc.set_writeback_item_for_pane.return_value = {"valid": True, "removed": [], "added": []}
    wc.apply_writeback_for_pane.return_value = {"applied_ids": ["md-1"], "written": {"md": ["r_f"], "ml_modules": [], "ml_waveforms": []}}
    wc.get_tab_snapshot = ctrl.get_tab_snapshot
    # attach to ctrl for dispatch helper fallback
    ctrl.tab_control = MagicMock()
    ctrl.tab_control.get_tab_snapshot.side_effect = lambda tab_id: _snap(tab_id)
    ctrl.tab_control.has_tab.return_value = True
    ctrl.writeback_control = wc
    ctrl.writeback_control.has_tab.return_value = True
    return ctrl


from ._helpers import dispatch_handler as _dispatch  # noqa: E402


def test_preview_serializes_metadict_and_module():
    ctrl = _ctrl()
    res = _dispatch(ctrl, "tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"})
    item_list = list(res["items"])  # type: ignore[call-overload]
    items = {it["id"]: it for it in item_list}

    md = items["md-1"]
    assert md["kind"] == "metadict"
    assert md["id"] == "md-1"
    assert md["target_name"] == "r_f"
    assert "key" not in md
    assert md["proposed_value"] == 6012.3

    mod = items["ml-1"]
    assert mod["kind"] == "module"
    assert mod["target_name"] == "readout_rf"
    assert mod["editor_id"] == "editor-9"
    assert mod["has_edit_schema"] is True
    assert mod["role_id"] == "readout"


def test_apply_reads_persistent_draft():
    ctrl = _ctrl()
    res = _dispatch(ctrl, "tab.writeback_apply", {"tab_id": "t", "subtab_id": "analysis"})
    assert res["applied_ids"] == ["md-1"]
    # apply now enriches: written (by kind) + post-apply context version + destination_context.
    assert res["written"] == {"md": ["r_f"], "ml_modules": [], "ml_waveforms": []}
    assert res["context_version"] == 7
    assert "destination_context" in res
    ctrl.writeback_control.apply_writeback_for_pane.assert_called_once_with("t", "analysis")


def test_preview_delegates_to_writeback_control_without_ctrl_fallback():
    ctrl = _ctrl()
    writeback_control = _ctrl().writeback_control
    tab_control = ctrl.tab_control
    ctrl.writeback_control = writeback_control
    # preview now reads via tab_control snapshot (pane-qualified), not directly via writeback_control items
    # but still requires writeback_control.has_tab for existence check
    ctrl.has_tab = MagicMock(
        side_effect=AssertionError("tab.writeback_preview must use writeback_control/tab_control")
    )
    ctrl.get_tab_writeback_items = MagicMock(
        side_effect=AssertionError("tab.writeback_preview must use writeback_control/tab_control")
    )

    res = _dispatch(ctrl, "tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"})

    assert res["has_draft"] is True
    # has_tab still via writeback_control
    writeback_control.has_tab.assert_called_once_with("t")
    # snapshot via tab_control
    tab_control.get_tab_snapshot.assert_called_with("t")
    ctrl.has_tab.assert_not_called()
    ctrl.get_tab_writeback_items.assert_not_called()


def test_set_delegates_to_writeback_control_without_ctrl_fallback():
    ctrl = _ctrl()
    writeback_control = _ctrl().writeback_control
    tab_control = ctrl.tab_control
    ctrl.writeback_control = writeback_control
    ctrl.has_tab = MagicMock(
        side_effect=AssertionError("tab.writeback_set must use writeback_control")
    )
    ctrl.set_writeback_item = MagicMock(
        side_effect=AssertionError("tab.writeback_set must use writeback_control")
    )
    ctrl.get_tab_writeback_items = MagicMock(
        side_effect=AssertionError("tab.writeback_set must use writeback_control")
    )

    res = _dispatch(
        ctrl,
        "tab.writeback_set",
        {"tab_id": "t", "subtab_id": "analysis", "id": "md-1", "selected": False},
    )

    assert res["item"]["id"] == "md-1"  # type: ignore[index]
    writeback_control.has_tab.assert_called_once_with("t")
    writeback_control.set_writeback_item_for_pane.assert_called_once_with(
        "t", "analysis", "md-1", selected=False
    )
    ctrl.has_tab.assert_not_called()
    ctrl.set_writeback_item.assert_not_called()
    ctrl.get_tab_writeback_items.assert_not_called()


def test_apply_delegates_to_writeback_control_without_ctrl_fallback():
    ctrl = _ctrl()
    writeback_control = _ctrl().writeback_control
    ctrl.writeback_control = writeback_control
    ctrl.has_tab = MagicMock(
        side_effect=AssertionError("tab.writeback_apply must use writeback_control")
    )
    ctrl.apply_writeback = MagicMock(
        side_effect=AssertionError("tab.writeback_apply must use writeback_control")
    )
    ctrl.resources_versions = MagicMock(
        side_effect=AssertionError("tab.writeback_apply must use writeback_control")
    )

    res = _dispatch(ctrl, "tab.writeback_apply", {"tab_id": "t", "subtab_id": "analysis"})

    assert res["context_version"] == 7
    assert "destination_context" in res
    writeback_control.has_tab.assert_called_once_with("t")
    writeback_control.apply_writeback_for_pane.assert_called_once_with("t", "analysis")
    writeback_control.get_context_version.assert_called_once_with()
    ctrl.has_tab.assert_not_called()
    ctrl.apply_writeback.assert_not_called()
    ctrl.resources_versions.assert_not_called()


def test_set_edits_metadict_item():
    ctrl = _ctrl()
    res = _dispatch(
        ctrl,
        "tab.writeback_set",
        {"tab_id": "t", "subtab_id": "analysis", "id": "md-1", "selected": False, "proposed_value": 6015.0},
    )
    ctrl.writeback_control.set_writeback_item_for_pane.assert_called_once()
    args, kwargs = ctrl.writeback_control.set_writeback_item_for_pane.call_args
    assert args[:3] == ("t", "analysis", "md-1")
    assert kwargs["selected"] is False
    assert kwargs["proposed_value"] == 6015.0
    # The reply echoes the (re-read) edited item.
    assert res["item"]["id"] == "md-1"  # type: ignore[index]


def test_set_module_cfg_edits_facet():
    """The module/waveform-only ``edits`` facet forwards an ordered {path,value}
    list to the service and folds the aggregated {valid, removed, added}; the
    agent never supplies an editor_id (it is resolved internally, ADR-0008)."""
    ctrl = _ctrl()
    ctrl.set_writeback_item.return_value = {
        "valid": True,
        "removed": ["readout_rf.old"],
        "added": ["readout_rf.new"],
    }
    ctrl.writeback_control.set_writeback_item_for_pane.return_value = {
        "valid": True,
        "removed": ["readout_rf.old"],
        "added": ["readout_rf.new"],
    }
    edits = [{"path": "readout_rf.freq", "value": 6012.3}]
    res = _dispatch(
        ctrl,
        "tab.writeback_set",
        {"tab_id": "t", "subtab_id": "analysis", "id": "ml-1", "edits": edits},
    )
    _, kwargs = ctrl.writeback_control.set_writeback_item_for_pane.call_args
    assert kwargs["edits"] == edits
    assert "editor_id" not in kwargs  # internalized — never on the wire
    # The aggregate is folded into the reply alongside the echoed item.
    assert res["item"]["id"] == "ml-1"  # type: ignore[index]
    assert res["valid"] is True
    assert res["removed"] == ["readout_rf.old"]
    assert res["added"] == ["readout_rf.new"]


def test_set_proposed_value_and_edits_mutually_exclusive():
    ctrl = _ctrl()
    with pytest.raises(RemoteError) as exc:
        _dispatch(
            ctrl,
            "tab.writeback_set",
            {
                "tab_id": "t", "subtab_id": "analysis",
                "id": "md-1",
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
            {"tab_id": "t", "subtab_id": "analysis", "id": "ml-1", "edits": [{"path": "p"}]},
        )
    assert exc.value.code is ErrorCode.INVALID_PARAMS


def test_set_target_name_override():
    ctrl = _ctrl()
    _dispatch(
        ctrl,
        "tab.writeback_set",
        {"tab_id": "t", "subtab_id": "analysis", "id": "ml-1", "target_name": "readout_v2"},
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
        {"tab_id": "t", "subtab_id": "analysis", "id": "md-1", "selected": False},
    )
    _, kwargs = ctrl.writeback_control.set_writeback_item_for_pane.call_args
    assert kwargs == {"selected": False}


def test_set_ignores_null_optionals():
    """The wire collapses 'omitted optional' to an explicit null-valued key.

    A retarget-only call on a module item must not forward ``selected`` or
    ``proposed_value`` just because they arrive as null — otherwise it flips the
    selection off and trips the metadict-only guard on a module item.
    """
    ctrl = _ctrl()
    _dispatch(
        ctrl,
        "tab.writeback_set",
        {
            "tab_id": "t", "subtab_id": "analysis",
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
            ctrl, "tab.writeback_set", {"tab_id": "t", "subtab_id": "analysis", "id": "md-1", "target_name": ""}
        )
    assert exc.value.code is ErrorCode.INVALID_PARAMS


def test_set_unknown_id_rejected():
    ctrl = _ctrl()
    ctrl.set_writeback_item.side_effect = InvalidInputError(
        "unknown writeback session_id"
    )
    with pytest.raises(RemoteError) as exc:
        _dispatch(
            ctrl, "tab.writeback_set", {"tab_id": "t", "subtab_id": "analysis", "id": "md-99", "selected": True}
        )
    assert exc.value.code is ErrorCode.INVALID_PARAMS


# ---------------------------------------------------------------------------
# Complex writeback scalar wire round-trip (WIRE 23). A complex metadict
# proposed_value serializes losslessly as {"__complex__": [re, im]} and a
# matching tag on tab.writeback_set coerces back to a Python complex.
# ---------------------------------------------------------------------------


def _complex_items() -> list:
    md = MetaDictWriteback(
        target_name="g_center",
        description="|g> IQ centre",
        proposed_value=complex(1.5, -2.25),
    )
    md.session_id = "md-1"
    return [md]


def test_preview_serializes_complex_as_tag():
    ctrl = _ctrl()
    # New handler reads via tab_control snapshot; mock that to return complex items
    from unittest.mock import MagicMock
    from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode
    from zcu_tools.gui.app.main.services.ports import AnalysisPaneSnapshot, PathResourceSnapshot, PostAnalysisPaneSnapshot, TabSnapshot
    from zcu_tools.gui.cfg import CfgSchema
    def _snap_complex(tab_id="t"):
        items = _complex_items()
        caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True, load_data=False, requires_soc=False)
        ana = AnalysisPaneSnapshot(params=None, result=object(), figure=None, writeback_items=tuple(items), image_path=PathResourceSnapshot(override=None, path=None))
        post = PostAnalysisPaneSnapshot(params=None, result=None, figure=None, writeback_items=tuple(), image_path=PathResourceSnapshot(override=None, path=None))
        return TabSnapshot(adapter_name="fake", cfg_schema=MagicMock(spec=CfgSchema), save_paths_override=None, tab_id=tab_id, interaction=MagicMock(), capabilities=caps, analyze_params=None, post_analyze_params=None, post_figure=None, writeback_items=tuple(items), figure=None, save_paths=None, result_source_path=None, run=None, analysis=ana, post_analysis=post, save=None, paths=None)
    ctrl.tab_control.get_tab_snapshot.side_effect = lambda tab_id: _snap_complex(tab_id)
    ctrl.writeback_control.get_tab_writeback_items.side_effect = lambda tab_id: _complex_items()
    res = _dispatch(ctrl, "tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"})
    item = list(res["items"])[0]  # type: ignore[call-overload]
    assert item["proposed_value"] == {"__complex__": [1.5, -2.25]}


def test_set_coerces_complex_tag_back_to_complex():
    ctrl = _ctrl()
    _dispatch(
        ctrl,
        "tab.writeback_set",
        {
            "tab_id": "t", "subtab_id": "analysis",
            "id": "md-1",
            "proposed_value": {"__complex__": [1.5, -2.25]},
        },
    )
    _, kwargs = ctrl.writeback_control.set_writeback_item_for_pane.call_args
    assert kwargs["proposed_value"] == complex(1.5, -2.25)
    assert isinstance(kwargs["proposed_value"], complex)


def test_complex_preview_set_round_trip_is_lossless():
    """preview tag -> set -> the same complex the service would apply."""
    ctrl = _ctrl()
    from unittest.mock import MagicMock
    from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode
    from zcu_tools.gui.app.main.services.ports import AnalysisPaneSnapshot, PathResourceSnapshot, PostAnalysisPaneSnapshot, TabSnapshot
    from zcu_tools.gui.cfg import CfgSchema
    def _snap_complex(tab_id="t"):
        items = _complex_items()
        caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True, load_data=False, requires_soc=False)
        ana = AnalysisPaneSnapshot(params=None, result=object(), figure=None, writeback_items=tuple(items), image_path=PathResourceSnapshot(override=None, path=None))
        post = PostAnalysisPaneSnapshot(params=None, result=None, figure=None, writeback_items=tuple(), image_path=PathResourceSnapshot(override=None, path=None))
        return TabSnapshot(adapter_name="fake", cfg_schema=MagicMock(spec=CfgSchema), save_paths_override=None, tab_id=tab_id, interaction=MagicMock(), capabilities=caps, analyze_params=None, post_analyze_params=None, post_figure=None, writeback_items=tuple(items), figure=None, save_paths=None, result_source_path=None, run=None, analysis=ana, post_analysis=post, save=None, paths=None)
    ctrl.tab_control.get_tab_snapshot.side_effect = lambda tab_id: _snap_complex(tab_id)
    ctrl.writeback_control.get_tab_writeback_items.side_effect = lambda tab_id: _complex_items()
    preview = _dispatch(ctrl, "tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"})
    wire_value = list(preview["items"])[0]["proposed_value"]  # type: ignore[call-overload]

    _dispatch(
        ctrl,
        "tab.writeback_set",
        {"tab_id": "t", "subtab_id": "analysis", "id": "md-1", "proposed_value": wire_value},
    )
    _, kwargs = ctrl.writeback_control.set_writeback_item_for_pane.call_args
    assert kwargs["proposed_value"] == complex(1.5, -2.25)


# ---------------------------------------------------------------------------
# Non-scalar (nested-list) writeback wire round-trip. A confusion matrix is
# already JSON-safe, so it needs no wire tag: it serializes verbatim through
# preview and passes through tab.writeback_set untouched (no _coerce_wire_value tag).
# ---------------------------------------------------------------------------


_CONFUSION = [[0.95, 0.03, 0.02], [0.03, 0.95, 0.02], [0.0, 0.0, 1.0]]


def _matrix_items() -> list:
    md = MetaDictWriteback(
        target_name="confusion_matrix",
        description="3x3 confusion matrix",
        proposed_value=_CONFUSION,
    )
    md.session_id = "md-1"
    return [md]


def test_preview_serializes_nested_list_verbatim():
    ctrl = _ctrl()
    from unittest.mock import MagicMock
    from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode
    from zcu_tools.gui.app.main.services.ports import AnalysisPaneSnapshot, PathResourceSnapshot, PostAnalysisPaneSnapshot, TabSnapshot
    from zcu_tools.gui.cfg import CfgSchema
    def _snap_matrix(tab_id="t"):
        items = _matrix_items()
        caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True, load_data=False, requires_soc=False)
        ana = AnalysisPaneSnapshot(params=None, result=object(), figure=None, writeback_items=tuple(items), image_path=PathResourceSnapshot(override=None, path=None))
        post = PostAnalysisPaneSnapshot(params=None, result=None, figure=None, writeback_items=tuple(), image_path=PathResourceSnapshot(override=None, path=None))
        return TabSnapshot(adapter_name="fake", cfg_schema=MagicMock(spec=CfgSchema), save_paths_override=None, tab_id=tab_id, interaction=MagicMock(), capabilities=caps, analyze_params=None, post_analyze_params=None, post_figure=None, writeback_items=tuple(items), figure=None, save_paths=None, result_source_path=None, run=None, analysis=ana, post_analysis=post, save=None, paths=None)
    ctrl.tab_control.get_tab_snapshot.side_effect = lambda tab_id: _snap_matrix(tab_id)
    ctrl.writeback_control.get_tab_writeback_items.side_effect = lambda tab_id: _matrix_items()
    res = _dispatch(ctrl, "tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"})
    item = list(res["items"])[0]  # type: ignore[call-overload]
    assert item["proposed_value"] == _CONFUSION


def test_set_passes_nested_list_through_untouched():
    ctrl = _ctrl()
    _dispatch(
        ctrl,
        "tab.writeback_set",
        {"tab_id": "t", "subtab_id": "analysis", "id": "md-1", "proposed_value": _CONFUSION},
    )
    _, kwargs = ctrl.writeback_control.set_writeback_item_for_pane.call_args
    assert kwargs["proposed_value"] == _CONFUSION


def test_nested_list_preview_set_round_trip_is_lossless():
    """preview verbatim -> set -> the same nested list the service would apply."""
    ctrl = _ctrl()
    from unittest.mock import MagicMock
    from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode
    from zcu_tools.gui.app.main.services.ports import AnalysisPaneSnapshot, PathResourceSnapshot, PostAnalysisPaneSnapshot, TabSnapshot
    from zcu_tools.gui.cfg import CfgSchema
    def _snap_matrix(tab_id="t"):
        items = _matrix_items()
        caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True, load_data=False, requires_soc=False)
        ana = AnalysisPaneSnapshot(params=None, result=object(), figure=None, writeback_items=tuple(items), image_path=PathResourceSnapshot(override=None, path=None))
        post = PostAnalysisPaneSnapshot(params=None, result=None, figure=None, writeback_items=tuple(), image_path=PathResourceSnapshot(override=None, path=None))
        return TabSnapshot(adapter_name="fake", cfg_schema=MagicMock(spec=CfgSchema), save_paths_override=None, tab_id=tab_id, interaction=MagicMock(), capabilities=caps, analyze_params=None, post_analyze_params=None, post_figure=None, writeback_items=tuple(items), figure=None, save_paths=None, result_source_path=None, run=None, analysis=ana, post_analysis=post, save=None, paths=None)
    ctrl.tab_control.get_tab_snapshot.side_effect = lambda tab_id: _snap_matrix(tab_id)
    ctrl.writeback_control.get_tab_writeback_items.side_effect = lambda tab_id: _matrix_items()
    preview = _dispatch(ctrl, "tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"})
    wire_value = list(preview["items"])[0]["proposed_value"]  # type: ignore[call-overload]

    _dispatch(
        ctrl,
        "tab.writeback_set",
        {"tab_id": "t", "subtab_id": "analysis", "id": "md-1", "proposed_value": wire_value},
    )
    _, kwargs = ctrl.writeback_control.set_writeback_item_for_pane.call_args
    assert kwargs["proposed_value"] == _CONFUSION
