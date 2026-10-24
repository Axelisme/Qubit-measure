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
from zcu_tools.core.element import ElementIdentity
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


def test_preview_preserves_author_element_identity_without_retargeting():
    ctrl = _ctrl()
    items = _items()
    items[0].element = ElementIdentity(index=2, name="Q3")
    ctrl.tab_control.get_tab_snapshot.side_effect = lambda tab_id: _snapshot(
        tab_id, items, has_writeback_draft=True
    )
    result = _dispatch(
        ctrl, "tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"}
    )
    candidates = result["items"]
    if not isinstance(candidates, list):
        raise TypeError("Expected writeback candidate list")
    candidate = candidates[0]
    assert candidate.get("element") == {"index": 2, "name": "Q3"}
    assert candidate["target_name"] == "r_f"
    assert candidate["actual_path"] == "r_f"
    assert candidate["proposed"] == 6012.3
    assert candidate["selected"] is True


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


# ---------------------------------------------------------------------------
# Complex parameter preview encodes proposed_value as {"__complex__": [re, im]}.
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


# ---------------------------------------------------------------------------
# Nested-list preview preserves a JSON-safe confusion matrix without a wire tag.
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
