"""Writeback preview and explicit apply against the real owner-thread socket."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.experiment.v2_gui.measure.adapters.fake import FakeAdapter
from zcu_tools.experiment.v2_gui.measure.adapters.fake.stub import FakeAnalyzeParams
from zcu_tools.gui.app.measure.adapter import (
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
)
from zcu_tools.gui.app.measure.cfg_schemas import (
    module_cfg_to_value,
    waveform_cfg_to_value,
)
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.program.v2 import ModuleCfgFactory, WaveformCfgFactory
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from ._helpers import Fixture, call, open_client


@pytest.mark.parametrize(
    "before,after,wire_before,wire_after",
    [
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6]]), [[1, 2], [3, 4]], [[5, 6]]),
        (1 + 2j, 3 + 4j, {"__complex__": [1.0, 2.0]}, {"__complex__": [3.0, 4.0]}),
        (
            {"nested": np.array([1 + 2j])},
            {"nested": [3 + 4j]},
            {"nested": [{"__complex__": [1.0, 2.0]}]},
            {"nested": [{"__complex__": [3.0, 4.0]}]},
        ),
    ],
)
def test_complete_md_values_survive_preview_and_write_socket(
    qapp,
    monkeypatch,
    before,
    after,
    wire_before,
    wire_after,
):
    monkeypatch.setattr(
        FakeAdapter,
        "get_writeback_items",
        lambda self, request: [
            MetaDictWriteback(
                target_name="value", description="V", proposed_value=after
            ),
        ],
    )
    fx = Fixture(active_label="ctx001")
    md = MetaDict()
    md.value = before
    fx.state.set_context(replace(fx.state.session_env, md=md, ml=ModuleLibrary()))
    fx.start()
    sock = open_client(fx.service.port)
    try:
        tab = fx.ctrl.new_tab("fake")
        for operation in (
            lambda: fx.ctrl.start_run(tab),
            lambda: fx.ctrl.analyze(tab, FakeAnalyzeParams()),
        ):
            assert (
                call(
                    sock,
                    "operation.await",
                    {
                        "operation_id": operation(),
                        "timeout": 2,
                    },
                )["result"]["status"]
                == "finished"
            )
        params = {"tab_id": tab, "subtab_id": "analysis"}
        preview = call(sock, "tab.writeback_preview", params)
        assert preview["ok"], preview
        item = preview["result"]["items"][0]
        assert item["current"] == wire_before
        assert item["proposed"] == wire_after
        assert item["proposed_value"] == wire_after
        assert call(sock, "tab.snapshot", {"tab_id": tab})["ok"]
        assert call(sock, "context.snapshot", {})["ok"]
        written = call(
            sock, "tab.writeback_write", {**params, "write": [{"id": "md-1"}]}
        )
        assert written["ok"], written
        assert written["result"] == {
            "written": [
                {
                    "id": "md-1",
                    "kind": "md",
                    "target": "value",
                    "before": wire_before,
                    "after": wire_after,
                }
            ]
        }
        # Unsupported live context values must fail rather than claim a full preview.
        md.value = object()
        rejected = call(sock, "tab.writeback_preview", params)
        assert not rejected["ok"]
        assert rejected["error"]["code"] == "precondition_failed"
        assert rejected["error"]["reason"] == "unserializable_context"
        for invalid in (
            np.array([float("nan")]),
            {"nested": [float("inf")]},
            complex(float("inf"), 1),
            complex(1, float("nan")),
        ):
            md.value = 1.0
            fx.ctrl.set_writeback_item_for_pane(
                tab, "analysis", "md-1", proposed_value=invalid
            )
            rejected_preview = call(sock, "tab.writeback_preview", params)
            assert not rejected_preview["ok"]
            assert rejected_preview["error"]["code"] == "precondition_failed"
            assert rejected_preview["error"]["reason"] == "unserializable_context"
            assert call(sock, "tab.snapshot", {"tab_id": tab})["ok"]
            assert call(sock, "context.snapshot", {})["ok"]
            rejected_write = call(
                sock,
                "tab.writeback_write",
                {
                    **params,
                    "write": [{"id": "md-1"}],
                },
            )
            assert not rejected_write["ok"]
            assert rejected_write["error"]["code"] == "precondition_failed"
            assert rejected_write["error"]["reason"] == "unserializable_context"
            # Reply projection fails after apply; no rollback is promised.
            rejected_current = call(sock, "tab.writeback_preview", params)
            assert not rejected_current["ok"]
            assert rejected_current["error"]["reason"] == "unserializable_context"
    finally:
        sock.close()
        fx.stop()


def test_batch_write_keeps_same_named_module_and_waveform_results(qapp, monkeypatch):
    module = ModuleCfgFactory.from_raw(
        {
            "type": "pulse",
            "ch": 0,
            "nqz": 1,
            "freq": 100.0,
            "gain": 0.5,
            "phase": 0.0,
            "pre_delay": 0.0,
            "post_delay": 0.0,
            "waveform": {"style": "const", "length": 0.1},
        }
    )
    waveform = WaveformCfgFactory.from_raw({"style": "const", "length": 0.1})
    module_schema = CfgSchema(*module_cfg_to_value(module))
    waveform_schema = CfgSchema(*waveform_cfg_to_value(waveform))
    monkeypatch.setattr(
        FakeAdapter,
        "get_writeback_items",
        lambda self, request: [
            ModuleWriteback(
                target_name="same", description="M", edit_schema=module_schema
            ),
            WaveformWriteback(
                target_name="same", description="W", edit_schema=waveform_schema
            ),
        ],
    )
    fx = Fixture(active_label="ctx001")
    ml = ModuleLibrary()
    ml.modules["same"] = module
    ml.waveforms["same"] = waveform
    fx.state.set_context(replace(fx.state.session_env, md=MetaDict(), ml=ml))
    before_module, before_waveform = module.to_dict(), waveform.to_dict()
    fx.start()
    sock = open_client(fx.service.port)
    try:
        tab = fx.ctrl.new_tab("fake")
        for operation in (
            lambda: fx.ctrl.start_run(tab),
            lambda: fx.ctrl.analyze(tab, FakeAnalyzeParams()),
        ):
            assert (
                call(
                    sock,
                    "operation.await",
                    {
                        "operation_id": operation(),
                        "timeout": 2,
                    },
                )["result"]["status"]
                == "finished"
            )
        assert call(sock, "tab.snapshot", {"tab_id": tab})["ok"]
        assert call(sock, "context.snapshot", {})["ok"]
        result = call(
            sock,
            "tab.writeback_write",
            {
                "tab_id": tab,
                "subtab_id": "analysis",
                "write": [
                    {"id": "ml-1", "edits": [{"path": "gain", "value": 0.8}]},
                    {"id": "wf-1", "edits": [{"path": "length", "value": 0.2}]},
                ],
            },
        )
        assert result["ok"], result
        assert ml.modules["same"].to_dict()["gain"] == 0.8
        assert ml.waveforms["same"].to_dict()["length"] == 0.2
        assert result["result"]["written"] == [
            {
                "id": "ml-1",
                "kind": "module",
                "target": "same",
                "before": before_module,
                "after": ml.modules["same"].to_dict(),
            },
            {
                "id": "wf-1",
                "kind": "waveform",
                "target": "same",
                "before": before_waveform,
                "after": ml.waveforms["same"].to_dict(),
            },
        ]
    finally:
        sock.close()
        fx.stop()


def test_live_preview_and_explicit_apply_preserve_other_gui_selection(
    qapp, monkeypatch
):
    monkeypatch.setattr(
        FakeAdapter,
        "get_writeback_items",
        lambda self, request: [
            MetaDictWriteback(target_name="a", description="A", proposed_value=11.0),
            MetaDictWriteback(target_name="b", description="B", proposed_value=22.0),
        ],
    )
    fx = Fixture(active_label="ctx001")
    md = MetaDict()
    md.a, md.b = 1.0, 2.0
    fx.state.set_context(replace(fx.state.session_env, md=md, ml=ModuleLibrary()))
    fx.start()
    sock = open_client(fx.service.port)
    try:
        tab = fx.ctrl.new_tab("fake")
        run = fx.ctrl.start_run(tab)
        assert (
            call(sock, "operation.await", {"operation_id": run, "timeout": 2})[
                "result"
            ]["status"]
            == "finished"
        )
        analyze = fx.ctrl.analyze(tab, FakeAnalyzeParams())
        assert (
            call(sock, "operation.await", {"operation_id": analyze, "timeout": 2})[
                "result"
            ]["status"]
            == "finished"
        )
        assert call(sock, "tab.snapshot", {"tab_id": tab})["ok"]
        assert call(sock, "context.snapshot", {})["ok"]
        params = {"tab_id": tab, "subtab_id": "analysis"}
        before = call(sock, "tab.writeback_preview", params)
        assert before["ok"]
        assert [
            (item["current"], item["proposed"]) for item in before["result"]["items"]
        ] == [(1.0, 11.0), (2.0, 22.0)]
        applied = call(sock, "tab.writeback_apply", {**params, "ids": ["md-1"]})
        assert applied["ok"]
        assert applied["result"]["applied_ids"] == ["md-1"]
        assert md.a == 11.0 and md.b == 2.0
        after = call(sock, "tab.writeback_preview", params)["result"]
        assert [item["current"] for item in after["items"]] == [11.0, 2.0]
        assert all(item["selected"] for item in after["items"])
        assert fx.ctrl.get_writeback_applied_for_pane(tab, "analysis") == {
            "md-1": True,
            "md-2": False,
        }
        assert call(sock, "tab.snapshot", {"tab_id": tab})["ok"]
        assert call(sock, "context.snapshot", {})["ok"]
        failed = call(
            sock,
            "tab.writeback_write",
            {
                **params,
                "write": [{"id": "md-2", "value": 33.0}, {"id": "missing"}],
            },
        )
        assert failed["ok"] is False
        assert failed["error"]["code"] == "invalid_params"
        assert md.b == 2.0
        retained = call(sock, "tab.writeback_preview", params)["result"]
        assert retained["items"][1]["proposed"] == 33.0
        assert all(item["selected"] for item in retained["items"])
        assert call(sock, "tab.snapshot", {"tab_id": tab})["ok"]
        assert call(sock, "context.snapshot", {})["ok"]
        written = call(
            sock,
            "tab.writeback_write",
            {
                **params,
                "write": [{"id": "md-2"}],
            },
        )
        assert written["ok"], written
        assert written["result"] == {
            "written": [
                {
                    "id": "md-2",
                    "kind": "md",
                    "target": "b",
                    "before": 2.0,
                    "after": 33.0,
                }
            ]
        }
        assert md.a == 11.0 and md.b == 33.0
    finally:
        sock.close()
        fx.stop()
