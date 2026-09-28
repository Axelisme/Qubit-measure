"""Writeback preview and explicit apply against the real owner-thread socket."""

from dataclasses import replace

from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.experiment.v2_gui.adapters.fake.stub import FakeAnalyzeParams
from zcu_tools.gui.app.main.adapter import MetaDictWriteback
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

from ._helpers import Fixture, call, open_client


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
    fx.state.set_context(replace(fx.state.exp_context, md=md, ml=ModuleLibrary()))
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
