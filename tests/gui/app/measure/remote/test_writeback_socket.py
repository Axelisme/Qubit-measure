"""Writeback preview and explicit apply against the real owner-thread socket."""

import numpy as np
import pytest
from pydantic import ValidationError
from zcu_tools.cfg_model import ConstWaveformCfg, ModuleCfgFactory, PulseSpec
from zcu_tools.gui.app.measure.adapter import (
    ModuleWriteback,
    ParameterWriteback,
    WaveformWriteback,
)
from zcu_tools.gui.app.measure.cfg_schemas import (
    module_cfg_to_value,
    waveform_cfg_to_value,
)
from zcu_tools.gui.app.measure.services.writeback import WritebackEdit
from zcu_tools.gui.app.measure.state import RunPaneState
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.resources.entry import AcceptedWrite, RoleSpec

from tests.gui.app.measure._synthetic import SampleAdapter, SampleParams

from ._helpers import Fixture, call, open_client
from ._point_application import PointAdapter, PointApplication


def assert_rejected_write_preserves_context(fx, sock, tab, value):
    """Check typed native admission and pane write rejection with no storage effects."""
    context = fx.ctrl.context_control
    control = fx.ctrl.writeback_control
    control.set_writeback_item_for_pane(
        tab, "analysis", "param-1", proposed_value=value
    )
    captured = context.read_context()
    storage = context.capture_storage(captured)
    before_context = call(sock, "context.snapshot", {})
    assert before_context["ok"], before_context
    before_point = captured.document()
    before_events = storage.ledger.events()
    before_version = control.get_context_version()
    before_versions = fx.state.version.snapshot()
    for operation in (
        lambda: AcceptedWrite.model_validate(
            {"path": "general.ext.value", "value": value}
        ),
        lambda: control.get_writeback_values_for_pane(tab, "analysis"),
        lambda: control.apply_writeback_for_pane(
            tab, "analysis", item_ids=("param-1",)
        ),
        lambda: control.write_writeback_for_pane(
            tab, "analysis", (WritebackEdit(session_id="param-1"),)
        ),
    ):
        with pytest.raises(ValidationError) as rejected:
            operation()
        assert [(error["loc"], error["type"]) for error in rejected.value.errors()] == [
            (("value",), "value_error")
        ]
        assert call(sock, "context.snapshot", {}) == before_context
        current = context.read_context()
        assert current.document() == before_point
        assert current.target == captured.target
        assert current.roles == captured.roles
        assert current.aliases == captured.aliases
        assert context.capture_storage(current).ledger.events() == before_events
        assert control.get_context_version() == before_version
        assert fx.state.version.snapshot() == before_versions


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
        SampleAdapter,
        "get_writeback_items",
        lambda self, request: [
            ParameterWriteback(
                target_name="general.ext.value", description="V", proposed_value=after
            ),
        ],
    )
    fx = Fixture(active_label="ctx001")
    context = fx.ctrl.context_control
    target = context.read_context().target
    if isinstance(before, complex):
        admitted = AcceptedWrite.model_validate(
            {"path": "general.ext.value", "value": before}
        )
        context.set_parameter(target, "general.ext.value", admitted.value)
    else:
        # These original non-native values are rejection inputs, not stored seeds.
        context.set_parameter(target, "general.ext.value", 1.0)
    fx.start()
    sock = open_client(fx.service.port)
    try:
        tab = fx.ctrl.new_tab("fake")
        for operation in (
            lambda: fx.ctrl.start_run(tab),
            lambda: fx.ctrl.analyze(tab, SampleParams()),
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
        if isinstance(before, complex):
            preview = call(sock, "tab.writeback_preview", params)
            assert preview["ok"], preview
            item = preview["result"]["items"][0]
            assert item["current"] == wire_before
            assert item["proposed"] == wire_after
            assert item["proposed_value"] == wire_after
            assert call(sock, "tab.snapshot", {"tab_id": tab})["ok"]
            assert call(sock, "context.snapshot", {})["ok"]
            written = call(
                sock, "tab.writeback_write", {**params, "write": [{"id": "param-1"}]}
            )
            assert written["ok"], written
            assert written["result"] == {
                "written": [
                    {
                        "id": "param-1",
                        "kind": "parameter",
                        "target": "general.ext.value",
                        "before": wire_before,
                        "after": wire_after,
                    }
                ]
            }
        else:
            for invalid in (before, after):
                assert_rejected_write_preserves_context(fx, sock, tab, invalid)
            fx.ctrl.set_writeback_item_for_pane(
                tab, "analysis", "param-1", proposed_value=1.0
            )
        for invalid in (
            np.array([float("nan")]),
            {"nested": [float("inf")]},
            complex(float("inf"), 1),
            complex(1, float("nan")),
        ):
            context.set_parameter(target, "general.ext.value", 1.0)
            assert_rejected_write_preserves_context(fx, sock, tab, invalid)
    finally:
        sock.close()
        fx.stop()


def test_batch_write_keeps_same_named_module_and_waveform_results(
    qapp, monkeypatch, point_models, point_registry_guard
):
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
    waveform = ConstWaveformCfg(length=0.1)
    module_schema = CfgSchema(*module_cfg_to_value(module))
    waveform_schema = CfgSchema(*waveform_cfg_to_value(waveform))
    monkeypatch.setattr(
        SampleAdapter,
        "get_writeback_items",
        lambda self, request: [
            ModuleWriteback(
                target_name="Q1.pulses.same", description="M", edit_schema=module_schema
            ),
            WaveformWriteback(
                target_name="Q1.waveforms.same",
                description="W",
                edit_schema=waveform_schema,
            ),
        ],
    )
    fx = Fixture(active_label="ctx001")
    context = fx.ctrl.context_control
    target = context.read_context().target
    context.add_component("Q1", "fake/drive/a", {})
    context.set_module(
        target, "Q1.pulses.same", PulseSpec.model_validate(module.to_dict())
    )
    context.set_waveform(target, "Q1.waveforms.same", waveform)
    before_module = PulseSpec.model_validate(module.to_dict()).model_dump(mode="python")
    before_waveform = waveform.model_dump(mode="python")
    after_module = PulseSpec.model_validate(
        {**module.to_dict(), "gain": 0.8}
    ).model_dump(mode="python")
    after_waveform = ConstWaveformCfg(length=0.2).model_dump(mode="python")
    fx.start()
    sock = open_client(fx.service.port)
    try:
        tab = fx.ctrl.new_tab("fake")
        for operation in (
            lambda: fx.ctrl.start_run(tab),
            lambda: fx.ctrl.analyze(tab, SampleParams()),
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
        assert (
            fx.ctrl.read_context()
            .library.module_spec("Q1.pulses.same")
            .model_dump(mode="python")["gain"]
            == 0.8
        )
        assert (
            fx.ctrl.read_context()
            .library.waveform_spec("Q1.waveforms.same")
            .model_dump(mode="python")["length"]
            == 0.2
        )
        assert result["result"]["written"] == [
            {
                "id": "ml-1",
                "kind": "module",
                "target": "Q1.pulses.same",
                "before": before_module,
                "after": after_module,
            },
            {
                "id": "wf-1",
                "kind": "waveform",
                "target": "Q1.waveforms.same",
                "before": before_waveform,
                "after": after_waveform,
            },
        ]
    finally:
        sock.close()
        fx.stop()


def test_live_preview_and_explicit_apply_preserve_other_gui_selection(
    qapp, monkeypatch
):
    monkeypatch.setattr(
        SampleAdapter,
        "get_writeback_items",
        lambda self, request: [
            ParameterWriteback(
                target_name="general.ext.a", description="A", proposed_value=11.0
            ),
            ParameterWriteback(
                target_name="general.ext.b", description="B", proposed_value=22.0
            ),
        ],
    )
    fx = Fixture(active_label="ctx001")
    context = fx.ctrl.context_control
    context.set_parameter(context.read_context().target, "general.ext.a", 1.0)
    context.set_parameter(context.read_context().target, "general.ext.b", 2.0)
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
        analyze = fx.ctrl.analyze(tab, SampleParams())
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
        applied = call(sock, "tab.writeback_apply", {**params, "ids": ["param-1"]})
        assert applied["ok"]
        assert applied["result"]["applied_ids"] == ["param-1"]
        assert (
            context.read_context().lookup("general.ext.a").value == 11.0
            and context.read_context().lookup("general.ext.b").value == 2.0
        )
        after = call(sock, "tab.writeback_preview", params)["result"]
        assert [item["current"] for item in after["items"]] == [11.0, 2.0]
        assert all(item["selected"] for item in after["items"])
        assert fx.ctrl.get_writeback_applied_for_pane(tab, "analysis") == {
            "param-1": True,
            "param-2": False,
        }
        assert call(sock, "tab.snapshot", {"tab_id": tab})["ok"]
        assert call(sock, "context.snapshot", {})["ok"]
        failed = call(
            sock,
            "tab.writeback_write",
            {
                **params,
                "write": [{"id": "param-2", "value": 33.0}, {"id": "missing"}],
            },
        )
        assert failed["ok"] is False
        assert failed["error"]["code"] == "invalid_params"
        assert context.read_context().lookup("general.ext.b").value == 2.0
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
                "write": [{"id": "param-2"}],
            },
        )
        assert written["ok"], written
        assert written["result"] == {
            "written": [
                {
                    "id": "param-2",
                    "kind": "parameter",
                    "target": "general.ext.b",
                    "before": 2.0,
                    "after": 33.0,
                }
            ]
        }
        assert (
            context.read_context().lookup("general.ext.a").value == 11.0
            and context.read_context().lookup("general.ext.b").value == 33.0
        )
    finally:
        sock.close()
        fx.stop()


def test_native_preview_and_window_keep_captured_path_and_stderr(
    application: PointApplication, qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qtpy.QtWidgets import QLabel

    # Only the domain fit is replaced. The real analysis owner creates its WB
    # draft, so State/facet/preview use the same opaque-draft lifetime owner.
    monkeypatch.setattr(PointAdapter, "analyze", lambda self, req, *, plots: 1.0)
    monkeypatch.setattr(
        PointAdapter,
        "get_writeback_items",
        lambda self, req: [
            ParameterWriteback(
                "$driver.ext.value", "Value", proposed_value=20.09, stderr=0.18
            ),
            ParameterWriteback("$driver.ext.other", "Other", proposed_value=23.0),
        ],
    )
    application.open_entry()
    controller = application.controller
    context = controller.context_control
    context.add_component(
        "Q1",
        "fake/drive/a",
        {"rate": 10.0, "ext": {"value": 19.8}},
        setup=True,
    )
    context.add_component(
        "Q2",
        "fake/drive/a",
        {"rate": 20.0, "ext": {"value": 29.8}},
        setup=True,
    )
    application.result("point.new", {"label": "a"})
    context.set_alias_rows((AliasRow("driver", "Q1"),))
    captured = context.read_context().bind_roles(
        {"driver": RoleSpec("fake/drive/*")}, focus="Q1", explicit={}
    )
    tab = controller.new_tab("pointed")
    application.state.replace_run_pane(
        tab, RunPaneState(result=1.0, parameters=captured)
    )
    analyzed = application.result(
        "operation.await",
        {"operation_id": controller.analyze(tab, SampleParams()), "timeout": 2},
    )
    assert analyzed["status"] == "finished", analyzed
    context.set_parameter(captured.target, "Q1.ext.value", 19.9)
    application.result("point.new", {"label": "b"})
    context.set_alias_rows((AliasRow("driver", "Q2"),))
    active = context.read_context().target
    window = MainWindow(controller)
    window.add_tab_widget(tab, "pointed")
    try:
        preview = application.result(
            "tab.writeback_preview", {"tab_id": tab, "subtab_id": "analysis"}
        )
        items = preview["items"]
        assert isinstance(items, list)
        item = items[0]
        assert isinstance(item, dict)
        assert item["target_name"] == "$driver.ext.value"
        assert item["actual_path"] == "Q1.ext.value"
        assert item["stderr"] == 0.18
        assert item["current"] == 19.9
        assert item["proposed"] == 20.09
        other = items[1]
        assert isinstance(other, dict)
        assert other["actual_path"] == "Q1.ext.other"
        assert other["stderr"] is None
        assert other["current"] is None
        window.refresh_tab_writeback(tab)
        qapp.processEvents()
        texts = [label.text() for label in window.findChildren(QLabel)]
        assert any("Q1.ext.value" in text for text in texts)
        assert any("±" in text and "0.18" in text for text in texts)
        assert any("19.9" in text for text in texts)
        assert context.read_context().target == active
        assert context.get_alias_rows() == (AliasRow("driver", "Q2"),)
    finally:
        controller.tab_control.close_tab(tab)
        window.deleteLater()
        qapp.processEvents()
