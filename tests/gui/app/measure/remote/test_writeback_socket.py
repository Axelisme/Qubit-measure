"""Writeback preview and explicit apply against the real owner-thread socket."""

from dataclasses import replace
from pathlib import Path
from socket import socket
from uuid import UUID

import numpy as np
import pytest
from pydantic import ValidationError
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.datafile import load, save
from zcu_tools.experiment.native_persistence import metadata_from_container
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    ModuleWriteback,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.state import RunPaneState
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.app.measure.writeback import WritebackEdit
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import CommandOrigin, PointCreateRequest
from zcu_tools.resources.entry import AcceptedWrite, RoleSpec

from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._load_support import write_native_file
from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    PaneSummary,
)
from tests.gui.app.measure._synthetic import (
    MODULE_PROPOSAL_TARGET,
    MODULE_PROPOSAL_WAVEFORM,
    ModuleProposalAdapter,
    SampleAdapter,
    SampleParams,
)

from ._helpers import Fixture, call, mcp_client, open_client
from ._point_application import PointAdapter, PointApplication


def assert_terminal_finished(fx: Fixture, client: socket, operation_id: int) -> None:
    """Deliver the arranged synthetic operation and assert its terminal wire status."""
    # The admitted offline run/fit returns; only its owner delivery ends this wait.
    qt_until(
        lambda: fx.ctrl.active_operation_count() == 0,
        label="writeback socket operation terminal",
    )
    reply = call(
        client, "operation.await", {"operation_id": operation_id, "timeout": 0}
    )
    assert reply["result"]["status"] == "finished", reply


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
    tmp_path,
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
    fx = Fixture(active_label="ctx001", project_root=str(tmp_path / "project"))
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
            assert_terminal_finished(fx, sock, operation())
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


@pytest.fixture
def module_fixture(tmp_path, qapp, point_models, point_registry_guard):
    """Return a started GUI fixture whose Q1 holds a pulse using a waveform key."""
    fx = Fixture(active_label="ctx001", project_root=str(tmp_path / "project"))
    fx.registry.register("synthetic/module_writeback", ModuleProposalAdapter)
    context = fx.ctrl.context_control
    target = context.read_context().target
    context.add_component("Q1", "fake/drive/a", {})
    context.set_waveform(target, MODULE_PROPOSAL_WAVEFORM, ConstWaveformCfg(length=0.1))
    context.set_module(target, MODULE_PROPOSAL_TARGET, _stored_pulse())
    fx.start()
    try:
        yield fx
    finally:
        fx.stop()


def _stored_pulse(**changes: object) -> PulseSpec:
    """Return the stored pulse; the synthetic proposal differs only in gain 0.3."""
    fields: dict[str, object] = {
        "ch": 0,
        "nqz": 1,
        "freq": 5000.0,
        "gain": 0.2,
        "waveform": MODULE_PROPOSAL_WAVEFORM,
    }
    return PulseSpec.model_validate(fields | changes)


def _analyzed_module_tab(fx: Fixture, sock) -> str:
    tab = fx.ctrl.new_tab("synthetic/module_writeback")
    for operation in (
        lambda: fx.ctrl.start_run(tab),
        lambda: fx.ctrl.analyze(tab, SampleParams()),
    ):
        assert_terminal_finished(fx, sock, operation())
    assert call(sock, "tab.snapshot", {"tab_id": tab})["ok"]
    assert call(sock, "context.snapshot", {})["ok"]
    return tab


def test_module_preview_proposes_the_current_shape(module_fixture):
    sock = open_client(module_fixture.service.port)
    try:
        tab = _analyzed_module_tab(module_fixture, sock)
        preview = call(
            sock, "tab.writeback_preview", {"tab_id": tab, "subtab_id": "analysis"}
        )
        assert preview["ok"], preview
        (item,) = preview["result"]["items"]
        assert (item["id"], item["kind"]) == ("ml-1", "module")
        assert item["actual_path"] == MODULE_PROPOSAL_TARGET
        assert item["current"] == _stored_pulse().model_dump(mode="json")
        assert item["proposed"] == _stored_pulse(gain=0.3).model_dump(mode="json")
    finally:
        sock.close()


def test_module_edits_by_path_reply_with_form_and_write_the_edited_spec(
    module_fixture,
):
    sock = open_client(module_fixture.service.port)
    try:
        tab = _analyzed_module_tab(module_fixture, sock)
        params = {"tab_id": tab, "subtab_id": "analysis"}
        edited = call(
            sock,
            "tab.writeback_set",
            {**params, "id": "ml-1", "edits": [{"path": ["freq"], "value": 5100.0}]},
        )
        assert edited["ok"], edited
        reply = edited["result"]
        assert reply["item"]["id"] == "ml-1"
        assert reply["errors"] == []
        assert reply["input"]["freq"] == 5100.0
        assert reply["input"]["waveform"] == {
            "kind": "ref",
            "name": MODULE_PROPOSAL_WAVEFORM,
        }
        assert {"path": ["freq"], "value": 5100.0} in reply["values"]
        assert set(reply) == {"item", "input", "values", "errors"}
        written = call(
            sock,
            "tab.writeback_write",
            {
                **params,
                "write": [
                    {
                        "id": "ml-1",
                        "edits": [{"path": ["post_delay"], "value": 0.05}],
                    }
                ],
            },
        )
        assert written["ok"], written
        expected = _stored_pulse(gain=0.3, freq=5100.0, post_delay=0.05)
        assert written["result"]["written"] == [
            {
                "id": "ml-1",
                "kind": "module",
                "target": MODULE_PROPOSAL_TARGET,
                "before": _stored_pulse().model_dump(mode="json"),
                "after": expected.model_dump(mode="json"),
            }
        ]
        library = module_fixture.ctrl.read_context().library
        assert library.module_spec(MODULE_PROPOSAL_TARGET) == expected
    finally:
        sock.close()


def test_invalid_module_edit_rejects_write_with_field_path(module_fixture):
    sock = open_client(module_fixture.service.port)
    try:
        tab = _analyzed_module_tab(module_fixture, sock)
        params = {"tab_id": tab, "subtab_id": "analysis"}
        edited = call(
            sock,
            "tab.writeback_set",
            {**params, "id": "ml-1", "edits": [{"path": ["gain"], "value": "bad"}]},
        )
        assert edited["ok"], edited
        assert [error["path"] for error in edited["result"]["errors"]] == [["gain"]]
        before = module_fixture.ctrl.read_context().library.document()
        rejected = call(
            sock, "tab.writeback_write", {**params, "write": [{"id": "ml-1"}]}
        )
        assert rejected["ok"] is False
        assert rejected["error"]["code"] == "invalid_params"
        assert [error["path"] for error in rejected["error"]["data"]["errors"]] == [
            ["gain"]
        ]
        assert module_fixture.ctrl.read_context().library.document() == before
        unknown = call(
            sock,
            "tab.writeback_set",
            {**params, "id": "ml-1", "edits": [{"path": ["unknown"], "value": 1}]},
        )
        assert unknown["ok"] is False
        assert unknown["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_apply_writeback_tool_writes_module_proposal_by_id(
    module_fixture, tmp_path, request
):
    sock = open_client(module_fixture.service.port)
    try:
        tab = _analyzed_module_tab(module_fixture, sock)
    finally:
        sock.close()
    _, invoke = mcp_client(module_fixture.service.port, tmp_path, request=request)
    invoke("connect", {"port": module_fixture.service.port})
    receipt = invoke("apply_writeback", {"tab": tab})
    assert receipt["status"] == "finished", receipt
    library = module_fixture.ctrl.read_context().library
    assert library.module_spec(MODULE_PROPOSAL_TARGET) == _stored_pulse(gain=0.3)


def test_live_preview_and_explicit_apply_preserve_other_gui_selection(
    tmp_path, qapp, monkeypatch
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
    fx = Fixture(active_label="ctx001", project_root=str(tmp_path / "project"))
    context = fx.ctrl.context_control
    context.set_parameter(context.read_context().target, "general.ext.a", 1.0)
    context.set_parameter(context.read_context().target, "general.ext.b", 2.0)
    fx.start()
    sock = open_client(fx.service.port)
    try:
        tab = fx.ctrl.new_tab("fake")
        run = fx.ctrl.start_run(tab)
        assert_terminal_finished(fx, sock, run)
        analyze = fx.ctrl.analyze(tab, SampleParams())
        assert_terminal_finished(fx, sock, analyze)
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


def terminal_result(
    application: PointApplication, operation_id: int
) -> dict[str, object]:
    """Deliver the arranged synthetic operation, then read its terminal wire result."""
    # The admitted synthetic fit returns; deliver its real terminal before polling.
    qt_until(
        lambda: application.controller.active_operation_count() == 0,
        label="captured writeback fit terminal",
    )
    return application.result(
        "operation.await", {"operation_id": operation_id, "timeout": 0}
    )


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
        tab, RunPaneState(result=RunRecord(None, 1.0), parameters=captured)
    )
    analyzed = terminal_result(application, controller.analyze(tab, SampleParams()))
    assert analyzed["status"] == "finished", analyzed
    context.set_parameter(captured.target, "Q1.ext.value", 19.9)
    application.result("point.new", {"label": "b"})
    context.set_alias_rows((AliasRow("driver", "Q2"),))
    active = context.read_context().target
    window = MainWindow(controller, templates=lambda: {})
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


class CapturedCouplerAdapter(AnalysisSavingAdapter):
    """Produce a concrete library destination and an explicit captured alias."""

    def get_writeback_items(
        self,
        req: WritebackRequest[RunRecord[SavingCfg, SavingResult], PaneSummary],
    ) -> list[WritebackItem]:
        parameters = req.ctx.parameters
        if parameters is None or req.run_result.cfg is None:
            raise ValueError("This proposal needs a captured source and native cfg")
        return [
            ModuleWriteback(
                parameters.library_key("coupler.pulses.cz_f2"),
                "Pulse",
                _stored_pulse(gain=req.run_result.cfg.gain),
            ),
            ParameterWriteback(
                "$coupler.rate",
                "Rate",
                proposed_value=parameters.value("$coupler.rate"),
            ),
        ]


@pytest.fixture
def coupler_load_application(qapp, tmp_path: Path, point_models, point_registry_guard):
    """Register a synthetic producer and native history without executing Run."""
    fx = Fixture(project_root=str(tmp_path), headless=True)
    fx.registry.register("captured-coupler", CapturedCouplerAdapter)
    context = fx.ctrl.context_control
    context.add_component("coupler", "fake/drive/a", {"rate": 0.4})
    target = context.read_context().target
    context.set_waveform(target, MODULE_PROPOSAL_WAVEFORM, ConstWaveformCfg(length=0.1))
    context.set_module(target, "coupler.pulses.cz_f2", _stored_pulse(gain=0.2))
    context.set_alias_rows((AliasRow("coupler", "coupler"),))
    path = tmp_path / "source.h5"
    write_native_file(path, tag=CapturedCouplerAdapter.experiment_tag or "", gain=0.4)
    stored = load(path)
    metadata = metadata_from_container(stored).model_copy(
        update={
            "snapshot": replace(
                metadata_from_container(stored).snapshot,
                entry_id=UUID(target.entry_id),
                point=target.point,
                roles={"coupler": "coupler"},
            )
        }
    )
    save(
        path,
        replace(
            stored, docs={**stored.docs, "metadata": metadata.model_dump(mode="json")}
        ),
        replace=True,
    )
    tab = fx.ctrl.new_tab("captured-coupler")
    port = fx.start()
    try:
        with open_client(port) as client:
            yield fx, client, tab, path
    finally:
        fx.stop()


def test_registered_source_only_load_analyze_preview_keeps_concrete_collision_target(
    coupler_load_application,
) -> None:
    fx, client, tab, path = coupler_load_application
    context = fx.ctrl.context_control
    source = context.read_context().target
    before_form = call(client, "tab.get_cfg", {"tab_id": tab})
    loaded = call(client, "tab.load_data", {"tab_id": tab, "data_path": str(path)})
    assert loaded["ok"], loaded
    assert call(client, "tab.get_cfg", {"tab_id": tab}) == before_form
    context.set_module(source, "coupler.pulses.cz_f2", _stored_pulse(gain=0.6))
    context.set_parameter(source, "coupler.rate", 0.7)
    fx.ctrl.setup_control.new_point(
        PointCreateRequest("other", "", CommandOrigin("gui"))
    )
    context.add_component("Q2", "fake/drive/a", {"rate": 9.0})
    context.set_alias_rows((AliasRow("coupler", "Q2"),))
    active = context.read_context().target
    started = call(client, "tab.analyze", {"tab_id": tab})
    assert started["ok"], started
    assert_terminal_finished(fx, client, started["result"]["operation_id"])
    preview = call(
        client, "tab.writeback_preview", {"tab_id": tab, "subtab_id": "analysis"}
    )
    assert preview["ok"], preview
    pulse, rate = preview["result"]["items"]
    assert pulse["actual_path"] == "coupler.pulses.cz_f2"
    assert pulse["current"]["gain"] == 0.6
    assert pulse["proposed"]["gain"] == 0.4
    assert rate["actual_path"] == "coupler.rate"
    assert (rate["current"], rate["proposed"]) == (0.7, 0.4)
    assert context.read_context().target == active
    assert context.get_alias_rows() == (AliasRow("coupler", "Q2"),)
