"""Full native parameter-context reads and error projections."""

from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from zcu_tools.cfg_model import ArbWaveformCfg, PulseSpec, waveform_spec_from_raw
from zcu_tools.gui.remote.errors import ErrorCode
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import ContextSourceFault
from zcu_tools.gui.session.events import MdChangedPayload, MlChangedPayload
from zcu_tools.gui.session.parameters import ParameterContext, RoleBinding
from zcu_tools.resources.entry import ComponentSchema, ResultEntry, component_registry
from zcu_tools.resources.waveform_assets import ArbWaveformError

from ._helpers import recv_push
from ._point_application import PointApplication


def test_active_point_label_and_real_window_follow_published_target_not_aliases(
    application: PointApplication, qapp, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    from qtpy.QtWidgets import QLabel
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    caplog.set_level(logging.ERROR)
    controller = application.controller
    window = MainWindow(controller)
    previous_view = application.remote.render_view
    application.remote.render_view = window

    def assert_visible_snapshot(entry: str, point: str) -> None:
        snapshot = window.get_view_snapshot()
        assert snapshot["entry_label"] == entry
        assert snapshot["point_label"] == point
        labels = window.findChildren(QLabel)
        assert any(
            label.objectName() == "current_entry" and label.text() == entry
            for label in labels
        )
        assert any(
            label.objectName() == "current_point" and label.text() == point
            for label in labels
        )
        assert application.result("view.snapshot", {}) == snapshot
        assert (
            application.result("tab.list_all", {})["active_tab_id"]
            == snapshot["active_tab_id"]
        )

    try:
        window.refresh_context_panel()
        assert_visible_snapshot("No entry open", "No point selected")
        assert application.result("context.active", {}) == {"label": None}
        assert application.result("context.labels", {}) == {"labels": []}
        info = application.open_entry()
        assert_visible_snapshot(info.name, "No point selected")
        context = controller.context_control
        context.add_component("Q1", "fake/drive/a", {"rate": 10.0}, setup=True)
        context.add_component("Q2", "fake/drive/a", {"rate": 20.0}, setup=True)
        assert application.result("context.active", {}) == {"label": None}
        for point in ("base", "next"):
            application.result("point.new", {"label": point, "description": point})
            assert application.result("context.active", {}) == {"label": point}
            assert_visible_snapshot(info.name, point)
            for name in ("Q1", "Q2", ""):
                context.set_alias_rows((AliasRow("driver", name),))
                assert application.result("context.active", {}) == {"label": point}
                assert_visible_snapshot(info.name, point)
        assert application.result("context.labels", {}) == {"labels": ["base", "next"]}
        assert context.get_entry_info() == info
        application.state.set_context(
            replace(controller.get_session_env(), parameters=None)
        )
        window.refresh_context_panel()
        assert_visible_snapshot(info.name, "No point selected")
        assert not caplog.records, (
            "Entry/point/alias facts must not raise in GUI subscribers"
        )
    finally:
        application.remote.render_view = previous_view
        window.deleteLater()
        qapp.processEvents()


def test_snapshot_returns_complete_native_documents_without_materializing_assets(
    application: PointApplication,
) -> None:
    info = application.open_entry()
    context = application.controller.context_control
    context.add_component("Q1", "fake/drive/a", {"rate": 10.0}, setup=True)
    context.add_component("S1", "fake/sensor", {"rate": 20.0}, setup=True)
    entry = ResultEntry.open(
        info.name,
        result_root=info.result_path.parent,
        database_root=info.database_path.parent,
    )
    point = entry.new_point("base", description="native snapshot")
    with point.edit() as draft:
        draft.set("general.ext.nested", {"trace": [1.0, 2.0], "unset": None})
    library = entry.module_library("base")
    with library.edit() as draft:
        draft.set_waveform(
            "Q1.waveforms.arbitrary", ArbWaveformCfg(data="missing_asset")
        )
        draft.set_module(
            "Q1.pulses.pi",
            PulseSpec(
                ch=0, nqz=1, freq=100.0, gain=0.25, waveform="Q1.waveforms.arbitrary"
            ),
        )
    application.result("point.use", {"label": "base"})
    captured = context.read_context()
    result = application.result("context.snapshot", {})
    assert result == {
        "target": captured.target.model_dump(),
        "point": captured.document().model_dump(serialize_as_any=True),
        "library": captured.library.document().model_dump(serialize_as_any=True),
        "aliases": [],
    }
    wire_point = result["point"]
    assert isinstance(wire_point, dict)
    components = wire_point["components"]
    assert isinstance(components, dict)
    q1 = components["Q1"]
    s1 = components["S1"]
    assert isinstance(q1, dict) and isinstance(s1, dict)
    assert q1["rate"] == 10.0 and s1["rate"] == 20.0
    assert wire_point["provenance"]
    wire_library = result["library"]
    assert isinstance(wire_library, dict)
    assert wire_library["provenance"]
    # The asset was never created: a full Spec read must still succeed.
    with pytest.raises(ArbWaveformError):
        _ = captured.library.get_waveform("Q1.waveforms.arbitrary").length


@pytest.mark.parametrize("event", ["md_changed", "ml_changed"])
def test_context_notification_requery_reads_complete_native_capture(
    application: PointApplication, event: str
) -> None:
    info = application.open_entry()
    context = application.controller.context_control
    context.add_component("Q1", "fake/drive/a", {"rate": 10.0}, setup=True)
    entry = ResultEntry.open(
        info.name,
        result_root=info.result_path.parent,
        database_root=info.database_path.parent,
    )
    entry.new_point("base")
    with entry.module_library("base").edit() as draft:
        draft.set_waveform(
            "Q1.waveforms.drive_wav",
            waveform_spec_from_raw({"style": "const", "length": 1.0}),
        )
    application.result("point.use", {"label": "base"})
    application.result("events.subscribe", {"events": [event]})
    application.controller.get_bus().emit(
        MdChangedPayload(binding_id=0, content_changed=True, fault=None)
        if event == "md_changed"
        else MlChangedPayload(binding_id=0, content_changed=True, fault=None)
    )
    notification = recv_push(application.client, event)
    # Consume the advertised query, not a hard-coded method or catalog entry.
    snapshot = application.result(notification["payload"]["requery"][0], {})
    point = snapshot["point"]
    library = snapshot["library"]
    assert isinstance(point, dict) and isinstance(library, dict)
    components = point["components"]
    partitions = library["components"]
    assert isinstance(components, dict) and isinstance(partitions, dict)
    q1 = components["Q1"]
    q1_library = partitions["Q1"]
    assert isinstance(q1, dict) and isinstance(q1_library, dict)
    assert q1["rate"] == 10.0
    waveforms = q1_library["waveforms"]
    assert isinstance(waveforms, dict)
    drive = waveforms["drive_wav"]
    assert isinstance(drive, dict)
    assert drive["style"] == "const" and drive["length"] == 1.0


def test_empty_snapshot_fails_without_unlocking_context(
    application: PointApplication,
) -> None:
    reply = application.rpc("context.snapshot", {})
    error = reply["error"]
    assert isinstance(error, dict)
    assert error["code"] == ErrorCode.PRECONDITION_FAILED
    assert error["reason"] == "no_context"


@pytest.mark.parametrize("document", ["point", "module_cfg"])
def test_source_unavailable_snapshot_does_not_unlock_context(
    application: PointApplication,
    document: Literal["point", "module_cfg"],
    tmp_path: Path,
) -> None:
    application.open_entry()
    application.result("point.new", {"label": "working"})
    captured = application.controller.context_control.read_context()
    fault = ContextSourceFault(
        document, tmp_path / f"{document}.yaml", "OSError", "source fault"
    )
    unavailable = ParameterContext(
        point=captured.document(),
        library=captured.library,
        target=captured.target,
        roles=None,
        source_faults=(fault,),
    )
    application.state.set_context(
        replace(application.state.session_env, parameters=unavailable)
    )
    reply = application.rpc("context.snapshot", {})
    error = reply["error"]
    assert isinstance(error, dict)
    assert error["code"] == ErrorCode.PRECONDITION_FAILED
    assert error["reason"] == "source_unavailable"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_snapshot_never_substitutes_null_or_unlocks_context(
    application: PointApplication, value: float
) -> None:
    application.open_entry()
    application.result("point.new", {"label": "working"})
    context = application.controller.context_control
    context.set_parameter(
        context.read_context().target, "general.ext.nested", {"values": [value]}
    )
    reply = application.rpc("context.snapshot", {})
    error = reply["error"]
    assert isinstance(error, dict)
    assert error["code"] == ErrorCode.PRECONDITION_FAILED
    assert error["reason"] == "unserializable_context"


def snapshot_mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict), value
    return value


def snapshot_rows(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list), value
    return [snapshot_mapping(row) for row in value]


@pytest.fixture
def browse_context(application: PointApplication) -> PointApplication:
    application.open_entry()
    application.result("point.new", {"label": "working"})
    context = application.controller.context_control
    context.add_component("Q1", "fake/drive/a", {"rate": 10.0, "sense": "S1"})
    context.add_component("S1", "fake/sensor", {"rate": 20.0})
    context.set_parameter(
        context.read_context().target,
        "general.ext.browse",
        {
            "scalar": 7.5,
            "unset": None,
            "list": [1, 2, 3],
            "matrix": [[1, 2], [3, 4]],
            "empty": [],
        },
    )
    context.set_alias_rows(
        (
            AliasRow("drive", "Q1"),
            AliasRow("sensor", "$drive.sense"),
            AliasRow("unbound", ""),
            AliasRow("missing", "Absent"),
            AliasRow("bad_link", "$drive.rate"),
        )
    )
    return application


def test_snapshot_summary_preserves_native_array_shape_and_numpy_scalars(
    application: PointApplication,
) -> None:
    class ArrayComponent(ComponentSchema):
        trace: object
        scalar: object

    component_registry.register("browse/array", ArrayComponent)
    try:
        application.open_entry()
        application.result("point.new", {"label": "working"})
        captured = application.controller.context_control.read_context()
        point = captured.document()
        point = point.model_copy(
            update={
                "components": {
                    **point.components,
                    "Trace": ArrayComponent(
                        kind="browse/array",
                        trace=np.arange(6).reshape(2, 3),
                        scalar=np.float64(4.5),
                    ),
                }
            }
        )
        native = ParameterContext(
            point=point,
            library=captured.library,
            target=captured.target,
            roles=RoleBinding(None, {}, {}, aliases=captured.aliases),
            source_faults=captured.source_faults,
            storage=captured.storage,
        )
        application.state.set_context(
            replace(application.state.session_env, parameters=native)
        )
        summary = application.result("context.snapshot", {"summaries": True})
        components = snapshot_mapping(snapshot_mapping(summary["point"])["components"])
        trace = snapshot_mapping(components["Trace"])
        assert trace["trace"] == "2 × 3 array"
        assert trace["scalar"] == 4.5
        value = application.result(
            "context.snapshot", {"path": "point.components.Trace.trace"}
        )
        assert value["value"] == [[0, 1, 2], [3, 4, 5]]
    finally:
        component_registry.unregister("browse/array")


def test_snapshot_alias_rows_isolate_unbound_and_invalid_links(
    browse_context: PointApplication,
) -> None:
    result = browse_context.result("context.snapshot", {})
    assert "aliases" in result, result
    rows = snapshot_rows(result["aliases"])
    assert rows[:2] == [
        {"alias": "drive", "value": "Q1", "resolved": "Q1"},
        {"alias": "sensor", "value": "$drive.sense", "resolved": "S1"},
    ]
    assert len(rows) == 5
    assert rows[3] == {"alias": "missing", "value": "Absent", "resolved": "Absent"}
    for row, alias, value in zip(
        (rows[2], rows[4]),
        ("unbound", "bad_link"),
        ("", "$drive.rate"),
        strict=True,
    ):
        assert row["alias"] == alias and row["value"] == value
        assert row["resolved"] is None
        assert alias in str(row["error"])
    assert "$unbound" in str(rows[2]["error"]) and "Aliases" in str(rows[2]["error"])
    assert "must name a component or device" in str(rows[4]["error"])


def test_snapshot_summary_keeps_scalars_and_complete_aliases(
    browse_context: PointApplication,
) -> None:
    full = browse_context.result("context.snapshot", {})
    summary = browse_context.result("context.snapshot", {"summaries": True})
    assert "aliases" in summary and "aliases" in full
    assert summary["target"] == full["target"]
    assert summary["aliases"] == full["aliases"]
    point = snapshot_mapping(summary["point"])
    components = snapshot_mapping(point["components"])
    assert snapshot_mapping(components["Q1"])["rate"] == 10.0
    general = snapshot_mapping(point["general"])
    assert snapshot_mapping(general["ext"])["browse"] == {
        "scalar": 7.5,
        "unset": None,
        "list": "3 items",
        "matrix": "2 × 2 matrix",
        "empty": "0 items",
    }


@pytest.mark.parametrize(
    "path, expected",
    [
        (
            "point.general.ext.browse",
            {
                "scalar": 7.5,
                "unset": None,
                "list": [1, 2, 3],
                "matrix": [[1, 2], [3, 4]],
                "empty": [],
            },
        ),
        ("point.general.ext.browse.matrix.1", [3, 4]),
        ("point.components.Q1.rate", 10.0),
        ("target.kind", "saved_point"),
        ("library.components", {}),
        ("aliases.1.resolved", "S1"),
    ],
)
def test_snapshot_path_reads_complete_subtree_before_summaries(
    browse_context: PointApplication,
    path: str,
    expected: object,
) -> None:
    result = browse_context.result(
        "context.snapshot", {"path": path, "summaries": True}
    )
    assert "path" in result and "value" in result, result
    assert result["path"] == path and result["value"] == expected
    assert set(result) == {"target", "aliases", "path", "value"}
    assert snapshot_rows(result["aliases"])[1] == {
        "alias": "sensor",
        "value": "$drive.sense",
        "resolved": "S1",
    }


@pytest.mark.parametrize(
    "path, bad_segment",
    [
        ("point.components.missing.rate", "missing"),
        ("aliases.-1.resolved", "-1"),
        ("aliases.01.resolved", "01"),
        ("aliases.99.resolved", "99"),
        ("aliases.x.resolved", "x"),
        ("point.components.Q1.rate.child", "child"),
        ("point..components", "''"),
        ("", "''"),
        ("$drive.rate", "$drive"),
        ("md.rate", "md"),
    ],
)
def test_snapshot_invalid_path_names_first_bad_segment_and_does_not_reveal(
    browse_context: PointApplication,
    path: str,
    bad_segment: str,
) -> None:
    reply = browse_context.rpc("context.snapshot", {"path": path})
    assert reply["ok"] is False
    error = snapshot_mapping(reply["error"])
    assert error["code"] == ErrorCode.INVALID_PARAMS
    assert bad_segment in str(error["message"])
