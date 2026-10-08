"""Editor drafts, native library writes, and observable session lifecycles."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, GaussWaveformCfg, PulseSpec
from zcu_tools.gui.app.measure.services.cfg_editor import (
    CfgEditorError,
    CfgEditorService,
    CfgEditorSession,
)
from zcu_tools.gui.app.measure.services.cfg_lowering import lower_module, lower_waveform
from zcu_tools.gui.app.measure.services.ports import CfgEdit
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CfgSchema,
    CfgSectionSpec,
    DirectValue,
    SweepSpec,
    make_default_value,
)
from zcu_tools.gui.cfg.binding import SettablePathError
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import LibraryChange, PointUseRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import ParameterContext, SessionEnv
from zcu_tools.gui.session.value_lookup import (
    MissingValue,
    ProviderError,
    UnavailableValue,
    ValueInfo,
    ValueTypeError,
)
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase


@pytest.fixture
def context(tmp_path):
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    entry.new_point("a")
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    context = ContextService(state, EventBus(), bind_assets=ArbWaveformDatabase.info_at)
    context.install_entry(entry)
    control = ContextControlFacet(context=context)
    control.use_point(PointUseRequest(entry.entry_id, "a"))
    target = control.read_context().target
    control.set_parameter(target, "general.ext.r_f", 6012.5)
    control.set_module(
        target,
        "Q1.pulses.seed",
        PulseSpec(
            ch=0,
            nqz=1,
            freq=0.0,
            gain=0.0,
            waveform=ConstWaveformCfg(length=0.0),
        ),
    )
    control.set_waveform(target, "Q1.waveforms.const", ConstWaveformCfg(length=0.0))
    control.set_waveform(
        target, "Q1.waveforms.gauss", GaussWaveformCfg(length=0.1, sigma=0.02)
    )
    return context


@pytest.fixture
def control(context):
    return ContextControlFacet(context=context)


@pytest.fixture
def ctrl(control):
    """Only peripheral options, value lookup, and versions are mocked."""
    c = MagicMock()
    c.read_context.side_effect = control.read_context
    c.list_device_names.return_value = []
    c.arb_waveforms.list_data_keys.return_value = []

    def apply(parameters: ParameterContext, change: LibraryChange) -> None:
        control.apply_library_change(
            parameters, change, lower_module=lower_module, lower_waveform=lower_waveform
        )

    c.apply_library_change.side_effect = apply
    return c


@pytest.fixture
def service(ctrl):
    return CfgEditorService(
        ctrl, read_port=ctrl, write_port=ctrl, versions=ctrl, bus=EventBus()
    )


def _paths(entries):
    from zcu_tools.gui.app.measure.remote.path_resolver import project_targets

    return {e["path"]: e for e in project_targets(entries)}


def _draft_paths(service, editor_id):
    return _paths(service.get_draft(editor_id).iter_settable_targets())


def _make_tab_seed():
    from zcu_tools.gui.app.measure.specs import make_pulse_spec

    spec = make_pulse_spec()
    return CfgSchema(spec=spec, value=make_default_value(spec))


def test_session_is_the_aggregate_with_behaviour(service, control):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    capture = control.read_context()
    session = CfgEditorSession(
        editor_id=editor_id,
        draft=service.get_draft(editor_id),
        resolve_value_ref=MagicMock(),
        gc=True,
        item_kind="module",
        parameters=capture,
        source_name="Q1.pulses.seed",
    )
    out = session.set_field("freq", 5000.0)
    assert out.valid is True
    assert out.removed == () and out.added == ()
    assert _paths(session.current_targets())["freq"]["value"] == 5000.0
    schema = session.commit_schema()
    assert schema.value.fields["freq"] == DirectValue(5000.0)
    service.commit(editor_id, "Q1.pulses.aggregate")
    assert (
        control.read_context().library.module_spec("Q1.pulses.aggregate").freq == 5000.0
    )


def test_seeded_session_rejects_commit_on_the_aggregate(service):
    editor_id, _ = service.open_seeded(_make_tab_seed(), parameters=None)
    session = CfgEditorSession(
        editor_id=editor_id,
        draft=service.get_draft(editor_id),
        resolve_value_ref=MagicMock(),
        gc=False,
    )
    with pytest.raises(CfgEditorError, match="seeded session"):
        session.commit_schema()


def test_open_existing_module_returns_paths(service):
    editor_id, paths = service.open("module", from_name="Q1.pulses.seed")
    assert editor_id.startswith("editor-")
    assert _paths(paths)["freq"]["kind"] == "scalar"


def test_open_unknown_kind_fails(service):
    with pytest.raises(CfgEditorError):
        service.open("widget", from_name="Q1.pulses.seed")


def test_open_unknown_module_name_fails(service):
    with pytest.raises(CfgEditorError):
        service.open("module", from_name="Q1.pulses.missing")


def test_open_from_existing_entry_returns_concrete_values(service):
    _, paths = service.open("waveform", from_name="Q1.waveforms.const")
    assert _paths(paths)["length"]["value"] == 0.0


def test_set_scalar_then_commit_lands_concrete(service, ctrl, control):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    assert service.set_field(editor_id, "freq", 5000.0).valid is True
    service.commit(editor_id, "Q1.pulses.agent")
    ctrl.apply_library_change.assert_called_once()
    assert control.read_context().library.module_spec("Q1.pulses.agent").freq == 5000.0


def test_eval_value_resolved_against_capture_on_commit(service, control):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    service.set_field(editor_id, "freq", {"__kind": "eval", "expr": "general.ext.r_f"})
    service.commit(editor_id, "Q1.pulses.eval")
    assert control.read_context().library.module_spec("Q1.pulses.eval").freq == 6012.5


def test_eval_value_requires_string_expr(service):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    with pytest.raises(CfgEditorError):
        service.set_field(editor_id, "freq", {"__kind": "eval", "expr": 123})


def test_value_ref_resolves_to_concrete_direct_value_on_set_field(
    service, ctrl, control
):
    ctrl.read_value_source.return_value = (
        ValueInfo("device.flux.value", float, "device:flux"),
        6.25,
    )
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    service.set_field(
        editor_id,
        "freq",
        {"__kind": "value_ref", "key": "device.flux.value", "type": "float"},
    )
    service.commit(editor_id, "Q1.pulses.ref")
    assert control.read_context().library.module_spec("Q1.pulses.ref").freq == 6.25
    ctrl.read_value_source.assert_called_once_with("device.flux.value", "float")


def test_value_ref_requires_string_key(service):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    with pytest.raises(CfgEditorError, match="string 'key'"):
        service.set_field(editor_id, "freq", {"__kind": "value_ref", "key": 123})


@pytest.mark.parametrize(
    "error",
    (
        MissingValue("device.flux.value", "missing"),
        UnavailableValue("device.flux.value", "unavailable"),
        ProviderError("device.flux.value", "device:flux", RuntimeError("boom")),
    ),
)
def test_value_ref_lookup_error_identity_and_category_are_preserved(
    service, ctrl, error
):
    ctrl.read_value_source.side_effect = error
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    with pytest.raises(type(error)) as exc:
        service.set_field(
            editor_id,
            "freq",
            {"__kind": "value_ref", "key": "device.flux.value", "type": "float"},
        )
    assert exc.value is error


def test_value_ref_type_error_keeps_nominal_invalid_input(service):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    with pytest.raises(ValueTypeError):
        service.set_field(
            editor_id,
            "freq",
            {"__kind": "value_ref", "key": "device.flux.value", "type": "str"},
        )


@pytest.mark.parametrize("path", ("sweep.start", "sweep.stop", "centered.center"))
def test_wire_eval_value_is_rejected_for_sweep_edges(service, path):
    spec = CfgSectionSpec(
        fields={"sweep": SweepSpec(), "centered": CenteredSweepSpec()}
    )
    editor_id, _ = service.open_seeded(
        CfgSchema(spec, make_default_value(spec)), parameters=None
    )
    with pytest.raises(SettablePathError, match="only valid for scalar"):
        service.set_field(
            editor_id, path, {"__kind": "eval", "expr": "general.ext.r_f"}
        )


def test_ref_switch_returns_new_subtree_and_diff(service):
    editor_id, paths = service.open("module", from_name="Q1.pulses.seed")
    assert "waveform.ref" in _paths(paths)
    assert "waveform.sigma" not in _paths(paths)
    res = service.set_field(editor_id, "waveform.ref", "<Custom:Gauss>")
    assert "waveform.sigma" in res.added
    assert "waveform.sigma" not in res.removed
    assert "waveform.sigma" in _draft_paths(service, editor_id)


def test_non_shape_batch_preserves_paths_and_updates_values(service):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    before = _draft_paths(service, editor_id)
    result = service.set_fields(
        editor_id, (CfgEdit("freq", 5000.0), CfgEdit("gain", 0.4))
    )
    after = _draft_paths(service, editor_id)
    assert result.removed == () and result.added == ()
    assert before.keys() == after.keys()
    assert after["freq"]["value"] == 5000.0
    assert after["gain"]["value"] == 0.4


def test_shape_batch_returns_net_diff(service):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    result = service.set_fields(
        editor_id,
        (
            CfgEdit("waveform.ref", "Gauss"),
            CfgEdit("waveform.ref", "Const"),
        ),
    )
    assert result.removed == () and result.added == ()
    assert "waveform.sigma" not in _draft_paths(service, editor_id)


def test_shape_batch_failure_keeps_successful_prefix(service):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    with pytest.raises(SettablePathError, match="unknown settable path"):
        service.set_fields(
            editor_id,
            (
                CfgEdit("waveform.ref", "Gauss"),
                CfgEdit("waveform.does_not_exist", 1),
            ),
        )
    draft = service.get_draft(editor_id)
    assert draft.resolve_target("waveform.ref").get_value() == "<Custom:Gauss>"
    assert "waveform.sigma" in _draft_paths(service, editor_id)


def test_batch_bumps_editor_version_once_per_successful_edit(service, ctrl):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    ctrl.bump_editor_version.reset_mock()
    service.set_fields(editor_id, (CfgEdit("freq", 5000.0), CfgEdit("gain", 0.25)))
    assert ctrl.bump_editor_version.call_args_list == [call(editor_id), call(editor_id)]


def test_batch_is_ordered_fail_fast_non_atomic(service, ctrl):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    ctrl.bump_editor_version.reset_mock()
    with pytest.raises(SettablePathError, match="unknown settable path"):
        service.set_fields(
            editor_id,
            (
                CfgEdit("freq", 4321.0),
                CfgEdit("does.not.exist", 1),
                CfgEdit("gain", 0.9),
            ),
        )
    paths = _draft_paths(service, editor_id)
    assert paths["freq"]["value"] == 4321.0
    assert paths["gain"]["value"] == 0.0
    ctrl.bump_editor_version.assert_called_once_with(editor_id)


def test_empty_batch_returns_live_validity_without_change(service, ctrl):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    before = service.get_draft(editor_id).snapshot()
    ctrl.bump_editor_version.reset_mock()
    result = service.set_fields(editor_id, ())
    assert result.valid is service.get_draft(editor_id).is_valid()
    assert result.removed == () and result.added == ()
    assert service.get_draft(editor_id).snapshot() == before
    ctrl.bump_editor_version.assert_not_called()


def test_unknown_editor_id_fails(service):
    with pytest.raises(CfgEditorError):
        service.set_field("editor-nope", "freq", 1.0)
    with pytest.raises(CfgEditorError):
        service.commit("editor-nope", "Q1.pulses.new")
    with pytest.raises(CfgEditorError):
        service.discard("editor-nope")


def test_discard_removes_session(service):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    service.discard(editor_id)
    with pytest.raises(CfgEditorError):
        service.get_draft(editor_id)


def test_discard_for_client_batch_ignores_unknown(service):
    id1, _ = service.open("module", from_name="Q1.pulses.seed")
    id2, _ = service.open("waveform", from_name="Q1.waveforms.const")
    service.discard_for_client([id1, id2, "editor-unknown"])
    for editor_id in (id1, id2):
        with pytest.raises(CfgEditorError):
            service.get_draft(editor_id)


def test_commit_failure_keeps_session(service, ctrl):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    ctrl.apply_library_change.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        service.commit(editor_id, "Q1.pulses.bad")
    assert service.get_draft(editor_id)


def test_replace_renames_and_commits_as_one_editor_operation(service, ctrl, control):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    service.set_field(editor_id, "freq", 5000.0)
    service.replace(editor_id, "Q1.pulses.seed", "Q1.pulses.renamed")
    library = control.read_context().library
    assert "Q1.pulses.seed" not in library.module_keys()
    assert library.module_spec("Q1.pulses.renamed").freq == 5000.0
    ctrl.apply_library_change.assert_called_once()
    with pytest.raises(CfgEditorError):
        service.get_draft(editor_id)


def test_replace_failure_keeps_editor_session(service, ctrl, control):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    service.set_field(editor_id, "freq", 5000.0)
    ctrl.apply_library_change.side_effect = RuntimeError("collision")
    with pytest.raises(RuntimeError, match="collision"):
        service.replace(editor_id, "Q1.pulses.seed", "Q1.pulses.existing")
    assert "Q1.pulses.seed" in control.read_context().library.module_keys()
    assert service.get_draft(editor_id)


def test_replace_rejects_a_source_name_that_is_not_the_opened_entry(service, ctrl):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    with pytest.raises(CfgEditorError, match="does not match"):
        service.replace(editor_id, "Q1.pulses.other", "Q1.pulses.renamed")
    ctrl.apply_library_change.assert_not_called()
    assert service.get_draft(editor_id)


def test_replace_rejects_an_entry_switch_and_keeps_the_draft(
    service, control, context, tmp_path
):
    editor_id, _ = service.open("module", from_name="Q1.pulses.seed")
    draft = service.get_draft(editor_id)
    service.set_field(editor_id, "freq", 5000.0)
    other = ResultEntry.create(
        "other", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    context.install_entry(other)
    with pytest.raises(FailedPreconditionError):
        service.replace(editor_id, "Q1.pulses.seed", "Q1.pulses.renamed")
    assert service.get_draft(editor_id) is draft
    assert draft.snapshot().value.fields["freq"] == DirectValue(5000.0)
    assert control.read_context().library.module_keys() == ()


def test_complex_cfg_edit_and_remote_projection_share_lossless_tag(service):
    import json

    from zcu_tools.gui.app.measure.remote.path_resolver import project_target_entries
    from zcu_tools.gui.cfg import CfgSectionValue, ScalarSpec

    schema = CfgSchema(
        CfgSectionSpec(fields={"center": ScalarSpec("Center", complex)}),
        CfgSectionValue(fields={"center": DirectValue(0j)}),
    )
    editor_id, _ = service.open_seeded(schema, parameters=None, owner_key="complex-tab")
    value = 1.234567891234567 - 0.5j
    service.set_field(editor_id, "center", {"__complex__": [value.real, value.imag]})
    draft = service.get_draft(editor_id)
    assert draft.snapshot().value.fields["center"] == DirectValue(value)
    entries = json.loads(json.dumps(project_target_entries(draft)))
    assert entries[0]["value"] == {"__complex__": [value.real, value.imag]}
    with pytest.raises(CfgEditorError, match="two components"):
        service.set_field(editor_id, "center", {"__complex__": [1]})
    assert draft.snapshot().value.fields["center"] == DirectValue(value)


def test_open_seeded_owns_model_and_is_addressable(service):
    editor_id, _ = service.open_seeded(
        _make_tab_seed(), parameters=None, owner_key="tab-1"
    )
    assert service.editor_id_for_owner("tab-1") == editor_id
    draft = service.get_draft(editor_id)
    service.set_field(editor_id, "freq", 4321.0)
    assert draft.snapshot().value.fields["freq"] == DirectValue(4321.0)


def test_teardown_seeded_session_drops_it(service):
    editor_id, _ = service.open_seeded(
        _make_tab_seed(), parameters=None, owner_key="tab-1"
    )
    service.teardown(editor_id)
    assert service.editor_id_for_owner("tab-1") is None
    with pytest.raises(CfgEditorError):
        service.get_draft(editor_id)


def test_reopen_same_owner_tears_down_previous(service):
    id1, _ = service.open_seeded(_make_tab_seed(), parameters=None, owner_key="tab-1")
    draft1 = service.get_draft(id1)
    id2, _ = service.open_seeded(_make_tab_seed(), parameters=None, owner_key="tab-1")
    assert service.editor_id_for_owner("tab-1") == id2
    assert service.get_draft(id2) is not draft1
    with pytest.raises(CfgEditorError):
        service.get_draft(id1)


def test_teardown_unknown_is_noop(service):
    service.teardown("editor-never")


def test_commit_rejects_seeded_session(service):
    editor_id, _ = service.open_seeded(
        _make_tab_seed(), parameters=None, owner_key="tab-1"
    )
    with pytest.raises(CfgEditorError):
        service.commit(editor_id, "Q1.pulses.new")


def test_discard_for_client_skips_gc_false_sessions(service):
    tab_id, _ = service.open_seeded(
        _make_tab_seed(), parameters=None, owner_key="tab-1"
    )
    gc_id, _ = service.open("module", from_name="Q1.pulses.seed")
    service.discard_for_client([tab_id, gc_id])
    with pytest.raises(CfgEditorError):
        service.get_draft(gc_id)
    assert service.editor_id_for_owner("tab-1") == tab_id


def test_gc_lru_evicts_oldest(service):
    ids = [service.open("module", from_name="Q1.pulses.seed")[0] for _ in range(20)]
    with pytest.raises(CfgEditorError):
        service.get_draft(ids[0])
    assert service.get_draft(ids[-1])


def test_lru_does_not_count_gc_false_sessions(service):
    gc_id, _ = service.open("module", from_name="Q1.pulses.seed")
    for i in range(20):
        service.open_seeded(_make_tab_seed(), parameters=None, owner_key=f"tab-{i}")
    assert service.get_draft(gc_id)


def test_change_stream_emits_editor_changed(service):
    events = []
    service.set_change_listener(
        lambda eid, ev, payload_factory: events.append((eid, ev, payload_factory()))
    )
    editor_id, _ = service.open_seeded(
        _make_tab_seed(), parameters=None, owner_key="tab-1"
    )
    service.set_field(editor_id, "freq", 5000.0)
    changed = [e for e in events if e[1] == "editor_changed" and e[0] == editor_id]
    assert changed
    assert any("freq" in {target.path for target in e[2]} for e in changed)


def test_change_stream_emits_editor_closed_with_reason(service):
    events = []
    service.set_change_listener(
        lambda eid, ev, payload_factory: events.append((eid, ev, payload_factory()))
    )
    tab_id, _ = service.open_seeded(
        _make_tab_seed(), parameters=None, owner_key="tab-1"
    )
    service.teardown(tab_id)
    h1, _ = service.open("module", from_name="Q1.pulses.seed")
    service.discard(h1)
    h2, _ = service.open("waveform", from_name="Q1.waveforms.const")
    service.commit(h2, "Q1.waveforms.copy")
    closed = {
        (eid, payload["reason"]) for eid, ev, payload in events if ev == "editor_closed"
    }
    assert (tab_id, "tab_closed") in closed
    assert (h1, "discarded") in closed
    assert (h2, "committed") in closed


def test_change_stream_no_listener_is_safe(service):
    editor_id, _ = service.open_seeded(
        _make_tab_seed(), parameters=None, owner_key="tab-1"
    )
    service.set_field(editor_id, "freq", 1.0)
    service.teardown(editor_id)


def test_closed_draft_rejects_residual_field_edits(service):
    events = []
    service.set_change_listener(
        lambda eid, ev, payload_factory: events.append((eid, ev, payload_factory()))
    )
    editor_id, _ = service.open_seeded(
        _make_tab_seed(), parameters=None, owner_key="tab-1"
    )
    field = service.get_draft(editor_id).root.fields["freq"]
    service.teardown(editor_id)
    events.clear()
    with pytest.raises(RuntimeError, match="closed"):
        field.set_value(99.0)
    assert events == []


def test_change_stream_defers_payload_but_always_bumps_version(service, ctrl):
    factories = []
    service.set_change_listener(lambda _eid, _ev, factory: factories.append(factory))
    editor_id, _ = service.open_seeded(
        _make_tab_seed(), parameters=None, owner_key="tab-1"
    )
    ctrl.bump_editor_version.reset_mock()
    service.set_field(editor_id, "freq", 5000.0)
    ctrl.bump_editor_version.assert_called_once_with(editor_id)
    assert len(factories) == 1
    payload = factories[0]()
    assert _paths(payload)["freq"]["value"] == 5000.0


def test_no_dangling_hook_after_every_removal_path(service):
    events = []
    service.set_change_listener(
        lambda eid, ev, factory: events.append((eid, ev, factory()))
    )
    for removal in ("teardown", "discard", "reopen"):
        if removal == "discard":
            eid, _ = service.open("module", from_name="Q1.pulses.seed")
        else:
            eid, _ = service.open_seeded(
                _make_tab_seed(), parameters=None, owner_key="tab-x"
            )
        field = service.get_draft(eid).root.fields["freq"]
        if removal == "reopen":
            service.open_seeded(_make_tab_seed(), parameters=None, owner_key="tab-x")
        elif removal == "discard":
            service.discard(eid)
        else:
            service.teardown(eid)
        events.clear()
        with pytest.raises(RuntimeError, match="closed"):
            field.set_value(5000.0)
        assert events == []


def _service_with_version_table(ctrl):
    from zcu_tools.gui.app.measure.state import VersionTable

    table = VersionTable()
    versions = MagicMock()
    versions.bump_editor_version.side_effect = lambda eid: table.bump(f"editor:{eid}")
    versions.drop_editor_version.side_effect = lambda eid: table.drop_prefix(
        f"editor:{eid}"
    )
    svc = CfgEditorService(ctrl, ctrl, ctrl, versions, EventBus())
    return svc, table


def test_commit_drops_editor_version(ctrl):
    svc, table = _service_with_version_table(ctrl)
    editor_id, _ = svc.open("module", from_name="Q1.pulses.seed")
    svc.set_field(editor_id, "freq", 5000.0)
    assert table.get(f"editor:{editor_id}") > 0
    svc.commit(editor_id, "Q1.pulses.committed")
    assert table.get(f"editor:{editor_id}") == 0


def test_discard_drops_editor_version(ctrl):
    svc, table = _service_with_version_table(ctrl)
    editor_id, _ = svc.open("waveform", from_name="Q1.waveforms.gauss")
    svc.set_field(editor_id, "length", 0.2)
    assert table.get(f"editor:{editor_id}") > 0
    svc.discard(editor_id)
    assert table.get(f"editor:{editor_id}") == 0


def test_teardown_seeded_drops_editor_version(ctrl):
    svc, table = _service_with_version_table(ctrl)
    editor_id, _ = svc.open_seeded(_make_tab_seed(), parameters=None, owner_key="tab-1")
    svc.set_field(editor_id, "freq", 7.0)
    assert table.get(f"editor:{editor_id}") > 0
    svc.teardown(editor_id)
    assert table.get(f"editor:{editor_id}") == 0
