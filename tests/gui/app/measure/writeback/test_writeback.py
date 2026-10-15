"""Persistent writeback drafts edit module proposals as Forms at a fixed capture."""

from __future__ import annotations

from pathlib import Path

import pytest
from zcu_tools.cfg_model import ArbWaveformCfg, ConstWaveformCfg, PulseSpec
from zcu_tools.gui.app.measure.writeback import (
    ModuleWriteback,
    ParameterWriteback,
    WritebackEdit,
    WritebackService,
    WritebackWritten,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.library_entry import LibraryEntryInvalid, ModuleEntryForm
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry, document_meta
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry

PULSE = "Q1.pulses.tuned"
WAVEFORM = "Q1.waveforms.flat"


def pulse_proposal(gain: float = 0.2) -> PulseSpec:
    """Return a pulse proposal whose waveform is the library key WAVEFORM."""
    return PulseSpec(ch=0, nqz=1, freq=5000.0, gain=gain, waveform=WAVEFORM)


@pytest.fixture
def session(tmp_path: Path):
    before = registry_state()
    try:
        register_fakes()
        installed = registry_state()
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "Result",
            database_root=tmp_path / "Database",
        )
        entry.new_point("a")
        bus = BaseEventBus()
        context = ContextService(
            SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
            bus,
            bind_assets=ArbWaveformDatabase.info_at,
        )
        context.install_entry(entry)
        control = ContextControlFacet(context=context)
        control.use_point(PointUseRequest("a"))
        control.add_component("Q1", "fake/drive/a", {})
        target = control.read_context().target
        control.set_waveform(target, WAVEFORM, ConstWaveformCfg(length=0.1))
        control.set_parameter(target, "general.ext.flux_value", 2.0)
        service = WritebackService(control, arb_names=lambda: ())
        yield entry, control, service
        assert registry_state() == installed
    finally:
        restore_registry(before)


def test_module_preview_proposes_the_spec_in_the_current_shape(session):
    _entry, control, service = session
    control.set_module(control.read_context().target, PULSE, pulse_proposal())
    capture = control.read_context()
    draft = service.create_draft(
        [ModuleWriteback(PULSE, "pulse", pulse_proposal(gain=0.4))],
        parameters=capture,
        experiment_name="test/writeback",
    )
    values = service.preview_values(draft)["ml-1"]
    assert values.actual_path == PULSE
    assert values.current == capture.library.entry_value(PULSE, kind="module")
    assert values.proposed == pulse_proposal(gain=0.4).model_dump(mode="python")
    assert isinstance(values.current, dict) and isinstance(values.proposed, dict)
    assert values.proposed.keys() == values.current.keys()
    assert [
        key for key in values.proposed if values.proposed[key] != values.current[key]
    ] == ["gain"]


def test_module_form_edits_apply_the_edited_spec_and_keep_child_keys(session):
    _entry, control, service = session
    capture = control.read_context()
    draft = service.create_draft(
        [ModuleWriteback(PULSE, "pulse", pulse_proposal())],
        parameters=capture,
        experiment_name="test/writeback",
    )
    draft.edit(
        "ml-1",
        edits={
            ("gain",): {"expr": "general.ext.flux_value / 10"},
            ("freq",): 5100.0,
        },
    )
    control.set_parameter(capture.target, "general.ext.flux_value", 8.0)
    assert draft.apply()["applied_ids"] == ["ml-1"]
    written = control.read_context().library.module_spec(PULSE)
    assert isinstance(written, PulseSpec)
    assert written.waveform == WAVEFORM
    assert written.freq == 5100.0
    assert written.gain == pytest.approx(0.2)
    assert service.get_all_applied(draft) == {"ml-1": True}


def test_invalid_module_edit_rejects_apply_with_field_paths(session):
    entry, control, service = session
    draft = service.create_draft(
        [ModuleWriteback(PULSE, "pulse", pulse_proposal())],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    draft.edit("ml-1", edits={("gain",): "bad"})
    count = len(entry.ledger.events())
    with pytest.raises(LibraryEntryInvalid, match="gain") as error:
        draft.apply()
    assert set(error.value.errors.messages) == {("gain",)}
    with pytest.raises(LibraryEntryInvalid):
        service.preview_values(draft)
    assert len(entry.ledger.events()) == count
    assert PULSE not in control.read_context().library.module_keys()
    assert service.get_all_applied(draft) == {"ml-1": False}


def test_structural_module_edit_is_rejected_without_changing_the_form(session):
    _entry, control, service = session
    draft = service.create_draft(
        [ModuleWriteback(PULSE, "pulse", pulse_proposal())],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    before = service.preview_values(draft)
    with pytest.raises(InvalidInputError, match="ml-1"):
        draft.edit("ml-1", edits={("gain",): 0.5, ("unknown",): 1.0})
    assert service.preview_values(draft) == before


def test_edits_require_a_module_item(session):
    _entry, control, service = session
    draft = service.create_draft(
        [ParameterWriteback("general.ext.value", "value", proposed_value=1)],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    with pytest.raises(InvalidInputError, match="not a module item"):
        draft.edit("param-1", edits={("gain",): 0.5})
    assert service.get_all_applied(draft) == {"param-1": False}


def test_context_free_module_draft_is_listed_but_not_editable(session):
    _entry, _control, service = session
    draft = service.create_draft(
        [ModuleWriteback(PULSE, "pulse", pulse_proposal())],
        parameters=None,
        experiment_name="test/writeback",
    )
    assert [item.session_id for item in draft.preview()] == ["ml-1"]
    with pytest.raises(FailedPreconditionError, match="captured context"):
        draft.edit("ml-1", edits={("gain",): 0.5})
    with pytest.raises(FailedPreconditionError, match="captured context"):
        service.get_item_form(draft, "ml-1")


def test_item_form_is_the_shared_draft_and_its_changes_mark_unapplied(session):
    _entry, control, service = session
    draft = service.create_draft(
        [ModuleWriteback(PULSE, "pulse", pulse_proposal())],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    draft.apply()
    assert service.get_all_applied(draft) == {"ml-1": True}
    form = service.get_item_form(draft, "ml-1")
    assert isinstance(form, ModuleEntryForm)
    form.state.write({("gain",): 0.4}, form.env)
    assert service.get_all_applied(draft) == {"ml-1": False}
    proposed = service.preview_values(draft)["ml-1"].proposed
    assert isinstance(proposed, dict) and proposed["gain"] == 0.4
    draft.edit("ml-1", edits={("gain",): 0.6})
    assert form.state.input.data["gain"] == 0.6


def test_arb_names_are_captured_when_the_draft_is_created(session):
    _entry, control, _service = session
    names = ["first"]
    service = WritebackService(control, arb_names=lambda: tuple(names))
    proposal = PulseSpec(
        ch=0, nqz=1, freq=5000.0, gain=0.2, waveform=ArbWaveformCfg(data="first")
    )
    draft = service.create_draft(
        [ModuleWriteback(PULSE, "pulse", proposal)],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    names.clear()
    proposed = service.preview_values(draft)["ml-1"].proposed
    assert isinstance(proposed, dict) and proposed["waveform"]["data"] == "first"
    draft.edit("ml-1", edits={("waveform", "module", "data"): "second"})
    with pytest.raises(LibraryEntryInvalid) as error:
        service.preview_values(draft)
    assert set(error.value.errors.messages) == {("waveform", "module", "data")}


def test_ordered_write_rejects_invalid_module_before_context_apply(session):
    entry, control, service = session
    draft = service.create_draft(
        [ModuleWriteback(PULSE, "pulse", pulse_proposal())],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    count = len(entry.ledger.events())
    with pytest.raises(LibraryEntryInvalid, match="gain"):
        service.write_draft(
            draft, (WritebackEdit("ml-1", edits={("gain",): "invalid"}),)
        )
    assert len(entry.ledger.events()) == count
    assert PULSE not in control.read_context().library.module_keys()


def test_ordered_module_write_returns_spec_values(session):
    _entry, control, service = session
    control.set_module(control.read_context().target, PULSE, pulse_proposal())
    draft = service.create_draft(
        [ModuleWriteback(PULSE, "pulse", pulse_proposal())],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    result = service.write_draft(
        draft, (WritebackEdit("ml-1", edits={("gain",): 0.7}),)
    )
    assert result == [
        WritebackWritten(
            "ml-1",
            "module",
            PULSE,
            pulse_proposal().model_dump(mode="python"),
            pulse_proposal(gain=0.7).model_dump(mode="python"),
        )
    ]


def test_apply_shared_flow_keeps_gui_selection_edits_and_failure_prefix(session):
    from zcu_tools.resources.entry import writeback

    entry, control, service = session
    draft = service.create_draft(
        [
            ModuleWriteback(PULSE, "module", pulse_proposal()),
            ParameterWriteback(
                "general.ext.before", "parameter", proposed_value=1.0, stderr=0.3
            ),
            ParameterWriteback("general.description", "invalid", proposed_value=42),
            ParameterWriteback("general.ext.after", "later", proposed_value=2.0),
        ],
        parameters=control.read_context(),
        experiment_name="test/gui-fit",
        run_id="run-fixed",
    )
    draft.edit("param-1", target_name="general.ext.tuned", proposed_value=3.0)
    draft.edit("param-3", selected=False)
    draft.edit("ml-1", edits={("gain",): 0.25})
    with pytest.raises(ValueError, match="description"):
        draft.apply()
    event = entry.ledger.events()[-1]
    assert service.get_all_applied(draft) == {
        "ml-1": True,
        "param-1": True,
        "param-2": False,
        "param-3": False,
    }
    current = control.read_context()
    assert current.value("general.ext.tuned") == 3.0
    assert not current.lookup("general.ext.after").present
    assert current.library.module_spec(PULSE) == pulse_proposal(gain=0.25)
    assert document_meta(current.document(), "general.ext.tuned") == {"stderr": 0.3}
    assert isinstance(event.payload, writeback.AcceptedPayload)
    assert [write.path for write in event.payload.writes] == [
        PULSE,
        "general.ext.tuned",
        "general.description",
    ]


def test_draft_teardown_is_idempotent(session):
    _entry, control, service = session
    draft = service.create_draft(
        [ModuleWriteback(PULSE, "pulse", pulse_proposal())],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    draft.teardown()
    draft.teardown()
    assert draft.is_active is False
    with pytest.raises(FailedPreconditionError):
        service.preview_values(draft)


def test_non_scalar_current_and_proposed_summaries_use_the_same_bounded_shape(session):
    _entry, control, service = session
    matrix = [[0.95, 0.03, 0.02], [0.03, 0.95, 0.02], [0.0, 0.0, 1.0]]
    control.set_parameter(
        control.read_context().target, "general.ext.confusion_matrix", matrix
    )
    draft = service.create_draft(
        [
            ParameterWriteback(
                "general.ext.confusion_matrix", "d", proposed_value=matrix
            )
        ],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    assert service.get_summaries(draft, "param-1") == ("3 × 3 matrix", "3 × 3 matrix")


@pytest.mark.parametrize(
    ("value", "summary"),
    [
        ([], "list[0]"),
        (((1, 2), (3, 4)), "2 × 2 matrix"),
        ([[1], [2, 3]], "list[2]"),
        ([[], []], "list[2]"),
        ([[{"value": 1}]], "list[1]"),
        ([1, 2, 3], "list[3]"),
    ],
)
def test_sequence_proposals_have_bounded_shape_summaries(session, value, summary):
    _entry, control, service = session
    draft = service.create_draft(
        [ParameterWriteback("general.ext.sequence", "sequence", proposed_value=value)],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    assert service.get_summaries(draft, "param-1")[1] == summary
    draft.teardown()


def test_applied_state_follows_successful_writes_and_content_edits(session):
    _entry, control, service = session
    draft = service.create_draft(
        [
            ParameterWriteback("general.ext.r_f", "d", proposed_value=1.0),
            ParameterWriteback("general.ext.skip", "d", proposed_value=2.0),
        ],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    assert service.get_all_applied(draft) == {"param-1": False, "param-2": False}
    draft.edit("param-2", selected=False)
    draft.apply()
    assert service.get_all_applied(draft) == {"param-1": True, "param-2": False}
    draft.edit("param-1", selected=False)
    assert service.get_all_applied(draft)["param-1"] is True
    draft.edit("param-1", target_name="general.ext.r_f", proposed_value=1.0)
    assert service.get_all_applied(draft)["param-1"] is True
    draft.edit("param-1", proposed_value=3.0)
    assert service.get_all_applied(draft)["param-1"] is False


def test_duplicate_selected_destination_is_rejected_before_write(session):
    entry, control, service = session
    draft = service.create_draft(
        [
            ParameterWriteback("general.ext.same", "first", proposed_value=1.0),
            ParameterWriteback("general.ext.same", "second", proposed_value=2.0),
        ],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    count = len(entry.ledger.events())
    with pytest.raises(InvalidInputError, match="duplicate writeback destination"):
        draft.apply()
    assert len(entry.ledger.events()) == count
    assert not control.read_context().lookup("general.ext.same").present
    assert service.get_all_applied(draft) == {"param-1": False, "param-2": False}


def test_failed_and_empty_apply_do_not_mark_items_applied(session):
    _entry, control, service = session
    draft = service.create_draft(
        [ParameterWriteback("general.description", "d", proposed_value=42)],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    with pytest.raises(ValueError, match="description"):
        draft.apply()
    assert service.get_all_applied(draft) == {"param-1": False}
    draft.edit("param-1", selected=False)
    assert draft.apply()["applied_ids"] == []
    assert service.get_all_applied(draft) == {"param-1": False}


def test_explicit_apply_ignores_selection_without_changing_checkboxes(session):
    _entry, control, service = session
    draft = service.create_draft(
        [
            ParameterWriteback("general.ext.a", "d", proposed_value=1.0),
            ParameterWriteback("general.ext.b", "d", proposed_value=2.0),
        ],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    draft.edit("param-1", selected=False)
    selection = [item.selected for item in draft.items]
    assert draft.apply(item_ids=("param-1",))["applied_ids"] == ["param-1"]
    current = control.read_context()
    assert current.value("general.ext.a") == 1.0
    assert not current.lookup("general.ext.b").present
    assert [item.selected for item in draft.items] == selection
    assert service.get_all_applied(draft) == {"param-1": True, "param-2": False}


@pytest.mark.parametrize("item_ids", [("param-1", "missing"), ("param-1", "param-1")])
def test_explicit_apply_rejects_invalid_ids_before_any_write(session, item_ids):
    entry, control, service = session
    draft = service.create_draft(
        [ParameterWriteback("general.ext.a", "d", proposed_value=1.0)],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    count = len(entry.ledger.events())
    with pytest.raises(InvalidInputError):
        draft.apply(item_ids=item_ids)
    assert len(entry.ledger.events()) == count
    assert not control.read_context().lookup("general.ext.a").present
    assert service.get_all_applied(draft) == {"param-1": False}


def test_explicit_empty_apply_does_not_fall_back_to_selected_items(session):
    entry, control, service = session
    draft = service.create_draft(
        [ParameterWriteback("general.ext.a", "d", proposed_value=1.0)],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    count = len(entry.ledger.events())
    assert draft.apply(item_ids=())["applied_ids"] == []
    assert len(entry.ledger.events()) == count
    assert not control.read_context().lookup("general.ext.a").present
    assert service.get_all_applied(draft) == {"param-1": False}


def test_ordered_write_returns_actual_values_and_leaves_selection_alone(session):
    _entry, control, service = session
    control.set_parameter(control.read_context().target, "general.ext.dest", [1])
    draft = service.create_draft(
        [
            ParameterWriteback("general.ext.a", "d", proposed_value=2),
            ParameterWriteback("general.ext.b", "d", proposed_value=3),
        ],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    draft.edit("param-1", selected=False)
    result = service.write_draft(
        draft,
        (
            WritebackEdit(
                "param-1", target_name="general.ext.dest", proposed_value=None
            ),
        ),
    )
    assert result == [
        WritebackWritten("param-1", "parameter", "general.ext.dest", [1], None)
    ]
    current = control.read_context()
    assert current.value("general.ext.dest") is None
    assert not current.lookup("general.ext.b").present
    assert [item.selected for item in draft.items] == [False, True]
    assert service.get_all_applied(draft) == {"param-1": True, "param-2": False}


def test_ordered_write_stops_on_unknown_item_and_preserves_draft_prefix(session):
    entry, control, service = session
    draft = service.create_draft(
        [
            ParameterWriteback("general.ext.a", "d", proposed_value=1),
            ParameterWriteback("general.ext.b", "d", proposed_value=2),
        ],
        parameters=control.read_context(),
        experiment_name="test/writeback",
    )
    count = len(entry.ledger.events())
    with pytest.raises(InvalidInputError):
        service.write_draft(
            draft,
            (
                WritebackEdit("param-1", proposed_value=11),
                WritebackEdit("missing"),
                WritebackEdit("param-2", proposed_value=22),
            ),
        )
    values = []
    for item in draft.items:
        assert isinstance(item, ParameterWriteback)
        values.append(item.proposed_value)
    assert values == [11, 2]
    assert len(entry.ledger.events()) == count
    assert not control.read_context().lookup("general.ext.a").present
    assert service.get_all_applied(draft) == {"param-1": False, "param-2": False}
