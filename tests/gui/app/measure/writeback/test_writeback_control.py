"""WritebackControlFacet public contract tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.catalog import ExperimentAccess
from zcu_tools.gui.app.measure.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.measure.services.guard import (
    GuardError,
    GuardService,
    TabAction,
)
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.app.measure.writeback import (
    CapturedWritebackPane,
    ModuleWriteback,
    ParameterWriteback,
    WritebackControlFacet,
    WritebackDraft,
    WritebackEdit,
    WritebackPane,
    WritebackService,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase


@dataclass
class ControlSession:
    facet: WritebackControlFacet
    state: State
    bus: BaseEventBus
    service: WritebackService
    draft: WritebackDraft
    foreign: WritebackDraft
    contexts: tuple[ContextService, ...]
    entries: tuple[ResultEntry, ...]
    versions: dict[str, int]


@pytest.fixture
def control_session(tmp_path: Path) -> ControlSession:
    contexts = []
    entries = []
    for name in ("local", "foreign"):
        entry = ResultEntry.create(
            name, result_root=tmp_path / "Result", database_root=tmp_path / "Database"
        )
        entry.new_point("a")
        context = ContextService(
            SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
            BaseEventBus(),
            bind_assets=ArbWaveformDatabase.info_at,
        )
        context.install_entry(entry)
        context.use_point(PointUseRequest("a"))
        entries.append(entry)
        contexts.append(context)
    service = WritebackService(contexts[0], arb_names=lambda: ())
    other = WritebackService(contexts[1], arb_names=lambda: ())
    proposals = [
        ParameterWriteback("general.ext.a", "parameter", proposed_value=1),
        ModuleWriteback(
            "Q1.pulses.test",
            "module",
            PulseSpec(
                ch=0,
                nqz=1,
                freq=5000.0,
                gain=0.2,
                waveform=ConstWaveformCfg(length=0.1),
            ),
            selected=False,
        ),
    ]
    draft = service.create_draft(
        proposals, parameters=contexts[0].read_context(), experiment_name="synthetic"
    )
    foreign = other.create_draft(
        proposals, parameters=contexts[1].read_context(), experiment_name="synthetic"
    )
    state = State(
        SessionEnv(parameters=contexts[0].read_context(), soc=None, soccfg=None)
    )
    state.add_tab(
        "tab-1", Session(adapter_name="synthetic", adapter=MagicMock(), cfg=MagicMock())
    )
    state.get_tab("tab-1").analysis.result = object()
    state.get_tab("tab-1").analysis.writeback_draft = draft
    bus = BaseEventBus()
    versions = {"context": 7}
    facet = WritebackControlFacet(
        state=state,
        guard=GuardService(state, access=ExperimentAccess()),
        writeback=service,
        resource_versions=lambda: versions,
        bus=bus,
    )
    return ControlSession(
        facet,
        state,
        bus,
        service,
        draft,
        foreign,
        tuple(contexts),
        tuple(entries),
        versions,
    )


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
@pytest.mark.parametrize(
    "operation",
    ["form", "edit", "apply", "write", "values", "summaries", "applied", "accept"],
)
def test_foreign_pane_draft_is_rejected_without_effects(
    control_session: ControlSession, pane: WritebackPane, operation: str
) -> None:
    session = control_session
    getattr(session.state.get_tab("tab-1"), pane).writeback_draft = session.foreign
    facet = session.facet
    events = []
    session.bus.subscribe(TabInteractionChangedPayload, events.append)
    before = [entry.ledger.events() for entry in session.entries]
    reservation = facet.begin_writeback("tab-1")
    operations = {
        "form": lambda: facet.get_writeback_item_form_for_pane("tab-1", pane, "ml-1"),
        "edit": lambda: facet.set_writeback_item_for_pane(
            "tab-1", pane, "param-1", proposed_value=9
        ),
        "apply": lambda: facet.apply_writeback_for_pane(
            "tab-1", pane, item_ids=("param-1",)
        ),
        "write": lambda: facet.write_writeback_for_pane(
            "tab-1",
            pane,
            (WritebackEdit("param-1", proposed_value=9),),
            submission_id=reservation.submission_id,
        ),
        "values": lambda: facet.get_writeback_values_for_pane("tab-1", pane),
        "summaries": lambda: facet.get_writeback_summaries_for_pane("tab-1", pane),
        "applied": lambda: facet.get_writeback_applied_for_pane("tab-1", pane),
        "accept": lambda: facet.accept_writeback(
            "tab-1",
            (CapturedWritebackPane(pane, ("param-1",)),),
            submission_id=reservation.submission_id,
        ),
    }
    with pytest.raises(InvalidInputError, match="unknown writeback draft"):
        operations[operation]()
    assert session.foreign.preview_values(session_id="param-1")["param-1"].proposed == 1
    assert not any(session.foreign.get_all_applied().values())
    assert [entry.ledger.events() for entry in session.entries] == before
    assert all(
        not context.read_context().lookup("general.ext.a").present
        for context in session.contexts
    )
    assert [event.fact for event in events] == (
        [TabInteractionFact.WRITEBACK_DRAFT_CHANGED]
        if operation in ("apply", "write", "accept")
        else []
    )
    receipt = facet.get_writeback_receipt(reservation.submission_id)
    assert receipt.status == (
        "failed" if operation in ("write", "accept") else "pending"
    )
    assert receipt.completed == []
    assert receipt.error == (
        "unknown writeback draft" if operation in ("write", "accept") else None
    )


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
def test_pane_writeback_reads_edits_and_applies_its_own_draft(
    control_session: ControlSession, pane: WritebackPane
) -> None:
    session = control_session
    post = session.service.create_draft(
        [ParameterWriteback("general.ext.a", "post", proposed_value=2)],
        parameters=session.contexts[0].read_context(),
        experiment_name="synthetic",
    )
    session.state.get_tab("tab-1").post_analysis.writeback_draft = post
    facet = session.facet
    events = []
    session.bus.subscribe(TabInteractionChangedPayload, events.append)
    form = facet.get_writeback_item_form_for_pane("tab-1", "analysis", "ml-1")
    assert form is session.draft.get_item_form("ml-1")
    facet.set_writeback_item_for_pane("tab-1", pane, "param-1", proposed_value=9)
    result = facet.apply_writeback_for_pane("tab-1", pane, item_ids=("param-1",))
    assert result["applied_ids"] == ["param-1"]
    assert session.contexts[0].read_context().value("general.ext.a") == 9
    assert session.draft.preview_values(session_id="param-1")["param-1"].proposed == (
        9 if pane == "analysis" else 1
    )
    assert post.preview_values()["param-1"].proposed == (
        9 if pane == "post_analysis" else 2
    )
    assert facet.get_writeback_applied_for_pane("tab-1", pane)["param-1"]
    assert [event.fact for event in events] == [
        TabInteractionFact.WRITEBACK_DRAFT_CHANGED
    ] * 2


@pytest.mark.parametrize("item_ids", [("param-1",), ()])
def test_explicit_item_ids_override_selection(
    control_session: ControlSession, item_ids: tuple[str, ...]
) -> None:
    facet = control_session.facet
    facet.set_writeback_item_for_pane("tab-1", "analysis", "param-1", selected=False)
    result = facet.apply_writeback_for_pane("tab-1", "analysis", item_ids=item_ids)
    assert result["applied_ids"] == list(item_ids)
    assert facet.get_writeback_applied_for_pane("tab-1", "analysis")["param-1"] == bool(
        item_ids
    )


@pytest.mark.parametrize("operation", ["form", "edit", "apply", "write", "accept"])
@pytest.mark.parametrize(
    "blocker", ["admission", "no_context", "unknown_tab", "closed", "foreign"]
)
def test_pane_checks_precede_item_validation(
    control_session: ControlSession, operation: str, blocker: str
) -> None:
    session = control_session
    session.state.get_tab("tab-1").analysis.writeback_draft = session.foreign
    events = []
    session.bus.subscribe(TabInteractionChangedPayload, events.append)
    tab_id = "missing" if blocker == "unknown_tab" else "tab-1"
    if blocker == "admission":
        session.state.set_tab_analyzing("tab-1", analyzing=True)
    if blocker == "no_context":
        session.state.set_context(replace(session.state.session_env, parameters=None))
    if blocker == "closed":
        session.foreign.teardown()
    facet = session.facet
    operations = {
        "form": lambda: facet.get_writeback_item_form_for_pane(
            tab_id, "analysis", "missing"
        ),
        "edit": lambda: facet.set_writeback_item_for_pane(
            tab_id, "analysis", "missing", selected=False
        ),
        "apply": lambda: facet.apply_writeback_for_pane(
            tab_id, "analysis", item_ids=("missing",)
        ),
        "write": lambda: facet.write_writeback_for_pane(
            tab_id, "analysis", (WritebackEdit("missing"),)
        ),
        "accept": lambda: facet.accept_writeback(
            tab_id, (CapturedWritebackPane("analysis", ("missing",)),)
        ),
    }
    if blocker in ("admission", "no_context", "unknown_tab"):
        with pytest.raises(GuardError) as caught:
            operations[operation]()
        assert (
            caught.value.reason_code
            == {
                "admission": "tab_busy",
                "no_context": "no_context",
                "unknown_tab": "unknown_tab",
            }[blocker]
        )
    elif blocker == "closed":
        with pytest.raises(FailedPreconditionError, match="torn down"):
            operations[operation]()
    else:
        with pytest.raises(InvalidInputError, match="unknown writeback draft"):
            operations[operation]()
    assert [event.fact for event in events] == (
        [TabInteractionFact.WRITEBACK_DRAFT_CHANGED]
        if blocker == "foreign" and operation in ("apply", "write", "accept")
        else []
    )


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
def test_pane_writeback_requires_an_existing_draft(
    control_session: ControlSession, pane: WritebackPane
) -> None:
    session = control_session
    session.state.get_tab("tab-1").analysis.writeback_draft = None
    session.state.get_tab("tab-1").post_analysis.writeback_draft = None
    with pytest.raises(FailedPreconditionError, match="No .*writeback draft"):
        session.facet.apply_writeback_for_pane("tab-1", pane)
    assert session.facet.get_writeback_summaries_for_pane("tab-1", pane) == {}
    assert session.facet.get_writeback_applied_for_pane("tab-1", pane) == {}


@pytest.mark.parametrize("pane", ["save"])
def test_unknown_pane_precedes_foreign_draft_validation(
    control_session: ControlSession, pane: WritebackPane
) -> None:
    """Invalid runtime pane values keep the existing rejection before draft checks."""
    session = control_session
    session.state.get_tab("tab-1").analysis.writeback_draft = session.foreign
    with pytest.raises(InvalidInputError, match="unknown writeback pane"):
        session.facet.apply_writeback_for_pane("tab-1", pane)
    with pytest.raises(InvalidInputError, match="unknown writeback pane"):
        session.facet.get_writeback_applied_for_pane("tab-1", pane)


def test_submission_identity_precedes_admission_and_foreign_draft(
    control_session: ControlSession,
) -> None:
    session = control_session
    session.state.get_tab("tab-1").analysis.writeback_draft = session.foreign
    session.state.get_tab("tab-1").is_analyzing = True
    with pytest.raises(InvalidInputError, match="unknown writeback submission_id"):
        session.facet.write_writeback_for_pane(
            "tab-1", "analysis", (), submission_id="unknown"
        )


def test_has_tab_reads_state(control_session: ControlSession) -> None:
    assert control_session.facet.has_tab("tab-1")
    assert not control_session.facet.has_tab("missing")


def test_get_context_version_reads_resource_versions(
    control_session: ControlSession,
) -> None:
    assert control_session.facet.get_context_version() == 7
    control_session.versions["context"] = 9
    assert control_session.facet.get_context_version() == 9


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
@pytest.mark.parametrize("fails", [False, True])
def test_batch_write_publishes_shared_draft_even_after_partial_failure(
    pane, fails, tmp_path: Path
):
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    entry.new_point("a")
    bus = BaseEventBus()
    context_service = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        bus,
        bind_assets=ArbWaveformDatabase.info_at,
    )
    context_service.install_entry(entry)
    control = context_service
    control.use_point(PointUseRequest("a"))
    parameters = control.read_context()
    state = State(SessionEnv(parameters=parameters, soc=None, soccfg=None))
    state.add_tab(
        "tab-1",
        Session(
            adapter_name="fake",
            adapter=MagicMock(),
            cfg=MagicMock(),
        ),
    )
    service = WritebackService(control, arb_names=lambda: ())
    draft = service.create_draft(
        [ParameterWriteback("general.ext.a", "d", proposed_value=1)],
        parameters=parameters,
        experiment_name="test/writeback",
    )
    tab = state.get_tab("tab-1")
    getattr(tab, pane).writeback_draft = draft
    accepted_before = len(entry.ledger.events())
    observed = []
    bus.subscribe(
        TabInteractionChangedPayload,
        lambda event: observed.append(
            (
                event.tab_id,
                event.fact,
                draft.preview_values()["param-1"].proposed,
            )
        ),
    )
    guard = MagicMock()
    guard.require_admission.side_effect = lambda tab_id, action: (
        GuardService.require_admission(guard, tab_id, action)
    )
    facet = WritebackControlFacet(
        state=state,
        guard=guard,
        writeback=service,
        resource_versions=lambda: {},
        bus=bus,
    )
    changes = (WritebackEdit("param-1", proposed_value=11),)
    if fails:
        with pytest.raises(InvalidInputError):
            facet.write_writeback_for_pane(
                "tab-1",
                pane,
                (*changes, WritebackEdit("missing")),
            )
        assert len(entry.ledger.events()) == accepted_before
        assert not control.read_context().lookup("general.ext.a").present
    else:
        result = facet.write_writeback_for_pane("tab-1", pane, changes)
        assert [item.id for item in result] == ["param-1"]
        assert len(entry.ledger.events()) == accepted_before + 1
        assert control.read_context().value("general.ext.a") == 11
    guard.admit.assert_called_once_with("tab-1", TabAction.WRITEBACK)
    assert observed == [
        ("tab-1", TabInteractionFact.WRITEBACK_DRAFT_CHANGED, 11),
    ]
