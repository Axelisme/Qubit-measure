"""Previews resolve captured paths but read the GUI's current point."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import Field
from zcu_tools.cfg_model import (
    ArbWaveformCfg,
    ConstWaveformCfg,
    FlatTopWaveformSpec,
    PulseSpec,
)
from zcu_tools.core.waveform_format import WaveformSamples
from zcu_tools.experiment import RunRecord
from zcu_tools.gui.app.measure.adapter import WritebackRequest
from zcu_tools.gui.app.measure.writeback import (
    ArbAssetWriteback,
    ModuleWriteback,
    ParameterWriteback,
    WritebackEdit,
    WritebackService,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.library_entry import LibraryEntryInvalid
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.resources.entry import ComponentSchema, ResultEntry, RoleSpec
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase, ArbWaveformError

from tests.gui._context_reads import capture_focus
from tests.gui.app.measure._record_support import require_target
from tests.gui.app.measure._synthetic import (
    ARB_PROPOSAL_KEY,
    MODULE_PROPOSAL_TARGET,
    MODULE_PROPOSAL_WAVEFORM,
    ArbProposalAdapter,
    SampleAnalysis,
    SampleCfg,
    SampleResult,
)
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry
from tests.resources.entry.view_support import registered_model


@pytest.fixture
def component_kinds():
    before = registry_state()
    try:
        register_fakes()
        installed = registry_state()
        yield
        assert registry_state() == installed
    finally:
        restore_registry(before)


@pytest.fixture
def session(tmp_path: Path, component_kinds):
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    entry.new_point("a")
    bus = BaseEventBus()
    context = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        bus,
        bind_assets=ArbWaveformDatabase.info_at,
    )
    context.install_entry(entry)
    control = context
    control.use_point("a")
    control.add_component("Q1", "fake/drive/a", {}, point=control.current_point_label())
    service = WritebackService(control, arb_names=lambda: ())
    return entry, control, service


@pytest.fixture
def make_arb_draft(session):
    _entry, control, service = session

    def create():
        run = RunRecord(
            SampleCfg(sweep=SweepCfg(start=0.0, stop=1.0, expts=2, step=1.0)),
            SampleResult(np.array([0.3])),
        )
        return service.create_draft(
            ArbProposalAdapter.get_writeback_items(
                WritebackRequest(
                    run, SampleAnalysis(peak=0.3), control.get_session_env()
                )
            ),
            parameters=capture_focus(control),
            experiment_name="synthetic/arb",
        )

    return create


def test_module_form_accepts_arb_key_proposed_later_in_same_draft(make_arb_draft):
    draft = make_arb_draft()
    assert draft.get_item_form("ml-1").spec().to_dict()["waveform"] == {
        "style": "arb",
        "data": ARB_PROPOSAL_KEY,
    }
    assert [item.session_id for item in draft.preview()] == ["ml-1", "arb-1"]


def test_unchecked_arb_publish_blocks_dependent_module_before_acceptance(
    session, make_arb_draft
):
    entry, control, _service = session
    draft = make_arb_draft()
    draft.edit("arb-1", selected=False)
    with pytest.raises(InvalidInputError, match=r"ml-1.*Q1\.pulses\.tuned.*optimized"):
        draft.apply()
    assert entry.ledger.events() == ()
    assert MODULE_PROPOSAL_TARGET not in capture_focus(control).library.module_keys()
    assert not (entry.database_path / "arb_waveforms" / "optimized.npz").exists()


def test_all_selected_arb_asset_commits_before_the_referencing_library_item(
    session, make_arb_draft
):
    entry, control, _service = session
    draft = make_arb_draft()
    result = draft.accept(("ml-1", "arb-1"))
    assert result.error is None
    assert [(row.id, row.kind) for row in result.values] == [
        ("arb-1", "arb_asset"),
        ("ml-1", "module"),
    ]
    stored = ArbWaveformDatabase.info_at(entry.database_path / "arb_waveforms")
    np.testing.assert_array_equal(
        stored.samples(ARB_PROPOSAL_KEY).qdata, [0.0, 0.4, 0.0]
    )
    saved = capture_focus(control).library.entry_value(
        MODULE_PROPOSAL_TARGET, kind="module"
    )
    assert saved["waveform"] == {"style": "arb", "data": ARB_PROPOSAL_KEY}


def test_apply_checks_arb_reference_in_active_library_not_creation_capture(session):
    entry, control, service = session
    target = require_target(capture_focus(control))
    control.set_waveform(target, MODULE_PROPOSAL_WAVEFORM, ConstWaveformCfg(length=0.1))
    captured = capture_focus(control)
    draft = service.create_draft(
        [
            ModuleWriteback(
                MODULE_PROPOSAL_TARGET,
                "pulse",
                PulseSpec(
                    ch=0,
                    nqz=1,
                    freq=5000.0,
                    gain=0.2,
                    waveform=MODULE_PROPOSAL_WAVEFORM,
                ),
            )
        ],
        parameters=captured,
        experiment_name="synthetic/arb",
    )
    control.set_waveform(
        target,
        MODULE_PROPOSAL_WAVEFORM,
        FlatTopWaveformSpec(length=1.0, raise_waveform=ArbWaveformCfg(data="missing")),
    )
    with pytest.raises(InvalidInputError, match=r"ml-1.*Q1\.pulses\.tuned.*missing"):
        draft.apply()
    assert entry.ledger.events() == ()
    repository = ArbWaveformDatabase(entry.database_path / "arb_waveforms")
    repository.save("missing", idata=[0.0, 0.4], time=[0.0, 0.25])
    draft.apply()
    assert (
        capture_focus(control).library.entry_value(
            MODULE_PROPOSAL_TARGET, kind="module"
        )["waveform"]
        == MODULE_PROPOSAL_WAVEFORM
    )


@pytest.mark.parametrize(
    "change",
    [
        WritebackEdit("arb-1", target_name="renamed"),
        WritebackEdit("arb-1", proposed_value={}),
        WritebackEdit("arb-1", edits={}),
    ],
    ids=["retarget", "replace-value", "form-edit"],
)
def test_arb_publish_refuses_edits_before_writing(session, make_arb_draft, change):
    entry, _control, _service = session
    draft = make_arb_draft()
    with pytest.raises(
        InvalidInputError, match=r"arb-1.*arb asset.*cannot be edited or retargeted"
    ):
        draft.writeback_write((change,))
    assert draft.preview()[1].target_name == ARB_PROPOSAL_KEY
    assert entry.ledger.events() == ()


def test_rejected_arb_edit_keeps_default_selection(make_arb_draft):
    draft = make_arb_draft()
    with pytest.raises(InvalidInputError, match="cannot be edited or retargeted"):
        draft.edit("arb-1", selected=False, target_name="renamed")

    asset = draft.preview()[1]
    assert asset.selected
    assert asset.target_name == ARB_PROPOSAL_KEY


def test_invalid_arb_key_rejects_draft_before_ledger_append(session):
    entry, control, service = session
    draft = service.create_draft(
        [
            ArbAssetWriteback(
                "bad-key",
                "Invalid asset key",
                samples=WaveformSamples(
                    np.array([0.0, 0.25]), np.array([0.0, 0.4]), None
                ),
            )
        ],
        parameters=capture_focus(control),
        experiment_name="synthetic/arb",
    )

    with pytest.raises(ArbWaveformError, match="data key must match"):
        draft.apply()

    assert entry.ledger.events() == ()
    assert not (entry.database_path / "arb_waveforms").exists()


def test_arb_preview_compares_live_samples_without_creating_or_overwriting_assets(
    session, make_arb_draft
):
    entry, _control, _service = session
    draft = make_arb_draft()
    root = entry.database_path / "arb_waveforms"

    def preview():
        return draft.preview_values(session_id="arb-1")["arb-1"]

    assert preview().current == "absent"
    assert not root.exists()
    repository = ArbWaveformDatabase(root)
    repository.save(
        ARB_PROPOSAL_KEY, [0.0, 0.3, -0.0], [0.0, 0.25, 0.75], [0.0, 0.4, 0.0]
    )
    assert preview().current == "identical"
    repository.save(ARB_PROPOSAL_KEY, [0.0, 0.4, 0.0], [0.0, 0.25, 0.75])
    facts = preview()
    assert facts.current == "different"
    assert facts.warning is not None and "Apply will fail" in facts.warning
    assert entry.ledger.events() == ()


def test_static_unit_facts_follow_selected_point_and_captured_path_without_conversion(
    session,
):
    entry, control, service = session

    class Measured(ComponentSchema):
        frequency: float = Field(default=19.8, json_schema_extra={"unit": "MHz"})
        duration: float | None = Field(default=None, json_schema_extra={"unit": "us"})

    with registered_model("fake/unit-preview", Measured) as kind:
        control.add_component("U1", kind, {}, point=control.current_point_label())
        target = capture_focus(control).target
        captured = capture_focus(control).bind_roles(
            {"driver": RoleSpec(kind)}, focus="U1", explicit={}
        )
        draft = service.create_draft(
            [
                ParameterWriteback(
                    "$driver.frequency", "Frequency", proposed_value=20.09, stderr=0.18
                )
            ],
            parameters=captured,
            experiment_name="test/writeback",
        )
        try:
            initial = draft.preview_values(session_id="param-1")["param-1"]
            assert (initial.actual_path, initial.unit) == ("U1.frequency", "MHz")
            assert (initial.current, initial.proposed) == (19.8, 20.09)
            control.set_parameter(target, "U1.frequency", 21.5)
            entry.new_point("b")
            control.use_point("b")
            control.add_component(
                "Q2", "fake/drive/a", {}, point=control.current_point_label()
            )
            control.add_component(
                "U1", kind, {"frequency": 30.0}, point=control.current_point_label()
            )
            control.apply_alias_rows(
                (AliasRow("driver", "Q2"),), point=control.current_point_label()
            )
            refreshed = draft.preview_values()["param-1"]
            assert (refreshed.actual_path, refreshed.unit) == ("U1.frequency", "MHz")
            assert (refreshed.current, refreshed.proposed) == (30.0, 20.09)
            draft.edit("param-1", target_name="$driver.duration")
            unset = draft.preview_values()["param-1"]
            assert (unset.actual_path, unset.unit) == ("U1.duration", "us")
            assert (unset.current, unset.proposed) == (None, 20.09)
            draft.edit("param-1", target_name="general.ext.value")
            ext = draft.preview_values()["param-1"]
            assert (ext.actual_path, ext.unit) == ("general.ext.value", None)
            assert (ext.current, ext.proposed) == (None, 20.09)
            assert draft.items[0].stderr == 0.18
            assert capture_focus(control).target != target
        finally:
            draft.teardown()


def test_concrete_destination_does_not_use_alias_or_historical_point_fallback(session):
    entry, control, service = session

    class Coupler(ComponentSchema):
        module: dict[str, str]

    with registered_model("fake/preview-coupler", Coupler) as kind:
        control.add_component(
            "coupler",
            kind,
            {"module": {"cz_f2": "coupler.pulses.original"}},
            point=control.current_point_label(),
        )
        control.apply_alias_rows(
            (AliasRow("coupler", "coupler"),), point=control.current_point_label()
        )
        captured = capture_focus(control)
        draft = service.create_draft(
            [
                ParameterWriteback(
                    "coupler.module.cz_f2",
                    "Calibrated pulse slot",
                    proposed_value="coupler.pulses.calibrated",
                )
            ],
            parameters=captured,
            experiment_name="test/writeback",
        )
        try:
            initial = draft.preview_values(session_id="param-1")["param-1"]
            assert initial.actual_path == "coupler.module.cz_f2"
            assert initial.current == "coupler.pulses.original"
            assert initial.proposed == "coupler.pulses.calibrated"
            control.set_parameter(
                captured.target, "coupler.module.cz_f2", "coupler.pulses.updated"
            )
            entry.new_point("b")
            control.use_point("b")
            control.add_component(
                "other_coupler",
                kind,
                {"module": {"cz_f2": "other.pulses.decoy"}},
                point=control.current_point_label(),
            )
            control.apply_alias_rows(
                (AliasRow("coupler", "other_coupler"),),
                point=control.current_point_label(),
            )
            refreshed = draft.preview_values()["param-1"]
            assert refreshed.actual_path == "coupler.module.cz_f2"
            assert refreshed.current is None
            assert refreshed.proposed == "coupler.pulses.calibrated"
            assert captured.value("coupler.module.cz_f2") == "coupler.pulses.original"
            assert capture_focus(control).target != captured.target
        finally:
            draft.teardown()


def test_values_follow_selected_point_retarget_edits_and_actual_current_values(session):
    entry, control, service = session
    target = capture_focus(control).target
    control.set_parameter(target, "general.ext.value", [1, 2])
    capture = capture_focus(control)
    draft = service.create_draft(
        [ParameterWriteback("general.ext.value", "d", proposed_value=[3, 4])],
        parameters=capture,
        experiment_name="test/writeback",
    )
    first = draft.preview_values()["param-1"]
    assert first.actual_path == "general.ext.value"
    assert first.current == [1, 2]
    assert first.proposed == [3, 4]
    control.set_parameter(target, "general.ext.value", [1, 2, 5])
    assert first.current == [1, 2]
    assert isinstance(first.proposed, list)
    first.proposed.append(99)
    assert draft.preview_values()["param-1"].proposed == [3, 4]
    assert draft.preview_values()["param-1"].current == [1, 2, 5]
    draft.edit("param-1", target_name="general.ext.other")
    assert draft.preview_values()["param-1"].current is None
    control.set_parameter(target, "general.ext.other", 7)
    assert draft.preview_values()["param-1"].current == 7
    entry.new_point("b")
    control.use_point("b")
    control.set_parameter(
        capture_focus(control).target,
        "general.ext.other",
        8,
    )
    retargeted = draft.preview_values()["param-1"]
    assert retargeted.actual_path == "general.ext.other"
    assert retargeted.current == 8
    assert draft.get_all_applied() == {"param-1": False}
    assert draft.items[0].selected is True


def test_module_preview_contains_complete_current_and_live_form_values(session):
    _entry, control, service = session
    name = "Q1.pulses.target"
    control.set_module(
        capture_focus(control).target,
        name,
        PulseSpec(
            ch=0, nqz=1, freq=100.0, gain=0.5, waveform=ConstWaveformCfg(length=0.1)
        ),
    )
    capture = capture_focus(control)
    proposal = ModuleWriteback(name, "d", capture.library.module_spec(name))
    draft = service.create_draft(
        [proposal], parameters=capture, experiment_name="test/writeback"
    )
    preview = draft.preview_values()["ml-1"]
    assert preview.actual_path == name
    assert preview.unit is None
    assert preview.current == capture.library.entry_value(name, kind="module")
    assert preview.proposed == preview.current
    draft.edit("ml-1", edits={("freq",): 250.0})
    live = draft.preview_values()["ml-1"]
    assert isinstance(live.proposed, dict)
    assert live.proposed["freq"] == 250.0
    assert live.current == preview.current
    draft.edit("ml-1", target_name=name.replace("target", "new"))
    missing = draft.preview_values()["ml-1"]
    assert missing.actual_path == name.replace("target", "new")
    assert missing.unit is None
    assert missing.current is None
    assert missing.proposed == live.proposed
    draft.teardown()
    with pytest.raises(FailedPreconditionError):
        draft.preview_values()


def test_item_preview_excludes_invalid_siblings_but_bulk_remains_strict(session):
    _entry, control, service = session
    proposals = []
    for suffix in ("a", "b"):
        name = f"Q1.pulses.{suffix}"
        control.set_module(
            capture_focus(control).target,
            name,
            PulseSpec(
                ch=0, nqz=1, freq=100.0, gain=0.5, waveform=ConstWaveformCfg(length=0.1)
            ),
        )
        proposals.append(
            ModuleWriteback(
                name,
                name,
                capture_focus(control).library.module_spec(name),
            )
        )
    draft = service.create_draft(
        proposals,
        parameters=capture_focus(control),
        experiment_name="test/writeback",
    )
    try:
        first, second = draft.items
        second.selected = False
        form = draft.get_item_form(second.session_id)
        form.state.write({("freq",): ""}, form.env)
        selected = draft.preview_values(session_id=first.session_id)
        assert list(selected) == [first.session_id]
        fact = selected[first.session_id]
        assert fact.actual_path == first.target_name
        assert isinstance(fact.current, dict) and isinstance(fact.proposed, dict)
        assert fact.current["freq"] == fact.proposed["freq"] == 100.0
        with pytest.raises(LibraryEntryInvalid):
            draft.preview_values()
        with pytest.raises(LibraryEntryInvalid):
            draft.preview_values(session_id=second.session_id)
        with pytest.raises(KeyError):
            draft.preview_values(session_id="missing")
        form.state.write({("freq",): 100.0}, form.env)
        assert len(draft.preview_values()) == 2
        assert second.selected is False
    finally:
        draft.teardown()
    empty = service.create_draft([], parameters=None, experiment_name="test/writeback")
    try:
        assert empty.preview_values() == {}
        with pytest.raises(KeyError):
            empty.preview_values(session_id="missing")
    finally:
        empty.teardown()


def test_preview_without_context_fails_instead_of_claiming_missing_targets():
    port = MagicMock()
    port.read_context.side_effect = FailedPreconditionError(
        "Select a point", reason_code="no_context"
    )
    service = WritebackService(port, arb_names=lambda: ())
    draft = service.create_draft(
        [ParameterWriteback("general.ext.value", "d", proposed_value=1)],
        parameters=None,
        experiment_name="test/writeback",
    )
    with pytest.raises(FailedPreconditionError):
        draft.preview_values()
