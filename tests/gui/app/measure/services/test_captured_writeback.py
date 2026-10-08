"""Accepted evidence and fixed destinations at the real writeback/owner seam."""

from collections.abc import Generator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.format_version import YamlValue
from zcu_tools.gui.app.measure.adapter import (
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
)
from zcu_tools.gui.app.measure.cfg_schemas import (
    module_cfg_to_value,
    waveform_cfg_to_value,
)
from zcu_tools.gui.app.measure.services.cfg_editor import CfgEditorService
from zcu_tools.gui.app.measure.services.writeback import (
    WritebackEdit,
    WritebackService,
    WritebackWritten,
)
from zcu_tools.gui.cfg import CfgSchema, EvalValue
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    ContextTarget,
    PointUseRequest,
)
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.entry.ledger import RecordsLedger
from zcu_tools.resources.entry.ledger_models import AcceptedPayload, LedgerEvent
from zcu_tools.resources.entry.provenance import Provenance
from zcu_tools.resources.entry.registry import RoleSpec
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


@dataclass
class WritebackSession:
    entry: ResultEntry
    control: ContextControlFacet
    service: WritebackService


@pytest.fixture
def session(tmp_path: Path) -> Generator[WritebackSession]:
    before = registry_state()
    try:
        register_fakes()
        installed = registry_state()
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "results",
            database_root=tmp_path / "Database",
        )
        bus = BaseEventBus()
        context = ContextService(
            SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
            bus,
            bind_assets=ArbWaveformDatabase.info_at,
        )
        context.install_entry(entry)
        control = ContextControlFacet(context=context)
        control.add_component("Q1", "fake/drive/a", {})
        editor = CfgEditorService(MagicMock(), control, MagicMock(), MagicMock(), bus)
        service = WritebackService(
            editor, write_port=control, entry_ledger=context.entry_ledger
        )
        yield WritebackSession(entry, control, service)
        assert registry_state() == installed
    finally:
        restore_registry(before)


def test_writeback_lowers_waveform_expression_from_creation_capture(
    session: WritebackSession,
) -> None:
    initial = session.control.read_context()
    session.control.set_parameter(initial.target, "general.flux_value", 1.0)
    captured = session.control.read_context()
    spec, value = waveform_cfg_to_value(
        ConstWaveformCfg(length=1.0).model_dump(mode="python")
    )
    value.fields["length"] = EvalValue("general.flux_value")
    draft = session.service.create_draft(
        [
            WaveformWriteback(
                "Q1.waveforms.base",
                "Captured waveform",
                edit_schema=CfgSchema(spec, value),
            )
        ],
        parameters=captured,
    )
    session.control.set_parameter(captured.target, "general.flux_value", 9.0)
    session.service.apply_draft(draft)
    current = session.control.read_context()
    assert current.library.get_waveform("Q1.waveforms.base").length == 1.0
    event = session.entry.ledger.events()[-1]
    assert event.kind == "accepted"
    assert isinstance(event.payload, AcceptedPayload)
    assert event.payload.writes[0].path == "Q1.waveforms.base"
    source = current.library.meta("Q1.waveforms.base.length")
    assert source is not None and source.source == event.id


def test_append_failure_leaves_parameter_untouched(
    session: WritebackSession, monkeypatch
) -> None:
    captured = session.control.read_context()
    draft = session.service.create_draft(
        [MetaDictWriteback("general.flux_value", "Value", proposed_value=2.0)],
        parameters=captured,
    )

    def fail_append(self: RecordsLedger, event: LedgerEvent, **_kwargs: object) -> None:
        raise OSError("ledger unavailable")

    monkeypatch.setattr(RecordsLedger, "append", fail_append)
    with pytest.raises(OSError, match="ledger unavailable"):
        session.service.apply_draft(draft)
    assert not session.control.read_context().lookup("general.flux_value").present
    assert session.entry.ledger.events() == ()


def test_setter_failure_retains_accepted_and_successful_prefix(
    session: WritebackSession, monkeypatch
) -> None:
    captured = session.control.read_context()
    draft = session.service.create_draft(
        [
            MetaDictWriteback("general.flux_value", "First", proposed_value=2.0),
            MetaDictWriteback(
                "general.description", "Failed", proposed_value="changed"
            ),
            MetaDictWriteback("general.flux_unit", "Not started", proposed_value="mA"),
        ],
        parameters=captured,
    )
    original = session.control.set_parameter

    def fail_second(
        target: ContextTarget,
        path: str,
        value: YamlValue,
        *,
        source: Provenance | None = None,
    ) -> None:
        # Accepted evidence must exist before even the first setter.
        assert session.entry.ledger.events()[-1].kind == "accepted"
        if path == "general.description":
            raise OSError("point unavailable")
        original(target, path, value, source=source)

    monkeypatch.setattr(session.control, "set_parameter", fail_second)
    with pytest.raises(OSError, match="point unavailable"):
        session.service.apply_draft(draft, origin=CommandOrigin("mcp", "native-call"))
    current = session.control.read_context()
    assert current.value("general.flux_value") == 2.0
    assert not current.lookup("general.description").present
    assert not current.lookup("general.flux_unit").present
    event = session.entry.ledger.events()[-1]
    assert (
        event.kind == "accepted"
        and event.origin == "mcp"
        and event.call_id == "native-call"
    )
    assert isinstance(event.payload, AcceptedPayload) and len(event.payload.writes) == 3
    source = current.meta("general.flux_value")
    assert source is not None and source.source == event.id


def test_receipt_reads_actual_role_target_after_point_and_focus_change(
    session: WritebackSession,
) -> None:
    session.entry.new_point("a")
    first = session.control.use_point(PointUseRequest(session.entry.entry_id, "a"))
    session.control.add_component("Q1", "fake/drive/a", {"rate": 0.1})
    captured = session.control.read_context().bind_roles(
        {"driver": RoleSpec("fake/drive/*")}, focus="Q1", explicit={}
    )
    draft = session.service.create_draft(
        [MetaDictWriteback("driver.rate", "Rate", proposed_value=0.2, stderr=0.05)],
        parameters=captured,
    )
    session.control.set_parameter(first, "Q1.rate", 0.6)
    session.entry.new_point("b")
    second = session.control.use_point(PointUseRequest(session.entry.entry_id, "b"))
    session.control.add_component("Q2", "fake/drive/a", {"rate": 0.8})
    session.control.set_component_focus("Q2")
    selected = session.control.read_context().document()
    assert session.service.preview_values(draft)["md-1"].current == 0.6
    draft.edit("md-1", selected=False)
    written = session.service.write_draft(
        draft, (WritebackEdit("md-1"),), origin=CommandOrigin("mcp", "native-receipt")
    )
    assert written == [WritebackWritten("md-1", "md", "Q1.rate", 0.6, 0.2)]
    assert captured.value("driver.rate") == 0.1
    assert session.control.read_context().target == second
    assert session.control.read_context().document() == selected
    assert session.control.get_component_focus().name == "Q2"
    assert not draft.items[0].selected
    assert session.service.get_all_applied(draft) == {"md-1": True}
    event = session.entry.ledger.events()[-1]
    assert event.origin == "mcp" and event.call_id == "native-receipt"
    source = session.control.read_target(first).meta("Q1.rate")
    assert source is not None and source.source == event.id and source.stderr == 0.05


@pytest.mark.parametrize("kind", ["module", "waveform"])
def test_library_receipt_reads_actual_raw_entry_and_uses_capture_for_proposal(
    session: WritebackSession,
    kind: str,
) -> None:
    initial = session.control.read_context()
    session.control.set_parameter(initial.target, "general.flux_value", 0.2)
    captured = session.control.read_context()
    if kind == "module":
        cfg = PulseSpec(
            ch=0, nqz=1, freq=100.0, gain=0.2, waveform=ConstWaveformCfg(length=0.1)
        )
        spec, value = module_cfg_to_value(cfg.model_dump(mode="python"))
        value.fields["gain"] = EvalValue("general.flux_value")
        item = ModuleWriteback(
            "Q1.pulses.pi", "Pulse", edit_schema=CfgSchema(spec, value), stderr=0.02
        )
        session.control.set_module(
            captured.target, item.target_name, cfg.model_copy(update={"gain": 0.6})
        )
        item_id, field = "ml-1", "gain"
    else:
        spec, value = waveform_cfg_to_value(
            ConstWaveformCfg(length=0.2).model_dump(mode="python")
        )
        value.fields["length"] = EvalValue("general.flux_value")
        item = WaveformWriteback(
            "Q1.waveforms.base",
            "Waveform",
            edit_schema=CfgSchema(spec, value),
            stderr=0.03,
        )
        session.control.set_waveform(
            captured.target, item.target_name, ConstWaveformCfg(length=0.6)
        )
        item_id, field = "wf-1", "length"
    draft = session.service.create_draft([item], parameters=captured)
    session.control.set_parameter(captured.target, "general.flux_value", 9.0)
    preview = session.service.preview_values(draft)[item_id]
    assert isinstance(preview.current, dict) and preview.current[field] == 0.6
    assert isinstance(preview.proposed, dict) and preview.proposed[field] == 0.2
    receipt = session.service.write_draft(draft, (WritebackEdit(item_id),))[0]
    assert receipt.target == item.target_name
    assert isinstance(receipt.before, dict) and receipt.before[field] == 0.6
    assert isinstance(receipt.after, dict) and receipt.after[field] == 0.2
    source = session.control.read_context().library.meta(f"{item.target_name}.{field}")
    assert source is not None and source.stderr == item.stderr


def test_each_parameter_write_preserves_its_own_stderr(
    session: WritebackSession,
) -> None:
    captured = session.control.read_context()
    draft = session.service.create_draft(
        [
            MetaDictWriteback("Q1.rate", "Rate", proposed_value=0.2, stderr=0.01),
            MetaDictWriteback(
                "Q1.duration", "Duration", proposed_value=3.0, stderr=0.4
            ),
            MetaDictWriteback("Q1.coherence", "Unknown error", proposed_value=5.0),
        ],
        parameters=captured,
    )
    session.service.apply_draft(draft)
    current = session.control.read_context()
    sources = [
        current.meta(path) for path in ("Q1.rate", "Q1.duration", "Q1.coherence")
    ]
    assert all(source is not None for source in sources)
    assert [source.stderr for source in sources if source is not None] == [
        0.01,
        0.4,
        None,
    ]
    assert len({source.source for source in sources if source is not None}) == 1


def test_duplicate_concrete_role_destinations_fail_before_accepted(
    session: WritebackSession,
) -> None:
    captured = session.control.read_context().bind_roles(
        {"driver": RoleSpec("fake/drive/*")}, focus="Q1", explicit={}
    )
    draft = session.service.create_draft(
        [
            MetaDictWriteback("driver.rate", "Role", proposed_value=0.2),
            MetaDictWriteback("Q1.rate", "Direct", proposed_value=0.3),
        ],
        parameters=captured,
    )
    with pytest.raises(InvalidInputError, match="duplicate writeback destination"):
        session.service.apply_draft(draft)
    assert session.entry.ledger.events() == ()
    assert not session.control.read_context().lookup("Q1.rate").present


def test_empty_draft_reads_and_writes_without_context(
    session: WritebackSession,
) -> None:
    draft = session.service.create_draft([], parameters=None)
    assert session.service.preview_values(draft) == {}
    assert session.service.write_draft(draft, ()) == []
    assert session.service.apply_draft(draft)["applied_ids"] == []
    assert session.entry.ledger.events() == ()


@pytest.mark.parametrize("operation", ["preview", "write"])
def test_stale_draft_target_is_not_replaced_by_selected_point(
    session: WritebackSession,
    operation: str,
) -> None:
    captured = session.control.read_context()
    draft = session.service.create_draft(
        [MetaDictWriteback("Q1.rate", "Rate", proposed_value=0.2)],
        parameters=captured,
    )
    session.entry.new_point("a")
    selected = session.control.use_point(PointUseRequest(session.entry.entry_id, "a"))
    events = session.entry.ledger.events()
    perform = (
        partial(session.service.preview_values, draft)
        if operation == "preview"
        else partial(session.service.write_draft, draft, (WritebackEdit("md-1"),))
    )
    with pytest.raises(FailedPreconditionError) as caught:
        perform()
    assert caught.value.reason_code == "draft_not_selected"
    assert session.control.read_context().target == selected
    assert session.entry.ledger.events() == events


def test_preview_proposed_data_is_isolated_at_validation_boundary(
    session: WritebackSession,
) -> None:
    draft = session.service.create_draft(
        [MetaDictWriteback("Q1.ext.samples", "Samples", proposed_value=[1, 2])],
        parameters=session.control.read_context(),
    )
    preview = session.service.preview_values(draft)["md-1"]
    assert preview.current is None and preview.proposed == [1, 2]
    assert isinstance(preview.proposed, list)
    preview.proposed.append(3)
    assert session.service.preview_values(draft)["md-1"].proposed == [1, 2]
