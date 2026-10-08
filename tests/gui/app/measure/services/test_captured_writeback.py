"""Accepted evidence and fixed destinations at the real writeback/owner seam."""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg
from zcu_tools.format_version import YamlValue
from zcu_tools.gui.app.measure.adapter import MetaDictWriteback, WaveformWriteback
from zcu_tools.gui.app.measure.cfg_schemas import waveform_cfg_to_value
from zcu_tools.gui.app.measure.services.cfg_editor import CfgEditorService
from zcu_tools.gui.app.measure.services.writeback import WritebackService
from zcu_tools.gui.cfg import CfgSchema, EvalValue
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import CommandOrigin, ContextTarget
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.entry.ledger import RecordsLedger
from zcu_tools.resources.entry.ledger_models import AcceptedPayload, LedgerEvent
from zcu_tools.resources.entry.provenance import Provenance
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase


@dataclass
class WritebackSession:
    entry: ResultEntry
    control: ContextControlFacet
    service: WritebackService


@pytest.fixture
def session(tmp_path: Path) -> WritebackSession:
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "Database"
    )
    bus = BaseEventBus()
    context = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        bus,
        bind_assets=ArbWaveformDatabase.info_at,
    )
    context.install_entry(entry)
    control = ContextControlFacet(context=context)
    editor = CfgEditorService(MagicMock(), control, MagicMock(), MagicMock(), bus)
    service = WritebackService(
        editor, write_port=control, entry_ledger=context.entry_ledger
    )
    return WritebackSession(entry, control, service)


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
    assert session.control.read_context().value("general.flux_value") is None
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
    assert current.value("general.description") is None
    assert current.value("general.flux_unit") is None
    event = session.entry.ledger.events()[-1]
    assert (
        event.kind == "accepted"
        and event.origin == "mcp"
        and event.call_id == "native-call"
    )
    assert isinstance(event.payload, AcceptedPayload) and len(event.payload.writes) == 3
    source = current.meta("general.flux_value")
    assert source is not None and source.source == event.id
