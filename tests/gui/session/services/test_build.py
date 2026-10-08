"""In-process entry binding through the shipped shared composition."""

from pathlib import Path
from unittest.mock import MagicMock

from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryDraft,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.services.build import build_session_services
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import ContextReadiness, SessionEnv
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase


def test_shared_build_routes_entry_and_point_commands_to_real_owners(
    tmp_path: Path,
) -> None:
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    services = build_session_services(
        state=state,
        bus=BaseEventBus(),
        gate=MagicMock(),
        handles=MagicMock(),
        background=MagicMock(),
        progress=MagicMock(),
        runner=MagicMock(),
        bind_assets=ArbWaveformDatabase.info_at,
        project_root=str(tmp_path),
    )

    entry = services.setup_control.new_entry(EntryRequest("entry"))

    assert state.session_env.readiness is ContextReadiness.DRAFT
    assert services.context_control.read_context().target == EntryDraft(
        entry_id=entry.entry_id
    )
    assert services.setup_control.get_entry_info() == entry

    target = services.setup_control.new_point(
        PointCreateRequest(
            entry_id=entry.entry_id,
            label="working",
            description="created through Setup",
            origin=CommandOrigin("gui"),
        )
    )

    assert state.session_env.readiness is ContextReadiness.ACTIVE
    assert services.context_control.read_context().target == target
    assert (
        services.setup_control.list_points()[0].description == "created through Setup"
    )
