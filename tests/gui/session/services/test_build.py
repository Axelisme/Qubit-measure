"""In-process entry binding through the shipped shared composition."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.hardware_gate import RunBlocksHardwareGate
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.build import build_session_services
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import ContextReadiness, SessionEnv
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase


@pytest.mark.parametrize("field", ("runner", "progress"))
def test_shared_build_preserves_session_infrastructure_instances(field: str) -> None:
    bus = BaseEventBus()
    gate = RunBlocksHardwareGate(run_kind="run", bus=bus)
    handles = OperationHandles()
    background = MagicMock()
    progress = ProgressService(MagicMock())
    runner = OperationRunner(gate, handles, progress, background, bus)

    services = build_session_services(
        state=SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        bus=bus,
        gate=gate,
        handles=handles,
        background=background,
        progress=progress,
        runner=runner,
        bind_assets=ArbWaveformDatabase.info_at,
    )

    assert getattr(services, field) is {"runner": runner, "progress": progress}[field]


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

    assert state.session_env.readiness is ContextReadiness.EMPTY
    assert state.session_env.parameters is None
    assert services.setup_control.get_entry_info() == entry

    target = services.setup_control.new_point(
        PointCreateRequest(
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
