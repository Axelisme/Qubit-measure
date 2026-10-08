"""ContextControlFacet exposes the current parameter-owner contract."""

import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.value_lookup import MissingValue
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase


@pytest.fixture
def control() -> ContextControlFacet:
    service = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
    )
    return ContextControlFacet(context=service)


def test_empty_owner_reads_and_value_sources(control: ContextControlFacet) -> None:
    assert control.get_entry_info() is None
    assert control.list_points() == ()
    assert control.get_source_binding() is None
    assert control.get_component_focus().name is None
    assert control.list_value_sources() == ()
    with pytest.raises(MissingValue):
        control.read_value_source("device.flux.value", "float")


def test_empty_owner_capture_and_write_fail_explicitly(
    control: ContextControlFacet,
) -> None:
    with pytest.raises(FailedPreconditionError) as capture:
        control.read_context()
    assert capture.value.reason_code == "no_context"
    with pytest.raises(FailedPreconditionError) as write:
        control.set_parameter(
            SavedPoint(entry_id="entry", point="a"), "Q1.rate", 4300.0
        )
    assert write.value.reason_code == "no_entry"
