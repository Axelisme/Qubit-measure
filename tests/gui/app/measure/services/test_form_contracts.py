"""Published source provenance survives device removal and recreation."""

from zcu_tools.gui.app.measure.services.form_contracts import (
    SourceRevision,
    source_basis,
)
from zcu_tools.gui.app.measure.state import (
    DEVICE_SET_VERSION_KEY,
    DeviceState,
    DeviceStatus,
    State,
)
from zcu_tools.gui.session.types import SessionEnv


def test_source_snapshot_tracks_context_and_device_set_aba_without_live_reads() -> None:
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    first = source_basis(state)
    assert first == (
        SourceRevision("context", 0),
        SourceRevision(DEVICE_SET_VERSION_KEY, 0),
    )

    state.version.bump("context")
    device = DeviceState(
        name="flux",
        type_name="YOKOGS200",
        address="addr",
        status=DeviceStatus.CONNECTED,
        remember=False,
        info=None,
    )
    state.put_device(device)
    second = source_basis(state)
    assert second == (
        SourceRevision("context", 1),
        SourceRevision(DEVICE_SET_VERSION_KEY, 1),
        SourceRevision("device:flux", 1),
    )
    state.remove_device("flux")
    state.put_device(device)
    recreated = source_basis(state)
    assert recreated[-1] == second[-1]
    assert recreated != second
