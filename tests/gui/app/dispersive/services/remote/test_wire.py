"""Dispersive remote wire and GUI revision constants."""

from zcu_tools.gui.app.dispersive.services.remote.wire_version import (
    GUI_VERSION,
    WIRE_VERSION,
)


def test_versions_preserve_wire_contract_and_track_lazy_push_revision() -> None:
    assert WIRE_VERSION == 4
    assert GUI_VERSION == 6
