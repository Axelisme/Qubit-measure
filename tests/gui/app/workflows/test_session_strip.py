"""Session chips above the workflow columns follow the shared session.

Chips refresh on session events, when Setup or Aliases close, and on Prepare
forms. Their text is read from the shown QLabel and its tooltip.
"""

from dataclasses import replace

import pytest
from qtpy.QtWidgets import QApplication, QLabel, QPushButton, QWidget
from zcu_tools.device.fake import FakeDevice
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.services.device import ConnectDeviceRequest
from zcu_tools.gui.session.ui.alias_dialog import AliasDialog
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.sim import DEFAULT_SIMPARAM

from ._window_support import QUBIT_KIND, WindowSession, bind_point, install_soc

CHIPS = (
    "workflow.session.entry",
    "workflow.source.status",
    "workflow.session.devices",
    "workflow.session.zcu",
)


def chip(window: QWidget, name: str) -> QLabel:
    found = window.findChild(QLabel, name)
    assert found is not None, name
    return found


def chip_texts(window: QWidget) -> tuple[str, ...]:
    return tuple(chip(window, name).text() for name in CHIPS)


def test_empty_session_shows_every_chip_without_context(
    window_session: WindowSession,
) -> None:
    window = window_session.window
    strip = window.findChild(QWidget, "workflow.session")
    assert strip is not None
    for name in CHIPS:
        assert strip.isAncestorOf(chip(window, name))
    assert chip_texts(window) == (
        "Entry: none",
        "Qubit: no context",
        "Devices: none",
        "ZCU: not connected",
    )


def test_context_events_refresh_entry_point_and_resolved_qubit(
    window_session: WindowSession,
) -> None:
    window = window_session.window
    bind_point(window_session.harness, window_session.root)
    assert chip(window, "workflow.session.entry").text() == "Entry: sample / source"
    assert chip(window, "workflow.source.status").text() == (
        f"Qubit: Q1 ({QUBIT_KIND})"
    )


@pytest.mark.parametrize(
    "rows", [(), (AliasRow("qubit", "Q9"),)], ids=["unbound", "missing"]
)
def test_unavailable_qubit_shows_a_reason_instead_of_a_path(
    window_session: WindowSession, rows: tuple[AliasRow, ...]
) -> None:
    window = window_session.window
    bind_point(window_session.harness, window_session.root)
    window_session.harness.services.context.set_alias_rows(rows)
    text = chip(window, "workflow.source.status").text()
    assert text.startswith("Qubit unavailable: ")
    assert len(text) > len("Qubit unavailable: ")
    assert ".kind" not in text


def test_long_reason_is_elided_without_widening_the_window(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = window_session.window
    bind_point(window_session.harness, window_session.root)
    missing = "Q" + "9" * 300
    window_session.harness.services.context.set_alias_rows(
        (AliasRow("qubit", missing),)
    )
    label = chip(window, "workflow.source.status")
    assert label.text().startswith("Qubit unavailable: ")
    assert missing not in label.text()
    assert missing in label.toolTip()
    window.resize(1280, 800)
    window.show()
    qapp.processEvents()
    assert window.minimumSizeHint().width() <= 1280
    assert label.width() < 1280 // 2


def test_device_and_simulator_events_refresh_their_chips(
    window_session: WindowSession,
) -> None:
    window, harness = window_session.window, window_session.harness
    harness.services.device.start_connect_device(
        ConnectDeviceRequest("FakeDevice", "flux", "none")
    )
    harness.background.execute()
    harness.background.deliver()
    assert chip(window, "workflow.session.devices").text() == "Devices: flux"
    driver = harness.services.device.get_connected_devices()["flux"]
    assert isinstance(driver, FakeDevice)
    soc, soccfg = make_mock_soc(sim=DEFAULT_SIMPARAM)
    soc.set_flux_source(driver.get_value)
    harness.services.soc_connection.install_prepared_mock(soc, soccfg)
    assert chip(window, "workflow.session.zcu").text() == "ZCU: simulator"


def test_setup_aliases_close_and_prepare_refresh_without_events(
    window_session: WindowSession, qapp: QApplication
) -> None:
    """A caller-owned state write publishes no event; the next refresh shows it."""
    window, harness = window_session.window, window_session.harness
    bind_point(harness, window_session.root)
    zcu = chip(window, "workflow.session.zcu")

    install_soc(harness)
    assert zcu.text() == "ZCU: not connected"
    window.setup_dialog.reject()
    qapp.processEvents()
    assert zcu.text() == "ZCU: connected"

    harness.state.set_context(replace(harness.state.session_env, soc=None))
    aliases = window.findChild(QPushButton, "workflow.source.aliases")
    assert aliases is not None
    aliases.click()
    dialog = next(d for d in window.findChildren(AliasDialog) if d.isVisible())
    assert zcu.text() == "ZCU: connected"
    dialog.reject()
    qapp.processEvents()
    assert zcu.text() == "ZCU: not connected"

    install_soc(harness)
    prepare = window.findChild(QPushButton, "workflow.source.use")
    assert prepare is not None
    prepare.click()
    assert zcu.text() == "ZCU: connected"


def test_closed_window_stops_following_session_events(
    window_session: WindowSession,
    qapp: QApplication,
    caplog: pytest.LogCaptureFixture,
) -> None:
    window = window_session.window
    bind_point(window_session.harness, window_session.root)
    before = chip(window, "workflow.source.status").text()
    assert window.close()
    qapp.processEvents()
    window_session.harness.services.context.set_alias_rows(())
    assert chip(window, "workflow.source.status").text() == before
    assert "raised" not in caplog.text
