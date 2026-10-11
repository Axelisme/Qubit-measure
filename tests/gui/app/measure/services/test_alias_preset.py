"""Caller-owned alias presets cross the public measure composition seams."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.app_services import build_app_services
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.types import SessionEnv

from tests.gui._progress_fakes import DirectProgressTransport


@contextmanager
def composed_context(
    seam: str,
    rows: tuple[AliasRow, ...],
    root: Path,
    qapp: QApplication,
) -> Iterator[ContextControlPort]:
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    registry = Registry()
    # Own the background adapter even when construction rejects the preset.
    background = BackgroundRunner()
    from unittest.mock import patch

    try:
        with patch(
            "zcu_tools.gui.app.measure.services.app_services.BackgroundRunner",
            return_value=background,
        ):
            if seam == "controller":
                controller = Controller(
                    state=state,
                    bus=bus,
                    registry=registry,
                    view=None,
                    project_root=str(root),
                    progress_transport=DirectProgressTransport(),
                    owner_scheduler=QtOwnerScheduler(),
                    alias_preset=rows,
                )
                context = controller.context_control
            else:
                services = build_app_services(
                    state=state,
                    bus=bus,
                    registry=registry,
                    cfg_editor_ctrl=MagicMock(),
                    progress_transport=DirectProgressTransport(),
                    resource_versions=state.version.snapshot,
                    render_host=lambda: None,
                    owner_scheduler=QtOwnerScheduler(),
                    project_root=str(root),
                    alias_preset=rows,
                )
                context = services.context_control
        yield context
    finally:
        background.quiesce()
        background.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("seam", ("services", "controller"))
@pytest.mark.parametrize(
    "rows",
    (
        (),
        (
            AliasRow("drive", "Q7"),
            AliasRow("sense", "$drive.resonator"),
            AliasRow("bias", ""),
        ),
    ),
    ids=("empty", "custom"),
)
def test_caller_preset_preserves_order_links_and_unbound_rows(
    seam: str, rows: tuple[AliasRow, ...], tmp_path: Path, qapp: QApplication
) -> None:
    with composed_context(seam, rows, tmp_path, qapp) as context:
        assert context.get_alias_rows() == rows
        context.set_alias_rows((AliasRow("replacement", "R2"),))
        assert context.get_alias_rows() == (AliasRow("replacement", "R2"),)
    # A second composition does not borrow the first session's edited table.
    with composed_context(seam, rows, tmp_path, qapp) as context:
        assert context.get_alias_rows() == rows


@pytest.mark.parametrize("seam", ("services", "controller"))
def test_invalid_caller_preset_fails_at_composition(
    seam: str, tmp_path: Path, qapp: QApplication
) -> None:
    with pytest.raises(InvalidInputError, match="duplicate"):
        with composed_context(
            seam, (AliasRow("drive", "Q1"), AliasRow("drive", "Q2")), tmp_path, qapp
        ):
            pytest.fail("Duplicate caller aliases must not construct a session")
