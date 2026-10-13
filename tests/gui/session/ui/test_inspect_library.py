"""Inspect library buttons persist structural changes through the real owner."""

import sys
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QInputDialog, QMessageBox, QPushButton, QTreeWidget
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.inspect_dialog import InspectDialog
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase


@dataclass
class _Inspector:
    dialog: InspectDialog
    control: ContextControlFacet
    entry: ResultEntry


@pytest.fixture
def inspector(qapp, tmp_path: Path) -> Generator[_Inspector]:
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "Database"
    )
    entry.new_point("point")
    library = entry.module_library("point")
    with library.edit() as draft:
        draft.set_waveform("Q1.waveforms.base", ConstWaveformCfg(length=0.1))
        draft.set_module(
            "Q1.pulses.pi",
            PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.25, waveform="Q1.waveforms.base"),
        )
    bus = BaseEventBus()
    service = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        bus,
        bind_assets=ArbWaveformDatabase.info_at,
    )
    service.install_entry(entry)
    control = ContextControlFacet(context=service)
    control.use_point(PointUseRequest("point"))
    dialog = InspectDialog(control, bus)
    try:
        yield _Inspector(dialog, control, entry)
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("kind", ["module", "waveform"])
@pytest.mark.parametrize("action", ["rename", "delete"])
def test_library_button_persists_selected_entry(
    inspector: _Inspector,
    kind: Literal["module", "waveform"],
    action: str,
    monkeypatch,
) -> None:
    dialog, control, entry = inspector.dialog, inspector.control, inspector.entry
    original = control.read_context()
    source = "Q1.pulses.pi" if kind == "module" else "Q1.waveforms.base"
    destination = "Q1.pulses.renamed" if kind == "module" else "Q1.waveforms.renamed"
    group = "modules" if kind == "module" else "waveforms"
    tree = next(
        tree
        for tree in dialog.findChildren(QTreeWidget)
        if tree.findItems(group, Qt.MatchFlag.MatchExactly)
    )
    tree.setCurrentItem(
        tree.findItems(source, Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive)[
            0
        ]
    )
    errors: list[str] = []
    monkeypatch.setattr(QInputDialog, "getText", lambda *_a, **_k: (destination, True))
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Yes
    )
    monkeypatch.setattr(
        QMessageBox, "critical", lambda _parent, _title, message: errors.append(message)
    )
    text = "Rename..." if action == "rename" else "Delete"
    button = next(
        button for button in dialog.findChildren(QPushButton) if button.text() == text
    )
    assert button.isEnabled()
    exceptions: list[BaseException] = []
    monkeypatch.setattr(
        sys, "excepthook", lambda _type, error, _trace: exceptions.append(error)
    )
    button.click()
    assert errors == [], f"Inspect reported a library command failure: {errors}"
    assert exceptions == [], f"Inspect library command raised: {exceptions}"
    # A fresh resource handle reads the persisted file, not the session projection.
    persisted = entry.module_library("point").snapshot()
    keys = persisted.module_keys() if kind == "module" else persisted.waveform_keys()
    assert keys == ((destination,) if action == "rename" else ())
    if action == "rename":
        assert persisted.entry_state(
            destination, kind=kind
        ) == original.library.entry_state(source, kind=kind)
    assert original.library.entry_value(source, kind=kind)
    assert control.read_context().library.document() == persisted.document()
