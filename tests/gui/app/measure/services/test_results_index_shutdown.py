"""The measure application drains its read-only index with no operation handle."""

from pathlib import Path
from threading import Event
from typing import Never

import h5py
import pytest
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import Container, ContainerAttrs, save
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.types import SessionEnv

from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui._qt_waits import qt_until


def _refuse_driver(type_name: str, address: str) -> Never:
    raise AssertionError(f"offline results test cannot construct {type_name}")


def test_close_drains_mid_entry_without_opening_the_rest(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small = tmp_path / "Database/alpha/runs/first"
    small.mkdir(parents=True)
    large = tmp_path / "Database/legacy-Axel.bak/runs"
    fixture = tmp_path / "native.h5"
    save(
        fixture,
        Container(
            {}, {"metadata": {}, "cfg": {}}, {}, ContainerAttrs("3.0", "test", None, {})
        ),
    )
    fixture_bytes = fixture.read_bytes()
    files = [large / f"run-{i:03d}" / "data.h5" for i in range(64)]
    for file in files:
        file.parent.mkdir(parents=True)
        file.write_bytes(fixture_bytes)
    in_flight, release, open_finished = Event(), Event(), Event()
    opened = 0
    open_file = h5py.File

    def hold_middle(path: Path, mode: str) -> h5py.File:
        nonlocal opened
        opened += 1
        if opened != len(files) // 2:
            return open_file(path, mode)
        in_flight.set()
        if not release.wait(timeout=15):
            raise TimeoutError("test never released in-flight native read")
        handle = open_file(path, mode)
        open_finished.set()
        return handle

    # Entry snapshots cannot pause or count opens within an unfinished entry.
    monkeypatch.setattr(h5py, "File", hold_middle)
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        Registry(),
        None,
        BaseEventBus(),
        project_root=str(tmp_path),
        progress_transport=DirectProgressTransport(),
        driver_factory=_refuse_driver,
    )
    try:
        retire = ctrl.watch_results_index(lambda _state: None)
        try:
            qt_until(in_flight.is_set, label="scanner paused midway through one entry")
            assert ctrl.active_operation_count() == 0
            reads_before_close = opened
            assert 1 < reads_before_close < len(files)
            # Only the nested owner-loop drain can deliver this release. It runs
            # after close() has sent the cancellation request.
            QTimer.singleShot(0, release.set)
            ctrl.close()
            assert release.is_set()
            assert open_finished.is_set()
            assert opened == reads_before_close
        finally:
            retire()
    finally:
        release.set()
        ctrl.close()
