"""Measure controller library entry editing uses the shared Form entry."""

from pathlib import Path

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.session.library_entry import LibraryEntryInvalid

from tests.gui._qt_waits import qt_until
from tests.gui.app.measure.remote._helpers import Fixture
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry

PULSE = "Q1.pulses.pi"
WAVEFORM = "Q1.waveforms.seed"


@pytest.fixture()
def library_app(qapp, tmp_path: Path):
    before = registry_state()
    fx = None
    try:
        register_fakes()
        installed = registry_state()
        fx = Fixture(active_label="ctx001", project_root=str(tmp_path))
        control = fx.ctrl.context_control
        control.add_component("Q1", "fake/drive/a", {})
        parameters = control.read_context()
        control.set_waveform(parameters.target, WAVEFORM, ConstWaveformCfg(length=0.1))
        control.set_module(
            parameters.target,
            PULSE,
            PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.2, waveform=WAVEFORM),
        )
        yield fx.ctrl
        assert registry_state() == installed
    finally:
        try:
            if fx is not None:
                closed = False

                def on_closed() -> None:
                    nonlocal closed
                    closed = True

                fx.ctrl.begin_shutdown(on_closed)
                # begin_shutdown() schedules the terminal owner notification.
                qt_until(lambda: closed, label="library fixture shutdown")
                fx.stop()
                qapp.processEvents()
        finally:
            restore_registry(before)


def test_replace_keeps_the_child_key_and_renames(library_app) -> None:
    ctrl = library_app
    entry = ctrl.open_library_entry("module", PULSE)
    entry.state.write({("gain",): 0.4}, entry.env)
    ctrl.replace_library_entry(entry, PULSE, "Q1.pulses.renamed")
    library = ctrl.read_context().library
    assert library.module_keys() == ("Q1.pulses.renamed",)
    written = library.module_spec("Q1.pulses.renamed")
    assert isinstance(written, PulseSpec)
    assert (written.gain, written.waveform) == (0.4, WAVEFORM)


def test_commit_writes_a_new_destination_and_keeps_the_source(library_app) -> None:
    ctrl = library_app
    entry = ctrl.open_library_entry("waveform", WAVEFORM)
    entry.state.write({("length",): 0.25}, entry.env)
    ctrl.commit_library_entry(entry, "Q1.waveforms.copy")
    library = ctrl.read_context().library
    assert library.waveform_spec("Q1.waveforms.copy").length == 0.25
    assert library.waveform_spec(WAVEFORM).length == 0.1


def test_invalid_entry_is_not_committed(library_app) -> None:
    ctrl = library_app
    before = ctrl.read_context().library.document()
    entry = ctrl.open_library_entry("module", PULSE)
    entry.state.write({("freq",): "bad"}, entry.env)
    with pytest.raises(LibraryEntryInvalid, match="freq"):
        ctrl.commit_library_entry(entry, PULSE)
    assert ctrl.read_context().library.document() == before
