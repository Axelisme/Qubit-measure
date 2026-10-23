"""Measure controller library entry editing uses the shared Form entry."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import JsonValue
from pydantic_core import PydanticCustomError
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.form import FormWritePath, field_error
from zcu_tools.gui.session.library_entry import LibraryEntryInvalid
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument

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


class _FailingLibraryProvider(LibrarySnapshot):
    """A captured library whose option enumeration fails at the Form boundary."""

    def __init__(self, document: ModuleLibraryDocument, failure: Exception) -> None:
        super().__init__(document)
        self.failure = failure

    def waveform_keys(self) -> tuple[str, ...]:
        raise self.failure


@pytest.fixture()
def library_provider_app(library_app, tmp_path: Path):
    controllers: list[Controller] = []

    def build(failure: Exception) -> Controller:
        source = library_app.read_context()
        parameters = ParameterContext(
            point=source.document(),
            library=_FailingLibraryProvider(source.library.document(), failure),
            target=source.target,
            roles=None,
        )
        controller = Controller(
            State(SessionEnv(parameters=parameters, soc=None, soccfg=None)),
            Registry(),
            None,
            BaseEventBus(),
            project_root=str(tmp_path),
        )
        controllers.append(controller)
        # Opening succeeds; the option provider is consumed by Form.write.
        controller.open_library_entry("module", PULSE)
        return controller

    yield build
    for controller in controllers:
        controller.close()


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("provider defect"),
        PydanticCustomError("provider_bug", "provider defect"),
    ],
    ids=["value_error", "unrelated_custom_error"],
)
def test_library_edit_preserves_unexpected_provider_failure(
    library_provider_app, failure: Exception
) -> None:
    controller = library_provider_app(failure)
    before = controller.read_context().library.document()
    with pytest.raises((ValueError, InvalidInputError)) as caught:
        controller.edit_library_entry("module", PULSE, {("gain",): 0.4})
    assert caught.value is failure
    assert controller.read_context().library.document() == before


def test_library_edit_translates_directed_source_refusal(library_provider_app) -> None:
    failure = field_error(("waveform",), "Missing captured waveform source")
    controller = library_provider_app(failure)
    before = controller.read_context().library.document()
    with pytest.raises(InvalidInputError) as caught:
        controller.edit_library_entry("module", PULSE, {("gain",): 0.4})
    assert caught.value.reason_code == "form_errors"
    assert caught.value.__cause__ is failure
    assert controller.read_context().library.document() == before


@pytest.mark.parametrize(
    "batch",
    [
        {("unknown",): 0.4},
        {
            ("waveform",): {"kind": "ref", "name": WAVEFORM},
            ("waveform", "name"): WAVEFORM,
        },
        {("waveform",): 7},
    ],
    ids=["unknown_path", "overlapping_paths", "invalid_structure"],
)
def test_library_edit_translates_owner_request_errors(
    library_app, batch: Mapping[FormWritePath, JsonValue]
) -> None:
    before = library_app.read_context().library.document()
    with pytest.raises(InvalidInputError) as caught:
        library_app.edit_library_entry("module", PULSE, batch)
    assert caught.value.reason_code == "form_errors"
    assert library_app.read_context().library.document() == before


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
