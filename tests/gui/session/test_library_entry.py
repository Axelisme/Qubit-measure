"""Library entries edit through a generated Form at one fixed capture."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from zcu_tools.cfg_model import ArbWaveformCfg, ConstWaveformCfg, PulseSpec
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.events import MlChangedPayload
from zcu_tools.gui.session.library_entry import (
    LibraryEntryInvalid,
    open_library_entry,
)
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import ParameterContext, SessionEnv
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

PULSE = "Q1.pulses.pi"
WAVEFORM = "Q1.waveforms.base"


@dataclass
class LibrarySession:
    entry: ResultEntry
    context: ContextService
    control: ContextControlFacet
    state: SessionState
    bus: BaseEventBus


@pytest.fixture
def session(tmp_path: Path) -> LibrarySession:
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    context = ContextService(state, bus, bind_assets=ArbWaveformDatabase.info_at)
    context.install_entry(entry)
    return LibrarySession(
        entry, context, ContextControlFacet(context=context), state, bus
    )


def _select(session: LibrarySession, label: str) -> None:
    session.entry.new_point(label)
    session.control.use_point(PointUseRequest(label))


def _seed(session: LibrarySession, pulse: PulseSpec | None = None) -> ParameterContext:
    """Select point a with flux 2.0, a const waveform and a pulse using its key."""
    _select(session, "a")
    capture = session.control.read_context()
    session.control.set_parameter(capture.target, "general.ext.flux_value", 2.0)
    session.control.set_waveform(capture.target, WAVEFORM, ConstWaveformCfg(length=1.0))
    session.control.set_module(
        capture.target,
        PULSE,
        pulse or PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.2, waveform=WAVEFORM),
    )
    return session.control.read_context()


def test_numeric_edit_keeps_the_child_waveform_key(session: LibrarySession) -> None:
    capture = _seed(session)
    entry = open_library_entry(capture, "module", PULSE, arb_names=())
    assert entry.state.input.data["waveform"] == WAVEFORM
    entry.state.write({("gain",): 0.5}, entry.env)
    session.control.apply_library_change(entry.parameters, entry.put(PULSE))
    written = session.control.read_context().library.module_spec(PULSE)
    assert isinstance(written, PulseSpec)
    assert written.waveform == WAVEFORM
    assert written.gain == 0.5


def test_expression_stores_the_number_from_the_open_capture(
    session: LibrarySession,
) -> None:
    capture = _seed(session)
    entry = open_library_entry(capture, "module", PULSE, arb_names=())
    entry.state.write({("gain",): {"expr": "general.ext.flux_value / 10"}}, entry.env)
    session.control.set_parameter(capture.target, "general.ext.flux_value", 8.0)
    session.control.apply_library_change(entry.parameters, entry.put(PULSE))
    session.control.set_parameter(capture.target, "general.ext.flux_value", 9.0)
    written = session.control.read_context().library.module_spec(PULSE)
    assert isinstance(written, PulseSpec)
    assert written.gain == pytest.approx(0.2)


def test_root_tag_is_fixed_to_the_entry_family(session: LibrarySession) -> None:
    capture = _seed(session)
    entry = open_library_entry(capture, "module", PULSE, arb_names=())
    tag = entry.state.input_schema(entry.env)["properties"]["type"]
    assert isinstance(tag, dict) and tag["const"] == "pulse"
    entry.state.write({("type",): "readout/pulse"}, entry.env)
    with pytest.raises(LibraryEntryInvalid) as error:
        entry.put(PULSE)
    assert ("type",) in error.value.errors.messages
    assert (
        session.control.read_context().library.document() == capture.library.document()
    )


def test_invalid_form_rejects_submission_with_field_paths(
    session: LibrarySession,
) -> None:
    capture = _seed(session)
    entry = open_library_entry(capture, "module", PULSE, arb_names=())
    entry.state.write({("gain",): "bad"}, entry.env)
    with pytest.raises(LibraryEntryInvalid, match="gain") as error:
        entry.replace(PULSE, "Q1.pulses.renamed")
    assert isinstance(error.value, InvalidInputError)
    assert set(error.value.errors.messages) == {("gain",)}
    assert (
        session.control.read_context().library.document() == capture.library.document()
    )
    assert (
        session.entry.module_library("a").snapshot().document()
        == capture.library.document()
    )


def test_replace_renames_and_writes_in_one_library_change(
    session: LibrarySession,
) -> None:
    capture = _seed(session)
    entry = open_library_entry(capture, "module", PULSE, arb_names=())
    entry.state.write({("gain",): 0.3}, entry.env)
    version = session.state.version.get("context")
    events: list[MlChangedPayload] = []
    session.bus.subscribe(MlChangedPayload, events.append)
    session.control.apply_library_change(
        entry.parameters, entry.replace(PULSE, "Q1.pulses.renamed")
    )
    library = session.control.read_context().library
    assert library.module_keys() == ("Q1.pulses.renamed",)
    written = library.module_spec("Q1.pulses.renamed")
    assert isinstance(written, PulseSpec)
    assert (written.gain, written.waveform) == (0.3, WAVEFORM)
    assert session.state.version.get("context") == version + 1
    assert len(events) == 1


def test_replace_onto_an_existing_name_is_rejected(session: LibrarySession) -> None:
    capture = _seed(session)
    session.control.set_module(
        capture.target,
        "Q1.pulses.occupied",
        PulseSpec(ch=1, nqz=1, freq=50.0, gain=0.1, waveform=WAVEFORM),
    )
    before = session.control.read_context().library.document()
    entry = open_library_entry(capture, "module", PULSE, arb_names=())
    entry.state.write({("gain",): 0.3}, entry.env)
    with pytest.raises(ValueError, match="already exists"):
        session.control.apply_library_change(
            entry.parameters, entry.replace(PULSE, "Q1.pulses.occupied")
        )
    assert session.control.read_context().library.document() == before


def test_arb_data_options_are_the_names_captured_at_open(
    session: LibrarySession,
) -> None:
    _select(session, "a")
    capture = session.control.read_context()
    session.control.set_waveform(
        capture.target, "Q1.waveforms.arb", ArbWaveformCfg(data="first")
    )
    capture = session.control.read_context()
    entry = open_library_entry(
        capture, "waveform", "Q1.waveforms.arb", arb_names=("first", "second")
    )
    data = entry.state.input_schema(entry.env)["properties"]["data"]
    assert isinstance(data, dict) and data["enum"] == ["first", "second"]
    entry.state.write({("data",): "third"}, entry.env)
    with pytest.raises(LibraryEntryInvalid) as error:
        entry.put("Q1.waveforms.arb")
    assert set(error.value.errors.messages) == {("data",)}
    entry.state.write({("data",): "second"}, entry.env)
    session.control.apply_library_change(
        entry.parameters, entry.put("Q1.waveforms.arb")
    )
    written = session.control.read_context().library.waveform_spec("Q1.waveforms.arb")
    assert isinstance(written, ArbWaveformCfg) and written.data == "second"


def test_inline_arb_data_in_a_pulse_uses_the_captured_names(
    session: LibrarySession,
) -> None:
    capture = _seed(
        session,
        PulseSpec(
            ch=0, nqz=1, freq=100.0, gain=0.2, waveform=ArbWaveformCfg(data="first")
        ),
    )
    entry = open_library_entry(capture, "module", PULSE, arb_names=("first",))
    entry.state.write({("waveform", "data"): "missing"}, entry.env)
    with pytest.raises(LibraryEntryInvalid) as error:
        entry.put(PULSE)
    assert set(error.value.errors.messages) == {("waveform", "data")}


def test_unsupported_fields_return_to_spec_defaults(session: LibrarySession) -> None:
    capture = _seed(
        session,
        PulseSpec(
            ch=0, nqz=1, freq=100.0, gain=0.2, waveform=WAVEFORM, desc="stored note"
        ),
    )
    entry = open_library_entry(capture, "module", PULSE, arb_names=())
    session.control.apply_library_change(entry.parameters, entry.put(PULSE))
    written = session.control.read_context().library.module_spec(PULSE)
    assert isinstance(written, PulseSpec) and written.desc is None


def test_submission_after_a_point_switch_writes_the_open_target(
    session: LibrarySession,
) -> None:
    first = _seed(session)
    entry = open_library_entry(first, "waveform", WAVEFORM, arb_names=())
    entry.state.write({("length",): 3.0}, entry.env)
    _select(session, "b")
    session.control.apply_library_change(
        entry.parameters, entry.put("Q1.waveforms.copy")
    )
    original = session.control.read_target(first.target).library
    assert original.waveform_spec("Q1.waveforms.copy").length == 3.0
    assert session.control.read_context().library.waveform_keys() == ()


def test_submission_to_a_different_open_entry_writes_nothing(
    session: LibrarySession, tmp_path: Path
) -> None:
    first = _seed(session)
    entry = open_library_entry(first, "waveform", WAVEFORM, arb_names=())
    other = ResultEntry.create(
        "other", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    session.context.install_entry(other)
    with pytest.raises(FailedPreconditionError) as error:
        session.control.apply_library_change(
            entry.parameters, entry.put("Q1.waveforms.copy")
        )
    assert error.value.reason_code == "entry_not_open"
    assert (
        session.entry.module_library("a").snapshot().document()
        == first.library.document()
    )


def test_opening_an_unknown_entry_is_rejected(session: LibrarySession) -> None:
    capture = _seed(session)
    with pytest.raises(InvalidInputError, match="Q1.pulses.missing"):
        open_library_entry(capture, "module", "Q1.pulses.missing", arb_names=())
