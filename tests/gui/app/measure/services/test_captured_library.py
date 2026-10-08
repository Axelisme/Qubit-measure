"""Library editors retain captured targets and resource atomicity."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.app.measure.services.cfg_editor import (
    CfgEditorError,
    CfgEditorService,
)
from zcu_tools.gui.app.measure.services.cfg_lowering import lower_module, lower_waveform
from zcu_tools.gui.app.measure.services.ports import CfgEdit
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import LibraryChange, PointUseRequest
from zcu_tools.gui.session.events import MlChangedPayload
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import ParameterContext, RoleBinding, SessionEnv
from zcu_tools.gui.session.value_lookup import ScalarValue, ValueInfo
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase


class EditorHost:
    """Real control commands with inert device/options and version collaborators."""

    def __init__(self, control: ContextControlFacet) -> None:
        self.control = control
        self.arb_waveforms = self
        self.versions: dict[str, int] = {}

    def read_context(self, *, roles: RoleBinding | None = None) -> ParameterContext:
        return self.control.read_context(roles=roles)

    def apply_library_change(
        self, parameters: ParameterContext, change: LibraryChange
    ) -> None:
        self.control.apply_library_change(
            parameters, change, lower_module=lower_module, lower_waveform=lower_waveform
        )

    def list_device_names(self) -> list[str]:
        return []

    def list_data_keys(self) -> list[str]:
        return []

    def read_value_source(
        self, key: str, type_name: str | None = None
    ) -> tuple[ValueInfo, ScalarValue]:
        return self.control.read_value_source(key, type_name)

    def bump_editor_version(self, editor_id: str) -> None:
        self.versions[editor_id] = self.versions.get(editor_id, 0) + 1

    def drop_editor_version(self, editor_id: str) -> None:
        self.versions.pop(editor_id, None)


@dataclass
class LibrarySession:
    entry: ResultEntry
    context: ContextService
    control: ContextControlFacet
    state: SessionState
    bus: BaseEventBus
    editor: CfgEditorService
    host: EditorHost


@pytest.fixture
def session(tmp_path: Path) -> LibrarySession:
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    context = ContextService(state, bus, bind_assets=ArbWaveformDatabase.info_at)
    context.install_entry(entry)
    control = ContextControlFacet(context=context)
    host = EditorHost(control)
    editor = CfgEditorService(host, control, host, host, bus)
    return LibrarySession(entry, context, control, state, bus, editor, host)


def _select(session: LibrarySession, name: str) -> None:
    session.entry.new_point(name)
    session.control.use_point(PointUseRequest(session.entry.entry_id, name))


def _seed(session: LibrarySession, *, saved: bool) -> ParameterContext:
    if saved:
        _select(session, "a")
    capture = session.control.read_context()
    session.control.set_parameter(capture.target, "general.flux_value", 2.0)
    session.control.set_waveform(
        capture.target, "Q1.waveforms.base", ConstWaveformCfg(length=1.0)
    )
    return session.control.read_context()


def test_editor_commit_after_point_switch_uses_original_target_and_values(
    session: LibrarySession,
) -> None:
    first = _seed(session, saved=True)
    editor_id, _ = session.editor.open("waveform", from_name="Q1.waveforms.base")
    session.editor.set_field(
        editor_id, "length", {"__kind": "eval", "expr": "general.flux_value"}
    )
    _select(session, "b")
    second = session.control.read_context()
    session.control.set_parameter(second.target, "general.flux_value", 9.0)
    version = session.state.version.get("context")
    events: list[MlChangedPayload] = []
    session.bus.subscribe(MlChangedPayload, events.append)
    session.editor.commit(editor_id, "Q1.waveforms.copy")
    original = session.control.read_target(first.target)
    assert original.library.waveform_spec("Q1.waveforms.copy").length == 2.0
    assert session.control.read_context().target == second.target
    assert session.control.read_context().library.waveform_keys() == ()
    assert session.state.version.get("context") == version
    assert events == []
    with pytest.raises(CfgEditorError, match="unknown editor"):
        session.editor.get_draft(editor_id)


@pytest.mark.parametrize("saved", [False, True], ids=["draft", "saved"])
def test_replace_relocates_and_publishes_one_complete_library(
    session: LibrarySession, saved: bool
) -> None:
    capture = _seed(session, saved=saved)
    editor_id, _ = session.editor.open("waveform", from_name="Q1.waveforms.base")
    session.editor.set_field(editor_id, "length", 3.0)
    version = session.state.version.get("context")
    events: list[MlChangedPayload] = []
    session.bus.subscribe(MlChangedPayload, events.append)
    session.editor.replace(editor_id, "Q1.waveforms.base", "Q1.waveforms.new")
    current = session.control.read_context()
    assert current.library.waveform_keys() == ("Q1.waveforms.new",)
    assert current.library.waveform_spec("Q1.waveforms.new").length == 3.0
    source = current.library.meta("Q1.waveforms.new.length")
    assert source is not None and source.source == "manual"
    assert capture.library.waveform_spec("Q1.waveforms.base").length == 1.0
    assert session.state.version.get("context") == version + 1
    assert len(events) == 1
    if saved:
        assert (
            session.entry.module_library("a").snapshot().document()
            == current.library.document()
        )


@pytest.mark.parametrize("saved", [False, True], ids=["draft", "saved"])
def test_replace_collision_retains_draft_and_entire_library(
    session: LibrarySession, saved: bool
) -> None:
    capture = _seed(session, saved=saved)
    session.control.set_waveform(
        capture.target, "Q1.waveforms.occupied", ConstWaveformCfg(length=8.0)
    )
    editor_id, _ = session.editor.open("waveform", from_name="Q1.waveforms.base")
    session.editor.set_field(editor_id, "length", 3.0)
    before = session.control.read_context()
    version = session.state.version.get("context")
    with pytest.raises(ValueError, match="already exists"):
        session.editor.replace(editor_id, "Q1.waveforms.base", "Q1.waveforms.occupied")
    assert session.editor.get_draft(editor_id).is_valid()
    assert (
        session.control.read_context().library.document() == before.library.document()
    )
    assert session.state.version.get("context") == version
    if saved:
        assert (
            session.entry.module_library("a").snapshot().document()
            == before.library.document()
        )


@pytest.mark.parametrize("saved", [False, True], ids=["draft", "saved"])
def test_invalid_lowering_leaves_original_library_and_editor(
    session: LibrarySession, saved: bool
) -> None:
    _seed(session, saved=saved)
    editor_id, _ = session.editor.open("waveform", from_name="Q1.waveforms.base")
    session.editor.set_field(editor_id, "length", {"__kind": "eval", "expr": "missing"})
    before = session.control.read_context()
    with pytest.raises(InvalidInputError, match="missing"):
        session.editor.replace(editor_id, "Q1.waveforms.base", "Q1.waveforms.new")
    assert (
        session.control.read_context().library.document() == before.library.document()
    )
    session.editor.get_draft(editor_id)
    if saved:
        assert (
            session.entry.module_library("a").snapshot().document()
            == before.library.document()
        )


def test_memory_editor_rejects_selection_of_saved_point_and_remains_editable(
    session: LibrarySession,
) -> None:
    _seed(session, saved=False)
    editor_id, _ = session.editor.open("waveform", from_name="Q1.waveforms.base")
    _select(session, "a")
    before = session.control.read_context()
    with pytest.raises(FailedPreconditionError) as error:
        session.editor.commit(editor_id, "Q1.waveforms.new")
    assert error.value.reason_code == "draft_not_selected"
    assert (
        session.control.read_context().library.document() == before.library.document()
    )
    session.editor.set_field(editor_id, "length", 4.0)


def test_editor_rejects_different_open_entry_without_writing_either(
    session: LibrarySession, tmp_path: Path
) -> None:
    original = _seed(session, saved=True)
    editor_id, _ = session.editor.open("waveform", from_name="Q1.waveforms.base")
    other = ResultEntry.create(
        "other", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    session.context.install_entry(other)
    with pytest.raises(FailedPreconditionError) as error:
        session.editor.commit(editor_id, "Q1.waveforms.new")
    assert error.value.reason_code == "entry_not_open"
    assert session.control.read_context().library.waveform_keys() == ()
    assert (
        session.entry.module_library("a").snapshot().document()
        == original.library.document()
    )
    session.editor.get_draft(editor_id)


@pytest.mark.parametrize("kind", ["module", "waveform"])
def test_detached_rename_then_delete_preserves_sources_without_materializing(
    session: LibrarySession, kind: Literal["module", "waveform"]
) -> None:
    _select(session, "a")
    capture = session.control.read_context()
    if kind == "module":
        old, new = "Q1.pulses.pi", "Q1.pulses.other"
        session.control.set_module(
            capture.target,
            old,
            PulseSpec(
                ch=0, nqz=1, freq=100.0, gain=0.2, waveform="Q1.waveforms.missing"
            ),
        )
    else:
        old, new = "Q1.waveforms.base", "Q1.waveforms.other"
        session.control.set_waveform(capture.target, old, ConstWaveformCfg(length=1.0))
    before = session.control.read_context().library
    leaf = ".gain" if kind == "module" else ".length"
    source = before.meta(old + leaf)
    _select(session, "b")
    version = session.state.version.get("context")
    session.host.apply_library_change(capture, LibraryChange(kind, "rename", old, new))
    relocated = session.control.read_target(capture.target).library
    assert relocated.entry_value(new, kind=kind) == before.entry_value(old, kind=kind)
    assert relocated.meta(new + leaf) == source
    session.host.apply_library_change(capture, LibraryChange(kind, "delete", new))
    deleted = session.control.read_target(capture.target).library
    assert (
        deleted.module_keys() if kind == "module" else deleted.waveform_keys()
    ) == ()
    assert session.state.version.get("context") == version


def test_module_editor_lowers_values_and_children_from_capture(
    session: LibrarySession,
) -> None:
    capture = _seed(session, saved=True)
    session.control.set_module(
        capture.target,
        "Q1.pulses.pi",
        PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.2, waveform="Q1.waveforms.base"),
    )
    editor_id, _ = session.editor.open("module", from_name="Q1.pulses.pi")
    session.editor.set_field(
        editor_id, "gain", {"__kind": "eval", "expr": "general.flux_value / 10"}
    )
    session.control.set_parameter(capture.target, "general.flux_value", 8.0)
    session.control.set_waveform(
        capture.target, "Q1.waveforms.base", ConstWaveformCfg(length=8.0)
    )
    session.editor.replace(editor_id, "Q1.pulses.pi", "Q1.pulses.new")
    written = session.control.read_context().library.module_spec("Q1.pulses.new")
    assert isinstance(written, PulseSpec)
    assert written.gain == 0.2
    assert isinstance(written.waveform, ConstWaveformCfg)
    assert written.waveform.length == 1.0


def test_ordered_library_edit_keeps_successful_prefix(session: LibrarySession) -> None:
    _seed(session, saved=True)
    result = session.editor.edit_library(
        "waveform",
        "Q1.waveforms.base",
        [CfgEdit("length", 3.0), CfgEdit("unknown", 5.0)],
    )
    assert result.applied == 1 and not result.valid
    assert (
        session.control.read_context().library.waveform_spec("Q1.waveforms.base").length
        == 3.0
    )
