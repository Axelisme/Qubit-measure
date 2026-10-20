"""Fixed context/storage through Context, Tab, synchronous Save and RunInputs."""

from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.cfg_model import ConfigBase, ConstWaveformCfg, PulseSpec
from zcu_tools.gui.app.measure.artifacts import SaveControlFacet, SaveService
from zcu_tools.gui.app.measure.services.guard import GuardService
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import (
    ContextSourceFault,
    ContextSourceUnavailableError,
    PointUseRequest,
)
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry, RoleSpec
from zcu_tools.resources.entry.save_layout import ArtifactKey, SaveLayout
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui.app.measure._form_fakes import form_resources, form_sources
from tests.gui.app.measure._native_save_support import GUI_ORIGIN, native_save_session
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


@dataclass
class SaveSession:
    entry: ResultEntry
    context: ContextService
    control: ContextControlFacet
    state: State
    bus: BaseEventBus


@pytest.fixture
def session(tmp_path: Path) -> Generator[SaveSession]:
    before = registry_state()
    try:
        register_fakes()
        installed = registry_state()
        entry = ResultEntry.create(
            "entry-a",
            result_root=tmp_path / "Result",
            database_root=tmp_path / "Database",
        )
        entry.setup.add_component("Q1", kind="fake/drive/a")
        entry.setup.add_component("Q2", kind="fake/drive/a")
        entry.new_point("p")
        state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
        bus = BaseEventBus()
        context = ContextService(state, bus, bind_assets=ArbWaveformDatabase.info_at)
        context.install_entry(entry)
        control = ContextControlFacet(context=context)
        control.use_point(PointUseRequest("p"))
        yield SaveSession(entry, context, control, state, bus)
        assert registry_state() == installed
    finally:
        restore_registry(before)


def _bound(session: SaveSession) -> ParameterContext:
    return session.control.read_context().bind_roles(
        {"driver": RoleSpec("fake/drive/*")}, focus="Q1", explicit={}
    )


def test_storage_retains_entry_point_description_and_roles_after_switch(
    session: SaveSession, tmp_path: Path
) -> None:
    target = session.control.read_context().target
    session.control.set_parameter(target, "general.description", "original")
    capture = _bound(session)
    session.control.set_parameter(target, "general.description", "edited")
    session.control.set_alias_rows((AliasRow("driver", "Q2"),))
    session.entry.new_point("q")
    session.control.use_point(PointUseRequest("q"))
    other = ResultEntry.create(
        "entry-b", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    session.context.install_entry(other)

    storage = session.control.capture_storage(capture)
    assert str(storage.entry_id) == session.entry.entry_id
    assert storage.entry_name == "entry-a"
    assert storage.point == "p"
    assert storage.description == "original"
    assert dict(storage.roles) == {"driver": "Q1"}
    assert storage.result_path == session.entry.result_path
    assert storage.database_path == session.entry.database_path
    layout = SaveLayout(
        result_path=storage.result_path,
        database_path=storage.database_path,
        run_id="run-a",
        experiment="synthetic",
        point=storage.point,
        saved_at=datetime(2026, 10, 8, tzinfo=timezone.utc),
    )
    outputs = layout.outputs(ArtifactKey("run", "T1", "data"))
    assert (
        outputs[0].path == session.entry.database_path / "runs/synthetic_run-a/data.h5"
    )
    assert storage.ledger is capture.capture_storage().ledger


def test_current_storage_description_updates_without_rewriting_old_capture(
    session: SaveSession,
) -> None:
    old = session.control.read_context()
    session.control.set_parameter(old.target, "general.description", "new")
    fresh = session.control.read_context()
    assert session.control.capture_storage(fresh).description == "new"
    assert session.control.capture_storage(old).description != "new"
    assert (
        session.control.capture_storage(
            session.control.read_target(old.target)
        ).description
        == "new"
    )


@pytest.mark.parametrize("kind", ["module", "waveform"])
def test_library_publication_preserves_storage_origin(
    session: SaveSession, kind: str
) -> None:
    original = session.control.read_context()
    if kind == "module":
        session.control.set_module(
            original.target,
            "Q1.pulses.pi",
            PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.2, waveform="Q1.waveforms.base"),
        )
    else:
        session.control.set_waveform(
            original.target, "Q1.waveforms.base", ConstWaveformCfg(length=1.0)
        )
    storage = session.control.capture_storage(session.control.read_context())
    assert storage == session.control.capture_storage(original)


def test_storage_requires_saved_origin_and_readable_sources(
    session: SaveSession,
) -> None:
    saved = session.control.read_context()
    fault = ContextSourceFault(
        "point", Path("/unavailable/point.yaml"), "ValueError", "broken"
    )
    unavailable = ParameterContext(
        point=saved.document(),
        library=saved.library,
        target=saved.target,
        roles=None,
        storage=saved.storage,
        source_faults=(fault,),
    )
    with pytest.raises(ContextSourceUnavailableError):
        session.control.capture_storage(unavailable)
    unbound = ParameterContext(
        point=saved.document(), library=saved.library, target=saved.target, roles=None
    )
    with pytest.raises(FailedPreconditionError, match="saved-point storage"):
        session.control.capture_storage(unbound)
    session.context.install_entry(session.entry)
    with pytest.raises(FailedPreconditionError) as caught:
        session.control.read_context()
    assert caught.value.reason_code == "no_context"


def test_storage_rejects_mismatched_target(session: SaveSession) -> None:
    saved = session.control.read_context()
    session.entry.new_point("q")
    session.control.use_point(PointUseRequest("q"))
    other = session.control.read_context()
    with pytest.raises(ValueError, match="origin"):
        ParameterContext(
            point=saved.document(),
            library=saved.library,
            target=other.target,
            roles=None,
            storage=saved.storage,
        )


def test_tab_created_without_context_does_not_gain_later_save_origin(
    tmp_path: Path, qapp: QApplication
) -> None:
    with native_save_session(tmp_path, qapp) as native:
        empty_state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
        adapter = MagicMock()
        adapter.form = Form(ConfigBase, {})
        registry = MagicMock()
        registry.create.return_value = adapter
        tabs = TabService(
            empty_state,
            registry,
            MagicMock(),
            form_resources(empty_state),
            form_sources=lambda aliases: form_sources(empty_state, aliases),
            captured_parameters=lambda _aliases: None,
        )
        tab_id = tabs.new_tab("pure")
        empty_state.update_tab_result(
            tab_id,
            native.state.get_tab("tab").run.result,
            metadata=native.metadata,
        )
        empty_state.set_context(native.state.session_env)
        control = SaveControlFacet(
            state=empty_state,
            guard=GuardService(empty_state),
            save=SaveService(empty_state, native.bus),
            context=native.context,
        )
        before = empty_state.get_artifact_snapshots(tab_id)
        with pytest.raises(FailedPreconditionError, match="captured storage"):
            control.save_data(tab_id, comment="rejected", origin=GUI_ORIGIN)
        assert empty_state.get_tab(tab_id).run.storage is None
        assert empty_state.get_tab(tab_id).save.comment == ""
        assert empty_state.get_artifact_snapshots(tab_id) == before
        assert not (native.entry.database_path / "runs").exists()
        assert native.entry.ledger.events() == ()
