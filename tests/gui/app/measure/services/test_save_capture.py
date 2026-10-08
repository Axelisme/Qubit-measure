"""Fixed context/storage at the Context, Tab and Save worker seams."""

from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.datafile import RunMetadata, RunSnapshot, SoftwareProvenance
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    SaveDataRequest,
    SavePaths,
)
from zcu_tools.gui.app.measure.services.guard import GuardService, SavePermit
from zcu_tools.gui.app.measure.services.save import SaveService
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.cfg import CfgSchema, CfgSectionSpec, CfgSectionValue
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import (
    ContextSourceFault,
    ContextSourceUnavailableError,
    PointUseRequest,
)
from zcu_tools.gui.session.operation_runner import OperationSpec
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry, RoleSpec
from zcu_tools.resources.entry.save_layout import ArtifactKey, SaveLayout
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui.app.measure._cfg_fakes import cfg_resolution, cfg_resources, make_cfg
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
        control.use_point(PointUseRequest(entry.entry_id, "p"))
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
    session.control.set_component_focus("Q2")
    session.entry.new_point("q")
    session.control.use_point(PointUseRequest(session.entry.entry_id, "q"))
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
        point=storage.point,
        saved_at=datetime(2026, 10, 8, tzinfo=timezone.utc),
    )
    outputs = layout.outputs(ArtifactKey("run", "T1", "data"))
    assert outputs[0].path == session.entry.database_path / "runs/run-a/data.h5"
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
    with pytest.raises(FailedPreconditionError, match="saved-point storage"):
        session.control.capture_storage(session.control.read_context())


def test_storage_rejects_mismatched_target(session: SaveSession) -> None:
    saved = session.control.read_context()
    session.entry.new_point("q")
    session.control.use_point(PointUseRequest(session.entry.entry_id, "q"))
    other = session.control.read_context()
    with pytest.raises(ValueError, match="origin"):
        ParameterContext(
            point=saved.document(),
            library=saved.library,
            target=other.target,
            roles=None,
            storage=saved.storage,
        )


@pytest.mark.parametrize("roleless", [False, True])
def test_save_worker_uses_tab_capture_after_entry_switch(
    session: SaveSession, tmp_path: Path, roleless: bool
) -> None:
    capture = _bound(session)
    adapter = MagicMock()
    adapter.parameter_roles.return_value = (
        {} if roleless else {"driver": RoleSpec("fake/drive/*")}
    )
    adapter.make_default_cfg.return_value = CfgSchema(
        CfgSectionSpec(), CfgSectionValue()
    )
    adapter.make_save_paths.return_value = SavePaths("unused.h5", "unused.png")
    registry = MagicMock()
    registry.create.return_value = adapter
    tabs = TabService(
        session.state,
        registry,
        MagicMock(),
        cfg_resources(session.state),
        cfg_resolution=lambda parameters: cfg_resolution(session.state, parameters),
        captured_parameters=lambda _roles: None if roleless else capture,
    )
    tab_id = tabs.new_tab("T1")
    result = object()
    storage = capture.capture_storage()
    metadata = RunMetadata(
        run_id="20261008T000000Z-a1b2c3",
        experiment="test/native",
        started_at="2026-10-08T00:00:00+00:00",
        finished_at="2026-10-08T00:01:00+00:00",
        completion="complete",
        labber_path=None,
        snapshot=RunSnapshot(
            entry_id=storage.entry_id,
            entry_name=storage.entry_name,
            point=storage.point,
            description=storage.description,
            roles=storage.roles,
        ),
        provenance=SoftwareProvenance(
            software_versions={},
            git_commit=None,
            git_dirty=None,
            qick_version=None,
            soc_fingerprint=None,
            hostname=None,
        ),
    )
    session.state.update_tab_result(tab_id, result, metadata=metadata)
    destination = tmp_path / "out/data.h5"
    session.state.get_tab(tab_id).save.data_path_override = str(destination)
    other = ResultEntry.create(
        "entry-b", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    session.context.install_entry(other)
    received: list[SaveDataRequest[object]] = []
    adapter.save.side_effect = received.append
    runner = MagicMock()

    def begin(spec: OperationSpec) -> int:
        spec.work(MagicMock())
        return 1

    runner.begin.side_effect = begin
    service = SaveService(
        session.state, runner, session.bus, owner_scheduler=MagicMock()
    )
    service.start_save_data(SavePermit(tab_id), str(destination), "annotation")

    request = received[0]
    assert request.run_result is result
    assert request.metadata is metadata
    assert request.parameters is (None if roleless else capture)
    assert request.storage.point == "p"
    assert str(request.storage.entry_id) == session.entry.entry_id
    assert dict(request.storage.roles) == ({} if roleless else {"driver": "Q1"})
    assert request.comment == "annotation"
    assert Path(request.data_path).parent == destination.parent


def test_tab_created_without_context_does_not_gain_later_save_origin(
    session: SaveSession, tmp_path: Path
) -> None:
    empty_state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    adapter = MagicMock()
    adapter.parameter_roles.return_value = {}
    adapter.make_default_cfg.return_value = CfgSchema(
        CfgSectionSpec(), CfgSectionValue()
    )
    registry = MagicMock()
    registry.create.return_value = adapter
    tabs = TabService(
        empty_state,
        registry,
        MagicMock(),
        cfg_resources(empty_state),
        cfg_resolution=lambda parameters: cfg_resolution(empty_state, parameters),
        captured_parameters=lambda _roles: None,
    )
    tab_id = tabs.new_tab("pure")
    empty_state.update_tab_result(tab_id, object())
    empty_state.set_context(session.state.session_env)
    service = SaveService(
        empty_state, MagicMock(), session.bus, owner_scheduler=MagicMock()
    )
    destination = tmp_path / "not-created/data.h5"
    with pytest.raises(FailedPreconditionError, match="captured save context"):
        service.start_save_data(SavePermit(tab_id), str(destination))
    assert not destination.parent.exists()


def test_guard_keeps_original_tab_entry_point_and_roles_in_run_permit(
    session: SaveSession, tmp_path: Path
) -> None:
    capture = _bound(session)
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(requires_soc=False)
    adapter.experiment_tag = "test/native"
    tab_id = "native"
    session.state.add_tab(
        tab_id,
        Session(
            adapter_name="native",
            adapter=adapter,
            cfg=make_cfg(
                CfgSchema(CfgSectionSpec(), CfgSectionValue()), state=session.state
            ),
            parameters=capture,
            save_context=capture,
        ),
    )
    other = ResultEntry.create(
        "entry-b", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    other.new_point("q")
    session.context.install_entry(other)
    session.control.use_point(PointUseRequest(other.entry_id, "q"))
    tab = session.state.get_tab(tab_id)
    permit = GuardService(session.state).acquire_run_permit(
        tab_id, expected_revision=tab.cfg.observe().ref.revision
    )
    assert permit.snapshot is not None
    assert str(permit.snapshot.entry_id) == session.entry.entry_id
    assert permit.snapshot.entry_name == "entry-a"
    assert permit.snapshot.point == "p"
    assert dict(permit.snapshot.roles) == {"driver": "Q1"}
    assert permit.experiment_tag == "test/native"
    session.context.clear_context()
    assert permit.snapshot.point == "p"
    assert dict(permit.snapshot.roles) == {"driver": "Q1"}
