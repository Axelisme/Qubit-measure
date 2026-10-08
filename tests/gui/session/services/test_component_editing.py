"""Component commands use the owning session and independent setup template."""

from collections.abc import Generator
from pathlib import Path

import pytest
from pydantic import ValidationError
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import CommandOrigin, PointCreateRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ComponentSchema, ResultEntry, component_registry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


class RequiredComponent(ComponentSchema):
    reference: str


@pytest.fixture
def component_session(
    tmp_path: Path,
) -> Generator[tuple[ContextControlFacet, ResultEntry]]:
    before = registry_state()
    try:
        register_fakes()
        component_registry.register("test/required", RequiredComponent)
        installed = registry_state()
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "results",
            database_root=tmp_path / "Database",
        )
        service = ContextService(
            SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
            BaseEventBus(),
            bind_assets=ArbWaveformDatabase.info_at,
        )
        service.install_entry(entry)
        yield ContextControlFacet(context=service), entry
        assert registry_state() == installed
    finally:
        restore_registry(before)


def test_add_draft_component_and_invalid_required_fields_are_atomic(component_session):
    control, entry = component_session
    control.add_component("Q1", "fake/drive/a", {"rate": 4300.0})
    assert control.read_context().value("Q1.rate") == 4300.0
    assert entry.list_points() == []
    with pytest.raises(ValidationError, match="reference"):
        control.add_component("Invalid", "test/required", {})
    assert "Invalid" not in control.inspect_document().components


def test_component_reference_is_an_ordinary_string(component_session):
    control, _ = component_session
    control.add_component("Ref", "test/required", {"reference": "not-yet-created"})
    assert control.read_context().value("Ref.reference") == "not-yet-created"
    capture = control.read_context()
    control.set_parameter(capture.target, "Ref.reference", "another-name")
    assert control.read_context().value("Ref.reference") == "another-name"


def test_setup_edits_do_not_change_existing_points(component_session):
    control, entry = component_session
    control.add_component("Q1", "fake/drive/a", {"rate": 4300.0}, setup=True)
    target = control.new_point(
        PointCreateRequest(entry.entry_id, "a", "", CommandOrigin("gui"))
    )
    capture = control.read_context()
    control.set_setup_parameter("Q1.rate", 5100.0)
    assert (
        control.inspect_document(setup=True).components["Q1"].model_dump()["rate"]
        == 5100.0
    )
    assert control.read_context().value("Q1.rate") == 4300.0
    assert capture.value("Q1.rate") == 4300.0
    assert control.read_context().target == target
    assert entry.use_point("a").Q1.rate == 4300.0


def test_add_saved_component_updates_focus_options_and_persists(component_session):
    control, entry = component_session
    control.new_point(PointCreateRequest(entry.entry_id, "a", "", CommandOrigin("gui")))
    control.add_component("Q1", "fake/drive/a", {"rate": 4300.0})
    assert entry.use_point("a").Q1.rate == 4300.0
    control.set_component_focus("Q1")
    assert control.get_component_focus().name == "Q1"
