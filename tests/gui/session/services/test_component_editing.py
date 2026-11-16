"""Component commands use the owning session and independent global documents."""

from collections.abc import Generator
from pathlib import Path

import pytest
from pydantic import ValidationError
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.context_types import CommandOrigin, PointCreateRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ComponentSchema, ResultEntry, component_registry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui._context_reads import capture_focus, inspect_focus
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


class RequiredComponent(ComponentSchema):
    reference: str


@pytest.fixture
def component_session(
    tmp_path: Path,
) -> Generator[tuple[ContextControlPort, ResultEntry]]:
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
        yield service, entry
        assert registry_state() == installed
    finally:
        restore_registry(before)


def test_add_saved_component_and_invalid_required_fields_are_atomic(component_session):
    control, entry = component_session
    control.new_point(PointCreateRequest("a", "", CommandOrigin("gui")))
    control.add_component(
        "Q1", "fake/drive/a", {"rate": 4300.0}, point=control.current_point_label()
    )
    assert capture_focus(control).value("Q1.rate") == 4300.0
    assert entry.list_points() == ["a"]
    with pytest.raises(ValidationError, match="reference"):
        control.add_component(
            "Invalid", "test/required", {}, point=control.current_point_label()
        )
    assert "Invalid" not in inspect_focus(control).components


def test_component_reference_is_an_ordinary_string(component_session):
    control, _ = component_session
    control.new_point(PointCreateRequest("a", "", CommandOrigin("gui")))
    control.add_component(
        "Ref",
        "test/required",
        {"reference": "not-yet-created"},
        point=control.current_point_label(),
    )
    assert capture_focus(control).value("Ref.reference") == "not-yet-created"
    capture = capture_focus(control)
    control.set_parameter(capture.target, "Ref.reference", "another-name")
    assert capture_focus(control).value("Ref.reference") == "another-name"


def test_setup_edits_do_not_change_existing_points(component_session):
    control, entry = component_session
    control.add_component("Q1", "fake/drive/a", {"rate": 4300.0}, point=None)
    target = control.new_point(PointCreateRequest("a", "", CommandOrigin("gui")))
    capture = capture_focus(control)
    control.set_parameter(control.read_context().target, "Q1.rate", 5100.0)
    assert (
        control.inspect_document(point=None).components["Q1"].model_dump()["rate"]
        == 5100.0
    )
    assert capture_focus(control).value("Q1.rate") == 4300.0
    assert capture.value("Q1.rate") == 4300.0
    assert capture_focus(control).target == target
    assert entry.parameters("a").Q1.rate == 4300.0


def test_add_saved_component_persists_and_can_bind_global_alias(component_session):
    control, entry = component_session
    control.new_point(PointCreateRequest("a", "", CommandOrigin("gui")))
    control.add_component(
        "Q1", "fake/drive/a", {"rate": 4300.0}, point=control.current_point_label()
    )
    assert entry.parameters("a").Q1.rate == 4300.0
    control.apply_alias_rows(
        (AliasRow("driver", "Q1"),), point=control.current_point_label()
    )
    assert (
        capture_focus(control)
        .bind_aliases(control.get_alias_rows(point=control.current_point_label()), {})
        .value("$driver.rate")
        == 4300.0
    )
