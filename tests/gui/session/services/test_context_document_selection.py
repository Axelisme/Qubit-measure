"""Global-default and focused-only document selection through the context port."""

import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import DocumentSetRef, PointUseRequest
from zcu_tools.gui.session.events import ContextSwitchedPayload
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.entry.aliases import EntryAliases
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase


def test_default_capture_reads_global_while_focus_remains_point(
    selected: tuple[ContextService, BaseEventBus, ResultEntry],
) -> None:
    context, _, rich_entry = selected
    point = rich_entry.new_point("local")
    point.Q1.rate = 5200.0
    context.install_entry(rich_entry)
    context.use_point(PointUseRequest("local"))
    before = context.get_source_binding()
    revisions = context.get_installed_file_revisions()

    capture = context.read_context()

    assert capture.value("Q1.rate") == 4300.0
    assert capture.target == DocumentSetRef(entry_name=rich_entry.result_path.name)
    assert context.current_point_label() == "local"
    assert context.get_source_binding() == before
    assert context.get_installed_file_revisions() == revisions


def test_default_document_edits_use_global_not_focused_point(
    selected: tuple[ContextService, BaseEventBus, ResultEntry],
) -> None:
    context, _, rich_entry = selected
    rich_entry.aliases().save(EntryAliases(aliases={"driver": "Q1"}))
    point = rich_entry.new_point("local")
    point.Q1.rate = 5200.0
    rich_entry.aliases("local").save(EntryAliases(aliases={"driver": "Q2"}))
    context.install_entry(rich_entry)
    context.use_point(PointUseRequest("local"))
    old_point = context.read_context(point="local")
    revisions = context.get_file_revisions()

    assert context.get_alias_rows() == (AliasRow("driver", "Q1"),)
    assert context.inspect_document().components["Q1"].model_dump()["rate"] == 4300.0
    context.apply_alias_rows((AliasRow("driver", "Q3"),))
    context.add_component("Q3", "fake/drive/a", {"rate": 6100.0, "sense": "R3"})

    assert context.read_context().value("$driver.rate") == 6100.0
    assert rich_entry.aliases().read().require_valid().aliases == {"driver": "Q3"}
    assert context.read_context(point="local").value("Q1.rate") == 5200.0
    assert old_point.value("Q1.rate") == 5200.0
    assert context.get_file_revisions() == revisions
    assert rich_entry.aliases("local").read().require_valid().aliases == {
        "driver": "Q2"
    }


@pytest.mark.parametrize(
    "command", ["capture", "inspect", "alias-read", "alias-write", "add"]
)
def test_document_commands_reject_nonfocused_point_without_effects(
    selected: tuple[ContextService, BaseEventBus, ResultEntry], command: str
) -> None:
    _, _, rich_entry = selected
    rich_entry.new_point("local")
    rich_entry.new_point("other")
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    context = ContextService(state, bus, bind_assets=ArbWaveformDatabase.info_at)
    context.install_entry(rich_entry)
    context.use_point(PointUseRequest("local"))
    capture = context.read_context(point="local")
    binding = context.get_source_binding()
    revisions = context.get_installed_file_revisions()
    version = state.version.get("context")
    files = {path: path.read_bytes() for path in rich_entry.result_path.rglob("*.yaml")}
    facts: list[ContextSwitchedPayload] = []
    subscription = bus.subscribe(ContextSwitchedPayload, facts.append)
    commands = {
        "capture": lambda: context.read_context(point="other"),
        "inspect": lambda: context.inspect_document(point="other"),
        "alias-read": lambda: context.get_alias_rows(point="other"),
        "alias-write": lambda: context.apply_alias_rows(
            (AliasRow("driver", "Q1"),), point="other"
        ),
        "add": lambda: context.add_component("Q3", "fake/drive/a", {}, point="other"),
    }
    try:
        with pytest.raises(InvalidInputError, match="not the focused point"):
            commands[command]()
        assert context.current_point_label() == "local"
        assert context.get_source_binding() == binding
        assert context.get_installed_file_revisions() == revisions
        assert state.version.get("context") == version
        assert facts == []
        assert {
            path: path.read_bytes() for path in rich_entry.result_path.rglob("*.yaml")
        } == files
        assert context.read_context(point="local").value("Q1.rate") == 4300.0
        assert capture.value("Q1.rate") == 4300.0
    finally:
        subscription.unsubscribe()
