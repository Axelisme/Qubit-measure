"""Entry selection is explicit when its qubit alias cannot choose a target."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from zcu_tools.gui.app.analysis_project.entry_dialog import EntryDialog
from zcu_tools.resources.entry import ResultEntry, component_registry
from zcu_tools.resources.entry.aliases import EntryAliases
from zcu_tools.resources.entry.schema import ComponentSchema

from tests.gui._user_input import ComponentDriver


@pytest.fixture
def dialog(qapp, tmp_path: Path, request) -> Iterator[ComponentDriver]:
    component_registry.register("qubit/fluxonium", ComponentSchema)
    component_registry.register("resonator", ComponentSchema)
    resource = ResultEntry.create(
        "synthetic",
        result_root=tmp_path / "result",
        database_root=tmp_path / "Database",
    )
    resource.setup.add_component("R1", kind="resonator")
    alias = request.param
    if alias != "no_components":
        resource.setup.add_component("Q1", kind="qubit/fluxonium")
        resource.setup.add_component("Q2", kind="qubit/fluxonium")
    aliases = {} if alias in (None, "no_components") else {"qubit": alias}
    resource.save_aliases(EntryAliases(aliases=aliases))
    view = EntryDialog(None, project_root=tmp_path)
    view.show()
    qapp.processEvents()
    try:
        yield ComponentDriver(view)
    finally:
        view.close()
        view.deleteLater()
        qapp.processEvents()
        component_registry.unregister("qubit/fluxonium")
        component_registry.unregister("resonator")


@pytest.mark.parametrize(
    ("dialog", "diagnostic"),
    [(None, "$qubit"), ("", "$qubit"), ("missing", "missing"), ("R1", "fluxonium")],
    indirect=["dialog"],
)
def test_invalid_qubit_alias_requires_explicit_component_choice(
    dialog: ComponentDriver,
    diagnostic: str,
    tmp_path: Path,
) -> None:
    dialog.choose_option(object_name="analysisEntry", value="synthetic")
    dialog.click("Open entry")
    dialog.screenshot(tmp_path / "01-no-default.png")
    assert dialog.observe(object_name="analysisComponent").text == ""
    assert not dialog.observe("&OK").enabled
    assert diagnostic in "\n".join(item.text for item in dialog.visible().widgets)
    try:
        dialog.choose_option(object_name="analysisComponent", value="Q1")
    except LookupError as error:
        pytest.fail(f"Open entry must list valid components despite its alias: {error}")
    assert dialog.observe("&OK").enabled
    dialog.screenshot(tmp_path / "02-explicit-choice.png")
    dialog.choose_option(object_name="analysisComponent", value="Q2")
    assert dialog.observe(object_name="analysisComponent").text == "Q2"
    dialog.screenshot(tmp_path / "03-other-component.png")
    dialog.click("&OK")
    assert not dialog.observe("&OK").visible


@pytest.mark.parametrize("dialog", ["no_components"], indirect=True)
def test_entry_without_fluxonium_components_reports_the_cause(
    dialog: ComponentDriver,
    tmp_path: Path,
) -> None:
    dialog.choose_option(object_name="analysisEntry", value="synthetic")
    dialog.click("Open entry")
    dialog.screenshot(tmp_path / "no-components.png")
    assert not dialog.observe("&OK").enabled
    assert "no fluxonium" in "\n".join(item.text for item in dialog.visible().widgets)
    dialog.click("&Cancel")
    assert not dialog.observe("&Cancel").visible
