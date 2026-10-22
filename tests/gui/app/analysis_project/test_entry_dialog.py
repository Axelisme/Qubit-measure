"""Entry selection is explicit when its qubit alias cannot choose a target."""

import sys
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from types import TracebackType

import pytest
from zcu_tools.gui.app.analysis_project.entry_dialog import EntryDialog
from zcu_tools.resources.entry import ResultEntry, component_registry
from zcu_tools.resources.entry.aliases import EntryAliases
from zcu_tools.resources.entry.schema import ComponentSchema

from tests.gui._user_input import ComponentDriver


@pytest.fixture
def dialog(qapp, tmp_path: Path, request) -> Iterator[ComponentDriver]:
    with ExitStack() as cleanup:
        component_registry.register("qubit/fluxonium", ComponentSchema)
        cleanup.callback(component_registry.unregister, "qubit/fluxonium")
        component_registry.register("resonator", ComponentSchema)
        cleanup.callback(component_registry.unregister, "resonator")
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
        try:
            view.show()
            qapp.processEvents()
            yield ComponentDriver(view)
        finally:
            view.close()
            view.deleteLater()
            qapp.processEvents()


def _caption(driver: ComponentDriver, text: str) -> str:
    return next(
        widget.text
        for widget in driver.visible().widgets
        if widget.kind == "QPushButton" and widget.text.replace("&", "") == text
    )


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
    assert not dialog.observe(_caption(dialog, "OK")).enabled
    assert diagnostic in "\n".join(item.text for item in dialog.visible().widgets)
    try:
        dialog.choose_option(object_name="analysisComponent", value="Q1")
    except LookupError as error:
        pytest.fail(f"Open entry must list valid components despite its alias: {error}")
    assert dialog.observe(_caption(dialog, "OK")).enabled
    dialog.screenshot(tmp_path / "02-explicit-choice.png")
    dialog.choose_option(object_name="analysisComponent", value="Q2")
    assert dialog.observe(object_name="analysisComponent").text == "Q2"
    dialog.screenshot(tmp_path / "03-other-component.png")
    dialog.click(_caption(dialog, "OK"))
    assert not dialog.visible().widgets


@pytest.mark.parametrize("dialog", [None], indirect=True)
def test_malformed_alias_reports_validation_and_allows_explicit_choice(
    dialog: ComponentDriver,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback_errors: list[BaseException] = []

    def record_callback_error(
        _type: type[BaseException],
        error: BaseException,
        _traceback: TracebackType | None,
    ) -> None:
        callback_errors.append(error)

    monkeypatch.setattr(sys, "excepthook", record_callback_error)
    (tmp_path / "result" / "synthetic" / "aliases.yaml").write_text(
        "format: zcu.aliases\nformat_version: '1.0'\naliases:\n  qubit: null\n",
        encoding="utf-8",
    )
    dialog.choose_option(object_name="analysisEntry", value="synthetic")
    dialog.screenshot(tmp_path / "01-entry-selected.png")
    dialog.click("Open entry")
    dialog.screenshot(tmp_path / "02-invalid-alias.png")
    visible_text = "\n".join(item.text for item in dialog.visible().widgets)
    assert "aliases.qubit" in visible_text
    assert "valid string" in visible_text
    assert not callback_errors
    assert dialog.observe(object_name="analysisComponent").text == ""
    assert not dialog.observe(_caption(dialog, "OK")).enabled
    for index, component in enumerate(("Q1", "Q2"), start=3):
        dialog.choose_option(object_name="analysisComponent", value=component)
        dialog.screenshot(tmp_path / f"{index:02}-{component}.png")
        assert dialog.observe(object_name="analysisComponent").text == component
        assert dialog.observe(_caption(dialog, "OK")).enabled
    dialog.click(_caption(dialog, "OK"))
    dialog.screenshot(tmp_path / "05-submitted.png")
    assert not dialog.visible().widgets


@pytest.mark.parametrize("dialog", ["no_components"], indirect=True)
def test_entry_without_fluxonium_components_reports_the_cause(
    dialog: ComponentDriver,
    tmp_path: Path,
) -> None:
    dialog.choose_option(object_name="analysisEntry", value="synthetic")
    dialog.click("Open entry")
    dialog.screenshot(tmp_path / "no-components.png")
    assert not dialog.observe(_caption(dialog, "OK")).enabled
    assert "no fluxonium" in "\n".join(item.text for item in dialog.visible().widgets)
    dialog.click(_caption(dialog, "Cancel"))
    assert not dialog.visible().widgets
