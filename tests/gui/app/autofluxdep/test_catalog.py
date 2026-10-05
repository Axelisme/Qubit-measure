"""Injected catalog declaration, placement and Controller contracts."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest
from qtpy.QtWidgets import QApplication, QInputDialog, QPushButton, QWidget
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.catalog import ExperimentCatalog
from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.ui.node_list import NodeListPane

from tests.gui.app.autofluxdep._helpers import ProduceFn, make_builder


def _builder(
    name: str, *, provides: tuple[str, ...] = (), produce_fn: ProduceFn | None = None
) -> Builder:
    builder = make_builder(name, provides=provides, produce_fn=produce_fn)
    # The existing declaration contract relates module stem to the Builder name.
    # make_builder creates a fresh class per call, so this mutates no shared state.
    type(builder).__module__ = f"catalog_test.{name or 'empty'}"
    return builder


@pytest.fixture
def injected_controller() -> Iterator[Controller]:
    catalog = ExperimentCatalog((_builder("second"), _builder("first")))
    ctrl = build_core(catalog)
    yield ctrl
    ctrl.quiesce_background()


def test_injected_catalog_controls_add_without_reordering(
    injected_controller: Controller,
) -> None:
    ctrl = injected_controller
    assert ctrl.experiment_catalog.names() == ("second", "first")

    ctrl.add_node_by_type("first")
    ctrl.add_node_by_type("second")

    assert tuple(node.type_name for node in ctrl.state.nodes) == ("first", "second")
    assert ctrl.experiment_catalog.builders()[1] is ctrl.state.nodes[0].builder


def test_fake_catalog_restores_workflow_order(
    injected_controller: Controller,
) -> None:
    ctrl = injected_controller
    ctrl.add_node_by_type("first")
    ctrl.add_node_by_type("second")
    saved = ctrl.capture_persisted_state()

    restored = build_core(ctrl.experiment_catalog)
    try:
        report = restored.restore_persisted_state(saved)

        assert report.rejected_nodes == ()
        assert tuple(node.type_name for node in restored.state.nodes) == (
            "first",
            "second",
        )
        assert restored.state.nodes[0].builder is ctrl.experiment_catalog.builders()[1]
        assert restored.state.nodes[1].builder is ctrl.experiment_catalog.builders()[0]
    finally:
        restored.quiesce_background()


def test_fake_catalog_populates_add_menu_and_placement(
    injected_controller: Controller,
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offered: list[tuple[str, ...]] = []

    def choose_item(
        _parent: QWidget,
        _title: str,
        _label: str,
        items: Sequence[str],
        _current: int,
        _editable: bool,
    ) -> tuple[str, bool]:
        offered.append(tuple(items))
        return "first", True

    monkeypatch.setattr(QInputDialog, "getItem", choose_item)
    pane = NodeListPane(injected_controller)
    try:
        pane.show()
        qapp.processEvents()
        add_button = next(
            button for button in pane.findChildren(QPushButton) if button.text() == "+"
        )
        add_button.click()

        assert offered == [("second", "first")]
        assert tuple(node.type_name for node in injected_controller.state.nodes) == (
            "first",
        )
        assert (
            injected_controller.state.nodes[0].builder
            is injected_controller.experiment_catalog.builders()[1]
        )
    finally:
        pane.teardown()
        pane.close()


def test_injected_catalog_unknown_name_raises(injected_controller: Controller) -> None:
    with pytest.raises(KeyError):
        injected_controller.add_node_by_type("not_registered")


def test_fake_catalog_runs_in_process_in_workflow_order() -> None:
    produced: list[str] = []

    def produce(env, snapshot) -> Patch:
        del snapshot
        produced.append(env.node_name)
        return Patch()

    catalog = ExperimentCatalog(
        (_builder("second", produce_fn=produce), _builder("first", produce_fn=produce))
    )
    ctrl = build_core(catalog)
    try:
        ctrl.set_flux_values([0.0])
        ctrl.add_node_by_type("first")
        ctrl.add_node_by_type("second")
        ctrl.dry_run()
        assert produced == ["first", "second"]
    finally:
        ctrl.quiesce_background()


def test_catalog_creates_independent_placements() -> None:
    catalog = ExperimentCatalog((_builder("measurement"),))
    first = catalog.create_placement("measurement")
    second = catalog.create_placement("measurement")

    assert first is not second
    assert first.schema is not second.schema
    assert first.builder is second.builder


@pytest.mark.parametrize("declarations", [(object(),)])
def test_catalog_rejects_non_builders(declarations) -> None:
    with pytest.raises(TypeError, match="Builder instances"):
        ExperimentCatalog(declarations)


def test_catalog_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ExperimentCatalog((_builder(""),))


def test_catalog_rejects_duplicate_names() -> None:
    declaration = _builder("measurement")
    with pytest.raises(ValueError, match="duplicate experiment name"):
        ExperimentCatalog((declaration, declaration))


def test_catalog_rejects_mismatched_module_stem() -> None:
    declaration = _builder("declared")
    type(declaration).__module__ = "catalog_test.actual"
    with pytest.raises(ValueError, match="module stem"):
        ExperimentCatalog((declaration,))


def test_catalog_rejects_duplicate_output_declarations() -> None:
    with pytest.raises(ValueError, match="duplicate provides declaration"):
        ExperimentCatalog((_builder("measurement", provides=("output", "output")),))
