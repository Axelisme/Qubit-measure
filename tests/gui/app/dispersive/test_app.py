"""The dispersive entry registers synthetic lab schemas before opening its target."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from zcu_tools.gui.app.analysis_project.selection import AnalysisSelection
from zcu_tools.gui.app.dispersive.app import DispersiveGuiBehavior
from zcu_tools.gui.app.dispersive.controller import Controller
from zcu_tools.resources.entry import ResultEntry, component_registry
from zcu_tools.resources.entry.registry import ComponentRegistry

from tests._lab import synthetic_lab
from tests.gui.app._assembly import assembled_controller
from tests.gui.app.dispersive._entry_fakes import Fluxonium


def test_lab_registration_precedes_opening_the_selected_component(
    qapp, analysis_entry: ResultEntry
) -> None:
    root = analysis_entry.result_path.parent.parent
    kind = "qubit/fluxonium/test_plain"
    component_registry.unregister(kind)

    def register(registry: ComponentRegistry) -> None:
        registry.register(kind, Fluxonium)

    lab = replace(synthetic_lab(), register_components=register)
    try:
        behavior = DispersiveGuiBehavior(
            lab=lab, selection=AnalysisSelection(root, "synthetic", "q1")
        )
    finally:
        if kind not in component_registry.kinds():
            component_registry.register(kind, Fluxonium)
    with assembled_controller(behavior, Controller, qapp=qapp) as ctrl:
        target = ctrl.state.project
        assert target is not None
        assert target.component == "q1"
        assert target.result_root == root / "result"
        assert ctrl.list_predictors() == ("fit.json",)


def test_empty_selection_leaves_dispersive_unopened(qapp, tmp_path: Path) -> None:
    behavior = DispersiveGuiBehavior(
        lab=synthetic_lab(), selection=AnalysisSelection(tmp_path)
    )
    with assembled_controller(behavior, Controller, qapp=qapp) as ctrl:
        assert ctrl.state.project is None


def test_relative_project_root_is_anchored_at_construction(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    behavior = DispersiveGuiBehavior(synthetic_lab(), AnalysisSelection("relative"))
    monkeypatch.chdir(tmp_path.parent)
    with assembled_controller(behavior, Controller, qapp=qapp) as ctrl:
        assert ctrl.get_project_root() == str(tmp_path / "relative")


def test_component_without_entry_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--component requires --entry"):
        DispersiveGuiBehavior(
            synthetic_lab(), AnalysisSelection(tmp_path, component="Q1")
        )


def test_target_open_failure_propagates_from_construction(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not-created"):
        DispersiveGuiBehavior(
            synthetic_lab(), AnalysisSelection(tmp_path, "not-created")
        )


def construct_with_legacy_keyword(
    constructor: Callable[..., object], keyword: str
) -> None:
    """Exercise dynamic caller keywords at the public constructor boundary."""
    constructor(**{keyword: None})


@pytest.mark.parametrize("keyword", ["project"])
def test_behavior_rejects_legacy_keywords(keyword: str) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        construct_with_legacy_keyword(DispersiveGuiBehavior, keyword)
