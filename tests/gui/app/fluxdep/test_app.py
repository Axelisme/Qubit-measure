"""The fluxdep entry consumes lab definitions before opening a target."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from zcu_tools.gui.app.analysis_project.selection import AnalysisSelection
from zcu_tools.gui.app.fluxdep.app import FluxDepGuiBehavior
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.resources.entry import ResultEntry, component_registry
from zcu_tools.resources.entry.registry import ComponentRegistry

from tests._lab import synthetic_lab
from tests.gui.app._assembly import assembled_controller
from tests.gui.app.fluxdep._components import (
    SyntheticFluxonium,
    skip_calibration,
    synthetic_flux_context,
)
from tests.gui.app.fluxdep._export import aligned_export_state


def test_lab_registration_precedes_target_open_and_supplies_analysis_policies(
    qapp, fluxonium_entry: ResultEntry
) -> None:
    root = fluxonium_entry.result_path.parent.parent
    component_registry.unregister("qubit/fluxonium")

    def register(registry: ComponentRegistry) -> None:
        registry.register("qubit/fluxonium", SyntheticFluxonium)

    lab = replace(
        synthetic_lab(),
        register_components=register,
        fluxdep_export_planner=skip_calibration,
        component_flux_context_provider=synthetic_flux_context,
    )
    try:
        behavior = FluxDepGuiBehavior(
            lab=lab, selection=AnalysisSelection(root, "synthetic")
        )
    finally:
        if "qubit/fluxonium" not in component_registry.kinds():
            component_registry.register("qubit/fluxonium", SyntheticFluxonium)
    with assembled_controller(behavior, Controller, qapp=qapp) as ctrl:
        target = ctrl.state.project
        assert target is not None
        assert target.component == "Q2"
        assert target.result_root == root / "result"
        assert ctrl.flux_context() == synthetic_flux_context(SyntheticFluxonium())
        ctrl.state.put_spectrum(
            replace(
                aligned_export_state(target).get_spectrum("aligned"),
                filter_mask=None,
                brush_mask=None,
            )
        )
        ctrl.state.set_fit_result((5.0, 1.2, 0.4))
        receipt = ctrl.export_params_receipt()
        assert receipt.flux_writeback.status == "skipped"
        assert receipt.savepath == str(target.predictor_path)


def test_empty_selection_leaves_fluxdep_unopened(qapp, tmp_path: Path) -> None:
    behavior = FluxDepGuiBehavior(
        lab=synthetic_lab(), selection=AnalysisSelection(tmp_path)
    )
    with assembled_controller(behavior, Controller, qapp=qapp) as ctrl:
        assert ctrl.state.project is None


def test_component_without_entry_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--component requires --entry"):
        FluxDepGuiBehavior(synthetic_lab(), AnalysisSelection(tmp_path, component="Q1"))


def test_target_open_failure_propagates_from_construction(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not-created"):
        FluxDepGuiBehavior(synthetic_lab(), AnalysisSelection(tmp_path, "not-created"))


def construct_with_legacy_keyword(
    constructor: Callable[..., object], keyword: str
) -> None:
    """Exercise dynamic caller keywords at the public constructor boundary."""
    constructor(**{keyword: None})


@pytest.mark.parametrize(
    "keyword", ["project", "export_planner", "flux_context_provider"]
)
def test_behavior_rejects_legacy_keywords(keyword: str) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        construct_with_legacy_keyword(FluxDepGuiBehavior, keyword)
