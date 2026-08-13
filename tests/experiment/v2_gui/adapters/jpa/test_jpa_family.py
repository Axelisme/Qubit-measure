"""JPA-07 — the six JPA adapters as one discoverable calibration family.

Family-level integration contract over the generic Interface (not per-adapter
policy, which the sibling test modules own):

- A1: all six ``jpa/*`` names appear in bring-up order in the startup catalog
  (``ADAPTERS``) and in the generic registry listing that the remote
  ``view.adapter_list`` surface serves.
- A2: generic tab creation (``TabService.new_tab``) yields a live cfg for every
  family member, and analysis/writeback capabilities are consistent.
- A3: every family member exposes a complete operator guide through the
  existing ``guide()`` Interface and the generic ``TabService.adapter_guide``
  surface.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from zcu_tools.experiment.v2_gui.adapters.jpa import (
    JpaAutoAnalyzeResult,
    JpaFluxAnalyzeResult,
    JpaFreqAnalyzeResult,
    JpaPowerAnalyzeResult,
)
from zcu_tools.experiment.v2_gui.registry import ADAPTERS, register_all
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    ContextReadiness,
    ExpAdapterProtocol,
    ExpContext,
    MetaDictWriteback,
    WritebackRequest,
)
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.app.main.services.tab import TabService
from zcu_tools.gui.app.main.state import State
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

# Bring-up order (startup catalog insertion order, also the remote listing
# order): find the pump frequency, then its flux sweet spot, then the pump
# power, optionally joint-optimize all three, survey flux × readout frequency
# by eye, and finish with the pump off/on diagnostic.
JPA_FAMILY = [
    "jpa/freq",
    "jpa/flux",
    "jpa/power",
    "jpa/auto_optimize",
    "jpa/flux_onetone",
    "jpa/check",
]

_GUIDE_FIELDS = (
    "behavior",
    "expects_md",
    "expects_ml",
    "typical_writeback",
    "recommended",
)


def _empty_ctx() -> ExpContext:
    return ExpContext(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
        readiness=ContextReadiness.ACTIVE,
    )


def _generic_service() -> tuple[TabService, State]:
    registry = Registry()
    register_all(registry)
    state = State(_empty_ctx())
    return TabService(state, registry, MagicMock()), state


# --- A1: startup catalog + generic remote listing --------------------------


def test_startup_catalog_lists_jpa_family_in_bringup_order() -> None:
    assert [name for name in ADAPTERS if name.startswith("jpa/")] == JPA_FAMILY


def test_generic_registry_listing_serves_jpa_family_in_bringup_order() -> None:
    registry = Registry()
    register_all(registry)
    assert [name for name in registry.list_names() if name.startswith("jpa/")] == (
        JPA_FAMILY
    )


# --- A2: generic tab creation + capability consistency ---------------------


@pytest.mark.parametrize("name", JPA_FAMILY)
def test_generic_tab_creation_yields_cfg_for_every_jpa_adapter(name: str) -> None:
    service, state = _generic_service()

    tab_id = service.new_tab(name)
    tab = state.get_tab(tab_id)

    assert tab.adapter_name == name
    assert tab.cfg_schema is not None  # fresh default cfg, validated on creation
    assert isinstance(tab.adapter, ExpAdapterProtocol)
    assert isinstance(tab.adapter.capabilities, AdapterCapabilities)


def test_family_analysis_capabilities_are_consistent() -> None:
    # The four calibration adapters fit an optimum; the 2D survey and the
    # off/on diagnostic are look-at-the-data steps (NONE / figure-only FIT).
    assert ADAPTERS["jpa/freq"]().capabilities.analysis is AnalysisMode.FIT
    assert ADAPTERS["jpa/flux"]().capabilities.analysis is AnalysisMode.FIT
    assert ADAPTERS["jpa/power"]().capabilities.analysis is AnalysisMode.FIT
    assert ADAPTERS["jpa/auto_optimize"]().capabilities.analysis is AnalysisMode.FIT
    assert ADAPTERS["jpa/flux_onetone"]().capabilities.analysis is AnalysisMode.NONE
    assert ADAPTERS["jpa/check"]().capabilities.analysis is AnalysisMode.FIT


@pytest.mark.parametrize(
    ("name", "analyze_result", "expected_targets"),
    [
        (
            "jpa/freq",
            JpaFreqAnalyzeResult(best_freq=12900.0, figure=Figure()),
            ["best_jpa_freq"],
        ),
        (
            "jpa/flux",
            JpaFluxAnalyzeResult(best_flux=0.4, figure=Figure()),
            ["best_jpa_flux"],
        ),
        (
            "jpa/power",
            JpaPowerAnalyzeResult(best_power=-10.0, figure=Figure()),
            ["best_jpa_power"],
        ),
        (
            "jpa/auto_optimize",
            JpaAutoAnalyzeResult(
                best_flux=0.4,
                best_freq=12900.0,
                best_power=-10.0,
                figure=Figure(),
            ),
            ["best_jpa_flux", "best_jpa_freq", "best_jpa_power"],
        ),
    ],
)
def test_calibration_adapters_propose_family_writeback_targets(
    name: str, analyze_result: object, expected_targets: list[str]
) -> None:
    adapter = ADAPTERS[name]()
    items = list(
        adapter.get_writeback_items(
            WritebackRequest(
                run_result=MagicMock(),
                analyze_result=analyze_result,
                ctx=_empty_ctx(),
            )
        )
    )
    assert [item.target_name for item in items] == expected_targets
    assert all(isinstance(item, MetaDictWriteback) for item in items)


@pytest.mark.parametrize("name", ["jpa/flux_onetone", "jpa/check"])
def test_visual_diagnostic_adapters_propose_no_writeback(name: str) -> None:
    adapter = ADAPTERS[name]()
    items = list(
        adapter.get_writeback_items(
            WritebackRequest(
                run_result=MagicMock(),
                analyze_result=MagicMock(),
                ctx=_empty_ctx(),
            )
        )
    )
    assert items == []


# --- A3: operator guides through the existing Interface ---------------------


@pytest.mark.parametrize("name", JPA_FAMILY)
def test_every_jpa_adapter_exposes_complete_guide(name: str) -> None:
    guide = ADAPTERS[name].guide()
    assert isinstance(guide, AdapterGuide)
    assert all(len(getattr(guide, field)) > 0 for field in _GUIDE_FIELDS)


def test_generic_adapter_guide_surface_serves_jpa_family() -> None:
    service, _ = _generic_service()
    for name in JPA_FAMILY:
        guide = service.adapter_guide(name)
        assert set(guide) == set(_GUIDE_FIELDS)
        assert all(len(guide[field]) > 0 for field in _GUIDE_FIELDS)
