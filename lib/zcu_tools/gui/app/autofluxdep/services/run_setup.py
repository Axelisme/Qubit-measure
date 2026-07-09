"""Run setup helpers for autofluxdep-gui.

These helpers build the run-time objects that are pure consequences of State and
the enabled workflow nodes: provider order, tools, result containers, cfg
snapshots, RunSession, and headless dry-run Orchestrator. Controller owns the
state machine; this module owns the setup graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from zcu_tools.gui.app.autofluxdep.cfg import (
    RunCfgSnapshot,
    validate_override_plan_base_cfg,
)
from zcu_tools.gui.app.autofluxdep.feedback import build_feedback_runtime
from zcu_tools.gui.app.autofluxdep.nodes.builder import PlacedNode
from zcu_tools.gui.app.autofluxdep.nodes.predictor import PredictorBuilder
from zcu_tools.gui.app.autofluxdep.orchestrator import (
    InfoStore,
    ModuleSource,
    Notify,
    Orchestrator,
)
from zcu_tools.gui.app.autofluxdep.run_session import RunEventSink, RunSession
from zcu_tools.gui.app.autofluxdep.services.run_store import RunStore
from zcu_tools.gui.app.autofluxdep.state import AutoFluxDepState, ProjectInfo
from zcu_tools.gui.app.autofluxdep.tools import (
    FluxoniumPredictorAdapter,
    SimplePredictor,
    Tools,
)


class MlModuleSource:
    """Transparent ``ModuleLibrary`` proxy honouring ``ModuleSource``.

    The run resolver wants ``get_module(name)`` to return None if absent, so the
    dependency resolver can fall back to a Node-produced module or dependency
    default. Node cfg builders still want the full ``ModuleLibrary`` surface
    (``get_waveform`` / ``make_cfg``), which should raise on missing references.
    This proxy overrides only ``get_module`` and forwards every other attribute.
    """

    def __init__(self, ml: Any) -> None:
        self._ml = ml

    def get_module(self, name: str) -> Any:
        if name not in self._ml.modules:
            return None
        return self._ml.get_module(name)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._ml, attr)


def build_run_providers(enabled_nodes: Sequence[PlacedNode]) -> list[PlacedNode]:
    """Return the execution sequence: predictor service + enabled user nodes."""
    service = PlacedNode(builder=PredictorBuilder())
    return [service, *enabled_nodes]


def build_run_tools(
    state: AutoFluxDepState,
    providers: Sequence[PlacedNode],
) -> Tools:
    """Build the sweep's run-lived predictor and feedback capabilities."""
    raw = state.exp_context.predictor
    predictor = (
        FluxoniumPredictorAdapter(fluxonium=raw)
        if raw is not None
        else SimplePredictor()
    )
    feedback = build_feedback_runtime(providers, md=state.exp_context.md)
    return Tools(predictor=predictor, feedback=feedback)


def allocate_run_results(
    state: AutoFluxDepState,
    enabled_nodes: Sequence[PlacedNode],
    flux: Any,
) -> dict[str, Any]:
    """Pre-allocate each user provider's sweep Result."""
    results: dict[str, Any] = {}
    md = state.exp_context.md
    for node in enabled_nodes:
        result = node.builder.make_init_result(node.schema, flux, md=md)
        if result is not None:
            results[node.name] = result
    return results


def build_run_cfg_snapshots(
    state: AutoFluxDepState,
    enabled_nodes: Sequence[PlacedNode],
) -> dict[str, RunCfgSnapshot]:
    """Lower each enabled node's run-start cfg and validate override paths."""
    ctx = state.exp_context
    snapshots: dict[str, RunCfgSnapshot] = {}
    for node in enabled_nodes:
        base_cfg = node.schema.lower_raw(ctx.ml, ctx.md)
        override_plan = node.builder.override_plan(node.schema)
        knobs = node.schema.lower(ctx.ml, ctx.md)
        validate_override_plan_base_cfg(
            override_plan,
            base_cfg,
            node_name=node.name,
        )
        snapshots[node.name] = RunCfgSnapshot(
            base_cfg=base_cfg,
            override_plan=override_plan,
            knobs=knobs,
        )
    return snapshots


def create_run_session(
    *,
    state: AutoFluxDepState,
    enabled_nodes: list[PlacedNode],
    flux_values: list[float],
    project: ProjectInfo,
    notify: Notify | None,
    event_sink: RunEventSink,
    progress_label: str,
) -> RunSession:
    """Create the sweep-lived RunSession from the current State snapshot."""
    ctx = state.exp_context
    enabled_names = {node.name for node in enabled_nodes}
    if not state.run_results or not set(state.run_results).issubset(enabled_names):
        state.run_results = allocate_run_results(state, enabled_nodes, flux_values)
    results = state.run_results
    providers = build_run_providers(enabled_nodes)
    tools = build_run_tools(state, providers)
    state.run_predictor = tools.predictor
    flux_device = state.flux_device_name
    cfg_snapshots = build_run_cfg_snapshots(state, enabled_nodes)
    store = RunStore.create(
        project=project,
        flux_values=flux_values,
        flux_device_name=flux_device,
        nodes=enabled_nodes,
        results=results,
        cfg_snapshots={
            name: snapshot.to_wire() for name, snapshot in cfg_snapshots.items()
        },
    )
    return RunSession(
        providers=providers,
        user_nodes=enabled_nodes,
        flux_values=flux_values,
        flux_device=flux_device,
        results=results,
        cfg_snapshots=cfg_snapshots,
        store=store,
        tools=tools,
        ml=MlModuleSource(ctx.ml),
        soc=ctx.soc,
        soccfg=ctx.soccfg,
        md=ctx.md,
        notify=notify,
        event_sink=event_sink,
        has_loaded_predictor=ctx.predictor is not None,
        progress_label=progress_label,
    )


def run_dry(
    *,
    state: AutoFluxDepState,
    enabled_nodes: list[PlacedNode],
    flux_values: list[float],
    tools: Tools | None = None,
    ml: ModuleSource | None = None,
) -> InfoStore:
    """Run the dependency model headless with the same provider/cfg setup."""
    providers = build_run_providers(enabled_nodes)
    ctx = state.exp_context
    run_ml = ml
    if run_ml is None and ctx.ml is not None:
        run_ml = MlModuleSource(ctx.ml)
    cfg_snapshots = build_run_cfg_snapshots(state, enabled_nodes)
    orch = Orchestrator(
        providers=providers,
        tools=tools or build_run_tools(state, providers),
        ml=run_ml,
        md=ctx.md,
        cfg_snapshots=cfg_snapshots,
    )
    return orch.run(flux_values)


__all__ = [
    "MlModuleSource",
    "allocate_run_results",
    "build_run_cfg_snapshots",
    "build_run_providers",
    "build_run_tools",
    "create_run_session",
    "run_dry",
]
