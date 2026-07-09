"""Read-only remote projection for autofluxdep-gui.

The remote bridge is an observation surface, not a command surface. This module
owns the JSON-friendly read model exposed through MCP/RPC so ``dispatch`` does
not need to know about Controller internals, node schemas, or numpy-backed
Result containers.
"""

from __future__ import annotations

from typing import Protocol

from zcu_tools.gui.app.autofluxdep.cfg import override_plan_to_wire
from zcu_tools.gui.app.autofluxdep.services.result_io import result_progress_summary
from zcu_tools.gui.app.autofluxdep.state import AutoFluxDepState, ProjectInfo
from zcu_tools.gui.project import DEFAULT_CHIP, DEFAULT_QUBIT


class _RemoteReadController(Protocol):
    @property
    def state(self) -> AutoFluxDepState: ...

    @property
    def is_running(self) -> bool: ...

    @property
    def is_paused(self) -> bool: ...

    @property
    def next_flux_idx(self) -> int | None: ...

    @property
    def run_status(self) -> str: ...


class RemoteReadModel(Protocol):
    """Narrow read-only query surface consumed by remote handlers."""

    def project_info(self) -> dict[str, object]: ...

    def state_check(self) -> dict[str, object]: ...

    def workflow_list(self) -> dict[str, object]: ...

    def node_cfg(self, name: str) -> dict[str, object]: ...

    def result_summary(self) -> dict[str, object]: ...

    def resources_versions(self) -> dict[str, object]: ...


class ControllerRemoteReadModel:
    """Controller-backed implementation of the autofluxdep remote read model."""

    def __init__(self, controller: _RemoteReadController) -> None:
        self._controller = controller

    @property
    def _state(self) -> AutoFluxDepState:
        return self._controller.state

    def project_info(self) -> dict[str, object]:
        return _project_info_payload(self._state.project)

    def state_check(self) -> dict[str, object]:
        state = self._state
        return {
            "has_project": _has_real_project(state.project),
            "has_soc": state.has_setup,
            "node_count": len(state.nodes),
            "flux_count": len(state.flux_values),
            "has_flux_device": state.flux_device_name is not None,
            "is_running": self._controller.is_running,
            "is_paused": self._controller.is_paused,
            "next_flux_idx": self._controller.next_flux_idx,
            "run_status": self._controller.run_status,
            "has_results": _has_enabled_results(state),
            "has_loaded_predictor": state.exp_context.predictor is not None,
            "has_run_predictor": state.run_predictor is not None,
        }

    def workflow_list(self) -> dict[str, object]:
        state = self._state
        nodes: list[dict[str, object]] = []
        for node in state.nodes:
            nodes.append(
                {
                    "name": node.name,
                    "type": node.type_name,
                    "enabled": node.enabled,
                    "provides": list(node.provides),
                    "provides_modules": list(node.provides_modules),
                    "requires": [dep.key for dep in node.all_dependencies()],
                    "has_result": node.enabled and node.name in state.run_results,
                }
            )
        return {"nodes": nodes}

    def node_cfg(self, name: str) -> dict[str, object]:
        for node in self._state.nodes:
            if node.name != name:
                continue
            return {
                "name": node.name,
                "type": node.type_name,
                "knobs": node.schema.read_knobs(),
                "override_plan": override_plan_to_wire(
                    node.builder.override_plan(node.schema)
                ),
            }
        raise KeyError(name)

    def result_summary(self) -> dict[str, object]:
        state = self._state
        results: list[dict[str, object]] = []
        for node in state.nodes:
            if not node.enabled:
                continue
            result = state.run_results.get(node.name)
            if result is None:
                continue
            summary = result_progress_summary(result)
            results.append({"name": node.name, **summary})
        return {"results": results}

    def resources_versions(self) -> dict[str, object]:
        return {"versions": self._state.version.snapshot()}


def _project_info_payload(project: ProjectInfo | None) -> dict[str, object]:
    if project is None:
        return {
            "chip_name": None,
            "qub_name": None,
            "result_dir": None,
            "database_path": None,
            "params_path": None,
        }
    return {
        "chip_name": project.chip_name,
        "qub_name": project.qub_name,
        "result_dir": project.result_dir,
        "database_path": project.database_path,
        "params_path": project.params_path,
    }


def _has_real_project(project: ProjectInfo | None) -> bool:
    return bool(
        project is not None
        and project.chip_name
        and project.qub_name
        and (project.chip_name, project.qub_name) != (DEFAULT_CHIP, DEFAULT_QUBIT)
    )


def _has_enabled_results(state: AutoFluxDepState) -> bool:
    return any(node.enabled and node.name in state.run_results for node in state.nodes)


__all__ = [
    "ControllerRemoteReadModel",
    "RemoteReadModel",
]
