"""Tool-test Scene fixtures backed by the existing small numeric test models."""

from contextlib import closing
from pathlib import Path

from zcu_tools.simulate.evolution import (
    Engine,
    ReferenceResult,
    RoutingHistory,
    write_frozen_manifest,
    write_reference,
)

from tests.simulate.evolution.contract.accuracy_cases import accuracy_case, history_case
from tools.family_pareto_inputs import Input, Scene, freeze_records


def make_scene(name: str) -> Scene:
    """Build a tool-owned record for a small accuracy fixture; caller closes source."""
    case = accuracy_case(name)
    return Scene(case.name, case.source, case.batch, case.families)


def engine_reference(scene: Scene) -> ReferenceResult:
    """Return small consumer-test data, not an independent accuracy oracle.

    Use a fixed exact-family fixture policy, independent of consumer --families.
    The zero convergence_error is a test-data sentinel, not solver evidence.
    The caller owns scene.source; this function closes its own Engine.
    """
    with closing(Engine(1, families=("t4", "t6", "dop853"))) as engine:
        return ReferenceResult(
            engine.evolve(scene.source, [scene.batch]).densities, 0.0
        )


def write_test_cache(inputs: list[Input], directory: Path) -> None:
    """Write prepared tiny synthetic consumer fixtures to a test-owned cache.

    No independent solver runs. Every source is fresh and closed before return.
    """
    for item in inputs:
        scene = item.build()
        with closing(scene.source):
            write_reference(
                directory, scene.source, scene.batch, engine_reference(scene)
            )


def write_test_manifest(inputs: list[Input], path: Path) -> None:
    """Write tiny test-owned numeric records through the public manifest codec."""
    write_frozen_manifest(path, freeze_records(inputs))


def make_history_scene(count: int, history: RoutingHistory) -> Scene:
    """Build a tool Scene borrowing the caller's history; caller closes source."""
    case = history_case(count, history)
    return Scene(case.name, case.source, case.batch, case.families)
