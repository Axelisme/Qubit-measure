"""Tool-test Scene fixtures backed by the existing small numeric test models."""

from zcu_tools.simulate.evolution import RoutingHistory

from tests.simulate.evolution.contract.accuracy_cases import accuracy_case, history_case
from tools.family_pareto_inputs import Scene


def make_scene(name: str) -> Scene:
    """Build a tool-owned record for a small accuracy fixture; caller closes source."""
    case = accuracy_case(name)
    return Scene(case.name, case.source, case.batch, case.families)


def make_history_scene(count: int, history: RoutingHistory) -> Scene:
    """Build a tool Scene borrowing the caller's history; caller closes source."""
    case = history_case(count, history)
    return Scene(case.name, case.source, case.batch, case.families)
