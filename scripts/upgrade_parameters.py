"""Upgrade entry components and predictors, or operator-declared record devices."""

from collections.abc import Sequence

from zcu_tools.resources.parameter_upgrade import UpgradePlan
from zcu_tools.resources.parameter_upgrade import main as run_upgrade


def main(argv: Sequence[str] | None = None, *, plan: UpgradePlan | None = None) -> int:
    """Run the dry-run/apply CLI with an optional caller-owned component plan.

    Omit plan for the lab's registered schema chain. Tests and other in-process
    callers can provide a synthetic plan without loading lab definitions.
    Return 0 on success, 1 on per-file rejection; invalid CLI raises SystemExit(2).
    """
    if plan is None:
        from zcu_tools.resources.entry import component_registry

        from zcu_lab.components import register_all
        from zcu_lab.components.parameter_upgrades import parameter_upgrade_plan

        register_all(component_registry)
        plan = parameter_upgrade_plan()
    return run_upgrade(argv, plan=plan)


if __name__ == "__main__":
    raise SystemExit(main())
