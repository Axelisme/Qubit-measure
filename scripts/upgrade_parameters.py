"""Upgrade component YAML or operator-declared record devices; dry-run by default."""

from zcu_tools.resources.entry import component_registry
from zcu_tools.resources.parameter_upgrade import main

from zcu_lab.components import register_all
from zcu_lab.components.parameter_upgrades import parameter_upgrade_plan

if __name__ == "__main__":
    register_all(component_registry)
    raise SystemExit(main(plan=parameter_upgrade_plan()))
