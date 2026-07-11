from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_FILES = (
    _ROOT / "lib/zcu_tools/gui/app/autofluxdep/controller.py",
    _ROOT / "lib/zcu_tools/gui/app/autofluxdep/services/run_setup.py",
)
_STATE_FIELDS = frozenset(
    {
        "nodes",
        "flux_values",
        "flux_start_expr",
        "flux_stop_expr",
        "flux_npts_expr",
        "auto_follow_tabs",
        "predictor_dialog_state",
        "project",
        "run_results",
        "run_predictor",
        "flux_device_name",
    }
)


def _is_state_expr(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Name)
        and node.id == "state"
        or isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == "_state"
    )


def test_autoflux_production_writes_use_state_mutators() -> None:
    violations: list[str] = []
    for path in _FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets.extend(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets.append(node.target)
            elif isinstance(node, ast.AugAssign):
                targets.append(node.target)
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr in _STATE_FIELDS
                    and _is_state_expr(target.value)
                ):
                    violations.append(
                        f"{path.name}:{getattr(node, 'lineno', 0)}:{target.attr}"
                    )

            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            receiver = node.func.value
            if (
                isinstance(receiver, ast.Attribute)
                and receiver.attr == "nodes"
                and _is_state_expr(receiver.value)
                and node.func.attr
                in {
                    "append",
                    "clear",
                    "extend",
                    "insert",
                    "pop",
                    "remove",
                    "reverse",
                    "sort",
                }
            ):
                violations.append(
                    f"{path.name}:{getattr(node, 'lineno', 0)}:nodes.{node.func.attr}"
                )

    assert violations == []
