"""M4 — no application service depends on another concrete application service
(ADR-0008 violation 2). Orchestrators / read models depend on ports (interfaces)
declared in ports.py, which also prevents a back-edge / cycle from forming.

This is an import-discipline gate: it scans the offender modules' source for a
concrete sibling-service import and asserts there is none.
"""

from __future__ import annotations

import ast
from pathlib import Path

import zcu_tools.gui.services as services_pkg

_SERVICES_DIR = Path(services_pkg.__file__).parent

# Modules that are application services (not ports / DTOs / infra adapters).
_APP_SERVICE_MODULES = {
    "analyze",
    "connection",
    "context",
    "device",
    "guard",
    "run",
    "save",
    "startup",
    "tab",
    "workspace",
    "writeback",
}


def _imported_service_modules(path: Path) -> set[str]:
    """Sibling .services.<mod> names imported by ``path`` (concrete deps)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        # `from .tab import TabService`  → level=1, module="tab"
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            head = node.module.split(".")[0]
            if head in _APP_SERVICE_MODULES:
                found.add(head)
    return found


def test_no_app_service_imports_another_app_service():
    offenders: dict[str, set[str]] = {}
    for mod in _APP_SERVICE_MODULES:
        path = _SERVICES_DIR / f"{mod}.py"
        if not path.exists():
            continue
        deps = _imported_service_modules(path) - {mod}
        if deps:
            offenders[mod] = deps
    assert not offenders, (
        "application services must depend on ports, not concrete sibling "
        f"services (ADR-0008 violation 2). Found concrete deps: {offenders}"
    )
