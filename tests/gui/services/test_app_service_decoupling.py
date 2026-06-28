"""M4 — no application service depends on another concrete application service
(ADR-0005 violation 2). Orchestrators / read models depend on ports (interfaces)
declared in ports.py, which also prevents a back-edge / cycle from forming.

This is an import-discipline gate: it scans the offender modules' source for a
concrete sibling-service import and asserts there is none.

It detects BOTH relative (``from .guard import ...``) and absolute
(``from zcu_tools.gui.app.main.services.guard import ...``) sibling imports — a
sibling dependency written either way is the same coupling, and an absolute path
must not let it slip past the gate.

What does NOT count as a forbidden dependency, and why:
- **Importing a value/credential TYPE** that happens to live in a sibling module
  (e.g. ``XxxPermit`` from ``guard``). The ADR-0001 Permit dataclasses live in
  ``guard.py`` next to ``GuardService``, but importing a Permit type is not a
  runtime dependency on the GuardService — it's a typed token the orchestrator
  receives. Only importing the concrete ``*Service`` class is the coupling we ban.
- **TYPE_CHECKING-only imports** (e.g. ``WritebackService`` under
  ``if TYPE_CHECKING:`` for an annotation). These have no runtime effect; the
  runtime wiring goes through the injected port (``WritebackQueryPort``).

Coverage includes the infrastructure modules that have session-layer ports:
``operation_gate`` (port: ``ExclusionGate``), ``cfg_editor`` (port:
``CfgEditorPort``), and ``writeback`` (port: ``WritebackLifecyclePort``).  These
did NOT end in "Service" before the infra denylist was added — the
``_is_infra_symbol`` check fills that gap. (The OffMain executor is the shared
``BackgroundRunner`` in ``gui/`` — outside ``services/`` — so it was never a
guard target here.)
"""

from __future__ import annotations

import ast
from pathlib import Path

import zcu_tools.gui.app.main.services as services_pkg

assert services_pkg.__file__ is not None
_SERVICES_DIR = Path(services_pkg.__file__).parent

# The dotted prefix of the moved services package, used to recognise an absolute
# sibling import (``from zcu_tools.gui.app.main.services.<mod> import ...``).
_SERVICES_PKG = services_pkg.__name__  # "zcu_tools.gui.app.main.services"

# Modules that are application services (not ports / DTOs / infra adapters).
# Includes the infrastructure modules that own concrete classes *with* declared
# ports: operation_gate, cfg_editor.  Runtime imports of their concrete classes
# by sibling app services are violations.
_APP_SERVICE_MODULES = {
    "analyze",
    "cfg_editor",
    "connection",
    "context",
    "device",
    "guard",
    "operation_gate",
    "run",
    "save",
    "startup",
    "tab",
    "workspace",
    "writeback",
}

# Explicit denylist of concrete infra class names that have declared ports.
# Names in this set are banned even when they don't end in "Service"
# (e.g. ``OperationGate`` whose port is ``ExclusionGate``).
_CONCRETE_INFRA_NAMES: frozenset[str] = frozenset(
    {
        "CfgEditorService",
        "OperationGate",
        "WritebackService",
    }
)


def _is_service_symbol(name: str) -> bool:
    """A concrete service class (the banned coupling), not a value/credential type.

    The forbidden dependency is importing another service's *implementation*
    (``GuardService``, ``WritebackService``). Importing a credential type that
    merely lives in the same module (``AnalyzePermit``) is not — it is a typed
    token, not a handle to the service.

    Also catches the explicit infra-with-port names in ``_CONCRETE_INFRA_NAMES``
    (e.g. ``OperationGate`` which does not end in "Service").
    """
    return name.endswith("Service") or name in _CONCRETE_INFRA_NAMES


def _imported_service_modules(path: Path) -> set[str]:
    """Sibling ``services.<mod>`` whose concrete *service class* ``path`` imports.

    Covers both relative (``from .tab import TabService``) and absolute
    (``from zcu_tools.gui.app.main.services.tab import TabService``) forms.
    Skips imports inside ``if TYPE_CHECKING:`` blocks (annotation-only, no runtime
    coupling) and imports of non-service symbols (Permit/Error value types).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    # Collect the line ranges of TYPE_CHECKING blocks to exclude them.
    type_checking_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            )
            if is_tc:
                end = getattr(node, "end_lineno", node.lineno)
                type_checking_ranges.append((node.lineno, end))

    def in_type_checking(lineno: int) -> bool:
        return any(lo <= lineno <= hi for lo, hi in type_checking_ranges)

    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if in_type_checking(node.lineno):
            continue

        # Determine the sibling module name, for relative or absolute form.
        head: str | None = None
        if node.level == 1:  # from .<mod> import ...
            head = node.module.split(".")[0]
        elif node.level == 0 and node.module.startswith(_SERVICES_PKG + "."):
            # from zcu_tools.gui.app.main.services.<mod>... import ...
            rest = node.module[len(_SERVICES_PKG) + 1 :]
            head = rest.split(".")[0]
        if head not in _APP_SERVICE_MODULES:
            continue

        # Only a concrete service-class import is the banned coupling.
        if any(_is_service_symbol(alias.name) for alias in node.names):
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
        f"services (ADR-0005 violation 2). Found concrete deps: {offenders}"
    )
