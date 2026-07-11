"""Ratchet the source boundary around direct Qt imports."""

from __future__ import annotations

import ast
from pathlib import Path

GUI_ROOT = Path(__file__).resolve().parents[2] / "lib" / "zcu_tools" / "gui"
QT_PACKAGES = frozenset({"qtpy", "PyQt6", "PySide6"})

KNOWN_QT_DEBT = frozenset(
    {
        # batch 3: migrate background execution behind a Qt adapter.
        Path("background.py"),
        # batch 3: migrate connection notifications with the operation layer.
        Path("session/services/connection.py"),
        # batch 3: migrate device notifications with the operation layer.
        Path("session/services/device.py"),
        # batch 3: isolate application bootstrap Qt ownership.
        Path("app/main/app.py"),
        # batch 3: move cfg binding widget access into the UI layer.
        Path("app/main/cfg_binding.py"),
        # batch 3: migrate analysis execution with the operation layer.
        Path("app/main/services/analyze.py"),
        # batch 3: migrate post-analysis execution with the operation layer.
        Path("app/main/services/post_analyze.py"),
        # batch 3: migrate run execution with the operation layer.
        Path("app/main/services/run.py"),
        # batch 3: migrate save execution with the operation layer.
        Path("app/main/services/save.py"),
        # batch 3: migrate staged analysis with the operation layer.
        Path("app/main/services/staged_analyze.py"),
        # batch 3: replace utility-level Qt error presentation with a UI port.
        Path("app/main/utils/error_handler.py"),
        # batch 3: isolate autofluxdep controller Qt ownership.
        Path("app/autofluxdep/controller.py"),
    }
)


def _imports_qt(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.partition(".")[0] in QT_PACKAGES for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.partition(".")[0] in QT_PACKAGES:
                return True
    return False


def _is_permanently_allowed(path: Path) -> bool:
    parts = path.parts
    return (
        "ui" in parts[:-1]
        or parts[0] == "widgets"
        or path == Path("runtime.py")
        or path
        in {
            Path("plotting/backend.py"),
            Path("plotting/container.py"),
            Path("plotting/host.py"),
        }
        or (path.parent == Path("session/adapters") and path.name.startswith("qt_"))
    )


def test_direct_qt_imports_match_known_debt_exactly() -> None:
    direct_imports = {
        path.relative_to(GUI_ROOT)
        for path in GUI_ROOT.rglob("*.py")
        if _imports_qt(path)
    }
    violations = direct_imports - {
        path for path in direct_imports if _is_permanently_allowed(path)
    }

    new_debt = violations - KNOWN_QT_DEBT
    cleared_debt = KNOWN_QT_DEBT - violations
    assert violations == KNOWN_QT_DEBT, (
        "Direct Qt imports outside adapter/ui boundaries changed. "
        f"New violations (move Qt dependencies into an adapter/ui layer): "
        f"{sorted(map(str, new_debt))}; "
        f"cleared debt (remove from KNOWN_QT_DEBT): "
        f"{sorted(map(str, cleared_debt))}"
    )
