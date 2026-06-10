"""ParamForm — a dynamic settings form generated from a provider's base_params.

Prototype: each ``Builder.base_params`` name becomes a labelled line edit; the
value is stored/read as text (typed widgets per param are §7 future work). The
form also shows a read-only dependency / provides summary so the user sees what
the provider consumes and produces. ``set_read_only`` locks every field during a
run (values stay visible — "what this run used"). Mirrors the fluxdep
``transitions_form`` convention (QFormLayout + per-field widget + get/set).
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.autofluxdep.nodes.builder import PlacedNode


class ParamForm(QWidget):
    """Settings form for one PlacedNode, plus a read-only dep/provides summary."""

    def __init__(self, node: PlacedNode, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._node = node
        self._edits: dict[str, QLineEdit] = {}

        root = QVBoxLayout(self)

        form = QFormLayout()
        for key in node.builder.base_params:
            edit = QLineEdit(str(node.params.get(key, "")))
            self._edits[key] = edit
            form.addRow(key, edit)
        root.addLayout(form)

        root.addWidget(_hline())
        root.addWidget(QLabel(self._summary_text()))
        root.addStretch(1)

    def _summary_text(self) -> str:
        s = self._node.builder
        req = ", ".join(d.key for d in s.requires) or "—"
        opt = (
            ", ".join(
                d.key + (f" (smooth={d.smooth})" if d.smooth else "")
                for d in s.optional
            )
            or "—"
        )
        mods = ", ".join(m.name for m in s.all_module_deps()) or "—"
        prov = ", ".join(s.provides) or "—"
        prov_m = ", ".join(s.provides_modules) or "—"
        return (
            f"requires:  {req}\n"
            f"optional:  {opt}\n"
            f"modules:   {mods}\n"
            f"provides:  {prov}\n"
            f"prov.mod:  {prov_m}"
        )

    def values(self) -> dict[str, str]:
        """Current field values (text)."""
        return {k: e.text() for k, e in self._edits.items()}

    def set_read_only(self, read_only: bool) -> None:
        for e in self._edits.values():
            e.setReadOnly(read_only)


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line
