"""NodeCfgForm — the typed cfg form for one PlacedNode's user knobs.

Replaces the prototype's text ``ParamForm`` (Phase 160b): the node's knobs are
now a typed ``NodeCfgSchema`` (the per-placement SSOT), so the form reuses the
measure-app cfg form machinery via the ``cfg/form`` seam — a ``SectionLiveField``
LiveModel over the placement's schema value tree, rendered by ``CfgFormWidget``.
This gives int/float spin widgets, a 3-field sweep editor, optional-blank → None,
and string scalars (the by-name waveform) for free, all WYSIWYG.

Edits flow back to the SSOT: ``CfgFormWidget.schema_changed`` fires a fresh draft
snapshot, and this widget writes each leaf into the placement schema through the
controller's typed ``set_node_params`` entry (a main-thread State write that bumps
the workflow version + emits ``WorkflowChangedPayload``). The LiveModel is a local
draft (like measure's inspect / writeback dialogs), not auto-committed by the
framework — this widget owns the commit.

``set_read_only`` disables the whole form during a run (values stay visible —
"what this run used"), preserving the prototype's lock-on-run intent. A read-only
dependency / provides summary footer is kept for at-a-glance context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.autofluxdep.cfg.form import (
    CfgFormWidget,
    LiveModelEnv,
    SectionLiveField,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.autofluxdep.controller import Controller
    from zcu_tools.gui.app.autofluxdep.nodes.builder import PlacedNode


class NodeCfgForm(QWidget):
    """Typed cfg form for one PlacedNode, plus a read-only dep/provides summary.

    Owns a ``SectionLiveField`` draft over the placement's schema and a
    ``CfgFormWidget`` rendering it; on edit it commits the changed leaves back to
    the placement's schema SSOT via ``controller.set_node_params``.
    """

    def __init__(
        self,
        controller: Controller,
        node: PlacedNode,
        index: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self._node = node
        self._index = index

        root = QVBoxLayout(self)

        # LiveModel draft over the placement's schema (spec + its current value
        # tree). The env fetches md/ml through the controller (ControllerProtocol);
        # the flat node schema carries no md-reference / ref fields, so only
        # get_bus is exercised in practice.
        self._model = SectionLiveField(
            node.schema.schema.spec,
            LiveModelEnv(ctrl=controller),
            node.schema.schema.value,
        )
        self._form = CfgFormWidget()
        self._form.attach(self._model)
        self._form.schema_changed.connect(self._on_schema_changed)
        root.addWidget(self._form, 1)

        root.addWidget(_hline())
        root.addWidget(QLabel(self._summary_text()))
        root.addStretch(0)

    def _on_schema_changed(self, schema: object) -> None:
        """Commit the form draft into the placement schema SSOT.

        ``schema`` is a fresh ``CfgSchema`` snapshot of the LiveModel; its value
        leaves (DirectValue / SweepValue) are written through the controller's
        typed entry, which coerces + fast-fails and bumps the workflow version.
        """
        from zcu_tools.gui.app.autofluxdep.cfg import CfgSchema

        assert isinstance(schema, CfgSchema)
        params = dict(schema.value.fields)
        self._ctrl.set_node_params(self._index, params)

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

    def set_read_only(self, read_only: bool) -> None:
        """Lock the form during a run (values stay visible, editing disabled)."""
        self._form.setEnabled(not read_only)

    def teardown(self) -> None:
        """Detach the CfgFormWidget + drop the LiveModel draft."""
        self._form.schema_changed.disconnect(self._on_schema_changed)
        self._form.detach()
        self._model.teardown()


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line
