"""NodeCfgForm — the typed cfg form for one PlacedNode's user knobs.

Replaces the prototype's text ``ParamForm`` (Phase 160b): the node's knobs are
now a typed ``NodeCfgSchema`` (the per-placement SSOT), so the form reuses the
measure-app cfg form machinery via the ``cfg/form`` seam. The placement's schema
value tree is split into a main "Default cfg" form and, when present, a
"Generation overrides" form, each backed by its own ``SectionLiveField`` and
rendered by ``CfgFormWidget``. This gives int/float spin widgets, a 3-field sweep
editor, optional-blank → None, and adapter-native module/waveform refs for free,
all WYSIWYG.

Edits flow back to the SSOT: ``CfgFormWidget.schema_changed`` fires a fresh draft
snapshot, and this widget writes the complete value tree into the placement schema
through the controller (a main-thread State write that bumps the workflow version
+ emits ``WorkflowChangedPayload``). The LiveModel is a local draft (like measure's
inspect / writeback dialogs), not auto-committed by the framework — this widget
owns the commit.

``set_read_only`` disables the whole form during a run (values stay visible —
"what this run used"), preserving the prototype's lock-on-run intent. A read-only
dependency / provides summary footer is kept for at-a-glance context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QFrame,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.autofluxdep.cfg import CfgSectionSpec, CfgSectionValue
from zcu_tools.gui.app.autofluxdep.cfg.form import (
    CfgFormWidget,
    LiveModelEnv,
    SectionLiveField,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.autofluxdep.controller import Controller
    from zcu_tools.gui.app.autofluxdep.nodes.builder import PlacedNode


NODE_FIELD_LABEL_MAX_WIDTH = 180
DEFAULT_CFG_BLOCK_MAX_HEIGHT = 520
GENERATION_BLOCK_MIN_HEIGHT = 180
GENERATION_BLOCK_MAX_HEIGHT = 340


class NodeCfgForm(QWidget):
    """Typed cfg form for one PlacedNode, plus a read-only dep/provides summary.

    Owns split ``SectionLiveField`` drafts over the placement's schema and
    ``CfgFormWidget`` renderers for them; on edit it commits the merged leaves back
    to the placement's schema SSOT via ``controller.set_node_cfg_value``.
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
        root.setSpacing(8)

        spec = node.schema.schema.spec
        value = node.schema.schema.value
        default_spec, default_value, generation_spec, generation_value = (
            _split_generation_section(spec, value)
        )

        # LiveModel draft over the placement's schema (spec + its current value
        # tree). The env fetches md/value sources through the controller so
        # numeric knobs can use the shared expression mode and @{...} resolver.
        env = LiveModelEnv(ctrl=controller)
        self._default_model = SectionLiveField(default_spec, env, default_value)
        self._generation_model: SectionLiveField | None = (
            SectionLiveField(generation_spec, env, generation_value)
            if generation_spec is not None
            else None
        )

        self._default_group = QGroupBox("Default cfg")
        self._default_group.setMaximumHeight(DEFAULT_CFG_BLOCK_MAX_HEIGHT)
        default_layout = QVBoxLayout(self._default_group)
        self._default_form = CfgFormWidget(
            field_label_max_width=NODE_FIELD_LABEL_MAX_WIDTH
        )
        self._default_form.attach(self._default_model)
        self._default_form.schema_changed.connect(self._on_schema_changed)
        default_layout.addWidget(self._default_form)
        root.addWidget(self._default_group, 3)

        self._generation_group: QGroupBox | None = None
        self._generation_form: CfgFormWidget | None = None
        if self._generation_model is not None:
            generation_group = QGroupBox("Generation overrides")
            generation_group.setMinimumHeight(GENERATION_BLOCK_MIN_HEIGHT)
            generation_group.setMaximumHeight(GENERATION_BLOCK_MAX_HEIGHT)
            generation_layout = QVBoxLayout(generation_group)
            self._generation_form = CfgFormWidget(
                field_label_max_width=NODE_FIELD_LABEL_MAX_WIDTH
            )
            self._generation_form.attach(self._generation_model)
            self._generation_form.schema_changed.connect(self._on_schema_changed)
            generation_layout.addWidget(self._generation_form)
            self._generation_group = generation_group
            root.addWidget(generation_group, 1)

        root.addWidget(_hline())
        root.addWidget(QLabel(self._summary_text()))
        root.addStretch(0)

    def _on_schema_changed(self, schema: object) -> None:
        """Commit the form draft into the placement schema SSOT.

        ``schema`` is a fresh ``CfgSchema`` snapshot of the LiveModel; its value
        leaves (DirectValue / SweepValue) are written through the controller's
        typed entry, which coerces + fast-fails and bumps the workflow version.
        """
        del schema
        self._ctrl.set_node_cfg_value(self._index, self._combined_value())

    def _combined_value(self) -> CfgSectionValue:
        """Merge the split UI drafts back into the schema's full root value tree."""
        fields = dict(self._default_model.get_value().fields)
        if self._generation_model is not None:
            fields["generation"] = self._generation_model.get_value()
        return CfgSectionValue(fields=fields)

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
        self._default_form.setEnabled(not read_only)
        if self._generation_form is not None:
            self._generation_form.setEnabled(not read_only)

    def teardown(self) -> None:
        """Detach the CfgFormWidget + drop the LiveModel draft."""
        self._default_form.schema_changed.disconnect(self._on_schema_changed)
        self._default_form.detach()
        self._default_model.teardown()
        if self._generation_form is not None:
            self._generation_form.schema_changed.disconnect(self._on_schema_changed)
            self._generation_form.detach()
        if self._generation_model is not None:
            self._generation_model.teardown()


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def _split_generation_section(
    spec: CfgSectionSpec, value: CfgSectionValue
) -> tuple[
    CfgSectionSpec,
    CfgSectionValue,
    CfgSectionSpec | None,
    CfgSectionValue | None,
]:
    """Split root cfg into default sections and the generation override section."""
    generation_node = spec.fields.get("generation")
    generation_value = value.fields.get("generation")
    default_spec = CfgSectionSpec(
        fields={
            key: field for key, field in spec.fields.items() if key != "generation"
        },
        label=spec.label,
    )
    default_value = CfgSectionValue(
        fields={
            key: field for key, field in value.fields.items() if key != "generation"
        }
    )
    if not isinstance(generation_node, CfgSectionSpec):
        return default_spec, default_value, None, None
    if generation_value is not None and not isinstance(
        generation_value, CfgSectionValue
    ):
        raise TypeError(
            "generation section value must be CfgSectionValue, "
            f"got {type(generation_value).__name__}"
        )
    generation_spec = CfgSectionSpec(
        fields=dict(generation_node.fields),
        label="",
        inherit_hook=generation_node.inherit_hook,
    )
    return (
        default_spec,
        default_value,
        generation_spec,
        cast(CfgSectionValue | None, generation_value),
    )
