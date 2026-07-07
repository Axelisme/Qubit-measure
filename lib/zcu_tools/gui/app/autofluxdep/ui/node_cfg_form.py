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
"what this run used"), preserving the prototype's lock-on-run intent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.autofluxdep.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    OverridePath,
    OverridePlan,
)
from zcu_tools.gui.app.autofluxdep.cfg.form import (
    CfgFormWidget,
    FieldDecorationPatch,
    LiveModelEnv,
    SectionLiveField,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.autofluxdep.controller import Controller
    from zcu_tools.gui.app.autofluxdep.nodes.builder import PlacedNode


NODE_FIELD_LABEL_MAX_WIDTH = 180


class NodeCfgForm(QWidget):
    """Typed cfg form for one PlacedNode.

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
        self._default_override_plan = self._node.builder.override_plan(
            self._node.schema
        )

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
        default_layout = QVBoxLayout(self._default_group)
        self._default_form = CfgFormWidget(
            field_label_max_width=NODE_FIELD_LABEL_MAX_WIDTH,
            decoration_provider=self._default_decoration_provider(),
        )
        self._default_form.attach(self._default_model)
        self._default_form.schema_changed.connect(self._on_default_schema_changed)
        default_layout.addWidget(self._default_form)
        root.addWidget(self._default_group, 1)

        self._generation_group: QGroupBox | None = None
        self._generation_form: CfgFormWidget | None = None
        if self._generation_model is not None:
            generation_group = QGroupBox("Generation overrides")
            generation_layout = QVBoxLayout(generation_group)
            self._generation_form = CfgFormWidget(
                field_label_max_width=NODE_FIELD_LABEL_MAX_WIDTH
            )
            self._generation_form.attach(self._generation_model)
            self._generation_form.schema_changed.connect(
                self._on_generation_schema_changed
            )
            generation_layout.addWidget(self._generation_form)
            self._generation_group = generation_group
            root.addWidget(generation_group, 1)

    def _on_default_schema_changed(self, schema: object) -> None:
        """Commit the form draft into the placement schema SSOT.

        ``schema`` is a fresh ``CfgSchema`` snapshot of the LiveModel; its value
        leaves (DirectValue / SweepValue) are written through the controller's
        typed entry, which coerces + fast-fails and bumps the workflow version.
        """
        del schema
        self._ctrl.set_node_cfg_value(self._index, self._combined_value())

    def _on_generation_schema_changed(self, schema: object) -> None:
        del schema
        self._ctrl.set_node_cfg_value(self._index, self._combined_value())
        self._refresh_default_decoration_provider()

    def _default_decoration_provider(self) -> _OverridePlanDecorationProvider:
        return _OverridePlanDecorationProvider(self._default_override_plan)

    def _refresh_default_decoration_provider(self) -> None:
        plan = self._node.builder.override_plan(self._node.schema)
        if plan == self._default_override_plan:
            return
        self._default_override_plan = plan
        self._default_form.set_decoration_provider(self._default_decoration_provider())

    def _combined_value(self) -> CfgSectionValue:
        """Merge the split UI drafts back into the schema's full root value tree."""
        fields = dict(self._default_model.get_value().fields)
        if self._generation_model is not None:
            fields["generation"] = self._generation_model.get_value()
        return CfgSectionValue(fields=fields)

    def set_read_only(self, read_only: bool) -> None:
        """Lock the form during a run (values stay visible, editing disabled)."""
        self._default_form.setEnabled(not read_only)
        if self._generation_form is not None:
            self._generation_form.setEnabled(not read_only)

    def refresh_external(self, event: object) -> None:
        """Refresh expression/ref snapshots after context, md, ml, or device changes."""
        self._default_model.refresh_external(event)
        if self._generation_model is not None:
            self._generation_model.refresh_external(event)

    def teardown(self) -> None:
        """Detach the CfgFormWidget + drop the LiveModel draft."""
        self._default_form.schema_changed.disconnect(self._on_default_schema_changed)
        self._default_form.detach()
        self._default_model.teardown()
        if self._generation_form is not None:
            self._generation_form.schema_changed.disconnect(
                self._on_generation_schema_changed
            )
            self._generation_form.detach()
        if self._generation_model is not None:
            self._generation_model.teardown()


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


class _OverridePlanDecorationProvider:
    def __init__(self, plan: OverridePlan) -> None:
        self._entries = {entry.path: entry for entry in plan.paths}
        self._children_by_template_path = _partial_template_children(plan)

    def decoration_for(
        self,
        path: str,
        spec: object,
        value: object,
    ) -> FieldDecorationPatch | None:
        del spec, value
        entry = self._entries.get(path)
        if entry is None:
            template_children = self._children_by_template_path.get(path)
            if template_children is None:
                return None
            return FieldDecorationPatch(
                hidden=False,
                enabled=True,
                tone="info",
                badge="template",
                tooltip=_template_tooltip(template_children),
            )
        if entry.mode == "after_first_point":
            return FieldDecorationPatch(
                hidden=False,
                enabled=True,
                tone="warning",
                badge="initial",
                tooltip=(
                    "Initial value is used at flux point 0; later points use a "
                    f"generated value. Source: {entry.source}. {entry.reason}"
                ),
            )
        if entry.mode == "fallback":
            return FieldDecorationPatch(
                hidden=False,
                enabled=True,
                tone="info",
                badge="fallback",
                tooltip=(
                    "Default value is used unless a runtime dependency provides an "
                    f"overlay value. Source: {entry.source}. {entry.reason}"
                ),
            )
        return FieldDecorationPatch(
            hidden=False,
            enabled=False,
            tone="muted",
            badge="generated",
            tooltip=(
                "Template value is stored for review; each flux point uses a "
                f"generated value. Source: {entry.source}. {entry.reason}"
            ),
        )


def _partial_template_children(
    plan: OverridePlan,
) -> dict[str, tuple[OverridePath, ...]]:
    by_path: dict[str, list[OverridePath]] = {}
    for entry in plan.paths:
        template_path = _template_module_path(entry.path)
        if template_path is None:
            continue
        by_path.setdefault(template_path, []).append(entry)
    return {
        path: tuple(sorted(children, key=lambda child: child.path))
        for path, children in by_path.items()
    }


def _template_module_path(path: str) -> str | None:
    parts = path.split(".")
    if len(parts) < 4:
        return None
    if parts[0] != "modules":
        return None
    # Readout dependencies overlay only the fallback frequency/gain/length leaves;
    # the module row remains the user-editable template for channel/trigger/etc.
    if parts[2] in {"pulse_cfg", "ro_cfg"}:
        return ".".join(parts[:2])
    return None


def _template_tooltip(children: tuple[OverridePath, ...]) -> str:
    overlays = "; ".join(_mode_group_summary(children))
    sources = ", ".join(sorted({child.source for child in children}))
    return (
        "This module is an editable template. Runtime overlay leaves: "
        f"{overlays}. Other module fields still apply. Source: {sources}."
    )


def _mode_group_summary(children: tuple[OverridePath, ...]) -> list[str]:
    summaries: list[str] = []
    for mode in ("all_points", "after_first_point", "fallback"):
        paths = [_relative_path(child.path) for child in children if child.mode == mode]
        if paths:
            summaries.append(f"{_mode_badge(mode)}: {', '.join(paths)}")
    return summaries


def _mode_badge(mode: str) -> str:
    if mode == "all_points":
        return "generated"
    if mode == "after_first_point":
        return "initial"
    if mode == "fallback":
        return "fallback"
    return mode


def _relative_path(path: str) -> str:
    parts = path.split(".")
    if len(parts) <= 2:
        return path
    return ".".join(parts[2:])
