"""Recording adapters for section-refresh behavior tests."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import MagicMock

from qtpy.QtWidgets import QWidget
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.cfg.binding import CfgField, ReferenceField, ScalarField
from zcu_tools.gui.widgets.cfg import CfgFormWidget, FieldDecorationPatch
from zcu_tools.gui.widgets.cfg.registry import (
    FieldRenderContext,
    FieldRendererRegistry,
    FieldWidgetProtocol,
    default_cfg_renderers,
)


class RecordingRenderers:
    """Delegate to real editors and retain observations from the renderer seam."""

    def __init__(self) -> None:
        self.widgets: dict[str, list[QWidget]] = defaultdict(list)
        self.fields: dict[str, CfgField] = {}
        self.defaults = default_cfg_renderers()
        self.registry = (
            FieldRendererRegistry()
            .register(ScalarField, self.render)
            .register(ReferenceField, self.render)
            .freeze()
        )

    def render(
        self, field: CfgField, context: FieldRenderContext
    ) -> FieldWidgetProtocol:
        widget = self.defaults.resolve(field)(field, context)
        assert isinstance(widget, QWidget)
        self.widgets[context.path].append(widget)
        self.fields[context.path] = field
        return widget


@dataclass(frozen=True)
class BadgeProvider:
    path: str
    badge: str

    def decoration_for(
        self, path: str, spec: object, value: object
    ) -> FieldDecorationPatch | None:
        if path == self.path:
            return FieldDecorationPatch(badge=self.badge)
        return None


@contextmanager
def attached_form(
    schema: CfgSchema, ctrl: MagicMock, rendering: RecordingRenderers
) -> Generator[CfgFormWidget]:
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget(renderers=rendering.registry)
    try:
        form.attach(draft)
        yield form
    finally:
        form.detach()
        draft.close()
        form.deleteLater()
