from __future__ import annotations

from ..model import CfgSchema
from .fields import CallbackList, SectionField
from .ports import ExpressionEvaluator, OptionProvider, ReferenceCatalog
from .targets import SettableTarget, iter_settable_targets, resolve_settable_target


class CfgDraft:
    """Owns one mutable cfg field tree and its complete refresh lifecycle."""

    def __init__(
        self,
        schema: CfgSchema,
        *,
        evaluate_expression: ExpressionEvaluator,
        provide_options: OptionProvider,
        references: ReferenceCatalog,
    ) -> None:
        self._closed = False
        self._root = SectionField(
            schema.spec,
            evaluate_expression=evaluate_expression,
            provide_options=provide_options,
            references=references,
            initial_val=schema.value,
        )
        self.on_change = CallbackList()
        self.on_validity_changed = CallbackList()
        self._root.on_change.connect(self._on_root_change)
        self._root.on_validity_changed.connect(self._on_root_validity_changed)

    @property
    def root(self) -> SectionField:
        self._require_open()
        return self._root

    def snapshot(self) -> CfgSchema:
        self._require_open()
        return CfgSchema(spec=self._root.spec, value=self._root.get_value())

    def is_valid(self) -> bool:
        self._require_open()
        return self._root.is_valid()

    def resolve_target(self, path: str) -> SettableTarget:
        self._require_open()
        return resolve_settable_target(self._root, path)

    def iter_settable_targets(self):
        self._require_open()
        yield from iter_settable_targets(self._root)

    def set_target(self, path: str, value: object) -> SettableTarget:
        target = self.resolve_target(path)
        target.set_value(value)
        return target

    def refresh_expressions(self) -> None:
        self._require_open()
        self._root.refresh_expressions()

    def refresh_options(self, source_id: str | None = None) -> None:
        self._require_open()
        self._root.refresh_options(source_id)

    def refresh_references(self, kind: str | None = None) -> None:
        self._require_open()
        self._root.refresh_references(kind)

    def close(self) -> None:
        if self._closed:
            return
        self._root.on_change.disconnect(self._on_root_change)
        self._root.on_validity_changed.disconnect(self._on_root_validity_changed)
        self._root.teardown()
        self.on_change.clear()
        self.on_validity_changed.clear()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("CfgDraft is closed")

    def _on_root_change(self) -> None:
        self.on_change.emit()

    def _on_root_validity_changed(self, valid: bool) -> None:
        self.on_validity_changed.emit(valid)
