"""Registry for FieldWidgets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Protocol, Type, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget
    from ..live_model import LiveField


@runtime_checkable
class FieldWidgetProtocol(Protocol):
    """Protocol for UI widgets that represent a LiveField."""

    @property
    def field(self) -> LiveField: ...
    def teardown(self) -> None: ...


T = TypeVar("T", bound=Type[Any])

# Map of LiveField subclass -> Widget class
WIDGET_REGISTRY: Dict[Any, Any] = {}


def register_widget(field_cls: Any):
    def wrapper(widget_cls: T) -> T:
        WIDGET_REGISTRY[field_cls] = widget_cls
        return widget_cls

    return wrapper


def get_widget_cls(field: LiveField) -> Type[FieldWidgetProtocol]:
    cls = type(field)
    if cls in WIDGET_REGISTRY:
        return WIDGET_REGISTRY[cls]

    # Fallback for inheritance and deferred registration (string keys)
    for base, widget_cls in WIDGET_REGISTRY.items():
        if isinstance(base, type) and isinstance(field, base):
            return widget_cls
        if isinstance(base, str) and type(field).__name__ == base:
            return widget_cls

    raise TypeError(f"No widget registered for field type: {cls}")
