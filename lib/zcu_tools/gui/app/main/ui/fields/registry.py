"""Registry for FieldWidgets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from ..live_model import LiveField


@runtime_checkable
class FieldWidgetProtocol(Protocol):
    """Protocol for UI widgets that represent a LiveField."""

    @property
    def field(self) -> LiveField: ...
    def refresh_section(self, path: str) -> bool: ...
    def teardown(self) -> None: ...


class FieldDecorationProtocol(Protocol):
    """Typed presentation surface consumed by field widgets."""

    @property
    def hidden(self) -> bool: ...
    @property
    def enabled(self) -> bool: ...
    @property
    def tone(self) -> str: ...
    @property
    def badge(self) -> str: ...
    @property
    def tooltip(self) -> str: ...
    @property
    def label_suffix(self) -> str: ...


T = TypeVar("T", bound=type[Any])

# Map of LiveField subclass -> Widget class
WIDGET_REGISTRY: dict[Any, Any] = {}


def register_widget(field_cls: Any):
    def wrapper(widget_cls: T) -> T:
        WIDGET_REGISTRY[field_cls] = widget_cls
        return widget_cls

    return wrapper


def get_widget_cls(field: LiveField) -> type[FieldWidgetProtocol]:
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
