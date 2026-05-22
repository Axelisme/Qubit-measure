"""Registry for FieldWidgets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Type

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget
    from ..live_model import LiveField

# Map of LiveField subclass -> Widget class (implementing build_widget(field))
WIDGET_REGISTRY: Dict[Type[LiveField], Type[QWidget]] = {}

def register_widget(field_cls: Type[LiveField]):
    def wrapper(widget_cls: Type[QWidget]):
        WIDGET_REGISTRY[field_cls] = widget_cls
        return widget_cls
    return wrapper

def get_widget_cls(field: LiveField) -> Type[QWidget]:
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
