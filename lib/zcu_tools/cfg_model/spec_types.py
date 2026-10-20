"""Persistent Spec type inspection shared by templates and library consumers."""

from typing import TypeIs, get_args

from .base import ConfigBase
from .module import ModuleSpec
from .waveform import WaveformSpec


def concrete_spec_types(annotation: object) -> tuple[type[ConfigBase], ...]:
    """Return the concrete ConfigBase classes declared by a Spec annotation.

    Pass ModuleSpec, WaveformSpec, a nested Spec union or a concrete Spec class.
    This function centralizes recursive type-alias and Annotated expansion so
    consumers need not know how persistent Spec families are composed. Classes
    retain declaration order; subclasses not declared in the annotation are not
    added. Non-model leaves, including annotation metadata, contribute no types.
    Nothing is instantiated or validated, and no I/O occurs.
    """
    # Type aliases may resolve to a concrete Spec rather than a union.
    alias_value: object = getattr(annotation, "__value__", annotation)
    if isinstance(alias_value, type) and issubclass(alias_value, ConfigBase):
        return (alias_value,)
    return tuple(
        member
        for argument in get_args(alias_value)
        for member in concrete_spec_types(argument)
    )


def is_module_spec_type(spec_type: type[ConfigBase]) -> TypeIs[type[ModuleSpec]]:
    """Return whether spec_type is a persistent module Spec class or subclass.

    Pass a ConfigBase class, not an instance. Membership derives from ModuleSpec,
    including its nested families; runtime-only cfg classes return False.
    The class is not instantiated or validated. A non-class argument raises
    TypeError. No I/O occurs.
    """
    return issubclass(spec_type, concrete_spec_types(ModuleSpec))


def is_waveform_spec_type(spec_type: type[ConfigBase]) -> TypeIs[type[WaveformSpec]]:
    """Return whether spec_type is a persistent waveform Spec class or subclass.

    Pass a ConfigBase class, not an instance. Membership derives from WaveformSpec;
    runtime-only cfg classes return False. Shared scalar Spec/Cfg classes return
    True. The class is not instantiated or validated. A non-class argument
    raises TypeError. No I/O occurs.
    """
    return issubclass(spec_type, concrete_spec_types(WaveformSpec))
