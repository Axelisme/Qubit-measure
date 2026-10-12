"""Persistent template data and placeholder expansion, without GUI dependencies."""

from collections.abc import Mapping
from dataclasses import dataclass
from string import Formatter
from types import MappingProxyType
from typing import get_args

from .base import ConfigBase
from .module import ModuleSpec
from .waveform import WaveformSpec


@dataclass(frozen=True, slots=True)
class Expr:
    """Authored expression text; unlike a plain string, consumers evaluate it.

    ``text`` is the expression body, without a wire wrapper. Template expansion
    substitutes component placeholders but does not parse or evaluate the body.
    """

    text: str


type TemplateSeed = str | int | float | bool | complex | None | Expr


def expand_path(
    path: str, *, qubit: str = "$qubit", resonator: str = "$resonator"
) -> str:
    """Expand component placeholders in a slot path or seed string.

    Only ``{qubit}`` and ``{resonator}`` are accepted field names. Bindings are
    authored paths such as ``$q2``; defaults retain the primary aliases. Python
    ``str.format`` handles escaping braces and formatting. Malformed syntax or
    an unknown field raises ValueError. This function never resolves aliases.
    """
    try:
        for _, field_name, _, _ in Formatter().parse(path):
            if field_name is not None and field_name not in {"qubit", "resonator"}:
                raise ValueError(f"Unknown placeholder {field_name!r}")
        return path.format(qubit=qubit, resonator=resonator)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"Invalid template placeholder in {path!r}: {exc}") from exc


def _spec_members(annotation: object) -> tuple[type[ConfigBase], ...]:
    if isinstance(annotation, type) and issubclass(annotation, ConfigBase):
        return (annotation,)
    # Type aliases and Annotated unions can nest inside ModuleSpec.
    alias_value: object = getattr(annotation, "__value__", annotation)
    return tuple(
        member
        for argument in get_args(alias_value)
        for member in _spec_members(argument)
    )


@dataclass(frozen=True, slots=True)
class TemplateDef:
    """One immutable, caller-owned module or waveform creation template.

    ``module`` must be a concrete member of persistent ModuleSpec or
    WaveformSpec, not a composite runtime Cfg. ``label`` is dropdown text.
    ``seeds`` maps dotted paths relative to that Spec's root to literal scalars
    or Expr bodies. Both strings and Expr bodies can contain component
    placeholders; expansion preserves their distinct types. ``default_name``
    suggests a library entry name; empty means no suggestion, including blanks.

    Construction copies seeds, rejects invalid module types with TypeError and
    invalid placeholder syntax with ValueError. Paths and seed compatibility
    are validated by consumers against their own editable schema.
    """

    module: type[ConfigBase]
    label: str
    seeds: Mapping[str, TemplateSeed]
    default_name: str = ""

    def __post_init__(self) -> None:
        if self.module not in (
            *_spec_members(ModuleSpec),
            *_spec_members(WaveformSpec),
        ):
            raise TypeError("Template module must be a concrete persistent Spec type")
        object.__setattr__(self, "seeds", MappingProxyType(dict(self.seeds)))
        self.expanded_seeds()

    def expanded_seeds(
        self, *, qubit: str = "$qubit", resonator: str = "$resonator"
    ) -> dict[str, TemplateSeed]:
        """Return fresh seeds with component placeholders expanded.

        Defaults use primary aliases; pass authored paths to bind another pair
        of components. Ordinary strings remain literal, Expr remains Expr and
        other scalars remain unchanged. Expansion does not resolve values or
        mutate this definition. Invalid placeholders raise ValueError.
        """
        expanded: dict[str, TemplateSeed] = {}
        for path, seed in self.seeds.items():
            if isinstance(seed, Expr):
                seed = Expr(expand_path(seed.text, qubit=qubit, resonator=resonator))
            elif isinstance(seed, str):
                seed = expand_path(seed, qubit=qubit, resonator=resonator)
            expanded[path] = seed
        return expanded
