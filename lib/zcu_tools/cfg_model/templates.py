"""Persistent template data and placeholder expansion, without GUI dependencies."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from string import Formatter
from types import MappingProxyType

from .base import ConfigBase
from .module import ModuleSpec
from .spec_types import concrete_spec_types
from .template_nodes import Expr, Slot, TemplateSeed
from .waveform import WaveformSpec


def _validate_placeholders(text: str) -> None:
    try:
        for _, name, spec, conversion in Formatter().parse(text):
            if name is not None and not name.isidentifier():
                raise ValueError(f"Expected a plain placeholder name, got {name!r}")
            if conversion not in (None, "s", "r", "a"):
                raise ValueError(f"Invalid conversion {conversion!r}")
            if spec:
                _validate_placeholders(spec)
    except ValueError as exc:
        raise ValueError(f"Invalid template placeholder in {text!r}: {exc}") from exc


def expand_path(
    path: str,
    *,
    bindings: Mapping[str, str] | None = None,
) -> str:
    """Expand plain named placeholders using a fixed set of authored bindings.

    bindings supplies caller-authored replacements and defaults to empty.
    Replacements are literal text, not recursively expanded or resolved as
    aliases. Double braces
    escape literal braces. Standard string conversion and formatting are
    supported. Missing bindings, traversal fields or malformed placeholders
    raise ValueError naming the path and offending placeholder.
    """
    _validate_placeholders(path)
    try:
        return path.format_map(bindings or {})
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"Invalid template placeholder in {path!r}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class TemplateDef:
    """One immutable, caller-owned module or waveform creation template.

    ``module`` must be a concrete member of persistent ModuleSpec or
    WaveformSpec, not a composite runtime Cfg. ``label`` is dropdown text.
    ``seeds`` maps dotted paths relative to that Spec's root to literal scalars
    or Expr bodies or Slot nodes. Strings, Expr bodies and Slot paths can contain
    component placeholders; expansion preserves their distinct types. Slot's
    consumer-specific options are retained without expansion or source resolution.
    ``default_name`` suggests a library entry name; empty means no suggestion,
    including blanks.

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
            *concrete_spec_types(ModuleSpec),
            *concrete_spec_types(WaveformSpec),
        ):
            raise TypeError("Template module must be a concrete persistent Spec type")
        object.__setattr__(self, "seeds", MappingProxyType(dict(self.seeds)))
        for seed in self.seeds.values():
            if isinstance(seed, Expr):
                _validate_placeholders(seed.text)
            elif isinstance(seed, Slot):
                _validate_placeholders(seed.path)
            elif isinstance(seed, str):
                _validate_placeholders(seed)

    def expanded_seeds(
        self,
        *,
        bindings: Mapping[str, str] | None = None,
    ) -> dict[str, TemplateSeed]:
        """Return fresh seeds with component placeholders expanded.

        bindings supplies caller-authored replacements, empty when omitted.
        Missing bindings raise ValueError only at expansion, not construction.
        Ordinary strings remain literal, Expr remains Expr and
        Slot remains Slot with only its path expanded, and other scalars remain
        unchanged. Expansion does not resolve values or mutate this definition.
        Invalid placeholders raise ValueError.
        """
        expanded: dict[str, TemplateSeed] = {}
        for path, seed in self.seeds.items():
            if isinstance(seed, Expr):
                seed = Expr(expand_path(seed.text, bindings=bindings))
            elif isinstance(seed, Slot):
                seed = replace(
                    seed,
                    path=expand_path(seed.path, bindings=bindings),
                )
            elif isinstance(seed, str):
                seed = expand_path(seed, bindings=bindings)
            expanded[path] = seed
        return expanded
