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


def _placeholder_names(text: str) -> frozenset[str]:
    """Validate placeholder syntax and collect names, including format specs."""
    names: set[str] = set()
    try:
        for _, name, spec, conversion in Formatter().parse(text):
            if name is not None and not name.isidentifier():
                raise ValueError(f"Expected a plain placeholder name, got {name!r}")
            if conversion not in (None, "s", "r", "a"):
                raise ValueError(f"Invalid conversion {conversion!r}")
            if name is not None:
                names.add(name)
            if spec:
                names.update(_placeholder_names(spec))
    except ValueError as exc:
        raise ValueError(f"Invalid template placeholder in {text!r}: {exc}") from exc
    return frozenset(names)


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
    _placeholder_names(path)
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
        # Construction and queries share the same seed traversal and syntax checks.
        self.placeholder_names()

    def placeholder_names(self) -> frozenset[str]:
        """Return the binding names needed by expanded_seeds, without expansion.

        Names come from string seeds, Expr bodies and Slot paths, including
        nested format specs. Escaped braces and consumer-specific Slot options
        contribute no names. This does not resolve aliases, read sources or
        modify the definition. Invalid placeholder syntax raises ValueError.
        """
        names: set[str] = set()
        for seed in self.seeds.values():
            if isinstance(seed, Expr):
                names.update(_placeholder_names(seed.text))
            elif isinstance(seed, Slot):
                names.update(_placeholder_names(seed.path))
            elif isinstance(seed, str):
                names.update(_placeholder_names(seed))
        return frozenset(names)

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
