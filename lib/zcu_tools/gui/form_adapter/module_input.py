"""Generated module fields for handwritten forms, with persistent Spec projection."""

from typing import overload

from pydantic import TypeAdapter

from zcu_tools.cfg_model import ConfigBase, ModuleSpec
from zcu_tools.gui.form import (
    FormEnv,
    Option,
)

from ._modules import ModuleField
from ._seeds import persistent_input
from ._types import members, union_type
from .input_types import ModuleInline, ModuleRef

_MODULE_SPEC = TypeAdapter[ModuleSpec](ModuleSpec)


def _family_specs(family: object) -> tuple[type[ConfigBase], ...]:
    specs: list[type[ConfigBase]] = []
    for member in members(family):
        if not isinstance(member, type) or member not in members(ModuleSpec):
            raise TypeError(
                "Module input family must be a concrete ModuleSpec class or union; "
                "declare nullable separately"
            )
        if not issubclass(member, ConfigBase):
            raise TypeError("Module input family must contain persistent Specs")
        if member not in specs:
            specs.append(member)
    if not specs:
        raise ValueError("Module input family cannot be empty")
    return tuple(specs)


class ModuleInputField:
    """Generate a tagged module field for a handwritten FormModel.

    family is a public Python/Pydantic type declaration: a concrete member of
    persistent ModuleSpec (for example PulseSpec, PulseReadoutSpec or NoneResetCfg)
    or a union of those classes, including the tagged ModuleSpec/ReadoutSpec/
    ResetSpec aliases. Waveform, materialized/native cfgs, other classes and
    collections are not accepted. A nonempty family is required. nullable alone
    decides whether annotation accepts None; do not include None in family.
    ref_family uses the same declarations and must be a nonempty subset of family;
    omitted means family. It limits captured ref choices and key options only,
    not inline kinds. Invalid declarations raise TypeError, or ValueError for a
    ref family outside family, at construction, before any sources are read.

    annotation is the generated Pydantic field type: ModuleRef/ModuleInline
    subclasses selected by kind, plus None when nullable. Inline content uses
    generated tagged children and omits optional hardware fields; persistent
    conversion restores Spec defaults. Use annotation in create_model or runtime
    annotations. Static type checkers cannot reflect these generated types;
    ModuleRef | ModuleInline is their nominal value view, not a persistent Spec.
    options(env) supplies the same field's S1 key/inline/None selections and
    explicit copy command. A handwritten model declares them at its field path
    in its options hook. Existing most-local choices semantics still apply.

    Obtain values through FormState.result with the fixed captured FormEnv. For
    example a model with selection=(field.annotation, ...) and options returning
    {('selection',): field.options(env)} passes result.selection to field.to_spec.
    This object hides the same generation and projection used by core-cfg Forms;
    callers never inspect generated module content or assemble its layout.
    """

    def __init__(
        self,
        family: object,
        *,
        nullable: bool = False,
        ref_family: object | None = None,
    ) -> None:
        specs = _family_specs(family)
        refs = specs if ref_family is None else _family_specs(ref_family)
        if not set(refs).issubset(specs):
            raise ValueError("Reference family must be a subset of inline family")
        annotation = union_type((*specs, type(None))) if nullable else union_type(specs)
        self._field = ModuleField(annotation, (), None, specs=specs, ref_specs=refs)
        self.annotation: object = (
            union_type((self._field.annotation, type(None)))
            if nullable
            else self._field.annotation
        )

    def options(self, env: FormEnv) -> tuple[Option, ...]:
        """Return this field's S1 options against a fixed captured environment.

        None is first only when nullable. Captured keys use ref_family; editable
        inline kinds use family. Empty structural-only kinds remain accepted
        input but have no Inline option, as in generated core-cfg Forms. Key
        commands copy supported captured Spec content once, keeping child keys
        and discarding screened hardware values. Missing/unreadable sources
        propagate; no sources or input are mutated by declaring options.
        """
        return self._field.options(env)

    @overload
    def to_spec(self, value: ModuleRef | ModuleInline) -> str | ModuleSpec: ...

    @overload
    def to_spec(self, value: None) -> None: ...

    def to_spec(
        self, value: ModuleRef | ModuleInline | None
    ) -> str | ModuleSpec | None:
        """Project this generated field's ordinary result to persistent form.

        Pass the field value from a successful FormState.result, not input JSON
        or a different field's result. Ref returns its literal key string; inline
        returns a detached validated persistent Spec in family, preserving child
        key strings and filling screened fields from Spec defaults. None returns
        None only for nullable fields. No library, asset or native cfg is read or
        materialized; the ordinary result is not modified. Wrong generated
        classes or forbidden None raise TypeError. Invalid persistent content
        raises ValueError (including Pydantic ValidationError), without fallback.
        """
        if value is None:
            if not self._field.nullable:
                raise TypeError("None requires a nullable module input field")
            return None
        if isinstance(value, self._field.ref):
            return value.name
        if not isinstance(value, self._field.inline):
            raise TypeError("Expected this field's generated ref or inline result")
        raw = persistent_input(type(value.module), value.module.model_dump(mode="json"))
        spec = _MODULE_SPEC.validate_python(raw)
        if not isinstance(spec, self._field.specs):
            raise ValueError("Persistent Spec is outside this module input family")
        return spec
