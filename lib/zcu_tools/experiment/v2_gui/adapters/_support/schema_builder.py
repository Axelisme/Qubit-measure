"""Context-free schema authoring for measure experiment adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, Self, cast

from zcu_tools.gui.cfg import (
    CfgNodeSpec,
    CfgSchema,
    CfgSchemaAssembler,
    CfgSectionSpec,
    DirectValue,
    EvalValue,
    FloatSpec,
    IntSpec,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarLeafInput,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    read_value_path,
    resolve_spec_path,
)

from .defaults import ROLE_FACTORIES
from .seeds import Seed, SweepDefault, custom, literal, value_source
from .spec_helpers import (
    make_bath_reset_module_spec,
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_pulse_reset_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    make_two_pulse_reset_module_spec,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import ExpContext

ModuleOverrideInput = ScalarLeafInput | Seed[ScalarLeafInput]


class ModuleInit(Enum):
    """How a role-backed reference is initialized for a fresh cfg."""

    SMART = "smart"
    INLINE = "inline"
    DISABLED = "disabled"


@dataclass(frozen=True)
class _SectionDeclaration:
    path: str
    label: str


@dataclass(frozen=True)
class _FieldDeclaration:
    path: str
    spec: CfgNodeSpec
    default: Seed[object]


_Declaration = _SectionDeclaration | _FieldDeclaration


@dataclass(frozen=True)
class MeasureCfgDefinition:
    """Immutable static shape plus deferred defaults for one measure cfg."""

    _label: str
    _declarations: tuple[_Declaration, ...]
    _spec: CfgSectionSpec

    @property
    def spec(self) -> CfgSectionSpec:
        """Return an isolated copy of the context-free structural contract."""

        return deepcopy(self._spec)

    def instantiate(self, ctx: ExpContext) -> CfgSchema:
        """Materialize defaults against ``ctx`` without changing the shape."""

        assembler = CfgSchemaAssembler(
            label=self._label,
            section_labeler=_measure_section_label,
        )
        for declaration in self._declarations:
            if isinstance(declaration, _SectionDeclaration):
                assembler.ensure_section(
                    declaration.path,
                    label=declaration.label,
                )
                continue
            try:
                default = declaration.default.resolve(ctx)
                assembler.declare(
                    declaration.path,
                    deepcopy(declaration.spec),
                    default,
                )
            except Exception as exc:
                exc.add_note(
                    f"while materializing cfg path {declaration.path!r} "
                    f"from seed {declaration.default.description!r}"
                )
                raise
        schema = assembler.build()
        if schema.spec != self._spec:
            raise AssertionError(
                "MeasureCfgDefinition instantiate changed its spec shape"
            )
        return schema


class MeasureCfgBuilder:
    """One-shot domain builder that produces a context-free definition."""

    def __init__(self, *, label: str = "") -> None:
        self._label = label
        self._assembler = CfgSchemaAssembler(
            label=label,
            section_labeler=_measure_section_label,
        )
        self._declarations: list[_Declaration] = []
        self._sections: dict[str, str] = {}
        self._built = False

    def pulse(
        self,
        name: str,
        *,
        role_id: str | None = None,
        label: str = "Init Pulse",
        init: ModuleInit = ModuleInit.SMART,
        optional: bool = False,
        blank_overrides: Mapping[str, ModuleOverrideInput] | None = None,
        overrides: Mapping[str, ModuleOverrideInput] | None = None,
        locked: Mapping[str, object] | None = None,
    ) -> Self:
        return self._module(
            name,
            spec=make_pulse_module_spec(label=label, optional=optional),
            role_id=role_id or name,
            init=init,
            blank_overrides=blank_overrides,
            overrides=overrides,
            locked=locked,
        )

    def readout(
        self,
        name: str = "readout",
        *,
        role_id: str | None = None,
        label: str = "Readout",
        pulse_only: bool = False,
        init: ModuleInit = ModuleInit.SMART,
        optional: bool = False,
        blank_overrides: Mapping[str, ModuleOverrideInput] | None = None,
        overrides: Mapping[str, ModuleOverrideInput] | None = None,
        locked: Mapping[str, object] | None = None,
    ) -> Self:
        spec = (
            make_pulse_readout_module_spec(label=label, optional=optional)
            if pulse_only
            else make_readout_module_spec(label=label, optional=optional)
        )
        return self._module(
            name,
            spec=spec,
            role_id=role_id or name,
            init=init,
            blank_overrides=blank_overrides,
            overrides=overrides,
            locked=locked,
        )

    def reset(
        self,
        name: str = "reset",
        *,
        role_id: str | None = None,
        label: str = "Reset",
        shape: Literal["all", "pulse", "two_pulse", "bath"] = "all",
        init: ModuleInit = ModuleInit.SMART,
        optional: bool = False,
        blank_overrides: Mapping[str, ModuleOverrideInput] | None = None,
        overrides: Mapping[str, ModuleOverrideInput] | None = None,
        locked: Mapping[str, object] | None = None,
    ) -> Self:
        factories = {
            "all": make_reset_module_spec,
            "pulse": make_pulse_reset_module_spec,
            "two_pulse": make_two_pulse_reset_module_spec,
            "bath": make_bath_reset_module_spec,
        }
        spec = factories[shape](label=label, optional=optional)
        return self._module(
            name,
            spec=spec,
            role_id=role_id or name,
            init=init,
            blank_overrides=blank_overrides,
            overrides=overrides,
            locked=locked,
        )

    def relax_delay(
        self,
        default: ScalarLeafInput | Seed[ScalarLeafInput],
        *,
        decimals: int = 3,
    ) -> Self:
        return self.float(
            "relax_delay",
            label="Relax delay (us)",
            default=default,
            decimals=decimals,
        )

    def sweep(
        self,
        name: str,
        *,
        label: str,
        default: SweepValue | Seed[SweepValue] | SweepDefault,
        decimals: int | None = None,
        section_label: str = "Sweep",
        tooltip: str = "",
    ) -> Self:
        _validate_local_name(name, verb="sweep")
        self._ensure_section("sweep", section_label)
        if isinstance(default, SweepDefault):
            seed: Seed[SweepValue] = custom(
                default.resolve,
                description=f"sweep:{name}",
            )
        else:
            seed = _as_seed(default)
        return self.field(
            f"sweep.{name}",
            spec=SweepSpec(label=label, decimals=decimals, tooltip=tooltip),
            default=seed,
        )

    def reps(self, default: int, *, locked: bool = False) -> Self:
        spec: CfgNodeSpec = (
            LiteralSpec(value=default) if locked else IntSpec(label="Reps")
        )
        return self.field("reps", spec=spec, default=default)

    def rounds(self, default: int, *, locked: bool = False) -> Self:
        spec: CfgNodeSpec = (
            LiteralSpec(value=default) if locked else IntSpec(label="Rounds")
        )
        return self.field("rounds", spec=spec, default=default)

    def device(
        self,
        name: str,
        *,
        label: str,
        default: str | Seed[ScalarLeafInput],
        choices_source: str = "devices",
        required: bool = True,
    ) -> Self:
        _validate_local_name(name, verb="device")
        self._ensure_section("dev", "Device")
        return self.field(
            f"dev.{name}",
            spec=ScalarSpec(
                label=label,
                type=str,
                choices_source=choices_source,
                required=required,
            ),
            default=default,
        )

    def device_from_value_source(
        self,
        name: str,
        *,
        label: str,
        source_key: str,
        fallback: str,
        type_name: str | None = None,
    ) -> Self:
        return self.device(
            name,
            label=label,
            default=value_source(
                source_key,
                target_type=str,
                type_name=type_name,
                fallback=fallback,
            ),
        )

    def int(
        self,
        path: str,
        *,
        label: str,
        default: int | EvalValue | Seed[ScalarLeafInput],
        required: bool = False,
        optional: bool = False,
        tooltip: str = "",
    ) -> Self:
        return self.field(
            path,
            spec=IntSpec(
                label=label,
                required=required,
                optional=optional,
                tooltip=tooltip,
            ),
            default=default,
        )

    def float(
        self,
        path: str,
        *,
        label: str,
        default: ScalarLeafInput | Seed[ScalarLeafInput],
        decimals: int | None = None,
        required: bool = False,
        optional: bool = False,
        tooltip: str = "",
    ) -> Self:
        return self.field(
            path,
            spec=FloatSpec(
                label=label,
                decimals=decimals,
                required=required,
                optional=optional,
                tooltip=tooltip,
            ),
            default=default,
        )

    def bool(
        self,
        path: str,
        *,
        label: str,
        default: bool | Seed[ScalarLeafInput],
        tooltip: str = "",
    ) -> Self:
        return self.field(
            path,
            spec=ScalarSpec(label=label, type=bool, tooltip=tooltip),
            default=default,
        )

    def choice(
        self,
        path: str,
        *,
        label: str,
        choices: Sequence[str],
        default: str | Seed[ScalarLeafInput],
        tooltip: str = "",
    ) -> Self:
        return self.field(
            path,
            spec=ScalarSpec(
                label=label,
                type=str,
                choices=list(choices),
                tooltip=tooltip,
            ),
            default=default,
        )

    def field(
        self,
        path: str,
        *,
        spec: CfgNodeSpec,
        default: object | Seed[object],
    ) -> Self:
        self._check_mutable()
        declaration = _FieldDeclaration(
            path=path,
            spec=deepcopy(spec),
            default=_as_object_seed(default),
        )
        self._assembler.declare(path, deepcopy(spec))
        self._declarations.append(declaration)
        return self

    def build(self) -> MeasureCfgDefinition:
        self._check_mutable()
        schema = self._assembler.build()
        self._built = True
        return MeasureCfgDefinition(
            _label=self._label,
            _declarations=tuple(deepcopy(self._declarations)),
            _spec=schema.spec,
        )

    def _module(
        self,
        name: str,
        *,
        spec: ReferenceSpec,
        role_id: str,
        init: ModuleInit,
        blank_overrides: Mapping[str, ModuleOverrideInput] | None,
        overrides: Mapping[str, ModuleOverrideInput] | None,
        locked: Mapping[str, object] | None,
    ) -> Self:
        self._check_mutable()
        _validate_local_name(name, verb="module")
        if not isinstance(init, ModuleInit):
            raise TypeError(
                f"module init must be a ModuleInit, got {type(init).__name__}"
            )
        role = ROLE_FACTORIES.get(role_id)
        if role is None:
            raise ValueError(
                f"unknown role_id {role_id!r} "
                f"(available: {', '.join(sorted(ROLE_FACTORIES))})"
            )
        if role.kind != spec.kind:
            raise TypeError(
                f"module role {role_id!r} has kind {role.kind!r}, "
                f"but the spec expects {spec.kind!r}"
            )
        role_shape = role.shape()
        if not any(
            _shape_identity(allowed) == _shape_identity(role_shape)
            for allowed in spec.allowed
        ):
            expected = ", ".join(_describe_shape(shape) for shape in spec.allowed)
            actual = _describe_shape(role_shape)
            raise TypeError(
                f"cfg path 'modules.{name}' role_id {role_id!r} has incompatible "
                f"shape: expected one of [{expected}], actual {actual}"
            )
        if init is ModuleInit.DISABLED and not spec.optional:
            raise ValueError("ModuleInit.DISABLED requires an optional reference")

        blank = _normalize_overrides(blank_overrides)
        always = _normalize_overrides(overrides)
        locks = tuple((locked or {}).items())
        _validate_module_paths(
            spec, "blank_overrides", tuple(path for path, _ in blank)
        )
        _validate_module_paths(spec, "overrides", tuple(path for path, _ in always))
        _validate_module_paths(spec, "locked", tuple(path for path, _ in locks))
        overlap = set(path for path, _ in locks) & {
            path for path, _ in (*blank, *always)
        }
        if overlap:
            raise ValueError(
                "locked paths cannot also be overridden: " + ", ".join(sorted(overlap))
            )
        for relative_path, value in locks:
            spec = spec.lock_literal(relative_path, value)

        def resolve(ctx: ExpContext) -> ReferenceValue | None:
            return _materialize_module(
                ctx,
                cfg_path=f"modules.{name}",
                role_id=role_id,
                init=init,
                optional=spec.optional,
                blank_overrides=blank,
                overrides=always,
            )

        self._ensure_section("modules", "Modules")
        return self.field(
            f"modules.{name}",
            spec=spec,
            default=custom(resolve, description=f"module role:{role_id}"),
        )

    def _ensure_section(self, path: str, label: str) -> None:
        existing = self._sections.get(path)
        if existing is not None:
            if existing != label:
                raise ValueError(
                    f"cfg section {path!r} already has label {existing!r}, "
                    f"cannot replace it with {label!r}"
                )
            return
        self._assembler.ensure_section(path, label=label)
        self._declarations.append(_SectionDeclaration(path=path, label=label))
        self._sections[path] = label

    def _check_mutable(self) -> None:
        if self._built:
            raise RuntimeError("MeasureCfgBuilder is already built; create a new one")


def _materialize_module(
    ctx: ExpContext,
    *,
    cfg_path: str,
    role_id: str,
    init: ModuleInit,
    optional: bool,
    blank_overrides: tuple[tuple[str, Seed[ScalarLeafInput]], ...],
    overrides: tuple[tuple[str, Seed[ScalarLeafInput]], ...],
) -> ReferenceValue | None:
    role = ROLE_FACTORIES[role_id]
    if init is ModuleInit.DISABLED:
        node = None
    elif init is ModuleInit.INLINE:
        node = role.blank(ctx)
    elif role.ref is not None:
        node = role.ref(ctx, optional=optional)
    elif optional:
        node = None
    else:
        node = role.blank(ctx)

    if node is None:
        return None
    if not isinstance(node, ReferenceValue):
        raise TypeError(
            f"module role {role_id!r} produced {type(node).__name__}, "
            "expected ReferenceValue"
        )

    applicable = list(overrides)
    if node.chosen_key.startswith("<Custom:"):
        applicable = [*blank_overrides, *applicable]
    resolved: list[tuple[str, ScalarLeafInput]] = []
    for relative_path, seed in applicable:
        try:
            value = seed.resolve(ctx)
            if not isinstance(value, (int, float, str, bool, DirectValue, EvalValue)):
                raise TypeError(
                    f"module override {relative_path!r} resolved to "
                    f"{type(value).__name__}, expected a scalar leaf"
                )
            read_value_path(node.value, relative_path)
        except Exception as exc:
            exc.add_note(
                f"while materializing cfg path {cfg_path}.{relative_path!s} "
                f"from seed {seed.description!r}"
            )
            raise
        resolved.append((relative_path, value))
    for relative_path, value in resolved:
        node.value.with_field(relative_path, value)
    return node


def _validate_module_paths(
    spec: ReferenceSpec,
    operation: str,
    paths: tuple[str, ...],
) -> None:
    root = CfgSectionSpec(fields={"module": spec})
    for path in paths:
        if not path:
            raise ValueError(f"module {operation} path must not be empty")
        leaf = resolve_spec_path(root, f"module.{path}")
        if not isinstance(leaf, ScalarSpec):
            raise TypeError(
                f"module {operation} path {path!r} targets "
                f"{type(leaf).__name__}, expected ScalarSpec"
            )


def _normalize_overrides(
    values: Mapping[str, ModuleOverrideInput] | None,
) -> tuple[tuple[str, Seed[ScalarLeafInput]], ...]:
    return tuple((path, _as_seed(value)) for path, value in (values or {}).items())


def _as_seed(value: object | Seed[object]) -> Seed:
    return value if isinstance(value, Seed) else literal(value)


def _as_object_seed(value: object | Seed[object]) -> Seed[object]:
    return cast(Seed[object], _as_seed(value))


def _validate_local_name(name: str, *, verb: str) -> None:
    if not name or "." in name:
        raise ValueError(f"{verb} name must be one non-empty path segment: {name!r}")


def _measure_section_label(key: str) -> str:
    labels = {"modules": "Modules", "sweep": "Sweep", "dev": "Device"}
    return labels.get(key, key.replace("_", " ").title())


def _shape_identity(spec: CfgSectionSpec) -> tuple[str, object]:
    for key in ("type", "style"):
        leaf = spec.fields.get(key)
        if isinstance(leaf, LiteralSpec):
            return key, leaf.value
    return "label", spec.label


def _describe_shape(spec: CfgSectionSpec) -> str:
    key, value = _shape_identity(spec)
    return f"{spec.label!r} ({key}={value!r})"


__all__ = ["MeasureCfgBuilder", "MeasureCfgDefinition", "ModuleInit"]
