"""SMART module defaults shared by library creation and deferred cfg seeds."""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import TypeAlias

from zcu_tools.cfg_model import Expr, Slot, Template, TemplateDef, TemplateSeed
from zcu_tools.experiment.cfg_editing import PROGRAM_SHAPES
from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
    select_ref_value_spec,
)
from zcu_tools.gui.session.parameters import ModuleSlotError
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import RoleResolutionError

from .cfg_schemas import module_cfg_to_value
from .specs import MAIN_PROGRAM_SPEC_POLICY

LibraryReferenceFallback: TypeAlias = (
    Template | Callable[[SessionEnv], ReferenceValue] | None
)


def resolve_library_reference(
    path: str,
    env: SessionEnv,
    *,
    kind: str,
    fallback: LibraryReferenceFallback,
    templates: Mapping[str, TemplateDef] | None = None,
) -> ReferenceValue | None:
    """Build a fresh SMART reference from one fixed parameter/library capture.

    ``path`` is an authored component/role module slot or concrete library root.
    ``env`` supplies the fixed capture; no live selection or hardware read occurs.
    Module materialization uses that capture's library and asset capabilities.
    ``kind`` must be "module"; other kinds and empty paths raise ValueError
    before lookup. ``fallback`` is a pure factory called only for bound unset,
    null or dangling slots, missing direct roots, or a missing concrete capture.
    None disables such a reference. Its result is detached before returning.
    Alternatively, pass ``Template(id)`` and ``templates``, the injected mapping
    of IDs to TemplateDef. Missing bound sources then expand that definition's
    seeds into a fresh inline Custom reference, preserving Expr as EvalValue.
    Template.set/lock and nondefault qubit/resonator bindings raise TypeError;
    only default component placeholders are expanded. Unknown IDs, recursive
    fallbacks and invalid seed paths raise ValueError; incompatible module kinds
    and unsupported nested Slot options raise TypeError. Template fallbacks are
    validated before lookup, even if a source is available. A Template without
    ``templates`` raises ValueError. Callable fallbacks do not require a mapping.

    Unbound aliases, including $ paths without a capture, instead keep a linked
    empty section without calling fallback. Available modules retain the authored
    path and carry their existing materialized cfg values. Source failures,
    invalid paths, malformed modules and asset/materialization errors propagate;
    only ModuleSlotError triggers fallback. Subsequent source changes do not
    mutate returned inputs; callers may resolve again against a new capture.
    """
    if kind != "module":
        raise ValueError(f"Unsupported reference kind {kind!r}; expected 'module'")
    if not path.strip():
        raise ValueError("Library reference path must not be empty")
    factory = _fallback_factory(fallback, templates)
    parameters = env.parameters
    if parameters is None and path.startswith("$"):
        return ReferenceValue(chosen_key=path, value=CfgSectionValue())
    if parameters is not None:
        try:
            key = parameters.library_key(path)
        except RoleResolutionError:
            # Keep the authored link so refresh can later supply its source.
            return ReferenceValue(chosen_key=path, value=CfgSectionValue())
        except ModuleSlotError:
            # Only the path owner's explicit unset/dangling slot error recovers.
            key = None
        if key is not None:
            try:
                parameters.library.module_spec(key)
            except KeyError:
                # A valid direct root can name a not-yet-created entry.
                pass
            else:
                _, value = module_cfg_to_value(parameters.library.get_module(key))
                return ReferenceValue(chosen_key=path, value=value)
    return deepcopy(factory(env)) if factory is not None else None


def _fallback_factory(
    fallback: LibraryReferenceFallback,
    templates: Mapping[str, TemplateDef] | None,
) -> Callable[[SessionEnv], ReferenceValue] | None:
    if not isinstance(fallback, Template):
        return fallback
    if templates is None:
        raise ValueError("Template fallback requires an injected templates mapping")
    definition = templates.get(fallback.id)
    if definition is not None and "type" not in definition.module.model_fields:
        raise TypeError(
            f"Template {fallback.id!r} must define a module, not a waveform"
        )
    make_library_template_value(fallback, templates)

    def seeded_value(env: SessionEnv) -> ReferenceValue:
        return make_library_template_value(fallback, templates, env=env)

    return seeded_value


def make_library_template_value(
    template: Template,
    templates: Mapping[str, TemplateDef],
    *,
    env: SessionEnv | None = None,
) -> ReferenceValue:
    """Expand an injected definition into fresh inline library editing values.

    ``template`` names a persistent module or waveform Spec definition in
    ``templates``. Only Template(id) with default component bindings is accepted;
    set/lock or nondefault qubit/resonator bindings raise TypeError. Definitions'
    component placeholders expand to $qubit/$resonator; literal seeds stay direct
    and Expr stays EvalValue.
    ``env`` supplies the fixed capture for nested Slot defaults. With None, validate
    seeds and every fallback without reading sources; nested Slots use defaults.
    Raise ValueError for unknown IDs, recursive fallbacks or invalid seed paths,
    and TypeError for incompatible module families or unsupported Slot options.
    No definition or capture is mutated. The root always uses an inline Custom key.
    """
    return _template_reference(template, templates, None, (), env)


def _template_reference(
    template: Template,
    templates: Mapping[str, TemplateDef],
    allowed: Sequence[CfgSectionSpec] | None,
    active: tuple[str, ...],
    env: SessionEnv | None,
) -> ReferenceValue:
    if (
        template.set
        or template.lock
        or template.qubit != "$qubit"
        or template.resonator != "$resonator"
    ):
        raise TypeError(
            f"Template {template.id!r} supports only Template(id), without "
            "set/lock/qubit/resonator overrides"
        )
    # Nested Slot fallbacks expand other templates; active IDs reject cycles.
    if template.id in active:
        raise ValueError(f"Recursive template fallback: {(*active, template.id)!r}")
    if template.id not in templates:
        raise ValueError(f"Unknown template {template.id!r}")
    definition = templates[template.id]
    kind = "module" if "type" in definition.module.model_fields else "waveform"
    key = "type" if kind == "module" else "style"
    discriminator = definition.module.model_fields[key].default
    if not isinstance(discriminator, str):
        raise TypeError(f"Template {template.id!r} requires a string discriminator")
    if allowed is not None and not any(
        isinstance(literal := spec.fields.get(key), LiteralSpec)
        and literal.value == discriminator
        for spec in allowed
    ):
        raise TypeError(
            f"Template {template.id!r} has incompatible module {definition.module.__name__}"
        )
    spec = PROGRAM_SHAPES.get(kind, discriminator).make_spec(MAIN_PROGRAM_SPEC_POLICY)
    seeds = definition.expanded_seeds(
        qubit=template.qubit, resonator=template.resonator
    )
    value = _seeded_section(
        spec, seeds, templates, (*active, template.id), f"Template {template.id!r}", env
    )
    return ReferenceValue(make_custom_reference_key(spec.label), value)


def _seeded_section(
    spec: CfgSectionSpec,
    seeds: Mapping[str, TemplateSeed],
    templates: Mapping[str, TemplateDef],
    active: tuple[str, ...],
    location: str,
    env: SessionEnv | None = None,
) -> CfgSectionValue:
    value = make_default_value(spec)
    unknown = {path.split(".")[0] for path in seeds} - spec.fields.keys()
    if unknown:
        raise ValueError(f"{location}: Unknown template seed fields: {sorted(unknown)}")
    for key, node_spec in spec.fields.items():
        child_seeds = {
            path[len(key) + 1 :]: seed
            for path, seed in seeds.items()
            if path.startswith(f"{key}.")
        }
        seed = seeds.get(key)
        if isinstance(seed, Slot):
            if child_seeds:
                raise ValueError(f"Slot seed {key!r} cannot also have inline children")
            try:
                value.fields[key] = _slot_reference(
                    node_spec, value.fields[key], seed, templates, env
                )
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"{location}.{key}: {exc}") from exc
            continue
        if isinstance(node_spec, ReferenceSpec) and child_seeds:
            current = value.fields[key]
            selected = _seed_reference_shape(node_spec, current, child_seeds)
            value.fields[key] = ReferenceValue(
                make_custom_reference_key(selected.label),
                _seeded_section(
                    selected, child_seeds, templates, active, f"{location}.{key}", env
                ),
            )
        elif isinstance(node_spec, CfgSectionSpec) and child_seeds:
            value.fields[key] = _seeded_section(
                node_spec, child_seeds, templates, active, f"{location}.{key}", env
            )
        elif child_seeds:
            raise TypeError(f"Template seed field {key!r} has no children")
        if key in seeds:
            if isinstance(node_spec, (ReferenceSpec, CfgSectionSpec)):
                raise TypeError(f"Template seed field {key!r} requires child paths")
            if isinstance(node_spec, LiteralSpec) and seed != node_spec.value:
                raise ValueError(f"Template seed cannot change literal {key!r}")
            value.with_field(
                key,
                EvalValue(seed.text) if isinstance(seed, Expr) else DirectValue(seed),
            )
    return value


def _slot_reference(
    spec: object,
    current: object,
    slot: Slot,
    templates: Mapping[str, TemplateDef],
    env: SessionEnv | None,
) -> ReferenceValue | None:
    if not isinstance(spec, ReferenceSpec):
        raise TypeError("Slot seed requires a reference field")
    if spec.kind != "module":
        raise TypeError("Library Slot seeds support only module references")
    if slot.copy or slot.set or slot.lock:
        raise TypeError("Library seeds support Slot(path) without copy/set/lock")
    if not isinstance(current, ReferenceValue) and current is not None:
        raise TypeError("Slot field requires a structural reference default")
    if env is None:
        # Startup validates syntax/shape without consulting a session source.
        return current
    fallback: LibraryReferenceFallback = None
    if fallback is None and isinstance(current, ReferenceValue):

        def structural_default(_env: SessionEnv) -> ReferenceValue:
            return current

        fallback = structural_default
    return resolve_library_reference(
        slot.path, env, kind="module", fallback=fallback, templates=templates
    )


def _seed_reference_shape(
    spec: ReferenceSpec,
    current: object,
    seeds: Mapping[str, TemplateSeed],
) -> CfgSectionSpec:
    discriminator_key = spec.discriminator
    if discriminator_key is None:
        raise ValueError("Template reference shape requires a discriminator")
    discriminator = seeds.get(discriminator_key)
    if discriminator is not None:
        for allowed in spec.allowed:
            literal = allowed.fields.get(discriminator_key)
            if isinstance(literal, LiteralSpec) and literal.value == discriminator:
                return allowed
        raise ValueError(f"Unknown template seed shape {discriminator!r}")
    if isinstance(current, ReferenceValue):
        return select_ref_value_spec(spec, current)
    return spec.allowed[0]
