"""Translate core cfg declarations into Qt-free input models and typed cfgs."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType, new_class
from typing import Literal

from pydantic import BaseModel, JsonValue, ValidationError
from pydantic.fields import FieldInfo

from zcu_tools.cfg_model import (
    ConfigBase,
    FormNode,
    Template,
    TemplateDef,
)
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPathError,
    FormState,
    FormWritePath,
)

from . import _logical
from ._modules import (
    ModuleField,
    ModuleValues,
    bind_template_defaults,
    editable_spec,
)
from ._sections import PERSISTENT_SPECS, Section, SpecSection
from ._seeds import supported_input, template_seed_tree
from ._seeds import template_input as _template_input
from ._types import members as _members
from .elements import ElementObservation, ElementsObservation
from .fixed_elements import FixedElements, FixedElementsForm
from .logical import LogicalEdit, LogicalEditCandidate, LogicalFieldObservation


class Form[C: ConfigBase]:
    """Generate an input model from a core cfg and its authored starting tree.

    core_cfg is the validated result type; tree keys follow its nested field
    names. Unspecified fields derive shape and defaults from core declarations.
    Hidden requires a core default; unknown keys fail with ValueError at creation.
    model is a generated FormModel class, used only through the public form API.
    default_input resolves names once against the supplied captured FormEnv;
    hosts use the same env until Reset. The host calls bind_templates with its
    template catalog before resolving Template nodes. Slot copies
    and Template defaults are detached persistent Specs, not library-editor
    structural defaults. Only Expr nodes become expression envelopes. Ordinary
    strings, including $-prefixed strings, remain literal. Expr initial values
    remove static defaults in this form only; failed evaluation requires input
    instead of falling back to core defaults. Range endpoints authored as None
    are also blank required input, never inherited 0/1 defaults.
    Optional extras adds an 'extras' section and requires
    a named lower(extras, cfg) callable returning the concrete core cfg type.
    Generated module input omits optional desc; conversion restores its
    Spec default, and copying a library entry inline discards its description.
    Templates cannot seed desc (see bind_templates); there is no override.
    Readout ro_ch remains required input; Pulse has no persistent ro_ch. FieldMeta label, unit,
    decimals, description, group and readonly become Display metadata.
    Readonly fixes the current subtree, not an ancestor's source selection.
    Generated options(env) declares nullable None, family-limited captured keys
    and every editable Inline kind. Key values use kind=ref/name; inline uses
    kind=inline/module at every depth. Inline values contain only branch tags
    and fields names the selected
    child form. The key option command to_inline (Convert to inline) copies
    supported captured Spec content in one write. Ordinary selections use fresh
    Template/type defaults, never copy the key or retain another kind's draft.
    A core_cfg that is a concrete persistent module or waveform Spec builds the
    same Spec form as module_form and requires an empty tree.
    fixed_elements is a tuple of root-relative FixedElements declarations for
    core cfg Forms. It supplies owner identity admission and participants value
    validation; default_input rejects duplicate capture names with indices.
    """

    def __init__[E: BaseModel](
        self,
        core_cfg: type[C],
        tree: Mapping[str, FormNode],
        *,
        extras: type[E] | None = None,
        lower: Callable[[E, C], C] | None = None,
        fixed_elements: tuple[FixedElements, ...] = (),
    ) -> None:
        if (extras is None) != (lower is None):
            raise ValueError("extras and lower must be provided together")
        self._core_cfg = core_cfg
        self._section: Section | SpecSection = (
            SpecSection(core_cfg, tree)
            if core_cfg in PERSISTENT_SPECS
            else Section(core_cfg, tree, ())
        )
        self.model: type[FormModel] = self._section.model
        self._lower: Callable[[BaseModel, ConfigBase], C] | None = None
        if extras is not None and lower is not None:
            if "extras" in core_cfg.model_fields:
                raise ValueError("Core cfg already declares extras")

            def lower_model(value: BaseModel, cfg: ConfigBase) -> C:
                return lower(
                    extras.model_validate(value.model_dump()["extras"]),
                    core_cfg.model_validate(cfg),
                )

            self._lower = lower_model
            parent = self.model
            namespace: dict[str, object] = {
                "__module__": __name__,
                "__annotations__": {"extras": extras},
            }
            namespace["extras"] = FieldInfo(annotation=extras, default_factory=extras)
            model = new_class(
                f"{core_cfg.__name__}WithExtrasForm",
                (parent,),
                exec_body=lambda ns: ns.update(namespace),
            )
            if not issubclass(model, FormModel):
                raise TypeError("Generated extras model must implement FormModel")
            self.model = model
            self._section.model = model
        if fixed_elements:
            if not issubclass(self.model, FixedElementsForm):
                raise ValueError("Fixed elements require a core cfg Form")
            self.model.fixed_elements = fixed_elements

    @property
    def core_cfg(self) -> type[C]:
        """Return the fixed validated cfg result type; the definition cannot retarget it."""
        return self._core_cfg

    def bind_templates(
        self, templates: Mapping[str, TemplateDef], *, adapter: str
    ) -> None:
        """Bind a detached catalog and validate authored templates before publication.

        The host's registration entry calls this explicitly. adapter is the
        registered definition name, included with the consuming field path in ValueError for unknown IDs,
        incompatible families or malformed seed trees. A template seed for an
        unsupported module field (optional desc)
        raises the same ValueError, naming the template id and the dotted seed
        path; to_cfg fills desc only from its Spec default. Seeds for other
        fields, including $-prefixed strings, stay literal. No sources are read.
        Call before default_input when the tree contains Template nodes.
        Definitions remain immutable and the caller's mapping is never retained.
        """
        self._section.bind_templates(MappingProxyType(dict(templates)), adapter)

    def default_input(self, env: FormEnv) -> FormInput[FormModel]:
        """Build fresh input from captured env; required unresolved names stay absent.

        No sources are refreshed or mutated. Source and default factory failures
        propagate. The shared core evaluates expression defaults once into direct
        input against this env. Unresolved or invalid defaults stay absent.
        """
        return self.model.default_input(env)

    def _check_logical_input(self, input: FormInput[FormModel]) -> None:
        if not issubclass(input.model, self.model):
            raise TypeError(
                "Logical operations require this Form's input or template specialization"
            )

    def read_logical_field(
        self, input: FormInput[FormModel], path: FormWritePath, env: FormEnv
    ) -> LogicalFieldObservation:
        """Read one detached logical field from input and its fixed capture.

        path omits only generator-owned wrappers; () reads the logical root.
        Traverse active structure only. Unknown/inactive/whole-leaf descendants
        raise FormPathError. Ref descendants project captured Specs, never native
        cfg or assets; unresolved sources raise LogicalSourceError. An unresolved
        target ref itself remains observable with unavailable value and errors.
        Authored input remains separate from source values. No defaults, capture,
        state mutation, publication or materialization occurs. Wrong input model
        raises TypeError; template projections preserve initial values and
        requiredness.
        Declared fixed-list descendants include the deepest matching
        FixedElements identity from this same input, never from participants.
        """
        self._check_logical_input(input)
        observation = _logical.read(input, path, env)
        if issubclass(input.model, FixedElementsForm):
            # The deepest matching declaration owns a nested fixed-list field.
            for declaration in sorted(
                input.model.fixed_elements,
                key=lambda item: len(item.list_path),
                reverse=True,
            ):
                element = declaration.identify(input.data, path)
                if element is not None:
                    return replace(observation, element=element)
        return observation

    def read_elements(
        self,
        input: FormInput[FormModel],
        path: FormWritePath,
        env: FormEnv,
        *,
        fields: tuple[FormWritePath, ...] = (),
    ) -> ElementsObservation:
        """Read every model-list element once from one detached input and capture.

        path is a nonempty canonical typed model-list path. Unknown, inactive
        or non-list targets raise FormPathError preserving the request. Wrong
        input model raises TypeError. Returns authored inputs, available preview
        leaves and directed errors in original list order, even for invalid or
        nonparticipating elements. References are not materialized. No converter,
        source refresh, mutation, notification or focus occurs.
        fields optionally requests nonempty relative logical paths per element.
        Their observations (including refs and FixedElements identity) use this
        same detached input/env and preserve request order. Invalid paths raise
        FormPathError; unavailable values keep their ordinary directed errors.
        """
        self._check_logical_input(input)
        if any(not field for field in fields):
            raise FormPathError(path, "Element fields must be nonempty relative paths")
        state = FormState(input)
        target = state.describe(path, env).target
        if target is None or target.elements is None:
            raise FormPathError(path, "Aggregate requires a model-list field")
        preview = state.preview(env)
        names = None
        if issubclass(input.model, FixedElementsForm):
            declaration = next(
                (item for item in input.model.fixed_elements if item.list_path == path),
                None,
            )
            if declaration is not None:
                names = declaration.names(input.data)
        elements: list[ElementObservation] = []
        for item in target.elements:
            raw = self._input_at(input.data, item.path)
            if not isinstance(raw, dict):
                raise FormPathError(item.path, "Expected an authored model element")
            name = names[item.index] if names is not None else item.name
            errors = _logical.errors_for_path(preview.errors, item.path)
            elements.append(
                ElementObservation(
                    item.index,
                    name if isinstance(name, str) else item.name,
                    item.path,
                    raw,
                    {
                        key: value
                        for key, value in input.selections.items()
                        if key[: len(item.path)] == item.path
                    },
                    {
                        key: value
                        for key, value in preview.values.items()
                        if key[: len(item.path)] == item.path
                    },
                    errors,
                    tuple(
                        self.read_logical_field(input, (*item.path, *field), env)
                        for field in fields
                    ),
                )
            )
        return ElementsObservation(path, tuple(elements), preview.errors)

    def prepare_logical_edits(
        self, input: FormInput[FormModel], edits: Sequence[LogicalEdit], env: FormEnv
    ) -> LogicalEditCandidate:
        """Prepare one detached canonical candidate for a sequence of logical edits.

        Resolve every path against the initial active structure. Duplicate and
        overlapping paths, ref ancestors, selector replacements and structural/
        source admission reject the entire batch with LogicalEditError and the
        original edit index. Whole replacement is also rejected when any declared
        branch, nested model or container item contains a generated selector. Use
        reachable logical children and dedicated selector operations instead.
        Model-list items expose indexed logical paths; other sequence/mapping
        items remain whole leaves. Ordinary nullable, union and container fields
        without selectors keep core write semantics.
        Ordinary invalid values and Expr remain drafts.
        No live state is installed, captured or published. Use the candidate only
        within the same guarded owner turn/input/env, without await/reset. Wrong
        model raises TypeError; declaration/provider failures retain their types.
        """
        self._check_logical_input(input)
        return _logical.edits(input, edits, env)

    def prepare_library_selection(
        self, input: FormInput[FormModel], path: FormWritePath, key: str, env: FormEnv
    ) -> LogicalEditCandidate:
        """Prepare a generated selector's actual captured key, never an alias.

        Validate active ancestry, declaration policy, captured family and allowed
        keys. Ref ancestors and illegal selections raise LogicalEditError without
        modifying input. Source/provider failures propagate. Wrong model raises
        TypeError. The result is a same-turn detached candidate, not publication.
        """
        self._check_logical_input(input)
        return _logical.select(input, path, env, action="library", key=key)

    def prepare_inline_copy(
        self, input: FormInput[FormModel], path: FormWritePath, env: FormEnv
    ) -> LogicalEditCandidate:
        """Copy this selector's captured Spec into detached inline input explicitly.

        Preserve child keys, omit unsupported fields, never materialize
        cfg/assets. An inline target returns changed=False. Missing/disabled/
        unresolved targets raise LogicalEditError; missing/incompatible captured
        sources raise LogicalSourceError. Ref ancestors must be copied first.
        No fresh-default fallback, capture, installation or publication occurs.
        Wrong model raises TypeError. Candidate lifetime is one guarded turn.
        """
        self._check_logical_input(input)
        return _logical.select(input, path, env, action="copy")

    def prepare_module_disable(
        self, input: FormInput[FormModel], path: FormWritePath, env: FormEnv
    ) -> LogicalEditCandidate:
        """Prepare null only for a nullable generated selector with writable ancestry.

        Ref ancestors, nonselectors and declaration-fixed/nonnullable selections
        raise LogicalEditError without modifying input. Wrong model raises
        TypeError. Source/declaration faults propagate. No capture, installation
        or publication occurs; use this candidate only in the same guarded turn.
        """
        self._check_logical_input(input)
        return _logical.select(input, path, env, action="disable")

    def logical_errors(
        self, input: FormInput[FormModel], errors: FormErrors | None, env: FormEnv
    ) -> FormErrors | None:
        """Translate existing canonical directed errors, without evaluating input.

        Only generated wrappers/tags disappear. Real names, container indexes
        and unknown author error suffixes remain intact; cfg paths already in
        logical space are not stripped twice. None remains None. Wrong input
        model raises TypeError. No state, sources or errors are changed.
        """
        self._check_logical_input(input)
        return _logical.logical_errors(input, errors, env)

    def _module_field(
        self, input: FormInput[FormModel], path: FormWritePath
    ) -> ModuleField:
        section = self._section
        raw: JsonValue = input.data
        remaining = list(path)
        while remaining:
            name = remaining.pop(0)
            if not isinstance(name, str) or not name:
                break
            if not remaining:
                if name in section.modules:
                    return section.modules[name]
                if name in section.model.model_fields:
                    raise ValueError(f"Not a core module field: {path!r}")
            raw = raw.get(name) if isinstance(raw, dict) else None
            if name in section.children:
                if raw is None and type(None) in _members(
                    section.model.model_fields[name].annotation
                ):
                    break
                section = section.children[name]
            elif isinstance(section, Section) and name in section.lists and remaining:
                index = remaining.pop(0)
                if (
                    type(index) is not int
                    or not isinstance(raw, list)
                    or not 0 <= index < len(raw)
                ):
                    break
                raw = raw[index]
                section = section.lists[name]
            else:
                break
        raise FormPathError(path, f"Not an active core module field: {path!r}")

    @staticmethod
    def _input_at(raw: JsonValue, path: FormWritePath) -> JsonValue:
        for part in path:
            current: JsonValue = raw
            if isinstance(current, dict) and isinstance(part, str):
                raw = current.get(part)
            elif (
                isinstance(current, list)
                and isinstance(part, int)
                and not isinstance(part, bool)
                and 0 <= part < len(current)
            ):
                raw = current[part]
            else:
                return None
        return raw

    def module_input_kind(
        self, input: FormInput[FormModel], path: FormWritePath
    ) -> Literal["ref", "inline"]:
        """Read an authored core module's tag without evaluating its draft.

        input must belong to this Form; another model raises TypeError. path
        contains nonempty core field names and int original model-list indexes,
        not inline descendants or extras.
        Unknown/nonmodule paths raise ValueError, even if their input has a
        kind tag. Return ref or inline from the public tagged input, including
        partial modules and invalid expressions. Disabled, missing or untagged
        module input raises ValueError naming the field. No validation,
        expression evaluation, source read, input change or publication occurs.
        """
        if input.model is not self.model:
            raise TypeError("module_input_kind requires this Form's input")
        self._module_field(input, path)
        raw = self._input_at(input.data, path)
        kind = raw.get("kind") if isinstance(raw, dict) else None
        if kind == "ref":
            return "ref"
        if kind == "inline":
            return "inline"
        raise ValueError(f"Module input has no ref/inline tag: {path!r}")

    def read_module(
        self, input: FormInput[FormModel], path: FormWritePath, env: FormEnv
    ) -> ModuleValues | FormErrors:
        """Read this module field's evaluated value without mutation or refresh.

        input must belong to this Form; other models raise TypeError. path uses
        nonempty core field names and int original model-list indexes.
        Unknown/inactive paths raise FormPathError with the request; a known
        nonmodule field raises ValueError.
        env is the fixed capture used to evaluate input. Key mode materializes
        its captured library Spec; inline mode evaluates this field's input.
        Other invalid draft fields do not block this read. Only this module's
        validation/source failures return FormErrors at the consuming path.
        Values exclude full-cfg validation and extras lowering: this is not the
        native cfg after to_cfg. Unexpected source exceptions propagate.
        """
        if input.model is not self.model:
            raise TypeError("read_module requires this Form's input")
        field = self._module_field(input, path)
        return field.read(self._input_at(input.data, path), env, path=path)

    def template_input(
        self,
        template: Template,
        catalog: Mapping[str, TemplateDef],
        env: FormEnv,
    ) -> FormInput[FormModel]:
        """Build fresh creation input for this persistent module or waveform Spec.

        template names a catalog ID, with optional named placeholder bindings and
        relative-path seed overrides. Explicit bindings override the fixed
        defaults in env.placeholders; missing bindings raise ValueError.
        catalog maps IDs to
        TemplateDef values. env supplies captured parameters and library for
        Expr initial values and Slot resolution. Partial seeds overlay Spec
        defaults; failed Expr initial values leave required fields unfilled.
        Bound Slots keep child library keys; unbound Slots leave required input absent.
        No native cfg or asset is materialized. to_cfg returns the original Spec.

        input.model may be a subclass of this Form's model, specialized for
        this template's Expr requirements and initial values. Build a new
        FormState from the returned input. An inline kind switch uses template
        defaults at that path only when its branch ancestry matches; other
        branches use type defaults. The converter then inherits same-name,
        same-type fields from the previous draft.
        Recompute the same template's initial values with
        input.model.default_input(env); changing templates requires a new state.
        The original Form.model, template, catalog and sources are not modified.
        The input model retains a detached catalog mapping, not the caller's map.

        Core cfg Forms raise TypeError. Unknown IDs, incompatible Spec families,
        unsupported or malformed seed paths raise
        ValueError naming the template or path. Captured-source and default
        factory failures propagate; they never silently use a fallback.
        """
        if not isinstance(self._section, SpecSection):
            raise TypeError("template_input requires a persistent Spec form")
        bound = MappingProxyType(dict(catalog))
        _template_input(template, bound, (self.model,), None)
        editable = editable_spec(self.model, template_seed_tree(template, bound))

        model = new_class(
            f"{self.model.__name__}Template",
            (editable,),
            exec_body=lambda ns: ns.update({"__module__": __name__}),
        )
        if not issubclass(model, FormModel):
            raise TypeError("Template input model must implement FormModel")
        bind_template_defaults((model,), template, bound)
        return model.default_input(env)

    def input_from_spec(self, spec: ConfigBase, env: FormEnv) -> FormInput[FormModel]:
        """Project an existing persistent Spec into fresh input for this Spec form.

        Only Forms built for a persistent Spec accept this; a core cfg Form raises
        TypeError. spec accepts the shared ConfigBase boundary so Form remains
        covariant in its concrete result type. At runtime spec must be exactly
        core_cfg, not a subclass; otherwise raise
        TypeError. Ordinary values and child library keys are copied, and None
        values stay absent like static defaults. Child keys become ref objects
        with kind=ref/name; inline children become kind=inline/module wrappers.
        Values of unsupported module
        fields are discarded, so to_cfg restores Spec defaults instead of the
        stored values. env supplies the default input that the projection
        overlays; captured source failures propagate. spec is not modified and
        no library entry, asset or native cfg is read.
        """
        if not isinstance(self._section, SpecSection):
            raise TypeError("input_from_spec requires a persistent Spec form")
        if type(spec) is not self.core_cfg:
            raise TypeError(
                f"Expected {self.core_cfg.__name__}, got {type(spec).__name__}"
            )
        state = FormState(self.default_input(env))
        dropped = state.restore_input(
            FormInput(
                state.input.model,
                supported_input(
                    self._section.editable,
                    spec.model_dump(mode="json", exclude_none=True),
                ),
            ),
            env,
        )
        if dropped:
            raise ValueError(f"Spec values do not fit generated input: {dropped}")
        return state.input

    def to_cfg(self, value: FormModel, env: FormEnv) -> C | FormErrors:
        """Convert this form's ordinary result into a validated concrete core cfg.

        value must be a result of this generated model or its template_input
        subclass; another model raises TypeError. env is the same captured
        source used by result. Core validation
        failures return FormErrors at their cfg field paths. Conversion does not
        mutate input or sources. Exceptions from the author-provided lower hook
        propagate; its result is validated again against core_cfg.
        """
        if not isinstance(value, self.model):
            raise TypeError("to_cfg requires this form's generated model result")
        errors: dict[tuple[str | int, ...], str] = {}
        raw = self._section.to_raw(value, env, errors)
        raw.pop("extras", None)
        if errors:
            return FormErrors(errors)
        try:
            cfg = self.core_cfg.model_validate(raw)
        except ValidationError as exc:
            return FormErrors(
                {tuple(error["loc"]): error["msg"] for error in exc.errors()}
            )
        if self._lower is not None:
            cfg = self.core_cfg.model_validate(self._lower(value, cfg).model_dump())
        return cfg


def module_form[S: ConfigBase](spec_type: type[S]) -> Form[S]:
    """Build the Form for one concrete persistent module or waveform Spec class.

    spec_type is a member class of ModuleSpec or WaveformSpec, such as PulseSpec
    or PulseReadoutSpec. Union aliases, materialized or native runtime cfgs and
    other classes raise TypeError. The model is a FormModel usable as a
    FormState root or a ModuleInline module field; hosts add choices through
    their own choices(env). to_cfg returns spec_type and keeps child library
    keys without reading the library, assets or building native cfg. Generated
    str-or-model children use kind=ref/name or kind=inline/module, including
    nested waveform kinds; to_cfg reverses those wrappers. Generated
    input omits optional desc fields, whose Spec
    defaults return in to_cfg. Use input_from_spec to edit an existing Spec.
    """
    if spec_type not in PERSISTENT_SPECS:
        raise TypeError(
            f"Not a concrete persistent module or waveform Spec: {spec_type!r}"
        )
    return Form(spec_type, {})
