"""Headless editing of one persistent library entry through the shared Form.

The shared Form generator builds the form from the entry's concrete Spec
class, so the family tag is fixed and child library keys stay keys. A form
captures its FormEnv once; text expressions evaluate against that capture
when the Spec is produced, so the library stores numbers. Submission yields a
LibraryChange for the session library write contract; this module writes
nothing itself and does not depend on any app.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from types import UnionType, new_class
from typing import (
    Annotated,
    ClassVar,
    Literal,
    Self,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, JsonValue

from zcu_tools.cfg_model import (
    ArbWaveformCfg,
    BathResetSpec,
    ConfigBase,
    ConstWaveformCfg,
    CosineWaveformCfg,
    DragWaveformCfg,
    FlatTopWaveformSpec,
    GaussWaveformCfg,
    ModuleSpec,
    MuxReadoutSpec,
    NoneResetCfg,
    PulseReadoutSpec,
    PulseResetSpec,
    PulseSpec,
    Template,
    TemplateDef,
    TwoPulseResetSpec,
    WaveformSpec,
)
from zcu_tools.gui.expected_error import ExpectedError, InvalidInputError
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPath,
    FormState,
)
from zcu_tools.gui.form_adapter import module_form
from zcu_tools.gui.session.context_types import LibraryChange
from zcu_tools.gui.session.parameters import ParameterContext

type LibraryKind = Literal["module", "waveform"]

# What a rejected put/replace/delete raises through apply_library_change and
# put/replace: ExpectedError for the target, selection or invalid form input,
# KeyError/ValueError for a missing, taken or malformed root, OSError for
# storage. A dialog that reports a rejected write catches exactly these.
LIBRARY_WRITE_ERRORS = (ExpectedError, KeyError, ValueError, OSError)

_MODULE_SPECS = (
    PulseSpec,
    PulseReadoutSpec,
    MuxReadoutSpec,
    NoneResetCfg,
    PulseResetSpec,
    TwoPulseResetSpec,
    BathResetSpec,
)
_WAVEFORM_SPECS = (
    ConstWaveformCfg,
    CosineWaveformCfg,
    GaussWaveformCfg,
    DragWaveformCfg,
    ArbWaveformCfg,
    FlatTopWaveformSpec,
)


class LibraryEntryInvalid(InvalidInputError):
    """A library entry form was submitted while its input is invalid.

    errors holds every message by form field path, as FormState.result reports
    them; an empty path is a model-wide error. The exception text lists them
    as "dotted.path: message". Nothing was written.
    """

    def __init__(self, kind: LibraryKind, errors: FormErrors) -> None:
        self.errors = errors
        details = "; ".join(
            f"{'.'.join(str(part) for part in path) or kind}: {message}"
            for path, message in errors.messages.items()
        )
        super().__init__(f"Invalid {kind} entry: {details}")


class LibraryEntryForm[S: ConfigBase](ABC):
    """Form editing state for one persistent module or waveform Spec.

    Use ModuleEntryForm or WaveformEntryForm for a Spec in hand, such as a
    writeback proposal, or open_library_entry for a library root. kind is
    "module" or "waveform", the library family that put/replace write.
    parameters is the fixed capture given at construction; env is the FormEnv
    built from it once (its parameters and library). state is the FormState
    whose base and input start as the Spec's projection. Callers write
    state with env, for example state.write(batch, env) or a FormEditor
    built on (state, env); Revert is state.reset(state.base). Discarding the
    form is dropping it: it holds no session or resource. Single-threaded.
    """

    kind: ClassVar[LibraryKind]

    def __init__(
        self, spec: S, parameters: ParameterContext, *, arb_names: Sequence[str]
    ) -> None:
        """Project spec into a new form that evaluates against parameters.

        spec is a concrete persistent module or waveform Spec of this form's
        family; it is not modified. Child library keys stay keys. The shared
        generator leaves out optional desc fields; the produced
        Spec has their defaults, not spec's values. arb_names are the arb
        waveform asset keys captured by the host: every arb waveform data
        field in this Spec, at the root or inline, accepts only these names.
        An empty sequence accepts none.
        A spec that is not a concrete persistent Spec raises TypeError.
        """
        self._form = module_form(type(spec))
        self._parameters = parameters
        self._env = FormEnv(parameters=parameters, library=parameters.library)
        model = _with_arb_choices(self._form.model, type(spec), tuple(arb_names))
        projected = self._form.input_from_spec(spec, self._env)
        self._state: FormState[FormModel] = FormState(FormInput(model, projected.data))

    @classmethod
    def from_template(
        cls,
        spec_type: type[S],
        template: Template,
        catalog: Mapping[str, TemplateDef],
        parameters: ParameterContext,
        *,
        arb_names: Sequence[str],
    ) -> Self:
        """Build a fresh entry state from a template for this family's spec_type.

        template names a catalog definition matching spec_type. The public
        generator resolves Expr and Slot seeds against parameters. Its returned
        specialized input model owns the new state; no existing state is reset.
        arb_names supplies the same captured choices as the Spec constructor.
        Bad templates raise the generator's errors; incomplete inputs remain
        editable and spec()/put() reject them with LibraryEntryInvalid.
        """
        # Template input may be incomplete, so there is no Spec for __init__ yet.
        entry = cls.__new__(cls)
        entry._form = module_form(spec_type)
        entry._parameters = parameters
        entry._env = FormEnv(parameters=parameters, library=parameters.library)
        projected = entry._form.template_input(template, catalog, entry._env)
        model = _with_arb_choices(projected.model, spec_type, tuple(arb_names))
        entry._state = FormState(FormInput(model, projected.data))
        return entry

    @property
    def parameters(self) -> ParameterContext:
        """Return the capture that fixes expression values and the write target."""
        return self._parameters

    @property
    def env(self) -> FormEnv:
        """Return the FormEnv captured at construction; use it for every state call."""
        return self._env

    @property
    def state(self) -> FormState[FormModel]:
        """Return the editing state; its base is the projection of the opened Spec."""
        return self._state

    def result(self) -> S | FormErrors:
        """Evaluate the current input with env into the Spec, or field errors.

        Text expressions become numbers from the captured parameters. Nothing
        is written and input is unchanged.
        """
        value = self._state.result(self._env)
        if isinstance(value, FormErrors):
            return value
        return self._form.to_cfg(value, self._env)

    def spec(self) -> S:
        """Return the valid Spec; invalid input raises LibraryEntryInvalid."""
        value = self.result()
        if isinstance(value, FormErrors):
            raise LibraryEntryInvalid(self.kind, value)
        return value

    def put(self, name: str) -> LibraryChange:
        """Return a put change that writes the current Spec at root name.

        Put creates or overwrites name. Apply it with parameters through the
        session library write contract (apply_library_change). Invalid input
        raises LibraryEntryInvalid; nothing is written here.
        """
        return self._change("put", name, None)

    def replace(self, source: str, new_name: str) -> LibraryChange:
        """Return a replace change that moves root source to new_name with this Spec.

        source must exist; new_name may equal source. The library owner
        rejects a missing source or an occupied different destination when the
        change is applied, in one transaction. Invalid input raises
        LibraryEntryInvalid; nothing is written here.
        """
        return self._change("replace", source, new_name)

    @abstractmethod
    def _change(
        self, action: Literal["put", "replace"], path: str, new_path: str | None
    ) -> LibraryChange:
        """Build the family's put/replace change from spec()."""


class ModuleEntryForm(LibraryEntryForm[ModuleSpec]):
    """Library entry form for a module Spec: pulse, readout or reset."""

    kind = "module"

    def _change(
        self, action: Literal["put", "replace"], path: str, new_path: str | None
    ) -> LibraryChange:
        return LibraryChange("module", action, path, new_path, module=self.spec())


class WaveformEntryForm(LibraryEntryForm[WaveformSpec]):
    """Library entry form for a waveform Spec."""

    kind = "waveform"

    def _change(
        self, action: Literal["put", "replace"], path: str, new_path: str | None
    ) -> LibraryChange:
        return LibraryChange("waveform", action, path, new_path, waveform=self.spec())


def library_entry_kind(spec_type: type[ConfigBase]) -> LibraryKind:
    """Return the library family that stores entries of spec_type.

    spec_type is a concrete persistent module Spec (pulse, readout or reset)
    or waveform Spec, such as a TemplateDef's module. Any other class raises
    TypeError.
    """
    if issubclass(spec_type, _MODULE_SPECS):
        return "module"
    if issubclass(spec_type, _WAVEFORM_SPECS):
        return "waveform"
    raise TypeError(
        "Template must define a concrete persistent module or waveform Spec"
    )


def new_library_entry(
    parameters: ParameterContext,
    template: Template,
    catalog: Mapping[str, TemplateDef],
    *,
    arb_names: Sequence[str],
) -> ModuleEntryForm | WaveformEntryForm:
    """Build an editable library entry from the injected persistent template.

    Capture, input lifecycle and arb choices follow from_template. Unknown IDs
    raise KeyError; a definition outside both families raises TypeError (see
    library_entry_kind); invalid definitions propagate without writing any
    library. The returned entry produces the same Spec put/replace contract
    as an opened entry. The caller owns its state and commits explicitly.
    """
    spec_type = catalog[template.id].module
    if issubclass(spec_type, _MODULE_SPECS):
        return ModuleEntryForm.from_template(
            spec_type, template, catalog, parameters, arb_names=arb_names
        )
    if issubclass(spec_type, _WAVEFORM_SPECS):
        return WaveformEntryForm.from_template(
            spec_type, template, catalog, parameters, arb_names=arb_names
        )
    raise TypeError(
        "Template must define a concrete persistent module or waveform Spec"
    )


def open_library_entry(
    parameters: ParameterContext,
    kind: LibraryKind,
    name: str,
    *,
    arb_names: Sequence[str],
) -> ModuleEntryForm | WaveformEntryForm:
    """Open library root name of kind from the captured library as a form.

    parameters is the fixed capture whose library supplies the stored Spec;
    the form keeps it for evaluation and as the write target. arb_names
    follows LibraryEntryForm. An unknown kind, or a name that is missing or
    not a root of that kind, raises InvalidInputError. No I/O occurs.
    """
    library = parameters.library
    if kind == "module":
        spec = _read_entry(library.module_spec, kind, name)
        return ModuleEntryForm(spec, parameters, arb_names=arb_names)
    if kind == "waveform":
        spec = _read_entry(library.waveform_spec, kind, name)
        return WaveformEntryForm(spec, parameters, arb_names=arb_names)
    raise InvalidInputError(f"Unknown library kind: {kind!r}")


def _read_entry[T](read: Callable[[str], T], kind: LibraryKind, name: str) -> T:
    # The library raises KeyError for a missing root, ValueError for a bad path.
    try:
        return read(name)
    except (KeyError, ValueError) as exc:
        raise InvalidInputError(f"Unknown {kind} entry {name!r}: {exc}") from exc


def _with_arb_choices(
    model: type[FormModel], spec_type: type[BaseModel], arb_names: tuple[str, ...]
) -> type[FormModel]:
    """Derive model with arb_names as the options of every arb data leaf.

    Hosts supply options through choices(env); the generated model keeps its
    own options for other paths.
    """
    paths = _arb_data_paths(spec_type, ())

    def choices(
        cls: type[FormModel], env: FormEnv
    ) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        del cls
        return {**model.choices(env), **dict.fromkeys(paths, arb_names)}

    derived = new_class(
        f"{model.__name__}WithArbData",
        (model,),
        exec_body=lambda ns: ns.update(
            {"__module__": __name__, "choices": classmethod(choices)}
        ),
    )
    if not issubclass(derived, FormModel):
        raise TypeError("Derived library entry model must implement FormModel")
    return derived


def _arb_data_paths(
    model: type[BaseModel], prefix: tuple[str, ...]
) -> list[tuple[str, ...]]:
    """List the data field path of every arb waveform reachable from model."""
    if issubclass(model, ArbWaveformCfg):
        return [(*prefix, "data")]
    paths: list[tuple[str, ...]] = []
    for name, info in model.model_fields.items():
        for child in _model_members(info.annotation):
            child_path = (*prefix, name)
            if child in _MODULE_SPECS or child in _WAVEFORM_SPECS:
                # Generated inline Spec children live under a module wrapper.
                child_path = (*child_path, "module")
            paths.extend(_arb_data_paths(child, child_path))
    return paths


def _model_members(annotation: object) -> tuple[type[BaseModel], ...]:
    """Return the Pydantic models an annotation can hold, through aliases and unions."""
    if isinstance(annotation, TypeAliasType):
        return _model_members(annotation.__value__)
    origin = get_origin(annotation)
    if origin is Annotated:
        return _model_members(get_args(annotation)[0])
    if origin in (Union, UnionType):
        return tuple(
            model for member in get_args(annotation) for model in _model_members(member)
        )
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return (annotation,)
    return ()
