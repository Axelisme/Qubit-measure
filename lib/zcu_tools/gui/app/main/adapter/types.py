from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass, field, replace
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Protocol,
    Self,
    TypeAlias,
)

from typing_extensions import TypeVar

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.experiment.cfg_model import ExpCfgModel
    from zcu_tools.meta_tool import ModuleLibrary
    from zcu_tools.meta_tool.metadict import MetaDict
    from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor


# ``SocProtocol``/``SocCfgProtocol`` + the ``SocHandle``/``SocCfgHandle`` aliases,
# ``ContextReadiness``, and ``ExpContext`` are session-core value types — they live
# in ``gui/session/types`` (no experiment cfg-tree coupling). They are re-exported
# from the adapter package because ``ExpAdapterProtocol``'s signatures speak in
# them (``make_default_cfg(ctx: ExpContext)``, ``RunRequest.soc: SocHandle``).
from zcu_tools.gui.session.types import ExpContext, SocCfgHandle, SocHandle

T_Result = TypeVar("T_Result")
T_Cfg = TypeVar("T_Cfg", bound="ExpCfgModel")
T_Cfg_contra = TypeVar("T_Cfg_contra", bound="ExpCfgModel", contravariant=True)
T_Result_co = TypeVar("T_Result_co")


class ExperimentProtocol(Protocol, Generic[T_Cfg_contra, T_Result_co]):
    """Structural contract for experiment classes referenced by adapters."""

    def run(self, soc: Any, soccfg: Any, cfg: T_Cfg_contra) -> T_Result_co: ...

    def save(self, filepath: str, result: T_Result_co) -> None: ...


class AnalysisMode(Enum):
    """How an adapter's run result is turned into an analysis.

    - ``NONE``: no analysis (a raw 2D/1D acquisition — flux_dep / power_dep).
    - ``FIT``: a deterministic fit computed on a worker; the result is ready when
      the synchronous ``analyze`` returns.
    - ``INTERACTIVE``: the result is produced by the user interacting with the
      plot (e.g. picking flux sweet-spot lines); it is deferred until the user
      finishes, so the agent polls for it rather than getting it synchronously.
    """

    NONE = "none"
    FIT = "fit"
    INTERACTIVE = "interactive"


@dataclass(frozen=True)
class AdapterCapabilities:
    """Declared capability flags an adapter exposes to the framework."""

    requires_soc: bool = True
    analysis: AnalysisMode = AnalysisMode.FIT
    # ``post_analysis``: the adapter offers a *second* analysis layer that runs on
    # top of the primary ``analyze`` result (e.g. single-shot multi-backend
    # discrimination). Opt-in: an adapter declaring it must override
    # ``post_analyze_spec`` / ``post_analyze``; the framework only routes
    # post-analysis to a tab whose adapter sets this True AND whose primary
    # analyze result exists.
    post_analysis: bool = False


@dataclass(frozen=True)
class AdapterGuide:
    """Human-facing orientation for an adapter — read *before* running it.

    Prose, not a contract: it helps an agent/user grasp the experiment's
    behaviour and the adapter's built-in recommendations at a glance, but how
    they actually use it is their call. Written in present tense at the
    intent level (e.g. "assumes the resonator frequency is already calibrated"),
    NOT as a literal md-key list or concrete numbers — those would rot against
    the imperative ``md_get_*`` reads / default values they describe.
    """

    behavior: str  # what this experiment measures and roughly how it runs
    expects_md: str  # what it assumes is already in the MetaDict (intent level)
    expects_ml: str  # what it assumes is in the ModuleLibrary
    typical_writeback: str  # what a completed run tends to propose writing back
    recommended: str  # recommended analysis settings + typical usage (when/why)


def _is_json_safe(val: object) -> bool:
    """Return True if val can be serialized to JSON without loss."""
    if isinstance(val, (int, float, str, bool, type(None))):
        return True
    if isinstance(val, dict):
        return all(isinstance(k, str) and _is_json_safe(v) for k, v in val.items())
    if isinstance(val, (list, tuple)):
        return all(_is_json_safe(v) for v in val)
    return False


class AnalyzeResultBase:
    """Mixin providing a default to_summary_dict() via dataclass reflection.

    Automatically skips fields whose values are not JSON-safe (Figure, ndarray,
    etc.). Adapter authors may override for custom formatting.
    """

    def to_summary_dict(self) -> dict[str, object]:
        import dataclasses

        result: dict[str, object] = {}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            if _is_json_safe(val):
                result[f.name] = val
        return result


class AnalyzeResultWithFigure(Protocol):
    @property
    def figure(self) -> Figure | None: ...

    def to_summary_dict(self) -> dict[str, object]: ...


@dataclass
class NoAnalyzeParams:
    """Default analyze-params type for adapters without analysis."""


@dataclass
class NoAnalysisResult(AnalyzeResultBase):
    """Default analyze-result type for adapters without analysis."""

    figure: Figure | None = None


# PEP 696 defaults: adapters without analysis omit the last two generic args
# (BaseAdapter[Cfg, Result]) and these No* types fill in automatically.
T_AnalyzeResult = TypeVar(
    "T_AnalyzeResult", bound=AnalyzeResultWithFigure, default=NoAnalysisResult
)
T_AnalyzeParams = TypeVar("T_AnalyzeParams", default=NoAnalyzeParams)


@dataclass(frozen=True)
class RunRequest:
    md: MetaDict
    ml: ModuleLibrary
    soc: SocHandle | None
    soccfg: SocCfgHandle | None


@dataclass(frozen=True)
class LoadDataRequest:
    data_path: str
    md: MetaDict
    ml: ModuleLibrary


@dataclass(frozen=True)
class AnalyzeRequest(Generic[T_Result, T_AnalyzeParams]):
    run_result: T_Result
    analyze_params: T_AnalyzeParams
    md: MetaDict
    ml: ModuleLibrary
    predictor: FluxoniumPredictor | None


# ---------------------------------------------------------------------------
# Post-analysis (AdapterCapabilities.post_analysis) — a second analysis layer
# that operates *on top of* the primary ``analyze`` result. It mirrors the
# primary analyze chain (request + figure-carrying result), but carries the
# primary ``analyze_result`` in addition to the raw ``run_result`` because a
# post-analysis typically refines/recomputes from the primary fit (centres,
# threshold, …) plus the raw shots.
# ---------------------------------------------------------------------------


class PostAnalyzeResultBase(AnalyzeResultBase):
    """Mixin for post-analysis results — same JSON-safe ``to_summary_dict`` as
    :class:`AnalyzeResultBase`. A distinct type so the framework / adapters never
    confuse a post-analysis result with a primary analyze result, even though the
    summary projection is identical."""


# Post-analysis result / params type vars, mirroring T_AnalyzeResult /
# T_AnalyzeParams. Both carry PEP 696 defaults so they may trail the
# default-bearing T_AnalyzeResult in PostAnalyzeRequest's parameter list.
T_PostAnalyzeResult = TypeVar(
    "T_PostAnalyzeResult",
    bound=AnalyzeResultWithFigure,
    default=NoAnalysisResult,
)
T_PostAnalyzeParams = TypeVar("T_PostAnalyzeParams", default=NoAnalyzeParams)


@dataclass(frozen=True)
class PostAnalyzeRequest(Generic[T_Result, T_AnalyzeResult, T_PostAnalyzeParams]):
    """Mirror of :class:`AnalyzeRequest` for the post-analysis layer.

    Carries the raw ``run_result`` AND the primary ``analyze_result`` (the
    post-analysis depends on the primary fit being present), plus the
    post-analysis params and the md/ml/predictor context.
    """

    run_result: T_Result
    analyze_result: T_AnalyzeResult
    post_analyze_params: T_PostAnalyzeParams
    md: MetaDict
    ml: ModuleLibrary
    predictor: FluxoniumPredictor | None


# ---------------------------------------------------------------------------
# Interactive analysis (AnalysisMode.INTERACTIVE) — the result is produced by
# the user interacting with the plot, deferred until they finish.
# ---------------------------------------------------------------------------

_T_Bg = TypeVar("_T_Bg")


class InteractiveHost(Protocol):
    """Host-side capabilities an interactive analysis ``InteractiveSession`` draws
    on. The GUI implements it (a Qt canvas + worker pool); the adapter's session
    is handed one and calls these to draw / repaint / offload a heavy step,
    without knowing it is Qt. Adding a host capability extends this Protocol — not
    ``setup_interactive_analysis``'s signature."""

    @property
    def figure(self) -> Figure:
        """The matplotlib Figure the session draws its interactive plot on."""
        ...

    def redraw(self) -> None:
        """Repaint the canvas (and refresh any status display). The session calls
        it whenever IT decides the view should update — fine-grained, mid-action."""
        ...

    def run_background(
        self, compute: Callable[[], _T_Bg], on_done: Callable[[_T_Bg], None]
    ) -> None:
        """Run ``compute()`` off the main thread, then call ``on_done(result)`` on
        the MAIN thread. The session offloads exactly the heavy inner step(s) it
        chooses; the rest of the action flow stays on the main thread."""
        ...


class InteractiveSession(Protocol):
    """An in-progress interactive analysis the user drives on the plot. Created by
    ``adapter.setup_interactive_analysis(req, host)``; the GUI host forwards
    pointer events + action-button clicks to it and, on Done, calls ``finish()``.
    Qt-free: it deals in matplotlib coordinates + a host port, never Qt widgets."""

    def on_press(self, x: float | None) -> None: ...
    def on_move(self, x: float | None) -> None: ...
    def on_release(self, x: float | None, y: float | None) -> None: ...

    def actions(self) -> list[tuple[str, str]]:
        """``[(action_id, label)]`` — generic toolbar buttons the host renders; on
        click the host calls ``invoke_action(action_id)``. The host never learns
        what an action does."""
        ...

    def invoke_action(self, action_id: str) -> None: ...

    def info_text(self) -> str:
        """A status line the host displays verbatim (it does not interpret it)."""
        ...

    def finish(self) -> AnalyzeResultBase:
        """Build the analysis result from the user's current selection (called on
        Done). The result flows through the same path as a FIT analyze result."""
        ...


@dataclass(frozen=True)
class SaveDataRequest(Generic[T_Result]):
    run_result: T_Result
    data_path: str
    md: MetaDict
    ml: ModuleLibrary
    chip_name: str
    qub_name: str
    res_name: str
    active_label: str
    comment: str = ""


@dataclass(frozen=True)
class WritebackRequest(Generic[T_Result, T_AnalyzeResult]):
    run_result: T_Result
    analyze_result: T_AnalyzeResult
    ctx: ExpContext


@dataclass
class WritebackItem(ABC):
    # ``target_name`` is the apply destination name (md attr / ml module / ml
    # waveform — the item subtype names which namespace). It is mutable: agent/UI
    # may retarget the writeback before applying.
    target_name: str
    description: str
    # ``session_id`` (``<kind>-<n>``, e.g. ``md-1``) is the stable identifier for
    # UI/wire/dedup. Stamped once by WritebackService when items are computed at
    # analyze time (never by the adapter), and decoupled from target_name so
    # retargeting does not change the id.
    session_id: str = field(default="", init=False)
    selected: bool = field(default=True, init=False)


@dataclass
class MetaDictWriteback(WritebackItem):
    proposed_value: Any


@dataclass
class ModuleWriteback(WritebackItem):
    edit_schema: CfgSchema | None = None
    # editor_id of the service-owned (gc=False) cfg model that holds this item's
    # live draft (ADR-0008). Stamped by WritebackService at compute time; the
    # agent edits via editor.set_field(editor_id, …), the user's Edit dialog
    # attaches to the same model.
    editor_id: str | None = field(default=None, init=False)


@dataclass
class WaveformWriteback(WritebackItem):
    edit_schema: CfgSchema | None = None
    editor_id: str | None = field(default=None, init=False)


@dataclass
class SavePaths:
    data_path: str
    image_path: str


def default_value_for_type(type_: type) -> object:
    defaults: dict[type, object] = {int: 0, float: 0.0, bool: False, str: ""}
    return defaults.get(type_, None)


# ---------------------------------------------------------------------------
# Spec tree — static, defined by Adapter, never mutated
# ---------------------------------------------------------------------------

# A transform applied to the leaf spec node reached by a dotted path. Returns a
# replacement node (e.g. a LiteralSpec for lock_literal).
_LeafTransform = Callable[["CfgNodeSpec"], "CfgNodeSpec"]


def _split_spec_path(path: str) -> list[str]:
    parts = [p for p in path.split(".") if p]
    if not parts:
        raise RuntimeError("Spec override path must not be empty")
    return parts


def _path_exists(spec: CfgSectionSpec, parts: list[str]) -> bool:
    """True if the dotted ``parts`` resolve to a leaf within ``spec`` (descending
    CfgSectionSpec.fields and ModuleRefSpec/WaveformRefSpec.allowed). Used by the
    duck-type descent to decide which allowed shapes contain a path."""
    head, rest = parts[0], parts[1:]
    child = spec.fields.get(head)
    if child is None:
        return False
    if not rest:
        return True
    if isinstance(child, CfgSectionSpec):
        return _path_exists(child, rest)
    if isinstance(child, (ModuleRefSpec, WaveformRefSpec)):
        return any(_path_exists(shape, rest) for shape in child.allowed)
    return False


@dataclass(frozen=True)
class ScalarSpec:
    label: str
    type: type
    editable: bool = True
    choices: list | None = None
    choices_source: Literal["", "arb_waveforms"] = ""
    decimals: int | None = None
    required: bool = False
    # ``optional``: the field may be left empty (value ``None``) and is *valid*
    # while empty — at lowering an unset optional scalar is omitted so the model
    # default (typically ``None``) applies (e.g. PulseCfg.mixer_freq). This is
    # the opposite of ``required`` (which forces a value: empty = invalid), so
    # the two are mutually exclusive.
    optional: bool = False
    # ``group``: pure presentation hint — fields sharing a non-empty group label
    # render together under a collapsible sub-header (e.g. "Advanced"). It does
    # NOT nest the value tree; the field stays a flat leaf of its section.
    group: str = ""
    tooltip: str = ""

    def __post_init__(self) -> None:
        if self.required and self.optional:
            raise RuntimeError(
                f"ScalarSpec {self.label!r}: 'required' and 'optional' are "
                "mutually exclusive"
            )


def IntSpec(
    label: str,
    *,
    editable: bool = True,
    choices: list | None = None,
    required: bool = False,
    optional: bool = False,
    group: str = "",
    tooltip: str = "",
) -> ScalarSpec:
    """Sugar for ``ScalarSpec(label=..., type=int)`` — an integer field.

    A thin, explicit factory (not a default ``type``): callers see ``IntSpec``
    and know it is int, with no hidden default to remember. Mirrors the
    ``ScalarSpec`` fields relevant to integers (``decimals`` is float-only).
    """
    return ScalarSpec(
        label=label,
        type=int,
        editable=editable,
        choices=choices,
        required=required,
        optional=optional,
        group=group,
        tooltip=tooltip,
    )


def FloatSpec(
    label: str,
    *,
    decimals: int | None = None,
    editable: bool = True,
    choices: list | None = None,
    required: bool = False,
    optional: bool = False,
    group: str = "",
    tooltip: str = "",
) -> ScalarSpec:
    """Sugar for ``ScalarSpec(label=..., type=float)`` — a float field.

    Explicit counterpart to :func:`IntSpec`; carries the float-only ``decimals``.
    """
    return ScalarSpec(
        label=label,
        type=float,
        decimals=decimals,
        editable=editable,
        choices=choices,
        required=required,
        optional=optional,
        group=group,
        tooltip=tooltip,
    )


@dataclass(frozen=True)
class LiteralSpec:
    """A fixed-value field: no widget shown, value is always spec.value."""

    value: Any
    label: str = ""


@dataclass(frozen=True)
class SweepSpec:
    label: str = "Sweep"
    editable: bool = True
    decimals: int | None = None
    tooltip: str = ""


@dataclass(frozen=True)
class ModuleRefSpec:
    allowed: list[CfgSectionSpec]
    label: str = "Module"
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.allowed:
            raise RuntimeError("ModuleRefSpec.allowed must be non-empty")

    def lock_literal(self, path: str, value: object) -> Self:
        """Lock a leaf of this ref's allowed shapes (path is relative to the
        shape, e.g. ``pulse_cfg.freq``). Lets an adapter lock fields on the
        sub-tree as it is built, instead of from the root section. Returns a new
        frozen ModuleRefSpec; chains stay on this type."""
        return self._with_override(
            _split_spec_path(path), lambda leaf: LiteralSpec(value=value)
        )

    def _with_override(self, parts: list[str], fn: _LeafTransform) -> Self:
        # Duck-type descent: apply to every allowed shape that contains the path,
        # skip those that don't. Fail only if no allowed shape matches (real typo).
        new_allowed: list[CfgSectionSpec] = []
        matched = False
        for shape in self.allowed:
            if _path_exists(shape, parts):
                new_allowed.append(shape._with_override(parts, fn))
                matched = True
            else:
                new_allowed.append(shape)
        if not matched:
            allowed_labels = ", ".join(s.label for s in self.allowed)
            raise RuntimeError(
                f"Spec override path {'.'.join(parts)!r} not found in any allowed "
                f"shape of ModuleRefSpec (allowed: {allowed_labels})"
            )
        return replace(self, allowed=new_allowed)


@dataclass(frozen=True)
class WaveformRefSpec:
    allowed: list[CfgSectionSpec]
    label: str = "Waveform"
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.allowed:
            raise RuntimeError("WaveformRefSpec.allowed must be non-empty")

    def lock_literal(self, path: str, value: object) -> Self:
        """Lock a leaf of this ref's allowed shapes (path is relative to the
        shape, e.g. ``length``). Symmetric with ``ModuleRefSpec.lock_literal`` so
        a leaf nested inside a waveform ref can be locked from the parent spec.
        Returns a new frozen WaveformRefSpec; chains stay on this type."""
        return self._with_override(
            _split_spec_path(path), lambda leaf: LiteralSpec(value=value)
        )

    def _with_override(self, parts: list[str], fn: _LeafTransform) -> Self:
        # Duck-type descent: apply to every allowed shape that contains the path,
        # skip those that don't. Fail only if no allowed shape matches (real typo).
        new_allowed: list[CfgSectionSpec] = []
        matched = False
        for shape in self.allowed:
            if _path_exists(shape, parts):
                new_allowed.append(shape._with_override(parts, fn))
                matched = True
            else:
                new_allowed.append(shape)
        if not matched:
            allowed_labels = ", ".join(s.label for s in self.allowed)
            raise RuntimeError(
                f"Spec override path {'.'.join(parts)!r} not found in any allowed "
                f"shape of WaveformRefSpec (allowed: {allowed_labels})"
            )
        return replace(self, allowed=new_allowed)


@dataclass(frozen=True)
class CfgSectionSpec:
    fields: dict[str, CfgNodeSpec] = field(default_factory=dict)
    label: str = ""
    inherit_hook: (
        Callable[[CfgSectionValue, CfgSectionSpec], CfgSectionValue | None] | None
    ) = None

    # -- fluent spec overrides (return a new frozen spec; never mutate) -------
    #
    # Used inside an adapter's ``cfg_spec()`` to lock/restrict a deep leaf of a
    # spec tree returned by a shared helper. The result MUST be the value that
    # ``cfg_spec()`` returns — locking is part of the spec contract, and
    # ``cfg_spec`` is the sole owner of that contract. Locking the return value
    # of ``cfg_spec()`` from outside leaks the contract to the call site.

    def lock_literal(self, path: str, value: object) -> Self:
        """Replace the scalar leaf at ``path`` with a fixed ``LiteralSpec(value)``.

        The locked field shows no widget and always lowers to ``value`` (notebook
        ``freq: 0.0, # not used``). Returns a new frozen spec.
        """
        return self._with_override(
            _split_spec_path(path), lambda leaf: LiteralSpec(value=value)
        )

    def _with_override(self, parts: list[str], fn: _LeafTransform) -> Self:
        head, rest = parts[0], parts[1:]
        if head not in self.fields:
            raise RuntimeError(
                f"Spec override path segment {head!r} not found "
                f"(available: {', '.join(self.fields)})"
            )
        child = self.fields[head]
        if not rest:
            new_child: CfgNodeSpec = fn(child)
        elif isinstance(child, (CfgSectionSpec, ModuleRefSpec, WaveformRefSpec)):
            new_child = child._with_override(rest, fn)
        else:
            raise RuntimeError(
                f"Spec override path cannot descend into {type(child).__name__} "
                f"at segment {head!r}"
            )
        return replace(self, fields={**self.fields, head: new_child})


@dataclass(frozen=True)
class ChoiceBinding:
    """One selector-driven variant list inside a ChoiceSectionSpec.

    ``choices`` maps a selector value to the section spec whose fields should be
    visible for that value. The owner ChoiceSectionSpec still owns the complete
    union ``fields``; the choice specs are the display contract.
    """

    selector_key: str
    choices: Mapping[str, CfgSectionSpec]

    def __post_init__(self) -> None:
        if not self.selector_key:
            raise RuntimeError("ChoiceBinding.selector_key must be non-empty")
        if not self.choices:
            raise RuntimeError("ChoiceBinding.choices must be non-empty")

    def controlled_field_keys(self) -> set[str]:
        keys: set[str] = set()
        for spec in self.choices.values():
            keys.update(spec.fields)
        return keys


@dataclass(frozen=True)
class ChoiceSectionSpec(CfgSectionSpec):
    """A section that renders selector-specific child specs.

    The value tree remains a normal complete CfgSectionValue over the union of
    ``fields``. Only rendering is variant-aware: selector fields stay editable, and
    fields listed by inactive choice specs are omitted from the form.
    """

    bindings: tuple[ChoiceBinding, ...] = ()

    def __post_init__(self) -> None:
        if not self.bindings:
            raise RuntimeError("ChoiceSectionSpec.bindings must be non-empty")
        field_keys = set(self.fields)
        for binding in self.bindings:
            if binding.selector_key not in field_keys:
                raise RuntimeError(
                    f"Choice selector {binding.selector_key!r} is not a section field"
                )
            for choice, spec in binding.choices.items():
                unknown = set(spec.fields) - field_keys
                if unknown:
                    raise RuntimeError(
                        f"Choice {binding.selector_key!r}={choice!r} references "
                        "unknown field(s): " + ", ".join(sorted(unknown))
                    )
                if binding.selector_key in spec.fields:
                    raise RuntimeError(
                        f"Choice {binding.selector_key!r}={choice!r} must not include "
                        "its own selector field"
                    )


@dataclass(frozen=True)
class DeviceRefSpec:
    """A field that selects a registered device by name."""

    label: str = "Device"


CfgNodeSpec = (
    ScalarSpec
    | LiteralSpec
    | SweepSpec
    | ModuleRefSpec
    | WaveformRefSpec
    | CfgSectionSpec
    | ChoiceSectionSpec
    | DeviceRefSpec
)


# ---------------------------------------------------------------------------
# Value tree — mutable, holds user-editable state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectValue:
    """A directly-entered scalar value. ``value is None`` means *unset* (the
    field has no value yet) — there is no separate ``is_unset`` flag, the value
    itself is the single source of truth (ADR-0010). Scalar types are only
    int/float/str/bool, whose legal values are never ``None``, so ``None``
    unambiguously means unset. The ``DirectValue`` wrapper is kept even when
    unset so the scalar's *mode* (direct vs ``EvalValue``) survives."""

    value: Any | None = None


@dataclass(frozen=True)
class EvalValue:
    expr: str
    resolved: Any | None = None
    error: str | None = None


ScalarValue: TypeAlias = DirectValue | EvalValue

# Accepted input for the value-tree fluent ``with_field``: a raw scalar (wrapped
# in DirectValue) or an already-built scalar value.
ScalarLeafInput: TypeAlias = int | float | str | bool | DirectValue | EvalValue


@dataclass
class SweepValue:
    start: float | EvalValue
    stop: float | EvalValue
    expts: int
    step: float = 0.1
    # ``auto_norm`` (init-only) derives ``step`` from start/stop/expts at
    # construction so that any direct ``SweepValue(start, stop, expts=N)`` (the
    # 16 adapter defaults, session codec, inheritance) is self-consistent — step
    # is a derived view of expts, not an independent input. ``SweepEditor`` (the
    # canonicalisation authority, which also runs the reverse step→expts rule)
    # passes ``auto_norm=False`` so its already-computed value is not re-derived.
    # Only plain numeric bounds are normalised; EvalValue bounds are left to
    # ``SweepEditor`` (which owns the resolved-edge handling) — auto_norm never
    # touches an EvalValue's ``resolved`` (it may be unresolved or non-numeric).
    auto_norm: InitVar[bool] = True

    def __post_init__(self, auto_norm: bool) -> None:
        if self.expts < 1:
            raise ValueError("SweepValue.expts must be >= 1")
        if (
            auto_norm
            and isinstance(self.start, (int, float))
            and isinstance(self.stop, (int, float))
        ):
            self.step = (
                0.0
                if self.expts == 1
                else (float(self.stop) - float(self.start)) / (self.expts - 1)
            )


@dataclass
class ModuleRefValue:
    chosen_key: str
    value: CfgSectionValue
    # True when chosen_key names a library entry but the user has edited value
    # away from the library snapshot (LibraryBindingState.MODIFIED). Persisted so
    # the override survives reload; False for pure library refs and <Custom:> refs.
    is_overridden: bool = False

    def with_field(self, path: str, value: ScalarLeafInput) -> Self:
        """Set a scalar leaf inside this ref's value (in-place, returns self).

        Adapter-side default override sugar (replaces long factory params). The
        value tree is mutable by contract; this mutates and returns self for
        chaining — deliberately asymmetric with spec-side fluent (which returns
        new frozen specs). See CONTEXT.md "Value OO 覆寫".
        """
        self.value.with_field(path, value)
        return self


@dataclass
class WaveformRefValue:
    chosen_key: str
    value: CfgSectionValue
    is_overridden: bool = False

    def with_field(self, path: str, value: ScalarLeafInput) -> Self:
        self.value.with_field(path, value)
        return self


@dataclass
class CfgSectionValue:
    # The value tree is always *complete*: every spec field has a corresponding
    # entry (no missing keys, ADR-0010). A disabled optional ModuleRef/WaveformRef
    # is represented by ``None`` (the entry is present, its value is None) — never
    # by omitting the key. "None" here means "this optional ref is not enabled",
    # distinct from a "None Reset" library entry (a real, enabled reset choice).
    fields: dict[str, CfgNodeValue | None] = field(default_factory=dict)

    def with_field(self, path: str, value: ScalarLeafInput) -> Self:
        """Set the scalar leaf at dotted ``path`` (in-place, returns self).

        ``value`` may be a raw scalar (wrapped in ``DirectValue``) or an already-
        built ``DirectValue``/``EvalValue``. Descends ``CfgSectionValue.fields``
        and ``ModuleRefValue``/``WaveformRefValue`` (into their ``.value``).
        """
        parts = [p for p in path.split(".") if p]
        if not parts:
            raise RuntimeError("Value override path must not be empty")
        node: CfgSectionValue = self
        for seg in parts[:-1]:
            child = node.fields.get(seg)
            if isinstance(child, (ModuleRefValue, WaveformRefValue)):
                node = child.value
            elif isinstance(child, CfgSectionValue):
                node = child
            else:
                raise RuntimeError(
                    f"Value override path cannot descend into {type(child).__name__} "
                    f"at segment {seg!r}"
                )
        leaf_value: CfgNodeValue = (
            value if isinstance(value, (DirectValue, EvalValue)) else DirectValue(value)
        )
        node.fields[parts[-1]] = leaf_value
        return self


CfgNodeValue = (
    ScalarValue | SweepValue | ModuleRefValue | WaveformRefValue | CfgSectionValue
)


# ---------------------------------------------------------------------------
# CfgSchema — pairs a spec tree with a value tree
# ---------------------------------------------------------------------------


@dataclass
class CfgSchema:
    spec: CfgSectionSpec
    value: CfgSectionValue

    def validate(self, ml: ModuleLibrary | None) -> None:
        """Fast-fail if the value tree is structurally incomplete or violates the
        spec — the *static* contract (structure complete, every spec field has an
        entry, LiteralSpec == spec.value, DirectValue scalar type/choices).
        Called at finished-cfg boundaries (``make_default_cfg`` output,
        ``to_raw_dict`` before lowering); NOT in ``__post_init__`` (that would
        reject legal editing intermediates)."""
        from .lowering import validate_section

        validate_section(self.spec, self.value, ml, [])

    def validate_dynamic(self, md: MetaDict, ml: ModuleLibrary | None) -> None:
        """Fast-fail if the value tree cannot be lowered with the given md.

        The *dynamic* contract: every scalar must have a value (no
        DirectValue(None)), every EvalValue must resolve against md, every
        device ref must be selected. Called by ``to_raw_dict`` before lowering
        when md is available — the lowering itself has its own (overlapping)
        checks as a safety net."""
        from .lowering import validate_dynamic_section

        validate_dynamic_section(self.spec, self.value, md, ml, [])

    def to_raw_dict(
        self, md: MetaDict | None, ml: ModuleLibrary | None
    ) -> dict[str, object]:
        """Lower the current schema into a raw experiment config dictionary.

        ``md`` lets lowering resolve any ``EvalValue`` built without a snapshot
        ``resolved``; omit it (pass ``None``) only when every EvalValue is
        already resolved. This is the single lowering entry point — the former
        free function ``schema_to_dict`` was folded into it.
        """
        from .lowering import _section_to_dict_inner

        self.validate(ml)
        if md is not None:
            self.validate_dynamic(md, ml)
        return _section_to_dict_inner(self.spec, self.value, ml, [], md)
