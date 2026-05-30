from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Optional,
    Protocol,
    Union,
)

from typing_extensions import Generic, TypeAlias, TypeVar

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.experiment.cfg_model import ExpCfgModel
    from zcu_tools.meta_tool import ModuleLibrary
    from zcu_tools.meta_tool.metadict import MetaDict
    from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor


class SocProtocol(Protocol):
    """Minimal QICK SoC surface carried through GUI run requests."""

    def get_cfg(self) -> Mapping[str, object]: ...


class SocCfgProtocol(Protocol):
    """Minimal QICK config surface used by adapters and setup UI."""

    def description(self) -> str: ...
    def cycles2us(
        self, cycles: Any, /, gen_ch: Any = None, ro_ch: Any = None
    ) -> Any: ...
    def us2cycles(
        self,
        us: Any,
        /,
        gen_ch: Any = None,
        ro_ch: Any = None,
        as_float: bool = False,
    ) -> Any: ...
    def freq2reg(self, freq: Any, /, gen_ch: Any = None, ro_ch: Any = None) -> Any: ...
    def reg2freq(self, reg: Any, /, gen_ch: Any = None) -> Any: ...
    def deg2reg(self, deg: Any, /, gen_ch: Any = None, ro_ch: Any = None) -> Any: ...
    def reg2deg(self, reg: Any, /, gen_ch: Any = None, ro_ch: Any = None) -> Any: ...
    def get_maxv(self, ch: int, /) -> Any: ...
    def __getitem__(self, key: str) -> object: ...


SocHandle: TypeAlias = SocProtocol
SocCfgHandle: TypeAlias = SocCfgProtocol
T_Result = TypeVar("T_Result")
T_Cfg = TypeVar("T_Cfg", bound="ExpCfgModel")
T_Cfg_contra = TypeVar("T_Cfg_contra", bound="ExpCfgModel", contravariant=True)
T_Result_co = TypeVar("T_Result_co")


class ExperimentProtocol(Protocol, Generic[T_Cfg_contra, T_Result_co]):
    """Structural contract for experiment classes referenced by adapters."""

    def run(self, soc: Any, soccfg: Any, cfg: T_Cfg_contra) -> T_Result_co: ...

    def save(self, filepath: str, result: T_Result_co) -> None: ...


@dataclass(frozen=True)
class AdapterCapabilities:
    """Declared capability flags an adapter exposes to the framework."""

    requires_soc: bool = True
    supports_analysis: bool = True


class ContextReadiness(Enum):
    """Lifecycle readiness for operations that need a persisted context."""

    EMPTY = "empty"
    DRAFT = "draft"
    ACTIVE = "active"


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
    def figure(self) -> Optional["Figure"]: ...

    def to_summary_dict(self) -> dict[str, object]: ...


@dataclass
class NoAnalyzeParams:
    """Default analyze-params type for adapters without analysis."""


@dataclass
class NoAnalysisResult(AnalyzeResultBase):
    """Default analyze-result type for adapters without analysis."""

    figure: Optional["Figure"] = None


# PEP 696 defaults: adapters without analysis omit the last two generic args
# (BaseAdapter[Cfg, Result]) and these No* types fill in automatically.
T_AnalyzeResult = TypeVar(
    "T_AnalyzeResult", bound=AnalyzeResultWithFigure, default=NoAnalysisResult
)
T_AnalyzeParams = TypeVar("T_AnalyzeParams", default=NoAnalyzeParams)


@dataclass(frozen=True)
class ExpContext:
    md: MetaDict
    ml: ModuleLibrary
    soc: Optional[SocHandle]
    soccfg: Optional[SocCfgHandle]
    chip_name: str = "unknown_chip"
    qub_name: str = "unknown_qubit"
    res_name: str = "unknown_resonator"
    result_dir: str = ""
    database_path: str = ""
    active_label: str = ""
    predictor: Optional[FluxoniumPredictor] = None
    readiness: ContextReadiness = ContextReadiness.EMPTY

    # -- readiness predicates (the context answers about itself) -----------

    def has_context(self) -> bool:
        """Any valid context exists (startup DRAFT or file-backed ACTIVE)."""
        return self.readiness is not ContextReadiness.EMPTY

    def is_draft(self) -> bool:
        return self.readiness is ContextReadiness.DRAFT

    def is_active(self) -> bool:
        """File-backed context eligible for run and save."""
        return self.readiness is ContextReadiness.ACTIVE

    def has_soc(self) -> bool:
        return self.soc is not None


@dataclass(frozen=True)
class RunRequest:
    md: MetaDict
    ml: ModuleLibrary
    soc: Optional[SocHandle]
    soccfg: Optional[SocCfgHandle]


@dataclass(frozen=True)
class AnalyzeRequest(Generic[T_Result, T_AnalyzeParams]):
    run_result: T_Result
    analyze_params: T_AnalyzeParams
    md: MetaDict
    ml: ModuleLibrary
    predictor: Optional[FluxoniumPredictor]


@dataclass(frozen=True)
class SaveDataRequest(Generic[T_Result]):
    run_result: T_Result
    data_path: str
    md: "MetaDict"
    ml: "ModuleLibrary"
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
    key: str
    description: str
    current_value: Any
    selected: bool = field(default=True, init=False)


@dataclass
class MetaDictWriteback(WritebackItem):
    md_key: str
    proposed_value: Any

    def __post_init__(self) -> None:
        if not self.md_key:
            raise RuntimeError("MetaDictWriteback.md_key must be non-empty")


@dataclass
class ModuleWriteback(WritebackItem):
    module_name: str
    proposed_module: Any
    edit_schema: Optional["CfgSchema"] = None
    edited_schema: Optional["CfgSchema"] = None

    def __post_init__(self) -> None:
        if not self.module_name:
            raise RuntimeError("ModuleWriteback.module_name must be non-empty")


@dataclass
class WaveformWriteback(WritebackItem):
    waveform_name: str
    proposed_waveform: Any
    edit_schema: Optional["CfgSchema"] = None
    edited_schema: Optional["CfgSchema"] = None

    def __post_init__(self) -> None:
        if not self.waveform_name:
            raise RuntimeError("WaveformWriteback.waveform_name must be non-empty")


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


def _path_exists(spec: "CfgSectionSpec", parts: list[str]) -> bool:
    """True if the dotted ``parts`` resolve to a leaf within ``spec`` (descending
    CfgSectionSpec.fields and ModuleRefSpec.allowed). Used by the duck-type
    descent to decide which allowed shapes contain a path."""
    head, rest = parts[0], parts[1:]
    child = spec.fields.get(head)
    if child is None:
        return False
    if not rest:
        return True
    if isinstance(child, CfgSectionSpec):
        return _path_exists(child, rest)
    if isinstance(child, ModuleRefSpec):
        return any(_path_exists(shape, rest) for shape in child.allowed)
    return False


@dataclass(frozen=True)
class ScalarSpec:
    label: str
    type: type
    editable: bool = True
    choices: Optional[list] = None
    decimals: Optional[int] = None
    required: bool = False


@dataclass(frozen=True)
class LiteralSpec:
    """A fixed-value field: no widget shown, value is always spec.value."""

    value: Any
    label: str = ""


@dataclass(frozen=True)
class SweepSpec:
    label: str = "Sweep"
    editable: bool = True
    decimals: Optional[int] = None


@dataclass(frozen=True)
class MultiSweepSpec:
    axes: dict[str, SweepSpec]
    label: str = "Sweep"


@dataclass(frozen=True)
class ModuleRefSpec:
    allowed: list["CfgSectionSpec"]
    label: str = "Module"
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.allowed:
            raise RuntimeError("ModuleRefSpec.allowed must be non-empty")

    def _with_override(self, parts: list[str], fn: "_LeafTransform") -> "ModuleRefSpec":
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
    allowed: list["CfgSectionSpec"]
    label: str = "Waveform"
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.allowed:
            raise RuntimeError("WaveformRefSpec.allowed must be non-empty")


@dataclass(frozen=True)
class CfgSectionSpec:
    fields: dict[str, "CfgNodeSpec"] = field(default_factory=dict)
    label: str = ""
    inherit_hook: Optional[
        Callable[["CfgSectionValue", "CfgSectionSpec"], Optional["CfgSectionValue"]]
    ] = None

    # -- fluent spec overrides (return a new frozen spec; never mutate) -------
    #
    # Used inside an adapter's ``cfg_spec()`` to lock/restrict a deep leaf of a
    # spec tree returned by a shared helper. The result MUST be the value that
    # ``cfg_spec()`` returns — locking is part of the spec contract, and
    # ``cfg_spec`` is the sole owner of that contract. Locking the return value
    # of ``cfg_spec()`` from outside leaks the contract to the call site.

    def lock_literal(self, path: str, value: Any) -> "CfgSectionSpec":
        """Replace the scalar leaf at ``path`` with a fixed ``LiteralSpec(value)``.

        The locked field shows no widget and always lowers to ``value`` (notebook
        ``freq: 0.0, # not used``). Returns a new frozen spec.
        """
        return self._with_override(
            _split_spec_path(path), lambda leaf: LiteralSpec(value=value)
        )

    def _with_override(
        self, parts: list[str], fn: "_LeafTransform"
    ) -> "CfgSectionSpec":
        head, rest = parts[0], parts[1:]
        if head not in self.fields:
            raise RuntimeError(
                f"Spec override path segment {head!r} not found "
                f"(available: {', '.join(self.fields)})"
            )
        child = self.fields[head]
        if not rest:
            new_child: CfgNodeSpec = fn(child)
        elif isinstance(child, (CfgSectionSpec, ModuleRefSpec)):
            new_child = child._with_override(rest, fn)
        else:
            raise RuntimeError(
                f"Spec override path cannot descend into {type(child).__name__} "
                f"at segment {head!r}"
            )
        return replace(self, fields={**self.fields, head: new_child})


@dataclass(frozen=True)
class DeviceRefSpec:
    """A field that selects a registered device by name."""

    label: str = "Device"


CfgNodeSpec = Union[
    ScalarSpec,
    LiteralSpec,
    SweepSpec,
    MultiSweepSpec,
    ModuleRefSpec,
    WaveformRefSpec,
    CfgSectionSpec,
    DeviceRefSpec,
]


# ---------------------------------------------------------------------------
# Value tree — mutable, holds user-editable state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectValue:
    value: Any
    is_unset: bool = False


@dataclass(frozen=True)
class EvalValue:
    expr: str
    resolved: Optional[Any] = None
    error: Optional[str] = None


ScalarValue: TypeAlias = Union[DirectValue, EvalValue]


@dataclass
class SweepValue:
    start: Union[float, EvalValue]
    stop: Union[float, EvalValue]
    expts: int
    step: float = 0.1

    def __post_init__(self) -> None:
        if self.expts < 1:
            raise ValueError("SweepValue.expts must be >= 1")


@dataclass
class MultiSweepValue:
    axes: dict[str, SweepValue]


@dataclass
class ModuleRefValue:
    chosen_key: str
    value: "CfgSectionValue"
    # True when chosen_key names a library entry but the user has edited value
    # away from the library snapshot (LibraryBindingState.MODIFIED). Persisted so
    # the override survives reload; False for pure library refs and <Custom:> refs.
    is_overridden: bool = False

    def with_field(self, path: str, value: Any) -> "ModuleRefValue":
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
    value: "CfgSectionValue"
    is_overridden: bool = False

    def with_field(self, path: str, value: Any) -> "WaveformRefValue":
        self.value.with_field(path, value)
        return self


@dataclass
class CfgSectionValue:
    fields: dict[str, "CfgNodeValue"] = field(default_factory=dict)

    def with_field(self, path: str, value: Any) -> "CfgSectionValue":
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


CfgNodeValue = Union[
    ScalarValue,
    SweepValue,
    MultiSweepValue,
    ModuleRefValue,
    WaveformRefValue,
    CfgSectionValue,
]


# ---------------------------------------------------------------------------
# CfgSchema — pairs a spec tree with a value tree
# ---------------------------------------------------------------------------


@dataclass
class CfgSchema:
    spec: CfgSectionSpec
    value: CfgSectionValue

    def to_raw_dict(self, req: RunRequest) -> dict[str, object]:
        """Lower the current schema into a raw experiment config dictionary."""
        from .lowering import _section_to_dict_inner

        return _section_to_dict_inner(self.spec, self.value, req.ml, [])
