from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
)

from typing_extensions import Generic, TypeAlias, TypeVar

if TYPE_CHECKING:
    from matplotlib.figure import Figure

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


class AnalyzeResultWithFigure(Protocol):
    @property
    def figure(self) -> Optional["Figure"]: ...


T_AnalyzeResult = TypeVar("T_AnalyzeResult", bound=AnalyzeResultWithFigure)


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


@dataclass(frozen=True)
class RunRequest:
    md: MetaDict
    ml: ModuleLibrary
    soc: Optional[SocHandle]
    soccfg: Optional[SocCfgHandle]


@dataclass(frozen=True)
class AnalyzeRequest(Generic[T_Result]):
    run_result: T_Result
    analyze_params: dict[str, object]
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


@dataclass(frozen=True)
class WritebackRequest(Generic[T_Result, T_AnalyzeResult]):
    run_result: T_Result
    analyze_result: T_AnalyzeResult
    ctx: ExpContext


def _normalize_analyze_value(type_: type, value: object) -> object:
    if type_ is bool:
        if not isinstance(value, bool):
            raise RuntimeError(f"AnalyzeParam expects bool, got {type(value).__name__}")
        return value
    if type_ is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise RuntimeError(f"AnalyzeParam expects int, got {type(value).__name__}")
        return value
    if type_ is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RuntimeError(
                f"AnalyzeParam expects float, got {type(value).__name__}"
            )
        return float(value)
    if type_ is str:
        if not isinstance(value, str):
            raise RuntimeError(f"AnalyzeParam expects str, got {type(value).__name__}")
        return value
    raise RuntimeError(f"Unsupported AnalyzeParam type: {type_!r}")


@dataclass(frozen=True)
class AnalyzeParam:
    key: str
    label: str
    type: type
    default: object
    choices: Optional[list[object]] = None
    decimals: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.key:
            raise RuntimeError("AnalyzeParam.key must be non-empty")
        _normalize_analyze_value(self.type, self.default)
        if self.choices is not None:
            if not self.choices:
                raise RuntimeError("AnalyzeParam.choices must not be empty")
            for choice in self.choices:
                _normalize_analyze_value(self.type, choice)


def analyze_params_to_raw_dict(
    params: Sequence["AnalyzeParam"],
    values: dict[str, object],
) -> dict[str, object]:
    raw: dict[str, object] = {}
    param_by_key: dict[str, AnalyzeParam] = {}
    for param in params:
        if param.key in param_by_key:
            raise RuntimeError(f"Duplicate AnalyzeParam key: {param.key}")
        param_by_key[param.key] = param

    missing = set(param_by_key) - set(values)
    if missing:
        names = ", ".join(sorted(missing))
        raise RuntimeError(f"Missing analyze params: {names}")

    unknown = set(values) - set(param_by_key)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise RuntimeError(f"Unknown analyze params: {names}")

    for key, param in param_by_key.items():
        value = _normalize_analyze_value(param.type, values[key])
        if param.choices is not None and value not in param.choices:
            raise RuntimeError(
                f"AnalyzeParam {key!r} must be one of {param.choices}, got {value!r}"
            )
        raw[key] = value
    return raw


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


@dataclass(frozen=True)
class ScalarSpec:
    label: str
    type: type
    editable: bool = True
    choices: Optional[list] = None
    decimals: Optional[int] = None
    hidden: bool = False
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
    collapsible: bool = True
    inherit_hook: Optional[
        Callable[["CfgSectionValue", "CfgSectionSpec"], Optional["CfgSectionValue"]]
    ] = None


CfgNodeSpec = Union[
    ScalarSpec,
    LiteralSpec,
    SweepSpec,
    MultiSweepSpec,
    ModuleRefSpec,
    WaveformRefSpec,
    CfgSectionSpec,
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
    start: float
    stop: float
    expts: int
    step: Optional[float] = None


@dataclass
class MultiSweepValue:
    axes: dict[str, SweepValue]


@dataclass
class ModuleRefValue:
    chosen_key: str
    value: "CfgSectionValue"


@dataclass
class WaveformRefValue:
    chosen_key: str
    value: "CfgSectionValue"


@dataclass
class CfgSectionValue:
    fields: dict[str, "CfgNodeValue"] = field(default_factory=dict)


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
