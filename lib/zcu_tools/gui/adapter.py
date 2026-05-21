from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

from matplotlib.figure import Figure
from typing_extensions import Generic, TypeVar

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary
    from zcu_tools.meta_tool.metadict import MetaDict
    from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor


@dataclass(frozen=True)
class ExpContext:
    md: "MetaDict"
    ml: "ModuleLibrary"
    soc: Any
    soccfg: Any
    chip_name: str = "unknown_chip"
    qub_name: str = "unknown_qubit"
    res_name: str = "unknown_resonator"
    result_dir: str = ""
    database_path: str = ""
    active_label: str = ""
    predictor: Optional["FluxoniumPredictor"] = None


@dataclass
class ParamSpec:
    label: str
    default: Any
    type: type
    choices: Optional[list] = None


@dataclass
class WritebackItem:
    key: str
    target: str  # "md" or "ml"
    current_value: Any
    new_value: Any
    description: str
    edit_template: Optional["CfgSchema"] = None  # CfgSchema shown in Edit Config form


@dataclass
class SavePaths:
    data_path: str
    image_path: str


# ---------------------------------------------------------------------------
# Spec tree — static, defined by Adapter, never mutated
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalarSpec:
    label: str
    type: type
    editable: bool = True
    choices: Optional[list] = None
    decimals: Optional[int] = None  # float display precision; None = 6 (default)
    hidden: bool = (
        False  # skip widget in UI; value still included in schema_to_dict output
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


@dataclass(frozen=True)
class MultiSweepSpec:
    axes: dict[str, SweepSpec]
    label: str = "Sweep"


@dataclass(frozen=True)
class ModuleRefSpec:
    allowed: list["CfgSectionSpec"]  # each spec's label is its display name
    label: str = "Module"


@dataclass(frozen=True)
class WaveformRefSpec:
    allowed: list["CfgSectionSpec"]  # each spec's label is its display name
    label: str = "Waveform"


@dataclass(frozen=True)
class CfgSectionSpec:
    fields: dict[str, "CfgNodeSpec"] = field(default_factory=dict)
    label: str = ""
    collapsible: bool = True


CfgNodeSpec = Union[
    ScalarSpec,
    LiteralSpec,
    SweepSpec,
    MultiSweepSpec,
    ModuleRefSpec,
    WaveformRefSpec,
    "CfgSectionSpec",
]


# ---------------------------------------------------------------------------
# Value tree — mutable, holds user-editable state
# ---------------------------------------------------------------------------


@dataclass
class ScalarValue:
    value: Any


@dataclass
class SweepValue:
    start: float
    stop: float
    expts: int
    step: Optional[float] = None  # None = expts mode; non-None = step mode


@dataclass
class MultiSweepValue:
    axes: dict[str, SweepValue]


@dataclass
class ModuleRefValue:
    chosen_key: str  # ml module name, or "<Custom:label>"
    value: "CfgSectionValue"


@dataclass
class WaveformRefValue:
    chosen_key: str  # ml waveform name, or "<Custom:label>"
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
    "CfgSectionValue",
]


# ---------------------------------------------------------------------------
# CfgSchema — pairs a spec tree with a value tree
# ---------------------------------------------------------------------------


@dataclass
class CfgSchema:
    spec: CfgSectionSpec
    value: CfgSectionValue


# ---------------------------------------------------------------------------
# schema_to_dict — flatten (spec, value) into an exp_cfg dict
# ---------------------------------------------------------------------------


def _section_to_dict(
    spec: CfgSectionSpec,
    value: CfgSectionValue,
    ml: "Optional[ModuleLibrary]",
) -> dict:
    result: dict[str, Any] = {}
    for key, node_spec in spec.fields.items():
        node_val = value.fields.get(key)
        if node_val is None:
            continue

        if isinstance(node_spec, ScalarSpec):
            assert isinstance(node_val, ScalarValue)
            result[key] = node_val.value

        elif isinstance(node_spec, LiteralSpec):
            result[key] = node_spec.value

        elif isinstance(node_spec, SweepSpec):
            assert isinstance(node_val, SweepValue)
            from zcu_tools.notebook.utils import make_sweep

            if node_val.step is not None:
                result[key] = make_sweep(
                    node_val.start, node_val.stop, step=node_val.step
                )
            else:
                result[key] = make_sweep(
                    node_val.start, node_val.stop, expts=node_val.expts
                )

        elif isinstance(node_spec, MultiSweepSpec):
            assert isinstance(node_val, MultiSweepValue)
            from zcu_tools.notebook.utils import make_sweep

            result[key] = {
                axis: (
                    make_sweep(sv.start, sv.stop, step=sv.step)
                    if sv.step is not None
                    else make_sweep(sv.start, sv.stop, expts=sv.expts)
                )
                for axis, sv in node_val.axes.items()
            }

        elif isinstance(node_spec, (ModuleRefSpec, WaveformRefSpec)):
            assert isinstance(node_val, (ModuleRefValue, WaveformRefValue))
            chosen = node_val.chosen_key
            # If chosen_key is an ml-named entry, return its to_dict() directly
            if not chosen.startswith("<Custom:") and ml is not None:
                try:
                    obj = (
                        ml.get_module(chosen)
                        if isinstance(node_spec, ModuleRefSpec)
                        else ml.get_waveform(chosen)
                    )
                    result[key] = obj.to_dict()
                    continue
                except Exception:
                    pass
            result[key] = _section_to_dict(
                _find_allowed_spec(node_spec, node_val),
                node_val.value,
                ml,
            )

        elif isinstance(node_spec, CfgSectionSpec):
            assert isinstance(node_val, CfgSectionValue)
            result[key] = _section_to_dict(node_spec, node_val, ml)

        else:
            raise TypeError(f"Unknown CfgNodeSpec type: {type(node_spec)}")

    return result


def _find_allowed_spec(
    ref_spec: Union[ModuleRefSpec, WaveformRefSpec],
    ref_val: Union[ModuleRefValue, WaveformRefValue],
) -> CfgSectionSpec:
    """Return the CfgSectionSpec from allowed[] that matches chosen_key's label."""
    chosen = ref_val.chosen_key
    # Strip "<Custom:...>" prefix to get the label
    if chosen.startswith("<Custom:"):
        label = chosen[len("<Custom:") : -1]
    else:
        label = chosen
    for s in ref_spec.allowed:
        if s.label == label:
            return s
    # fallback: return first allowed spec
    if ref_spec.allowed:
        return ref_spec.allowed[0]
    return CfgSectionSpec()


def schema_to_dict(schema: CfgSchema, ml: "Optional[ModuleLibrary]") -> dict:
    """Recursively convert a CfgSchema to an exp_cfg dict."""
    return _section_to_dict(schema.spec, schema.value, ml)


# ---------------------------------------------------------------------------
# make_default_value — build a default CfgSectionValue from a CfgSectionSpec
# ---------------------------------------------------------------------------


def make_default_value(spec: CfgSectionSpec) -> CfgSectionValue:
    """Produce a default CfgSectionValue mirroring the given spec structure."""
    fields: dict[str, CfgNodeValue] = {}
    for key, node_spec in spec.fields.items():
        if isinstance(node_spec, LiteralSpec):
            fields[key] = ScalarValue(node_spec.value)
        elif isinstance(node_spec, ScalarSpec):
            # Use 0 / "" / False as a sensible zero-value per type
            defaults: dict[type, Any] = {int: 0, float: 0.0, bool: False, str: ""}
            fields[key] = ScalarValue(defaults.get(node_spec.type, None))
        elif isinstance(node_spec, SweepSpec):
            fields[key] = SweepValue(start=0.0, stop=1.0, expts=11)
        elif isinstance(node_spec, MultiSweepSpec):
            fields[key] = MultiSweepValue(
                axes={axis: SweepValue(0.0, 1.0, 11) for axis in node_spec.axes}
            )
        elif isinstance(node_spec, (ModuleRefSpec, WaveformRefSpec)):
            first = node_spec.allowed[0] if node_spec.allowed else CfgSectionSpec()
            label = first.label or "Custom"
            fields[key] = (
                ModuleRefValue(f"<Custom:{label}>", make_default_value(first))
                if isinstance(node_spec, ModuleRefSpec)
                else WaveformRefValue(f"<Custom:{label}>", make_default_value(first))
            )
        elif isinstance(node_spec, CfgSectionSpec):
            fields[key] = make_default_value(node_spec)
    return CfgSectionValue(fields=fields)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

T_Result = TypeVar("T_Result")
T_AnalyzeResult = TypeVar("T_AnalyzeResult")


class AbsExpAdapter(ABC, Generic[T_Result, T_AnalyzeResult]):
    @abstractmethod
    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        """Build a default CfgSchema from ctx."""

    @abstractmethod
    def get_run_params(self) -> dict[str, ParamSpec]:
        """Declare extra run params the GUI should collect from the user."""

    @abstractmethod
    def run(self, ctx: ExpContext, schema: CfgSchema, **user_params: Any) -> T_Result:
        """Run the experiment; internally calls schema_to_dict()."""

    @abstractmethod
    def get_analyze_params(self) -> dict[str, ParamSpec]:
        """Declare extra analyze params the GUI should collect from the user."""

    @abstractmethod
    def analyze(
        self, result: T_Result, ctx: ExpContext, **user_params: Any
    ) -> T_AnalyzeResult:
        """Run analysis."""

    @abstractmethod
    def get_writeback_spec(
        self, analyze_result: T_AnalyzeResult, ctx: ExpContext
    ) -> list[WritebackItem]: ...

    @abstractmethod
    def apply_writeback(
        self,
        ctx: ExpContext,
        analyze_result: T_AnalyzeResult,
        selected_keys: list[str],
        overrides: Optional[dict[str, Any]] = None,
    ) -> None: ...

    @abstractmethod
    def get_figure(self, analyze_result: T_AnalyzeResult) -> Optional[Figure]:
        """Extract a matplotlib Figure from the analyze result."""

    @abstractmethod
    def make_save_paths(self, ctx: ExpContext) -> SavePaths: ...

    @abstractmethod
    def save(self, data_path: str, result: T_Result, ctx: ExpContext) -> None: ...
