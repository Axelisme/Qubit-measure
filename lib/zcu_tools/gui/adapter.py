from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, Union, cast

from matplotlib.figure import Figure
from typing_extensions import Generic, TypeAlias, TypeVar

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.meta_tool.metadict import MetaDict
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor
from zcu_tools.utils.datasaver import create_datafolder

SocHandle: TypeAlias = object
SocCfgHandle: TypeAlias = object
T_Result = TypeVar("T_Result")


class AnalyzeResultWithFigure(Protocol):
    @property
    def figure(self) -> Optional[Figure]: ...


T_AnalyzeResult = TypeVar("T_AnalyzeResult", bound=AnalyzeResultWithFigure)


@dataclass(frozen=True)
class ExpContext:
    md: "MetaDict"
    ml: "ModuleLibrary"
    soc: Optional[SocHandle]
    soccfg: Optional[SocCfgHandle]
    chip_name: str = "unknown_chip"
    qub_name: str = "unknown_qubit"
    res_name: str = "unknown_resonator"
    result_dir: str = ""
    database_path: str = ""
    active_label: str = ""
    predictor: Optional["FluxoniumPredictor"] = None


@dataclass(frozen=True)
class RunRequest:
    md: "MetaDict"
    ml: "ModuleLibrary"
    soc: Optional[SocHandle]
    soccfg: Optional[SocCfgHandle]


@dataclass(frozen=True)
class AnalyzeRequest(Generic[T_Result]):
    run_result: T_Result
    analyze_params: dict[str, object]
    md: "MetaDict"
    ml: "ModuleLibrary"
    predictor: Optional["FluxoniumPredictor"]


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
    selected: bool = True


@dataclass
class MetaDictWriteback(WritebackItem):
    md_key: str = ""
    proposed_value: Any = None


@dataclass
class ModuleWriteback(WritebackItem):
    module_name: str = ""
    proposed_module: Any = None
    edit_schema: Optional["CfgSchema"] = None
    edited_schema: Optional["CfgSchema"] = None


@dataclass
class WaveformWriteback(WritebackItem):
    waveform_name: str = ""
    proposed_waveform: Any = None
    edit_schema: Optional["CfgSchema"] = None
    edited_schema: Optional["CfgSchema"] = None


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


@dataclass(frozen=True)
class DirectValue:
    value: Any
    is_unset: bool = False


@dataclass(frozen=True)
class EvalValue:
    expr: str
    resolved: Optional[Any] = None


ScalarValue: TypeAlias = Union[DirectValue, EvalValue]


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

    def to_raw_dict(self, req: RunRequest) -> dict[str, object]:
        """Lower the current schema into a raw experiment config dictionary."""
        return _section_to_dict(self.spec, self.value, req.ml)


# ---------------------------------------------------------------------------
# schema_to_dict — flatten (spec, value) into an exp_cfg dict
# ---------------------------------------------------------------------------


def _section_to_dict(
    spec: CfgSectionSpec,
    value: CfgSectionValue,
    ml: "Optional[ModuleLibrary]",
    path: Optional[list[str]] = None,
) -> dict:
    if path is None:
        path = []
    result: dict[str, Any] = {}
    extra_keys = set(value.fields.keys()) - set(spec.fields.keys())
    if extra_keys:
        section = ".".join(path) or "<root>"
        extras = ", ".join(sorted(extra_keys))
        raise RuntimeError(f"Config section '{section}' has unknown fields: {extras}")
    for key, node_spec in spec.fields.items():
        node_val = value.fields.get(key)
        if node_val is None:
            if isinstance(node_spec, LiteralSpec):
                result[key] = node_spec.value
                continue
            label = getattr(node_spec, "label", "") or key
            full_path = ".".join([*path, key])
            raise RuntimeError(f"Config field '{full_path}' ({label}) is missing")

        if isinstance(node_spec, ScalarSpec):
            assert isinstance(node_val, (DirectValue, EvalValue))
            if isinstance(node_val, DirectValue):
                if node_val.is_unset:
                    label = node_spec.label or key
                    full_path = ".".join([*path, key])
                    raise RuntimeError(f"Config field '{full_path}' ({label}) is unset")
                result[key] = node_val.value
            else:
                if node_val.resolved is None:
                    label = node_spec.label or key
                    full_path = ".".join([*path, key])
                    raise RuntimeError(
                        f"Config field '{full_path}' ({label}) expression "
                        f"{node_val.expr!r} is unresolved"
                    )
                result[key] = node_val.resolved

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
            if not isinstance(node_val.value, CfgSectionValue):
                label = node_spec.label or key
                full_path = ".".join([*path, key])
                raise RuntimeError(f"Config field '{full_path}' ({label}) is missing")
            result[key] = _section_to_dict(
                _find_allowed_spec(node_spec, node_val),
                node_val.value,
                ml,
                path=[*path, key],
            )

        elif isinstance(node_spec, CfgSectionSpec):
            assert isinstance(node_val, CfgSectionValue)
            result[key] = _section_to_dict(node_spec, node_val, ml, path=[*path, key])

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
    # Named module: match by LiteralSpec discriminators first (type/style)
    if isinstance(ref_val.value, CfgSectionValue):
        literal_matches = []
        for spec in ref_spec.allowed:
            ok = True
            for key, node_spec in spec.fields.items():
                if isinstance(node_spec, LiteralSpec):
                    node_val = ref_val.value.fields.get(key)
                    if (
                        not isinstance(node_val, DirectValue)
                        or node_val.value != node_spec.value
                    ):
                        ok = False
                        break
            if ok:
                literal_matches.append(spec)
        if literal_matches:
            return max(literal_matches, key=lambda s: len(s.fields))
    # Named module: infer spec from value shape (prefer most specific match)
    if isinstance(ref_val.value, CfgSectionValue):
        value_keys = set(ref_val.value.fields.keys())
        matches = [
            s for s in ref_spec.allowed if set(s.fields.keys()).issubset(value_keys)
        ]
        if matches:
            return max(matches, key=lambda s: len(s.fields))
    # fallback: return first allowed spec
    if ref_spec.allowed:
        return ref_spec.allowed[0]
    return CfgSectionSpec()


def schema_to_dict(schema: CfgSchema, ml: "Optional[ModuleLibrary]") -> dict:
    """Compatibility wrapper around CfgSchema.to_raw_dict()."""
    if ml is None:
        return _section_to_dict(schema.spec, schema.value, None)
    fake_req = RunRequest(md=cast(Any, None), ml=ml, soc=object(), soccfg=object())
    return schema.to_raw_dict(fake_req)


# ---------------------------------------------------------------------------
# make_default_value — build a default CfgSectionValue from a CfgSectionSpec
# ---------------------------------------------------------------------------


def make_default_value(spec: CfgSectionSpec) -> CfgSectionValue:
    """Produce a default CfgSectionValue mirroring the given spec structure."""
    fields: dict[str, CfgNodeValue] = {}
    for key, node_spec in spec.fields.items():
        if isinstance(node_spec, LiteralSpec):
            fields[key] = DirectValue(node_spec.value)
        elif isinstance(node_spec, ScalarSpec):
            # Use 0 / "" / False as a sensible zero-value per type
            defaults: dict[type, Any] = {int: 0, float: 0.0, bool: False, str: ""}
            fields[key] = DirectValue(defaults.get(node_spec.type, None))
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
# inherit_from — carry matching field values when switching Ref combo
# ---------------------------------------------------------------------------


def inherit_from(
    old_val: CfgSectionValue,
    old_spec: CfgSectionSpec,
    new_spec: CfgSectionSpec,
) -> CfgSectionValue:
    """Build a new CfgSectionValue from new_spec, inheriting old_val where compatible.

    Rules (per field key in new_spec):
    - LiteralSpec: always use new spec's fixed value (never inherit).
    - ScalarSpec: inherit if old has same key with ScalarSpec of identical type.
    - SweepSpec: inherit if old has same key with SweepSpec (copy entire SweepValue).
    - MultiSweepSpec: inherit per axis key; new axes fall back to defaults.
    - ModuleRefSpec / WaveformRefSpec: inherit the whole value (chosen_key + sub-value)
      if old has same key with the same ref type.
    - CfgSectionSpec: recurse into inherit_from.

    - No matching key / incompatible type: make_default_value for that field.

    Special cross-spec rules (hardcoded by label pair):
    - "Direct Readout" → "Pulse Readout": inject old_val into new ro_cfg sub-section.
    - "Pulse Readout" → "Direct Readout": extract old ro_cfg sub-section as new top-level.
    """
    # --- Hardcoded cross-spec rules (identified by spec label) ---
    old_label = old_spec.label
    new_label = new_spec.label

    if old_label == "Direct Readout" and new_label == "Pulse Readout":
        result = make_default_value(new_spec)
        result.fields["ro_cfg"] = old_val
        return result

    if old_label == "Pulse Readout" and new_label == "Direct Readout":
        ro_cfg_val = old_val.fields.get("ro_cfg")
        if isinstance(ro_cfg_val, CfgSectionValue):
            return ro_cfg_val
        return make_default_value(new_spec)

    # --- Generic field-by-field inheritance ---
    new_fields: dict[str, CfgNodeValue] = {}

    for key, new_node_spec in new_spec.fields.items():
        old_node_spec = old_spec.fields.get(key)
        old_node_val = old_val.fields.get(key)

        # LiteralSpec — always use spec's fixed value
        if isinstance(new_node_spec, LiteralSpec):
            new_fields[key] = DirectValue(new_node_spec.value)
            continue

        # ScalarSpec — inherit if same type
        if isinstance(new_node_spec, ScalarSpec):
            if (
                isinstance(old_node_spec, ScalarSpec)
                and old_node_spec.type is new_node_spec.type
                and isinstance(old_node_val, (DirectValue, EvalValue))
            ):
                new_fields[key] = old_node_val
            else:
                defaults: dict[type, Any] = {int: 0, float: 0.0, bool: False, str: ""}
                new_fields[key] = DirectValue(defaults.get(new_node_spec.type, None))
            continue

        # SweepSpec — inherit whole SweepValue
        if isinstance(new_node_spec, SweepSpec):
            if isinstance(old_node_spec, SweepSpec) and isinstance(
                old_node_val, SweepValue
            ):
                new_fields[key] = SweepValue(
                    old_node_val.start,
                    old_node_val.stop,
                    old_node_val.expts,
                    old_node_val.step,
                )
            else:
                new_fields[key] = SweepValue(start=0.0, stop=1.0, expts=11)
            continue

        # MultiSweepSpec — per-axis key matching
        if isinstance(new_node_spec, MultiSweepSpec):
            old_axes = (
                old_node_val.axes
                if isinstance(old_node_spec, MultiSweepSpec)
                and isinstance(old_node_val, MultiSweepValue)
                else {}
            )
            new_axes: dict[str, SweepValue] = {}
            for axis_key in new_node_spec.axes:
                if axis_key in old_axes:
                    old_sv = old_axes[axis_key]
                    new_axes[axis_key] = SweepValue(
                        old_sv.start, old_sv.stop, old_sv.expts, old_sv.step
                    )
                else:
                    new_axes[axis_key] = SweepValue(0.0, 1.0, 11)
            new_fields[key] = MultiSweepValue(axes=new_axes)
            continue

        # ModuleRefSpec / WaveformRefSpec — inherit whole ref value
        if isinstance(new_node_spec, ModuleRefSpec):
            if isinstance(old_node_spec, ModuleRefSpec) and isinstance(
                old_node_val, ModuleRefValue
            ):
                new_fields[key] = ModuleRefValue(
                    old_node_val.chosen_key, old_node_val.value
                )
            else:
                first = (
                    new_node_spec.allowed[0]
                    if new_node_spec.allowed
                    else CfgSectionSpec()
                )
                label = first.label or "Custom"
                new_fields[key] = ModuleRefValue(
                    f"<Custom:{label}>", make_default_value(first)
                )
            continue

        if isinstance(new_node_spec, WaveformRefSpec):
            if isinstance(old_node_spec, WaveformRefSpec) and isinstance(
                old_node_val, WaveformRefValue
            ):
                new_fields[key] = WaveformRefValue(
                    old_node_val.chosen_key, old_node_val.value
                )
            else:
                first = (
                    new_node_spec.allowed[0]
                    if new_node_spec.allowed
                    else CfgSectionSpec()
                )
                label = first.label or "Custom"
                new_fields[key] = WaveformRefValue(
                    f"<Custom:{label}>", make_default_value(first)
                )
            continue

        # CfgSectionSpec — recurse
        if isinstance(new_node_spec, CfgSectionSpec):
            if isinstance(old_node_spec, CfgSectionSpec) and isinstance(
                old_node_val, CfgSectionValue
            ):
                new_fields[key] = inherit_from(
                    old_node_val, old_node_spec, new_node_spec
                )
            else:
                new_fields[key] = make_default_value(new_node_spec)
            continue

    return CfgSectionValue(fields=new_fields)


class AbsExpAdapter(ABC, Generic[T_Result, T_AnalyzeResult]):
    exp_cls: Optional[type[Any]] = None

    @abstractmethod
    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        """Build a default CfgSchema from ctx."""

    @abstractmethod
    def build_exp_cfg(
        self, raw_cfg: dict[str, object], req: RunRequest
    ) -> "ExpCfgModel":
        """Convert lowered raw cfg into the concrete experiment config model."""

    def run(self, req: RunRequest, schema: CfgSchema) -> T_Result:
        """Default run pipeline for experiment-class adapters.

        Long-running experiment implementations must cooperate with GUI cancel by
        checking the active task stop flag inside their own acquisition loop.
        """
        if self.exp_cls is None:
            raise RuntimeError(
                f"{type(self).__name__} must define exp_cls or override run()"
            )
        raw_cfg = schema.to_raw_dict(req)
        exp_cfg = self.build_exp_cfg(raw_cfg, req)
        experiment = cast(Any, self.exp_cls())
        return cast(T_Result, experiment.run(exp_cfg))

    @abstractmethod
    def get_analyze_params(
        self, result: T_Result, ctx: ExpContext
    ) -> list[AnalyzeParam]:
        """Declare analysis params the GUI should collect for a run result."""

    @abstractmethod
    def analyze(
        self,
        req: AnalyzeRequest[T_Result],
    ) -> T_AnalyzeResult:
        """Run analysis."""

    @abstractmethod
    def get_writeback_items(
        self, req: WritebackRequest[T_Result, T_AnalyzeResult]
    ) -> Sequence[WritebackItem]: ...

    @abstractmethod
    def make_filename_stem(self, ctx: ExpContext) -> str:
        """Return the filename stem used by the default save path template."""

    def make_default_save_paths(self, ctx: ExpContext) -> SavePaths:
        """Default save path policy shared by most adapters."""
        if not ctx.database_path:
            raise RuntimeError("ExpContext.database_path is required for save paths")
        if not ctx.result_dir:
            raise RuntimeError("ExpContext.result_dir is required for save paths")
        if not ctx.active_label:
            raise RuntimeError("ExpContext.active_label is required for save paths")

        stem = self.make_filename_stem(ctx)
        data_dir = create_datafolder(ctx.database_path)
        image_dir = os.path.join(ctx.result_dir, "exps", ctx.active_label, "image")
        os.makedirs(image_dir, exist_ok=True)
        return SavePaths(
            data_path=os.path.join(data_dir, stem),
            image_path=os.path.join(image_dir, f"{stem}.png"),
        )

    def make_save_paths(self, ctx: ExpContext) -> SavePaths:
        return self.make_default_save_paths(ctx)

    @abstractmethod
    def save(self, req: SaveDataRequest[T_Result]) -> None: ...
