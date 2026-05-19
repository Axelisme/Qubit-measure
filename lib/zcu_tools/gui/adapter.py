from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

from matplotlib.figure import Figure
from typing_extensions import Generic, TypeVar

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ExperimentManager, ModuleLibrary
    from zcu_tools.meta_tool.metadict import MetaDict
    from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor


@dataclass(frozen=True)
class ExpContext:
    md: "MetaDict"
    ml: "ModuleLibrary"
    em: "ExperimentManager"
    soc: Any
    soccfg: Any
    database_path: str = ""
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


@dataclass
class SavePaths:
    data_path: str
    image_path: str


# --- CfgSchema type tree ---


@dataclass
class ScalarField:
    value: Any
    label: str
    type: type
    editable: bool = True
    choices: Optional[list] = None


@dataclass
class SweepField:
    start: float
    stop: float
    expts: int
    step: Optional[float] = None  # None = expts mode; non-None = step mode
    label: str = "Sweep"
    editable: bool = True


@dataclass
class MultiSweepField:
    sweeps: dict[str, SweepField]
    label: str = "Sweep"


@dataclass
class ModuleRefField:
    module_name: Optional[str]
    override: dict
    inline_cfg: Optional[dict]
    expanded_content: Optional["CfgSection"]  # eager load at make_default_cfg time
    available_modules: list
    label: str = "Module"


CfgNode = Union[ScalarField, SweepField, MultiSweepField, ModuleRefField, "CfgSection"]


@dataclass
class CfgSection:
    fields: dict[str, CfgNode] = field(default_factory=dict)
    label: str = ""
    collapsible: bool = True


@dataclass
class CfgSchema:
    root: CfgSection
    # reps/rounds live as ScalarField entries in root.fields


# --- schema_to_dict helper ---


def _section_to_dict(section: CfgSection, ml: "ModuleLibrary") -> dict:
    result: dict[str, Any] = {}
    for key, node in section.fields.items():
        if isinstance(node, ScalarField):
            result[key] = node.value
        elif isinstance(node, SweepField):
            from zcu_tools.notebook.utils import make_sweep

            if node.step is not None:
                result[key] = make_sweep(node.start, node.stop, step=node.step)
            else:
                result[key] = make_sweep(node.start, node.stop, expts=node.expts)
        elif isinstance(node, MultiSweepField):
            from zcu_tools.notebook.utils import make_sweep

            result[key] = {
                axis: (
                    make_sweep(f.start, f.stop, step=f.step)
                    if f.step is not None
                    else make_sweep(f.start, f.stop, expts=f.expts)
                )
                for axis, f in node.sweeps.items()
            }
        elif isinstance(node, ModuleRefField):
            if node.module_name is not None:
                result[key] = ml.get_module(node.module_name, node.override or None)
            else:
                result[key] = node.inline_cfg or {}
        elif isinstance(node, CfgSection):
            result[key] = _section_to_dict(node, ml)
        else:
            raise TypeError(f"Unknown CfgNode type: {type(node)}")
    return result


def schema_to_dict(schema: CfgSchema, ml: "ModuleLibrary") -> dict:
    """Recursively convert a CfgSchema to an exp_cfg dict for ml.make_cfg()."""
    return _section_to_dict(schema.root, ml)


# --- Abstract base ---

T_Result = TypeVar("T_Result")
T_AnalyzeResult = TypeVar("T_AnalyzeResult")


class AbsExpAdapter(ABC, Generic[T_Result, T_AnalyzeResult]):
    @abstractmethod
    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        """Build a default CfgSchema from ctx; ModuleRefField.expanded_content eager-loaded."""

    @abstractmethod
    def get_run_params(self) -> dict[str, ParamSpec]:
        """Declare extra run params the GUI should collect from the user."""

    @abstractmethod
    def run(self, ctx: ExpContext, schema: CfgSchema, **user_params: Any) -> T_Result:
        """Run the experiment; internally calls schema_to_dict() + ml.make_cfg()."""

    @abstractmethod
    def get_analyze_params(self) -> dict[str, ParamSpec]:
        """Declare extra analyze params the GUI should collect from the user."""

    @abstractmethod
    def analyze(
        self, result: T_Result, ctx: ExpContext, **user_params: Any
    ) -> T_AnalyzeResult:
        """Run analysis; params derivable from ctx are read internally."""

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
    ) -> None: ...

    @abstractmethod
    def get_figure(self, analyze_result: T_AnalyzeResult) -> Optional[Figure]:
        """Extract a matplotlib Figure from the analyze result (None if not available)."""

    @abstractmethod
    def make_save_paths(self, ctx: ExpContext) -> SavePaths: ...

    @abstractmethod
    def save(self, data_path: str, result: T_Result, ctx: ExpContext) -> None: ...
