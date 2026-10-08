from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from zcu_tools.gui.cfg import CfgSchema, CfgSectionSpec

if TYPE_CHECKING:
    from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
    from zcu_tools.experiment.context import RunContext
    from zcu_tools.experiment.native_persistence import PreparedRunData
    from zcu_tools.plotting.plots import Plots
    from zcu_tools.resources.entry import RoleSpec

from .interactive import InteractivePluginProvider
from .types import (
    AdapterCapabilities,
    AdapterGuide,
    AnalyzeRequest,
    LoadDataRequest,
    NoAnalysisResult,
    NoAnalyzeParams,
    PostAnalyzeRequest,
    PostWritebackRequest,
    RunRequest,
    SaveDataRequest,
    SavePaths,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)

__all__ = [
    "AdapterCfgDefinition",
    "ExpAdapterProtocol",
    "NoAnalysisResult",
    "NoAnalyzeParams",
]


class AdapterCfgDefinition(Protocol):
    """Context-free cfg authoring contract consumed by ``BaseAdapter``.

    Implementations own field policy and deferred defaults. The framework does
    not inspect declarations or seeds and validates the instantiated schema
    through its existing finished-cfg validator.
    """

    @property
    def spec(self) -> CfgSectionSpec:
        """Return an isolated copy of the static field shape, labels and locks."""
        ...

    def instantiate(self, ctx: SessionEnv) -> CfgSchema:
        """Resolve fresh defaults from ``ctx`` without changing the static spec.

        ``ctx`` supplies metadata, module-library and value-source lookups.
        Return an independently mutable schema. Propagate default-resolution
        failures; do not change this definition or the supplied context.
        """
        ...


@runtime_checkable
class ExpAdapterProtocol(InteractivePluginProvider, Protocol):
    """Structural contract the GUI framework requires from an experiment adapter.

    This Protocol lists *only* the members the framework actually calls; it
    carries no generics and no implementation. The shared default behaviour
    (build_exp_cfg delegation, save-path policy, no-op analysis, …) lives in
    ``zcu_tools.gui.app.measure.adapter.base.BaseAdapter``, which adapters
    inherit and which satisfies this Protocol structurally.

    Keeping the framework side generic-free is deliberate: the GUI handles
    every adapter as an opaque ``ExpAdapterProtocol``. Analyze/writeback
    results remain opaque. Native acquisition evidence checks only the RunRecord
    envelope described by run(), not its concrete cfg/result types. Results
    cross Qt ``Signal(object)`` boundaries as plain objects. The cross-method type linkage that pyright
    checks lives entirely inside a single concrete adapter via the generic
    ``BaseAdapter[T_Cfg, T_Result, T_AnalyzeResult, T_AnalyzeParams]``.
    """

    # Class-level capability declaration. Implementers (BaseAdapter and any
    # adapter overriding it) must annotate this as ``ClassVar`` too, or pyright
    # rejects the Protocol conformance.
    capabilities: ClassVar[AdapterCapabilities]
    # Declaring persistence spec tag for acquisition metadata; None means this
    # adapter does not produce native run evidence. Declaring a tag opts into
    # run()'s RunRecord envelope contract when the tab has a saved origin.
    # Never infer it from a name.
    experiment_tag: ClassVar[str | None]

    @classmethod
    def guide(cls) -> AdapterGuide:
        """Return the static human-facing orientation guide (no instance state)."""
        ...

    def parameter_roles(self) -> Mapping[str, RoleSpec]:
        """Declare ordered required roles; empty means pure calculation.

        The tab owner resolves and captures these once at creation. Changing
        session focus later does not rebind an existing tab.
        """
        ...

    def make_default_cfg(self, ctx: SessionEnv) -> CfgSchema:
        """Compose the static spec with context-derived default values."""
        ...

    def make_save_paths(self, ctx: SessionEnv) -> SavePaths:
        """Resolve the save-path policy for the given context."""
        ...

    @classmethod
    def analyze_params_cls(cls) -> type:
        """Return the analyze-params dataclass type (static, no instance)."""
        ...

    def get_analyze_params(self, result: Any, ctx: SessionEnv) -> Any:
        """Build the analyze parameter instance presented to the user."""
        ...

    def validate_run_request(
        self, req: RunRequest, raw_cfg: Mapping[str, ExperimentCfgValue]
    ) -> None:
        """Validate adapter-specific run constraints before opening an operation."""
        ...

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> Any:
        """Execute detached resolved inputs; never resolve from live context.

        req provides accepted device/SoC inputs; raw_cfg contains the accepted
        lowered cfg, and context owns this acquisition's plots and stop signal.
        With experiment_tag set and a saved tab origin, return RunRecord(cfg,
        result) for native evidence. A bare result then raises TypeError at the
        Run publication boundary and publishes neither result nor metadata.
        Without the tag or saved origin, the framework does not manufacture
        acquisition metadata and retains the adapter's opaque result contract.
        Acquisition errors propagate to the operation's failed outcome.
        """
        ...

    def load(self, req: LoadDataRequest) -> Any:
        """Load a canonical result file and return its run-result object."""
        ...

    def analyze(self, req: AnalyzeRequest[Any, Any], *, plots: Plots) -> Any:
        """Run analysis on a completed run result."""
        ...

    @classmethod
    def post_analyze_params_cls(cls) -> type:
        """Return the post-analysis param dataclass type (post_analysis cap)."""
        ...

    def get_post_analyze_params(self, analyze_result: Any, ctx: SessionEnv) -> Any:
        """Build the post-analysis param instance presented to the user."""
        ...

    def post_analyze(
        self, req: PostAnalyzeRequest[Any, Any, Any], *, plots: Plots
    ) -> Any:
        """Run the second-layer analysis on the primary analyze result."""
        ...

    def get_writeback_items(
        self, req: WritebackRequest[Any, Any]
    ) -> Sequence[WritebackItem]:
        """Return the writeback items proposed from run/analyze results."""
        ...

    def get_post_writeback_items(
        self, req: PostWritebackRequest[Any, Any, Any]
    ) -> Sequence[WritebackItem]:
        """Return the writeback items proposed from one post-analysis result."""
        ...

    def prepare_save_data(self, result: object, *, comment: str) -> PreparedRunData:
        """Prepare this experiment's RunRecord for both native data writers.

        No I/O or metadata capture occurs. Invalid record/cfg/Result/comment or
        a missing persistence declaration raises TypeError, ValueError or
        NotImplementedError. Do not mutate result arrays during saving.
        """
        ...

    def save(self, req: SaveDataRequest[Any]) -> None:
        """Persist the run result to disk."""
        ...
