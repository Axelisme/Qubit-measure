from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Protocol, runtime_checkable

from .types import (
    AdapterCapabilities,
    AdapterGuide,
    AnalyzeRequest,
    CfgSchema,
    CfgSectionSpec,
    ExpContext,
    InteractiveHost,
    InteractiveSession,
    NoAnalysisResult,
    NoAnalyzeParams,
    PostAnalyzeRequest,
    RunRequest,
    SaveDataRequest,
    SavePaths,
    WritebackItem,
    WritebackRequest,
)

__all__ = [
    "ExpAdapterProtocol",
    "NoAnalysisResult",
    "NoAnalyzeParams",
]


@runtime_checkable
class ExpAdapterProtocol(Protocol):
    """Structural contract the GUI framework requires from an experiment adapter.

    This Protocol lists *only* the members the framework actually calls; it
    carries no generics and no implementation. The shared default behaviour
    (build_exp_cfg delegation, save-path policy, no-op analysis, …) lives in
    ``zcu_tools.experiment.v2_gui.adapters.base.BaseAdapter``, which adapters
    inherit and which satisfies this Protocol structurally.

    Keeping the framework side generic-free is deliberate: the GUI handles
    every adapter as an opaque ``ExpAdapterProtocol`` and never narrows run /
    analyze / writeback result types (they cross Qt ``Signal(object)``
    boundaries as plain objects). The cross-method type linkage that pyright
    checks lives entirely inside a single concrete adapter via the generic
    ``BaseAdapter[T_Cfg, T_Result, T_AnalyzeResult, T_AnalyzeParams]``.
    """

    # Class-level capability declaration. Implementers (BaseAdapter and any
    # adapter overriding it) must annotate this as ``ClassVar`` too, or pyright
    # rejects the Protocol conformance.
    capabilities: ClassVar[AdapterCapabilities]

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        """Return the static cfg spec tree (no context, no instance state)."""
        ...

    @classmethod
    def guide(cls) -> AdapterGuide:
        """Return the static human-facing orientation guide (no instance state)."""
        ...

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        """Compose the static spec with context-derived default values."""
        ...

    def make_save_paths(self, ctx: ExpContext) -> SavePaths:
        """Resolve the save-path policy for the given context."""
        ...

    @classmethod
    def analyze_params_cls(cls) -> type:
        """Return the analyze-params dataclass type (static, no instance)."""
        ...

    def get_analyze_params(self, result: Any, ctx: ExpContext) -> Any:
        """Build the analyze parameter instance presented to the user."""
        ...

    def run(self, req: RunRequest, schema: CfgSchema) -> Any:
        """Run the experiment and return its result."""
        ...

    def analyze(self, req: AnalyzeRequest[Any, Any]) -> Any:
        """Run analysis on a completed run result."""
        ...

    def setup_interactive_analysis(
        self, req: AnalyzeRequest[Any, Any], host: InteractiveHost
    ) -> InteractiveSession:
        """Set up an interactive analysis on the host's figure (INTERACTIVE)."""
        ...

    @classmethod
    def post_analyze_params_cls(cls) -> type:
        """Return the post-analysis param dataclass type (post_analysis cap)."""
        ...

    def get_post_analyze_params(self, analyze_result: Any, ctx: ExpContext) -> Any:
        """Build the post-analysis param instance presented to the user."""
        ...

    def post_analyze(self, req: PostAnalyzeRequest[Any, Any, Any]) -> Any:
        """Run the second-layer analysis on the primary analyze result."""
        ...

    def get_writeback_items(
        self, req: WritebackRequest[Any, Any]
    ) -> Sequence[WritebackItem]:
        """Return the writeback items proposed from run/analyze results."""
        ...

    def save(self, req: SaveDataRequest[Any]) -> None:
        """Persist the run result to disk."""
        ...
