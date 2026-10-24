from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from zcu_tools.gui.app.measure.artifacts import ArtifactHandover
from zcu_tools.gui.form_adapter import Form, FormContext

if TYPE_CHECKING:
    from zcu_tools.datafile import Container
    from zcu_tools.experiment.context import RunContext
    from zcu_tools.experiment.native_persistence import RestoredRun
    from zcu_tools.experiment.records import RunRecord
    from zcu_tools.plotting.plots import Plots

from zcu_tools.gui.app.measure.writeback import WritebackItem

from .interactive import InteractivePluginProvider
from .types import (
    AdapterCapabilities,
    AdapterGuide,
    AnalyzeRequest,
    PostAnalyzeRequest,
    PostWritebackRequest,
    RunRequest,
    SessionEnv,
    WritebackRequest,
)

__all__ = [
    "ExpAdapterProtocol",
]


@runtime_checkable
class ExpAdapterProtocol(ArtifactHandover, InteractivePluginProvider, Protocol):
    """Structural contract the GUI framework requires from an experiment adapter.

    This Protocol lists *only* the members the framework actually calls; it
    carries no generics and no implementation. The shared default behaviour
    (run delegation, persistence payload preparation, no-op analysis, …) lives in
    ``zcu_tools.gui.app.measure.adapter.base.BaseAdapter``, which adapters
    inherit and which satisfies this Protocol structurally.

    Keeping the framework side generic-free is deliberate: the GUI handles
    every adapter as an opaque ``ExpAdapterProtocol``. Analyze/writeback
    results remain opaque. cfg is the output of this adapter's own Form;
    the framework does not check its type. Run and restore return a RunRecord
    envelope; its cfg and Result stay adapter-owned black boxes. Results
    cross Qt ``Signal(object)`` boundaries as plain objects. The cross-method type linkage that pyright
    checks lives entirely inside a single concrete adapter via the generic
    ``BaseAdapter[T_Cfg, T_Result, T_AnalyzeResult, T_AnalyzeParams]``.
    """

    # Class-level capability declaration. Implementers (BaseAdapter and any
    # adapter overriding it) must annotate this as ``ClassVar`` too, or pyright
    # rejects the Protocol conformance.
    capabilities: ClassVar[AdapterCapabilities]
    # Declaring persistence spec tag for acquisition metadata; None means this
    # adapter does not produce native run evidence. The tag describes evidence,
    # not the concrete shape of the adapter-owned run handle.
    # Never infer it from a name.
    experiment_tag: ClassVar[str | None]
    # Device alias path ("$name") supplying Sample writeback coordinates from
    # captured parameters and Run device evidence. None disables only samples;
    # resolution/missing-evidence failures omit them with an unavailable reason.
    sample_device_alias: ClassVar[str | None]

    @classmethod
    def guide(cls) -> AdapterGuide:
        """Return complete Markdown and its optional relative-resource directory.

        No instance state is needed. Source I/O and decoding errors propagate;
        an undeclared guide returns an unwritten-guide notice with no directory.
        Consumers decide how to render Markdown and whether to read images.
        """
        ...

    @staticmethod
    def make_form(ctx: FormContext) -> Form[Any]:
        """Build a definition using the newly captured Open/Reset context.

        ctx only queries captured reference existence. The framework binds its
        template catalog after return and owns editable state. Factory and
        template errors propagate at use, never during catalog registration.
        """
        ...

    @classmethod
    def analyze_params_cls(cls) -> type:
        """Return the analyze-params dataclass type (static, no instance)."""
        ...

    def get_analyze_params(self, result: Any, ctx: SessionEnv) -> Any:
        """Build the analyze parameter instance presented to the user."""
        ...

    def validate_run_request(self, req: RunRequest, cfg: Any) -> None:
        """Validate adapter-specific run constraints before opening an operation."""
        ...

    def run(
        self,
        req: RunRequest,
        cfg: Any,
        *,
        context: RunContext,
    ) -> RunRecord[Any, Any]:
        """Execute detached resolved inputs; never resolve from live context.

        req provides accepted device/SoC inputs; cfg is this adapter's validated
        Form output, forwarded unchanged, and context owns this acquisition's plots and stop signal.
        Return a RunRecord wrapping the adapter-owned cfg and Result. With
        experiment_tag set and a saved tab origin, the framework installs native
        evidence in record.metadata. Otherwise it does not manufacture metadata.
        Saving consumes ArtifactHandover content, never inferred cfg/result fields.
        Acquisition errors follow the operation's terminal policy: without
        accepted user cancellation they fail the operation. Accepted cancellation
        may instead produce a cancelled outcome; Schedule errors remain failures.
        """
        ...

    def restore(self, stored: Container) -> RestoredRun[RunRecord[Any, Any]]:
        """Restore framework-read native data admitted for this experiment tag.

        Return a RunRecord with historical metadata and cfg_issue: None means typed cfg
        is available; a nonempty string explains why it is unavailable. Valid
        data may load successfully without cfg. Do not inspect current inputs,
        read files or recapture history; the framework uses record.metadata.
        Raise NotImplementedError for unsupported restoration and ValueError for
        invalid experiment data; other core failures propagate before replacement.
        """
        ...

    def analysis_result_tone_count(self, result: Any) -> int:
        """Return the measured tone count for result-read budgeting; zero for non-mux.

        result is the captured Run/Load result, not current Form input. Do not
        compute analysis or access devices. Return a nonnegative integer.
        """
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
