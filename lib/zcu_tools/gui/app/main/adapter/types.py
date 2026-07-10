from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
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
from zcu_tools.gui.cfg import CfgSchema
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
    # Optional GUI/MCP role metadata for the proposed ModuleLibrary entry. This is
    # not persisted by ModuleLibrary; it identifies which role template the writeback
    # proposal represents while the user/agent reviews the draft.
    role_id: str | None = None
    # editor_id of the service-owned (gc=False) cfg model that holds this item's
    # live draft (ADR-0008). Stamped by WritebackService at compute time; the
    # agent edits via editor.set_field(editor_id, …), the user's Edit dialog
    # attaches to the same model.
    editor_id: str | None = field(default=None, init=False)


@dataclass
class WaveformWriteback(WritebackItem):
    edit_schema: CfgSchema | None = None
    role_id: str | None = None
    editor_id: str | None = field(default=None, init=False)


@dataclass
class SavePaths:
    data_path: str
    image_path: str
