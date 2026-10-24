from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Generic,
    Protocol,
    runtime_checkable,
)

from typing_extensions import TypeVar

if TYPE_CHECKING:
    from zcu_tools.cfg_model import CfgValidationContext, ConfigBase
    from zcu_tools.gui.session.parameters import ParameterContext


# ``SocProtocol``/``SocCfgProtocol`` + the ``SocHandle``/``SocCfgHandle`` aliases,
# ``ContextReadiness``, and ``SessionEnv`` are session-core value types — they live
# in ``gui/session/types`` (no experiment cfg-tree coupling). They are re-exported
# from the adapter package because callback signatures speak in
# them (``RunRequest.soc: SocHandle``, ``RunRequest.soccfg: SocCfgHandle``).
from zcu_tools.gui.session.types import SessionEnv, SocCfgHandle, SocHandle

T_Result = TypeVar("T_Result")
T_Cfg = TypeVar("T_Cfg", bound="ConfigBase")


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
    """Read-only capability projection from an Adapter value.

    requires_soc and load_data are True for every Adapter. analysis selects
    its primary stage mode; post_analysis records a PostAnalysis declaration.
    """

    requires_soc: bool = True
    analysis: AnalysisMode = AnalysisMode.FIT
    post_analysis: bool = False
    load_data: bool = False


@dataclass(frozen=True)
class AdapterGuide:
    """Human-facing Markdown and the directory for its relative resources.

    markdown is the complete guide text, without section parsing or rewriting.
    An adapter without a guide returns an unwritten-guide notice instead.
    resource_dir is the directory against which consumers resolve relative image
    paths, or None when the adapter declares no guide. It does not imply that
    any referenced image exists or has been read.
    """

    markdown: str
    resource_dir: Path | None


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
        # tolerated: Result subclasses are dataclasses; this mixin reflects their declared fields.
        for f in dataclasses.fields(self):  # pyright: ignore[reportArgumentType]
            val = getattr(self, f.name)
            if _is_json_safe(val):
                result[f.name] = val
        return result


@runtime_checkable
class AnalyzeResult(Protocol):
    """Numeric analysis payload; figures belong to the operation's plots."""

    def to_summary_dict(self) -> dict[str, object]: ...


# None means no numeric result or no parameters; omitted slots use it too.
T_AnalyzeResult = TypeVar("T_AnalyzeResult", bound=AnalyzeResult | None, default=None)
T_AnalyzeParams = TypeVar("T_AnalyzeParams", default=None)


@dataclass(frozen=True)
class RunRequest:
    """Observed SoC handles and the fixed cfg materialization capability.

    cfg_context is the fixed library/assets validation context captured for
    this Run; None permits only dependency-free inline cfg inputs.
    """

    soc: SocHandle | None
    soccfg: SocCfgHandle | None
    cfg_context: CfgValidationContext | None


@dataclass(frozen=True)
class AnalyzeRequest(Generic[T_Result, T_AnalyzeParams]):
    """Inputs accepted for one primary analysis, independent of later session edits.

    run_result is the selected run data. analyze_params holds adapter-specific
    inputs. parameters is the Run/Load result's fixed source, or None for pure
    calculation.
    """

    run_result: T_Result
    analyze_params: T_AnalyzeParams
    parameters: ParameterContext | None


# ---------------------------------------------------------------------------
# Post-analysis (AdapterCapabilities.post_analysis) — a second analysis layer
# that operates *on top of* the primary ``analyze`` result. It mirrors the
# primary analyze chain (request + numeric result), but carries the
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
    bound=AnalyzeResult | None,
    default=None,
)
T_PostAnalyzeParams = TypeVar("T_PostAnalyzeParams", default=None)


@dataclass(frozen=True)
class PostAnalyzeRequest(Generic[T_Result, T_AnalyzeResult, T_PostAnalyzeParams]):
    """Mirror of :class:`AnalyzeRequest` for the post-analysis layer.

    run_result is the selected run data. analyze_result is the required primary
    analysis result. post_analyze_params are adapter-specific inputs. parameters is the Run/Load
    result's fixed source, or None for pure calculation.
    Later tab bindings
    and session edits do not replace these request inputs.
    """

    run_result: T_Result
    analyze_result: T_AnalyzeResult
    post_analyze_params: T_PostAnalyzeParams
    parameters: ParameterContext | None


@dataclass(frozen=True)
class WritebackRequest(Generic[T_Result, T_AnalyzeResult]):
    run_result: T_Result
    analyze_result: T_AnalyzeResult
    ctx: SessionEnv


@dataclass(frozen=True)
class PostWritebackRequest(Generic[T_Result, T_AnalyzeResult, T_PostAnalyzeResult]):
    """Inputs for the optional post-analysis proposal hook.

    The workflow constructs this from the same operation-start values used by
    ``post_analyze``. Writeback remains unaware of this type; it only receives
    the resulting proposal items.
    """

    run_result: T_Result
    analyze_result: T_AnalyzeResult
    post_analyze_result: T_PostAnalyzeResult
    ctx: SessionEnv
