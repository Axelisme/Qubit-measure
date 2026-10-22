"""Detached owner evidence for full recipe execution queries."""

from dataclasses import dataclass
from typing import Literal

from zcu_tools.format_version import YamlValue
from zcu_tools.gui.app.measure.remote.form_description import RecipeFieldObservation
from zcu_tools.mcp.measure.analysis_execution import InvalidAnalysisValue
from zcu_tools.mcp.measure.recipe_api import WritebackProposal


@dataclass(frozen=True)
class RecipeAnalysisDetail:
    """One completed owner analysis, not a recomputed fit or estimate selection.

    stage identifies primary/post and op is its session handle. summary preserves
    all owner estimates, including nulls and nested values, in native units.
    invalid preserves the owner's paths/reasons without interpreting quality.
    """

    stage: Literal["primary", "post"]
    op: int
    summary: YamlValue
    invalid: tuple[InvalidAnalysisValue, ...]


@dataclass(frozen=True)
class RecipeRunDetail:
    """Evidence associated with one Run attempt, retained after terminal outcome.

    tab/adapter identify its authored target, run_id is the owner's acquisition
    identity or None until known. op is the session Run handle or None before
    start acknowledgement. conditions is the owner's root logical observation
    immediately before Run, including inputs, values and validation diagnostics;
    None means capture did not complete. analyses retains completed stage facts
    in order. writeback is the last explicit proposal for this Run, or None if
    the author never requested it. Reading details sends no RPC and saves nothing.
    """

    tab: str
    adapter: str
    run_id: str | None = None
    op: int | None = None
    conditions: RecipeFieldObservation | None = None
    analyses: tuple[RecipeAnalysisDetail, ...] = ()
    writeback: WritebackProposal | None = None
