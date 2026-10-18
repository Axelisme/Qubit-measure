"""Per-original-tone GE analysis using the existing single-qubit owner."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from zcu_tools.experiment import RunRecord
from zcu_tools.experiment.utils.single_shot.ge import GENumericalError
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.mux import MuxFitFailure
from zcu_lab.v2.singleshot.ge.core import (
    GE_Cfg,
    GE_Exp,
    GE_Result,
    GEAnalysis,
    GEAnalyzeOptions,
    GEPostAnalysis,
    GEPostAnalyzeOptions,
)

if TYPE_CHECKING:
    from .core import MuxGECfg, MuxGEResult


@dataclass(frozen=True)
class MuxGEAnalyzeOptions:
    """Existing GE fit options, common default plus original-index overrides.

    default applies to every active tone unless overrides contains its original
    index. Each GEAnalyzeOptions supplies analysis-only ground/excited initial
    state, PCA/center backend, degree angle, T1 alignment, dimensionless length
    ratio and histogram log scale. Unknown/nonactive override indices raise.
    """

    default: GEAnalyzeOptions = field(default_factory=GEAnalyzeOptions)
    overrides: Mapping[int, GEAnalyzeOptions] = field(default_factory=dict)


@dataclass(frozen=True)
class MuxGEPostOptions:
    """Existing GE radius policy, common default plus original-index overrides.

    radius is None for automatic selection or nonnegative finite normalized ADC
    units. consider_other changes only automatic-radius penalty, not the isolated
    other row of the 3x3 diagnostic. Unknown/nonactive override indices raise.
    """

    default: GEPostAnalyzeOptions = field(default_factory=GEPostAnalyzeOptions)
    overrides: Mapping[int, GEPostAnalyzeOptions] = field(default_factory=dict)


@dataclass(frozen=True)
class MuxGEAnalysis:
    """Original tone index and adopted physical g/e primary values/fit quality."""

    index: int
    primary: GEAnalysis


@dataclass(frozen=True)
class MuxGEPostAnalysis:
    """Original tone index and single-qubit 3x3 confusion diagnostic, not joint."""

    index: int
    post: GEPostAnalysis


def _tone_source(result: MuxGEResult, row: int) -> RunRecord[GE_Cfg, GE_Result]:
    # GE analysis accepts cfg=None and consumes only raw IQ and acquisition axes.
    return RunRecord(
        None,
        GE_Result(result.signals[row], result.shot_indices, result.prepared_states),
    )


def _failure_plot(plots: Plots, name: str, reason: str) -> None:
    if name not in plots:
        _, ax = plots.subplots(name)
        ax.text(0.02, 0.98, reason, transform=ax.transAxes, va="top", wrap=True)
    else:
        plots[name].suptitle(reason, wrap=True)


def analyze_ge(
    source: RunRecord[MuxGECfg, MuxGEResult],
    options: MuxGEAnalyzeOptions,
    *,
    plots: Plots,
) -> tuple[MuxGEAnalysis | MuxFitFailure, ...]:
    """Fit finite paired preparation rows without filtering or reordering shots.

    source is a validated explicit (K,2,S) raw IQ result in normalized ADC units.
    cfg=None is supported. Validate option keys before fitting; invalid options
    raise. Each original index gets fit_i and either existing GE calibration /
    quality or a numerical/missing-preparation failure reason. No calibration
    is invented for partial data. Only the GE owner's GENumericalError is
    isolated; plotting and other errors propagate unchanged. Caller owns plots
    finish/release. No hardware, current cfg, binding or proposal access occurs.
    """
    indices = [int(i) for i in source.result.tone_indices]
    if any(type(i) is not int or i not in indices for i in options.overrides):
        raise ValueError("GE overrides must name active original tone indices")
    if any(f"fit_{i}" in plots for i in indices):
        raise ValueError("fit_i figures already exist in this operation")
    selected = [options.overrides.get(i, options.default) for i in indices]
    for option in selected:
        if option.backend not in ("pca", "center") or option.initial_state not in (
            "ground",
            "excited",
        ):
            raise ValueError("invalid GE backend or initial_state")
        if option.angle is not None and not math.isfinite(option.angle):
            raise ValueError("GE angle must be finite degrees")
        if option.length_ratio is not None and (
            not math.isfinite(option.length_ratio) or option.length_ratio < 0
        ):
            raise ValueError("GE length_ratio must be finite and nonnegative")
    outcomes: list[MuxGEAnalysis | MuxFitFailure] = []
    for row, (index, option) in enumerate(zip(indices, selected, strict=True)):
        name = f"fit_{index}"
        if not np.isfinite(source.result.signals[row]).all():
            reason = "GE fit requires both complete probe-off/on preparations"
            _failure_plot(plots, name, reason)
            outcomes.append(MuxFitFailure(index, reason))
            continue
        try:
            primary = GE_Exp().analyze(
                _tone_source(source.result, row), option, plots=plots, figure_name=name
            )
            outcomes.append(MuxGEAnalysis(index, primary))
        except GENumericalError as exc:
            _failure_plot(plots, name, str(exc))
            outcomes.append(MuxFitFailure(index, str(exc)))
    return tuple(outcomes)


def post_analyze_ge(
    source: RunRecord[MuxGECfg, MuxGEResult],
    primary: tuple[MuxGEAnalysis | MuxFitFailure, ...],
    options: MuxGEPostOptions,
    *,
    plots: Plots,
) -> tuple[MuxGEPostAnalysis | MuxFitFailure, ...]:
    """Classify explicit paired IQ with the adopted same-source primary, no refit.

    primary must contain exactly one outcome per active original index in source
    order. Caller must retain its source association. cfg=None is supported.
    Invalid keys/radii or index mismatch raise before diagnostics. Publish post_i;
    retain primary failures and per-tone numerical failures without fabricating a
    calibration. Values are existing GE 3x3 diagnostics, not four-state assignment
    matrices. Only the GE owner's GENumericalError is isolated; plotting and
    other errors propagate unchanged. Caller finishes/releases plots; no hardware
    or live source is read.
    """
    indices = [int(i) for i in source.result.tone_indices]
    if [item.index for item in primary] != indices:
        raise ValueError("primary indices must match source active original order")
    if any(type(i) is not int or i not in indices for i in options.overrides):
        raise ValueError("post overrides must name active original tone indices")
    if any(f"post_{i}" in plots for i in indices):
        raise ValueError("post_i figures already exist in this operation")
    selected = [options.overrides.get(i, options.default) for i in indices]
    if any(
        o.radius is not None and (not math.isfinite(o.radius) or o.radius < 0)
        for o in selected
    ):
        raise ValueError("post radius must be finite and nonnegative ADC units")
    outcomes: list[MuxGEPostAnalysis | MuxFitFailure] = []
    for row, (item, option) in enumerate(zip(primary, selected, strict=True)):
        name = f"post_{item.index}"
        if isinstance(item, MuxFitFailure):
            _failure_plot(plots, name, item.reason)
            outcomes.append(item)
            continue
        if not np.isfinite(source.result.signals[row]).all():
            reason = "GE post requires both complete probe-off/on preparations"
            _failure_plot(plots, name, reason)
            outcomes.append(MuxFitFailure(item.index, reason))
            continue
        try:
            post = GE_Exp().post_analyze(
                _tone_source(source.result, row),
                item.primary,
                option,
                plots=plots,
                figure_name=name,
            )
            outcomes.append(MuxGEPostAnalysis(item.index, post))
        except GENumericalError as exc:
            _failure_plot(plots, name, str(exc))
            outcomes.append(MuxFitFailure(item.index, str(exc)))
    return tuple(outcomes)
