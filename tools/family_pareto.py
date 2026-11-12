"""Measure Engine.evolve families, not program.acquire latency."""

import argparse
import json
import logging
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, closing, contextmanager
from dataclasses import asdict, dataclass, field
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter, process_time
from typing import Protocol, runtime_checkable

from zcu_tools.simulate.evolution import (
    Engine,
    EvolutionResult,
    ModelSource,
    TimelineBatch,
)

from tests.simulate.evolution.contract.accuracy import ALL_FAMILIES
from tests.simulate.evolution.contract.accuracy_cases import AccuracyCase
from tests.simulate.evolution.contract.interaction_reference import (
    interaction_reference,
)
from tests.simulate.evolution.contract.reference import (
    ReferenceResult,
    converged_reference,
    density_error,
)
from tools.family_pareto_inputs import (
    Input,
    canonical_json,
    digest,
    freeze_manifest,
    make_inputs,
    physical_input,
)
from tools.family_pareto_reference import solve_reference


@runtime_checkable
class Workload(Protocol):
    """Provider-owned detached numeric input consumed without lab knowledge.

    name is stable. source supplies immutable physics and initial density; batch
    holds all points and their complete timelines. metadata describes why the
    input represents a workload; manifest contains reproducible provenance.
    domain_spans declares span counts from input physics. family_domains maps
    each family to its public candidate domains. companions declares the fixed
    family/domain companions, which cannot overlap the target's domains.
    Providers return a fresh record with an owned source. Tools detach its models
    and use new sources/histories for every measurement session.
    """

    name: str
    source: ModelSource
    batch: TimelineBatch
    metadata: dict[str, object]
    manifest: dict[str, object]
    domain_spans: dict[str, int]
    family_domains: dict[str, tuple[str, ...]]
    companions: dict[str, tuple[str, ...]]


def load_provider(path: str) -> Input:
    """Load a zero-argument module:attribute provider as a real-workload input.

    Import/factory failures propagate. The provider must return Workload or
    TypeError is raised. Its owned source is closed after detaching immutable
    models at every scheduled level plus DC. No acquire or lab details enter
    the generic measurement sessions.
    """
    module, attribute = path.split(":", 1)
    factory = getattr(import_module(module), attribute)
    workload = factory()
    if not isinstance(workload, Workload):
        raise TypeError(f"{path} does not provide the Workload protocol")
    with closing(workload.source):
        levels = {tuple(int(v) for v in level) for level in workload.batch.flux.level}
        levels.add(tuple(0 for _ in workload.batch.flux.level[0]))
        models = {level: workload.source.model(level) for level in levels}
        initial = workload.source.initial_density
        subdivisions = workload.source.slow_envelope_segments_per_period
        expansion = (
            workload.source.flux_expansion()
            if workload.batch.flux_waveform is not None
            else None
        )

    def build() -> AccuracyCase:
        return AccuracyCase(
            workload.name,
            ModelSource(
                initial,
                models.__getitem__,
                expansion,
                slow_envelope_segments_per_period=subdivisions,
            ),
            workload.batch,
            tuple(workload.family_domains),
        )

    return Input(
        workload.name,
        build,
        str(workload.metadata["reason"]),
        domain_spans=workload.domain_spans,
        family_domains=workload.family_domains,
        companions=workload.companions,
        metadata=workload.metadata,
        manifest=workload.manifest,
        category="real",
    )


@dataclass
class EngineSession:
    """Adapter borrowed from the caller's context-managed fresh Engine session.

    evolve calls public Engine.evolve and returns its result; it must not perform
    setup/cleanup. clear_history clears that source's routing history, untimed.
    The owning context manager releases the source and Engine on exit.
    """

    evolve: Callable[[], EvolutionResult]
    clear_history: Callable[[], None]
    subset_refusal: Callable[[tuple[str, ...]], str | None] | None = None
    """Optional full-input probe: public domain refusal text, or None on success.

    Other failures must propagate. It creates its own fresh session, untimed.
    Required only for policies with a distinct companion family.
    """


SessionOpener = Callable[[], AbstractContextManager[EngineSession]]


@dataclass(frozen=True)
class Sample:
    """One timed evolve, inspected after the clocks stop.

    wall_seconds and cpu_seconds are elapsed wall/process CPU seconds.
    density_error is the maximum absolute complex density difference to oracle,
    or None when measure_family receives reference=None for timing/path only.
    paths maps executed family names to segment counts.
    """

    wall_seconds: float
    cpu_seconds: float
    density_error: float | None
    paths: dict[str, int]


@dataclass(frozen=True)
class ModeResult:
    """One history mode's raw samples and separate timing medians.

    mode is first_evolve, warm_clear_history or warm_retain_history.
    samples contains five evolves, or one when its first evolve exceeds 60 wall
    seconds, in measurement order. wall_median and
    cpu_median are the respective elapsed-second medians of those samples.
    """

    mode: str
    samples: list[Sample]
    wall_median: float
    cpu_median: float


def measure_family(
    open_session: SessionOpener,
    reference: ReferenceResult | None,
    *,
    wall_clock: Callable[[], float] = perf_counter,
    cpu_clock: Callable[[], float] = process_time,
) -> list[ModeResult]:
    """Measure up to five fresh or warmed evolves per mode; failures propagate.

    If the first timed evolve exceeds 60 wall seconds, keep only that sample
    for that mode. Exactly 60 seconds retains all five. Warmups are untimed.

    Setup, cleanup, two identical warmups, clear and result inspection are
    untimed. open_session supplies a fresh source/history/Engine on every entry.
    Clocks return monotonic wall and process CPU seconds. No sample spawns a
    process or resets disk caches. Only public Engine.evolve is timed.
    reference is a real oracle result, or explicitly None to omit density error
    measurement (timing/path-only consumers). No oracle is solved here.
    """
    modes = []
    for mode in ("first_evolve", "warm_clear_history", "warm_retain_history"):
        samples = []
        for _ in range(5 if mode == "first_evolve" else 1):
            with open_session() as session:
                if mode != "first_evolve":
                    for _ in range(2):
                        session.evolve()
                for _ in range(1 if mode == "first_evolve" else 5):
                    if mode == "warm_clear_history":
                        session.clear_history()
                    wall, cpu = wall_clock(), cpu_clock()
                    result = session.evolve()
                    cpu_elapsed = cpu_clock() - cpu
                    wall_elapsed = wall_clock() - wall
                    samples.append(
                        Sample(
                            wall_elapsed,
                            cpu_elapsed,
                            None
                            if reference is None
                            else density_error(result.densities, reference.density),
                            dict(result.paths.segments_by_kernel),
                        )
                    )
                    if samples[0].wall_seconds > 60:
                        break
            if samples[0].wall_seconds > 60:
                break
        modes.append(
            ModeResult(
                mode,
                samples,
                median(s.wall_seconds for s in samples),
                median(s.cpu_seconds for s in samples),
            )
        )
    return modes


@dataclass(frozen=True)
class FamilyResult:
    """Pinned family or default policy; unavailable has reason and no samples.

    family is a pinned family name, or 'default' for normal routing.
    error is the maximum density error across probe and all timed evolves for
    overall accuracy reporting/PASS-FAIL, never an assumed exact-solver zero.
    unavailable is the refusal reason, with error=None and modes=[]; otherwise
    unavailable=None and modes contains each of the three history modes once,
    in any order. modes retain separate raw samples and wall/CPU medians.
    """

    family: str
    error: float | None
    unavailable: str | None
    modes: list[ModeResult]
    domains: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Pinned family -> disjoint owned domains, empty for synthetic/default."""
    expected_segments: dict[str, int] = field(default_factory=dict)
    """Family counts inferred from declared domains, not per-span observations."""
    subset_refusals: dict[str, str] = field(default_factory=dict)
    """Full-batch target-only and companions-only domain-refusal evidence."""
    attribution: str = "target"
    """target, default, fixed_family_group or target_plus_driving_companion.

    Group times must never be attributed to the target alone.
    """
    whitelist: tuple[str, ...] | None = None
    """Actual families restriction; None denotes unrestricted default routing."""


def _mode_error(mode: ModeResult) -> float:
    errors = []
    for sample in mode.samples:
        if sample.density_error is None:
            raise ValueError(f"{mode.mode}: require measured density errors")
        errors.append(sample.density_error)
    if not errors:
        raise ValueError(f"{mode.mode}: require measured density errors")
    return max(errors)


def _available_modes(rows: list[FamilyResult]) -> dict[str, dict[str, ModeResult]]:
    modes = {}
    for row in rows:
        if row.unavailable is not None or row.family == "default":
            continue
        if row.error is None:
            raise ValueError(f"{row.family}: require measured density error")
        by_name = {mode.mode: mode for mode in row.modes}
        if len(row.modes) != 3 or set(by_name) != {
            "first_evolve",
            "warm_clear_history",
            "warm_retain_history",
        }:
            raise ValueError(f"{row.family}: require all three unique history modes")
        for mode in row.modes:
            _mode_error(mode)
        modes[row.family] = by_name
    return modes


def dominance(rows: list[FamilyResult]) -> dict[str, dict[str, list[str]]]:
    """List dominators by family and mode/clock; omit unavailable and default.

    A competitor must have error <= target error and time < .9*target time.
    Error is the maximum of that history mode's raw sample density errors;
    probe/other-mode errors and FamilyResult.error do not enter this comparison.
    Equality at ten percent is equivalent timing, with no error epsilon.
    All three unique history modes are required per available row, in any order;
    missing/duplicate/unknown modes raise ValueError. Available rows also require
    an overall error and nonempty measured density errors in every mode;
    timing/path-only results raise ValueError, never become unavailable.
    """
    modes = _available_modes(rows)
    available = [r for r in rows if r.family in modes]
    result: dict[str, dict[str, list[str]]] = {}
    for target in available:
        cells = {}
        for mode in target.modes:
            for metric in ("wall", "cpu"):
                target_time = mode.wall_median if metric == "wall" else mode.cpu_median
                cells[f"{mode.mode}/{metric}"] = [
                    other.family
                    for other in available
                    if other.family != target.family
                    and _mode_error(modes[other.family][mode.mode]) <= _mode_error(mode)
                    and (
                        modes[other.family][mode.mode].wall_median
                        if metric == "wall"
                        else modes[other.family][mode.mode].cpu_median
                    )
                    < 0.9 * target_time
                ]
        result[target.family] = cells
    return result


@contextmanager
def engine_session(item: Input, family: str) -> Generator[EngineSession]:
    """Own a fresh source/history and single-worker Engine for one family policy.

    'default' uses normal routing. Other names pin the target plus declared
    companions, preserving numerical admission. Construction, cleanup and clear are outside timers.
    """
    families = (
        None
        if family == "default"
        else tuple(dict.fromkeys((family, *item.companions)))
    )
    with whitelist_session(item, families) as session:
        yield session


@contextmanager
def whitelist_session(
    item: Input, families: tuple[str, ...] | None
) -> Generator[EngineSession]:
    """Own a fresh source/history and Engine with exactly the supplied whitelist.

    None retains default routing. Unknown families and all numeric, evolve or
    cleanup failures propagate. Setup and cleanup remain outside timing.
    """
    case = item.build()
    with closing(case.source), closing(Engine(1, families=families)) as engine:
        yield EngineSession(
            lambda: engine.evolve(case.source, (case.batch,), budget=1e-3),
            case.source.history.clear,
            lambda subset: probe_subset_refusal(item, subset),
        )


def probe_subset_refusal(item: Input, families: tuple[str, ...]) -> str | None:
    """Probe the complete input with one whitelist, outside every sample timer.

    Return only a public missing-kernel refusal, or None when evolution succeeds.
    Other runtime, numerical and cleanup failures propagate to the caller.
    """
    case = item.build()
    with closing(case.source), closing(Engine(1, families=families)) as engine:
        try:
            engine.evolve(case.source, (case.batch,), budget=1e-3)
        except RuntimeError as error:
            if re.fullmatch(UNAVAILABLE_PATTERN, str(error)) is None:
                raise
            return str(error)
    return None


PolicyOpener = Callable[[Input, str], AbstractContextManager[EngineSession]]


def _subset_refusals(
    session: EngineSession, family: str, companions: tuple[str, ...]
) -> dict[str, str]:
    if family == "default" or not companions:
        return {}
    if session.subset_refusal is None:
        raise ValueError("companion proof requires a full-input subset probe")
    refusals = {}
    for label, subset in (
        ("target_only", (family,)),
        ("companions_only", companions),
    ):
        refusal = session.subset_refusal(subset)
        if refusal is None:
            raise RuntimeError(f"{label} subset unexpectedly accepted full input")
        if re.fullmatch(UNAVAILABLE_PATTERN, refusal) is None:
            raise RuntimeError(refusal)
        refusals[label] = refusal
    return refusals


def measure_policy(
    item: Input,
    family: str,
    reference: ReferenceResult,
    *,
    opener: PolicyOpener = engine_session,
    wall_clock: Callable[[], float] = perf_counter,
    cpu_clock: Callable[[], float] = process_time,
) -> FamilyResult:
    """Probe availability, then measure a fixed family or default policy.

    Only the public missing-kernel/domain refusal may become unavailable.
    Other setup/evolve/cleanup failures propagate. Every measured pinned path
    must execute the target, plus declared companions when present. Domains
    must be disjoint; counts must match input-declared domain spans. Both
    target-only and companions-only full-input probes must refuse publicly.
    This infers allocation from the public domain contract, not per-span
    observations. Oracle is supplied once per physical input by the caller;
    exact families still have measured density errors. Unavailable rows carry
    no synthetic zero errors or times.
    """
    domains = (
        {**item.companions, family: item.family_domains[family]}
        if family != "default" and item.domain_spans
        else {}
    )
    seen: set[str] = set()
    for owned in domains.values():
        if seen.intersection(owned):
            raise ValueError(f"pinned {family} has overlapping domains: {domains}")
        seen.update(owned)
    whitelist = (
        None
        if family == "default"
        else tuple(dict.fromkeys((family, *item.companions)))
    )
    with opener(item, family) as session:
        try:
            probe = session.evolve()
        except RuntimeError as error:
            if re.fullmatch(UNAVAILABLE_PATTERN, str(error)) is None:
                raise
            return FamilyResult(
                family, None, str(error), [], domains=domains, whitelist=whitelist
            )
        companions = tuple(name for name in item.companions if name != family)
        refusals = _subset_refusals(session, family, companions)
    modes = measure_family(
        lambda: opener(item, family),
        reference,
        wall_clock=wall_clock,
        cpu_clock=cpu_clock,
    )
    paths = [dict(probe.paths.segments_by_kernel)] + [
        s.paths for m in modes for s in m.samples
    ]
    expected = {}
    if family != "default":
        if item.domain_spans:
            expected = {
                name: sum(item.domain_spans.get(domain, 0) for domain in owned)
                for name, owned in domains.items()
            }
            expected = {name: count for name, count in expected.items() if count}
            if expected.get(family, 0) <= 0 or any(path != expected for path in paths):
                raise RuntimeError(
                    f"pinned {family} domain span counts differ: expected {expected}, got {paths}"
                )
        elif any(set(p) != {family} or p[family] <= 0 for p in paths):
            raise RuntimeError(f"pinned {family} did not execute exclusively: {paths}")
    error = max(
        density_error(probe.densities, reference.density),
        max(_mode_error(m) for m in modes),
    )
    driving = {"slow", "plateau", "shaped", "two", "multiple", "flux"}
    attribution = (
        "default"
        if family == "default"
        else "target_plus_driving_companion"
        if any(
            driving.intersection(owned)
            for name, owned in domains.items()
            if name != family
        )
        else "target"
    )
    return FamilyResult(
        family, error, None, modes, domains, expected, refusals, attribution, whitelist
    )


# Only this verified public refusal is recoverable. Numerical failures propagate.
UNAVAILABLE_PATTERN = (
    r"no valid kernel for segment \d+; domain [a-z_]+: "
    r"points \[\d+(?:, \d+)*\], segments \[\d+(?:, \d+)*\]"
)


@dataclass(frozen=True)
class InputResult:
    """Complete input row, with independent wall/CPU and history-mode cells.

    name is the stable manifest input name. physical_sha256 hashes the complete
    input; oracle_sha256 excludes nonphysical identity for shared references.
    oracle_convergence_error is the maximum density difference between the two
    oracle tolerance levels. policies contains pinned results and default.
    dominators maps pinned family -> 'mode/wall' or 'mode/cpu' -> dominating
    family names. timing_equivalents has the same keys and time-equivalent peers.
    Unavailable policies have no comparison cells; default is not a competitor.
    """

    name: str
    physical_sha256: str
    oracle_sha256: str
    oracle_convergence_error: float
    policies: list[FamilyResult]
    dominators: dict[str, dict[str, list[str]]]
    timing_equivalents: dict[str, dict[str, list[str]]]
    category: str = "synthetic"
    """real or synthetic; deletion conclusions use only real workloads."""
    oracle_wall_seconds: float = 0.0
    """Elapsed solve/cache-load wall seconds, zero for an in-memory reference hit."""
    oracle_cpu_seconds: float = 0.0
    """Process CPU seconds spent in that solve/cache load, excluding setup."""
    fixed_default_group: FamilyResult | None = None
    """Separately timed whitelist of the default route's observed family union."""


def timing_equivalents(rows: list[FamilyResult]) -> dict[str, dict[str, list[str]]]:
    """List time-equivalent peers when neither is >10% faster, ignoring error.

    This is symmetric: other_time >= .9*target_time and target_time >=
    .9*other_time. It is not Pareto dominance and never merges wall and CPU.
    All three unique history modes are required per available row, in any order;
    missing/duplicate/unknown modes raise ValueError. Unavailable and default
    rows are omitted. Available rows require an overall error and nonempty
    measured density errors in every mode; timing/path-only results raise
    ValueError rather than becoming unavailable.
    """
    modes = _available_modes(rows)
    available = [r for r in rows if r.family in modes]
    equivalents: dict[str, dict[str, list[str]]] = {}
    for target in available:
        cells = {}
        for mode in target.modes:
            for metric in ("wall", "cpu"):
                target_time = mode.wall_median if metric == "wall" else mode.cpu_median
                peers = []
                for other in available:
                    if other.family == target.family:
                        continue
                    other_time = (
                        modes[other.family][mode.mode].wall_median
                        if metric == "wall"
                        else modes[other.family][mode.mode].cpu_median
                    )
                    if (
                        other_time >= 0.9 * target_time
                        and target_time >= 0.9 * other_time
                    ):
                        peers.append(other.family)
                cells[f"{mode.mode}/{metric}"] = peers
        equivalents[target.family] = cells
    return equivalents


def measure_fixed_group(
    item: Input, families: tuple[str, ...], reference: ReferenceResult
) -> FamilyResult:
    """Time the default route's observed family union as one fixed whitelist.

    The complete input, budget and three history modes match measure_policy.
    This is a group latency, never attributed to any one member or included
    among pinned-family dominance competitors. Failures propagate.
    """
    modes = measure_family(lambda: whitelist_session(item, families), reference)
    return FamilyResult(
        "fixed_default_group",
        max(_mode_error(mode) for mode in modes),
        None,
        modes,
        attribution="fixed_family_group",
        whitelist=families,
    )


def run_table(
    inputs: list[Input],
    families: tuple[str, ...] = ALL_FAMILIES,
    *,
    reference_solver: Callable[[ModelSource, TimelineBatch], ReferenceResult]
    | None = None,
    reference_cache: Path | None = None,
    policy_runner: Callable[
        [Input, str, ReferenceResult], FamilyResult
    ] = measure_policy,
    group_runner: Callable[[Input, tuple[str, ...], ReferenceResult], FamilyResult]
    | None = None,
) -> list[InputResult]:
    """Measure every family and default for each input, sharing identical oracles.

    Inputs stay in caller order. Independent two-level full-Lindblad references
    are cached by complete physical content, not family or input alias. All
    failures except the precisely matched probe refusal stop the whole table.
    Real inputs use the exact interaction-picture solver; synthetic inputs keep
    the lab-frame solver. reference_cache optionally persists complete references;
    an injected reference_solver owns its own caching and tolerance contract.
    Real default paths are also timed with their observed family union pinned,
    separately from dominance. group_runner overrides only that group measurement.
    """
    references: dict[str, ReferenceResult] = {}
    rows = []
    for index, item in enumerate(inputs):
        case = item.build()
        with closing(case.source):
            key = digest(physical_input(case))
            oracle_key = digest(physical_input(case, include_identity=False))
            oracle_wall = oracle_cpu = 0.0
            if oracle_key not in references:
                wall, cpu = perf_counter(), process_time()
                references[oracle_key] = (
                    reference_solver(case.source, case.batch)
                    if reference_solver is not None
                    else solve_reference(
                        case,
                        solver=interaction_reference
                        if item.category == "real"
                        else converged_reference,
                        cache_dir=reference_cache,
                    )
                )
                oracle_wall, oracle_cpu = perf_counter() - wall, process_time() - cpu
        reference = references[oracle_key]
        policies = [
            policy_runner(item, family, reference) for family in (*families, "default")
        ]
        default = policies[-1]
        observed = tuple(
            sorted(
                {
                    name
                    for mode in default.modes
                    for sample in mode.samples
                    for name in sample.paths
                }
            )
        )
        fixed_group = (
            (measure_fixed_group if group_runner is None else group_runner)(
                item, observed, reference
            )
            if item.category == "real" and observed
            else None
        )
        rows.append(
            InputResult(
                item.name,
                key,
                oracle_key,
                reference.convergence_error,
                policies,
                dominance(policies),
                timing_equivalents(policies),
                item.category,
                oracle_wall,
                oracle_cpu,
                fixed_group,
            )
        )
        print(
            f"{index + 1}/{len(inputs)} {item.name}: "
            f"{sum(p.error is not None for p in policies)} available policies",
            flush=True,
        )
    return rows


def render_summary(
    rows: list[InputResult], families: tuple[str, ...] = ALL_FAMILIES
) -> str:
    """Render per-family conclusions and all measured mode/clock cells.

    Dominance is limited to measured inputs. No family deletion or accuracy
    exception follows from this report. Default is shown but never a competitor.
    """
    lines = [
        "# Engine family Pareto table",
        "",
        "Timers cover Engine.evolve only, not program.acquire. Each mode has up to five samples; a first timed evolve >60 wall seconds retains one.",
        "Errors compare complete complex density to the two-level converged full-Lindblad oracle.",
        "A dominator has mode-local maximum sample error <= target error and time < 0.9 * target time. Exactly 10% is equivalent.",
        "Displayed density error and PASS/FAIL use the overall maximum across probe and all modes, not the mode-local comparator error.",
        "No epsilon or weighted score is used. FAIL means error > 2e-3, not permission to keep an inaccurate route.",
        "Conclusions apply only to the measured input set; no deletion is authorized.",
        "",
        "## Family conclusions",
        "",
    ]
    conclusion_rows = [row for row in rows if row.category == "real"]
    if not conclusion_rows:
        conclusion_rows = rows
        lines += [
            "No real workloads measured. Synthetic results cannot justify deletion.",
            "",
        ]
    for family in families:
        available = [
            (r, p)
            for r in conclusion_rows
            for p in r.policies
            if p.family == family and p.error is not None
        ]
        nondominated = [
            (r.name, cell)
            for r, _ in available
            for cell, competitors in r.dominators[family].items()
            if not competitors
        ]
        lines += [
            f"### {family}",
            "",
            f"Available on {len(available)} inputs; unavailable on {len(conclusion_rows) - len(available)}.",
        ]
        if not available:
            lines.append("NOT EXERCISED. No dominance conclusion is possible.")
        else:
            worst = max(p.error for _, p in available if p.error is not None)
            lines.append(
                f"Maximum density error {worst:.12g}: {'PASS' if worst <= 2e-3 else 'FAIL'}."
            )
            if nondominated:
                lines.append("Not dominated on these input/mode/metric cells:")
                lines += [f"- `{name}`: {cell}" for name, cell in nondominated]
            else:
                lines.append(
                    "Every measured available input/mode/metric cell is dominated; this is not a proof for unmeasured inputs."
                )
        lines.append("")
    lines += [
        "## Complete measured matrix",
        "",
        "Unavailable cells retain their exact refusal; they have no error or timing value.",
        "Each measured cell lists its dominators. An empty list means not dominated.",
        "Raw samples, per-sample paths and full inputs are in the sibling JSON.",
        "",
    ]
    category = None
    for row in sorted(rows, key=lambda row: row.category):
        if row.category != category:
            category = row.category
            lines += [f"## {category.capitalize()} inputs", ""]
        lines += [
            f"### {row.name}",
            "",
            f"Oracle convergence error: {row.oracle_convergence_error:.12g}; solve/cache-load wall {row.oracle_wall_seconds:.9g}s, CPU {row.oracle_cpu_seconds:.9g}s.",
            "",
            *[
                f"- {policy.family} ({policy.attribution}): domains {policy.domains}; expected segments {policy.expected_segments}; full-input subset refusals {policy.subset_refusals}."
                for policy in row.policies
                if policy.domains
            ],
            "",
            "| Family | Density error | Status | Mode/metric | Median seconds | Samples | Actual path | Dominators | Time-equivalent peers |",
            "| --- | ---: | --- | --- | ---: | ---: | --- | --- | --- |",
        ]
        for policy in row.policies:
            if policy.error is None:
                lines.append(
                    f"| {policy.family} | — | unavailable: {policy.unavailable} | — | — | — | — | — | — |"
                )
                continue
            for mode in policy.modes:
                for metric, value in (
                    ("wall", mode.wall_median),
                    ("cpu", mode.cpu_median),
                ):
                    cell = f"{mode.mode}/{metric}"
                    competitors = row.dominators.get(policy.family, {}).get(cell, [])
                    peers = row.timing_equivalents.get(policy.family, {}).get(cell, [])
                    lines.append(
                        f"| {policy.family} | {policy.error:.12g} | {'PASS' if policy.error <= 2e-3 else 'FAIL'} | "
                        f"{cell} | {value:.9g} | {len(mode.samples)} | {mode.samples[0].paths} | {', '.join(competitors) or 'none'} | {', '.join(peers) or 'none'} |"
                    )
        lines.append("")
        lines += _render_fixed_group(row)
    return "\n".join(lines) + "\n"


def _render_fixed_group(row: InputResult) -> list[str]:
    fixed = row.fixed_default_group
    if fixed is None:
        return []
    default = next(policy for policy in row.policies if policy.family == "default")
    defaults = {mode.mode: mode for mode in default.modes}
    lines = [
        f"Default vs fixed same family group {fixed.whitelist} (group latency, no single-family attribution):",
        "",
        "| Mode/metric | Default seconds | Fixed seconds | Fixed samples | Fixed path | Default/fixed | >2x difference |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for mode in fixed.modes:
        original = defaults[mode.mode]
        for metric, default_time, fixed_time in (
            ("wall", original.wall_median, mode.wall_median),
            ("cpu", original.cpu_median, mode.cpu_median),
        ):
            ratio = default_time / fixed_time if fixed_time > 0 else None
            ratio_text = "undefined" if ratio is None else f"{ratio:.6g}"
            finding = ratio is not None and (ratio > 2 or ratio < 0.5)
            lines.append(
                f"| {mode.mode}/{metric} | {default_time:.9g} | {fixed_time:.9g} | "
                f"{len(mode.samples)} | {mode.samples[0].paths} | {ratio_text} | {finding} |"
            )
    return [*lines, ""]


def tested_state() -> dict[str, str | bool]:
    """Read Git HEAD/tree/dirty facts through the external Git process boundary."""

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], text=True).strip()

    return {
        "head": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "dirty": bool(git("status", "--porcelain")),
        "cwd": str(Path.cwd()),
    }


def main(argv: list[str] | None = None, *, inputs: list[Input] | None = None) -> int:
    """Freeze inputs or measure a matching manifest; output failures return nonzero.

    --freeze-inputs writes inputs before any oracle or timing. Otherwise require
    --manifest, --output and --summary. A manifest mismatch or measurement failure
    leaves output untouched. Output write failures can leave partial files and
    return 1 with the original traceback. Run measurement through run_heavy
    --exclusive; CLI never starts per-sample processes. Optional inputs replace
    only physical input construction for small in-process seam tests.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", action="append", default=[])
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--freeze-inputs", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    try:
        items = (
            (make_inputs() if inputs is None else inputs)
            if not args.provider or args.include_synthetic
            else []
        )
        items = [*items, *(load_provider(path) for path in args.provider)]
        manifest = freeze_manifest(items)
        if args.freeze_inputs is not None:
            args.freeze_inputs.write_text(
                canonical_json(manifest) + "\n", encoding="utf-8"
            )
            print(f"Frozen {len(items)} inputs; manifest sha256={digest(manifest)}")
            return 0
        if args.manifest is None or args.output is None or args.summary is None:
            raise ValueError("measurement requires --manifest, --output and --summary")
        frozen = json.loads(args.manifest.read_text(encoding="utf-8"))
        if digest(frozen) != digest(manifest):
            raise ValueError(
                "frozen manifest differs from current inputs; do not tune inputs after measurement"
            )
        state = tested_state()
        rows = run_table(items, reference_cache=args.reference_cache)
        result = {
            "schema_version": 1,
            "manifest": manifest,
            "manifest_sha256": digest(manifest),
            "tested_state": state,
            "argv": [
                sys.executable,
                "-m",
                "tools.family_pareto",
                *(sys.argv[1:] if argv is None else argv),
            ],
            "libraries": {
                "python": platform.python_version(),
                **{
                    name: version(name)
                    for name in ("zcu-tools", "numpy", "scipy", "numba", "pydantic")
                },
            },
            "native_thread_settings": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                )
            },
            "measurement": {
                "entry": "Engine.evolve",
                "budget": 1e-3,
                "workers": 1,
                "maximum_samples_per_mode": 5,
                "slow_first_evolve_wall_seconds": 60,
                "reference_cache": str(args.reference_cache)
                if args.reference_cache
                else None,
                "reference_total_wall_seconds": sum(
                    row.oracle_wall_seconds for row in rows
                ),
                "reference_total_cpu_seconds": sum(
                    row.oracle_cpu_seconds for row in rows
                ),
                "warmups_per_warm_mode": 2,
                "oracle_coarse": [2e-10, 2e-12],
                "oracle_fine": [2e-12, 2e-14],
                "oracle_convergence_limit": 2e-6,
                "accuracy_limit": 2e-3,
                "dominance": "competitor_error <= target_error and competitor_time < .9 * target_time",
                "unavailable_pattern": UNAVAILABLE_PATTERN,
            },
            "inputs": [asdict(row) for row in rows],
        }
        summary = render_summary(rows)
        args.output.write_text(canonical_json(result) + "\n", encoding="utf-8")
        args.summary.write_text(summary, encoding="utf-8")
    except Exception:
        # CLI isolation preserves failures; unavailable is handled only at probe.
        logging.exception("Pareto table failed; no complete report was written")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
