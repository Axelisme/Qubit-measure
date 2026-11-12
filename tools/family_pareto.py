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
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter, process_time

from zcu_tools.simulate.evolution import (
    Engine,
    EvolutionResult,
    ModelSource,
    TimelineBatch,
)

from tests.simulate.evolution.contract.accuracy import ALL_FAMILIES
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


@dataclass
class EngineSession:
    """Untimed ownership adapter: execute one evolve or clear routing history."""

    evolve: Callable[[], EvolutionResult]
    clear_history: Callable[[], None]


SessionOpener = Callable[[], AbstractContextManager[EngineSession]]


@dataclass(frozen=True)
class Sample:
    """One evolve's elapsed seconds, full density error and observed segment counts."""

    wall_seconds: float
    cpu_seconds: float
    density_error: float
    paths: dict[str, int]


@dataclass(frozen=True)
class ModeResult:
    """Five ordered samples and wall/CPU medians for one routing-history mode."""

    mode: str
    samples: list[Sample]
    wall_median: float
    cpu_median: float


def measure_family(
    open_session: SessionOpener,
    reference: ReferenceResult,
    *,
    wall_clock: Callable[[], float] = perf_counter,
    cpu_clock: Callable[[], float] = process_time,
) -> list[ModeResult]:
    """Measure five fresh or warmed evolves per mode; failures propagate.

    Setup, cleanup, two identical warmups, clear and result inspection are
    untimed. open_session supplies a fresh source/history/Engine on every entry.
    Clocks return monotonic wall and process CPU seconds. No sample spawns a
    process or resets disk caches. Only public Engine.evolve is timed.
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
                            density_error(result.densities, reference.density),
                            dict(result.paths.segments_by_kernel),
                        )
                    )
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

    error is the maximum full density error across measured evolves, never an
    assumed zero for exact solvers. modes retain separate wall/CPU measurements.
    """

    family: str
    error: float | None
    unavailable: str | None
    modes: list[ModeResult]


def dominance(rows: list[FamilyResult]) -> dict[str, dict[str, list[str]]]:
    """List dominators by family and mode/clock; omit unavailable and default.

    A competitor must have error <= target error and time < .9*target time.
    Equality at ten percent is equivalent timing, with no error epsilon.
    """
    available = [r for r in rows if r.error is not None and r.family != "default"]
    result: dict[str, dict[str, list[str]]] = {}
    for target in available:
        cells = {}
        for index, mode in enumerate(target.modes):
            for metric in ("wall", "cpu"):
                target_time = mode.wall_median if metric == "wall" else mode.cpu_median
                cells[f"{mode.mode}/{metric}"] = [
                    other.family
                    for other in available
                    if other.family != target.family
                    and other.error is not None
                    and target.error is not None
                    and other.error <= target.error
                    and (
                        other.modes[index].wall_median
                        if metric == "wall"
                        else other.modes[index].cpu_median
                    )
                    < 0.9 * target_time
                ]
        result[target.family] = cells
    return result


@contextmanager
def engine_session(item: Input, family: str) -> Generator[EngineSession]:
    """Own a fresh source/history and single-worker Engine for one family policy.

    'default' uses normal routing. Other names pin families=(family,), preserving
    numerical admission. Construction, cleanup and clear are outside timers.
    """
    case = item.build()
    with (
        closing(case.source),
        closing(
            Engine(1, families=None if family == "default" else (family,))
        ) as engine,
    ):
        yield EngineSession(
            lambda: engine.evolve(case.source, (case.batch,), budget=1e-3),
            case.source.history.clear,
        )


PolicyOpener = Callable[[Input, str], AbstractContextManager[EngineSession]]


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
    must execute only that family and at least one segment. Oracle is supplied
    once per physical input by the caller; exact families still have measured
    density errors. No unavailable row carries synthetic zero errors or times.
    """
    with opener(item, family) as session:
        try:
            probe = session.evolve()
        except RuntimeError as error:
            if re.fullmatch(UNAVAILABLE_PATTERN, str(error)) is None:
                raise
            return FamilyResult(family, None, str(error), [])
    modes = measure_family(
        lambda: opener(item, family),
        reference,
        wall_clock=wall_clock,
        cpu_clock=cpu_clock,
    )
    paths = [dict(probe.paths.segments_by_kernel)] + [
        s.paths for m in modes for s in m.samples
    ]
    if family != "default" and any(set(p) != {family} or p[family] <= 0 for p in paths):
        raise RuntimeError(f"pinned {family} did not execute exclusively: {paths}")
    error = max(
        density_error(probe.densities, reference.density),
        max(s.density_error for m in modes for s in m.samples),
    )
    return FamilyResult(family, error, None, modes)


# Only this verified public refusal is recoverable. Numerical failures propagate.
UNAVAILABLE_PATTERN = (
    r"no valid kernel for segment \d+; domain [a-z_]+: "
    r"points \[\d+(?:, \d+)*\], segments \[\d+(?:, \d+)*\]"
)


@dataclass(frozen=True)
class InputResult:
    """Complete input matrix row: oracle convergence, policies and cell dominators."""

    name: str
    physical_sha256: str
    oracle_sha256: str
    oracle_convergence_error: float
    policies: list[FamilyResult]
    dominators: dict[str, dict[str, list[str]]]
    timing_equivalents: dict[str, dict[str, list[str]]]


def timing_equivalents(rows: list[FamilyResult]) -> dict[str, dict[str, list[str]]]:
    """List time-equivalent peers when neither is >10% faster, ignoring error.

    This is symmetric: other_time >= .9*target_time and target_time >=
    .9*other_time. It is not Pareto dominance and never merges wall and CPU.
    """
    available = [r for r in rows if r.error is not None and r.family != "default"]
    equivalents: dict[str, dict[str, list[str]]] = {}
    for target in available:
        cells = {}
        for index, mode in enumerate(target.modes):
            for metric in ("wall", "cpu"):
                target_time = mode.wall_median if metric == "wall" else mode.cpu_median
                peers = []
                for other in available:
                    if other.family == target.family:
                        continue
                    other_time = (
                        other.modes[index].wall_median
                        if metric == "wall"
                        else other.modes[index].cpu_median
                    )
                    if (
                        other_time >= 0.9 * target_time
                        and target_time >= 0.9 * other_time
                    ):
                        peers.append(other.family)
                cells[f"{mode.mode}/{metric}"] = peers
        equivalents[target.family] = cells
    return equivalents


def run_table(
    inputs: list[Input],
    families: tuple[str, ...] = ALL_FAMILIES,
    *,
    reference_solver: Callable[
        [ModelSource, TimelineBatch], ReferenceResult
    ] = converged_reference,
    policy_runner: Callable[
        [Input, str, ReferenceResult], FamilyResult
    ] = measure_policy,
) -> list[InputResult]:
    """Measure every family and default for each input, sharing identical oracles.

    Inputs stay in caller order. Independent two-level full-Lindblad references
    are cached by complete physical content, not family or input alias. All
    failures except the precisely matched probe refusal stop the whole table.
    """
    references: dict[str, ReferenceResult] = {}
    rows = []
    for index, item in enumerate(inputs):
        case = item.build()
        with closing(case.source):
            key = digest(physical_input(case))
            oracle_key = digest(physical_input(case, include_identity=False))
            if oracle_key not in references:
                references[oracle_key] = reference_solver(case.source, case.batch)
        reference = references[oracle_key]
        policies = [
            policy_runner(item, family, reference) for family in (*families, "default")
        ]
        rows.append(
            InputResult(
                item.name,
                key,
                oracle_key,
                reference.convergence_error,
                policies,
                dominance(policies),
                timing_equivalents(policies),
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
        "Timers cover Engine.evolve only, not program.acquire. Each mode has five wall/CPU samples.",
        "Errors compare complete complex density to the two-level converged full-Lindblad oracle.",
        "A dominator has error <= target error and time < 0.9 * target time. Exactly 10% is equivalent.",
        "No epsilon or weighted score is used. FAIL means error > 2e-3, not permission to keep an inaccurate route.",
        "Conclusions apply only to the measured input set; no deletion is authorized.",
        "",
        "## Family conclusions",
        "",
    ]
    for family in families:
        available = [
            (r, p)
            for r in rows
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
            f"Available on {len(available)} inputs; unavailable on {len(rows) - len(available)}.",
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
    for row in rows:
        lines += [
            f"### {row.name}",
            "",
            f"Oracle convergence error: {row.oracle_convergence_error:.12g}.",
            "",
            "| Family | Density error | Status | Mode/metric | Median seconds | Dominators | Time-equivalent peers |",
            "| --- | ---: | --- | --- | ---: | --- | --- |",
        ]
        for policy in row.policies:
            if policy.error is None:
                lines.append(
                    f"| {policy.family} | — | unavailable: {policy.unavailable} | — | — | — | — |"
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
                        f"{cell} | {value:.9g} | {', '.join(competitors) or 'none'} | {', '.join(peers) or 'none'} |"
                    )
        lines.append("")
    return "\n".join(lines) + "\n"


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
    parser.add_argument("--freeze-inputs", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    try:
        items = make_inputs() if inputs is None else inputs
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
        rows = run_table(items)
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
                "samples_per_mode": 5,
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
