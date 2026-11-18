"""Paired Engine timing controller and importable synthetic contract seams.

Engine timing follows ADR-0072 and ADR-0078; no acquire or solver is used.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, ExitStack, closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING, Literal, Self, TextIO

from pydantic import BaseModel, Field, TypeAdapter

from tools.simulator_measurement import Sample, SampleRunner, Session

if TYPE_CHECKING:
    from zcu_tools.simulate.evolution import Engine, EvolutionResult

    from tools.family_pareto_inputs import Scene
    from tools.family_pareto_reference import PreparedReference

PROTOCOL_VERSION = 2
"""Wire and tools scene-declaration semantics; bump when either interpretation changes."""


class Ready(BaseModel):
    """Child setup receipt, emitted after scene and cache-only reference preflight.

    Compare protocol_version, physical_version, cache_version and ordered records.
    segment_numeric_version is provenance only: numerical optimization may change
    it without changing the input contract. provenance contains HEAD/tree/dirty
    and runtime facts. No path or container file hash participates in identity.
    """

    status: Literal["ready"] = "ready"
    protocol_version: int
    physical_version: int
    cache_version: int
    records: list[dict[str, str]]
    segment_numeric_version: str
    provenance: dict[str, str | bool]


def validate_ready(expected: Ready, actual: Ready) -> None:
    """Reject any wire/format/ordered record mismatch before requesting samples.

    expected is the other side's initial receipt, or this side's initial receipt
    for a fresh process. Git and numerical revision are deliberately not gates.
    Both receipts are borrowed; ValueError identifies the mismatched field.
    """
    for name in ("protocol_version", "physical_version", "cache_version", "records"):
        if getattr(expected, name) != getattr(actual, name):
            raise ValueError(f"child handshake mismatch: {name}")


@dataclass(frozen=True)
class RatioSummary:
    """Block ratios for one clock; below one means candidate B is faster.

    ratios retains one (B1+B2)/(A1+A2) value per ordered ABBA block.
    median is their ordinary median. iqr contains linear Q1 and Q3, using
    percentile index (K-1)*p. Small K is descriptive, not calibration evidence.
    """

    ratios: list[float]
    median: float
    iqr: tuple[float, float]


@dataclass(frozen=True)
class BlockSummary:
    """Independent wall and whole-process CPU ratio statistics, never family CPU."""

    wall: RatioSummary
    """Elapsed wall clock ratios."""
    process: RatioSummary
    """Whole-process CPU ratios."""


@dataclass(frozen=True)
class Calibration:
    """One side's on/off overhead and off/off noise, using the same estimator.

    on_off takes A=off, B=on. off_off uses both flags off. Each clock needs
    more than two raw block ratios before it may select a telemetry mode.
    """

    on_off: BlockSummary
    off_off: BlockSummary


def choose_telemetry(calibrations: tuple[Calibration, Calibration]) -> bool:
    """Enable both sides only when all four overhead medians lie in noise IQRs.

    calibrations contains baseline and candidate evidence, in that order.
    IQR endpoints are inclusive, including degenerate single-point intervals.
    Any clock with K<=2 or unequal calibration sizes raises ValueError; missing
    data is never zero-filled. False requires separate empty-history diagnostics.
    This calculation borrows its inputs and has no resource side effects.
    """
    decisions = []
    for calibration in calibrations:
        for overhead, noise in (
            (calibration.on_off.wall, calibration.off_off.wall),
            (calibration.on_off.process, calibration.off_off.process),
        ):
            if len(overhead.ratios) <= 2 or len(noise.ratios) <= 2:
                raise ValueError("telemetry calibration needs more than two blocks")
            if len(overhead.ratios) != len(noise.ratios):
                raise ValueError("telemetry calibration block counts must agree")
            decisions.append(noise.iqr[0] <= overhead.median <= noise.iqr[1])
    return all(decisions)


@dataclass(frozen=True)
class SampleRequest:
    """One serial evolve request; callbacks return only after inspection completes.

    telemetry selects Engine accounting. empty_history clears routing history
    before this evolve; use it for untimed, separate family diagnostics.
    """

    telemetry: bool
    empty_history: bool = False
    action: Literal["sample", "reset", "close"] = "sample"
    """sample evolves, reset releases retained state, close ends the child stream."""


def serve_child[T, O](
    ready: Ready,
    runner: SampleRunner[T, O],
    commands: TextIO,
    replies: TextIO,
) -> None:
    """Serve newline JSON requests synchronously after emitting a ready receipt.

    runner and streams are borrowed; the caller must close them in finally or
    contexts. reset closes retained state and acknowledges it; close returns.
    EOF also returns. Invalid requests, evolve/inspection and I/O failures
    propagate, so the process boundary reports failure rather than fake samples.
    """
    replies.write(ready.model_dump_json() + "\n")
    replies.flush()
    adapter = TypeAdapter(SampleRequest)
    for line in commands:
        request = adapter.validate_json(line)
        if request.action == "close":
            return
        if request.action == "reset":
            runner.close()
            replies.write('{"reset":true}\n')
        else:
            result = runner.sample(
                telemetry=request.telemetry, empty_history=request.empty_history
            )
            replies.write(result.model_dump_json() + "\n")
        replies.flush()


class Observation(BaseModel):
    """Untimed complete density, reference evidence and optional family diagnostics.

    points retains ordered Engine coordinates. density_real/density_imag contain
    the complete complex (points,M,M) density, not populations or a projection.
    reference_error is max absolute complex deviation from the same-coordinate
    cached density. reference_key/convergence_error identify the cache evidence.
    routing contains original public FamilyRouting fields. aggregate, when
    present, sums prepare_cpu_s/apply_cpu_s/rows/cache_hits over those reports;
    CPU is calling-thread CPU, not wall/process. None means unavailable, not zero.
    """

    points: list[int]
    density_real: list[list[list[float]]]
    density_imag: list[list[list[float]]]
    reference_error: float
    reference_key: str
    convergence_error: float
    routing: dict[str, dict[str, int | float | str | None]]
    aggregate: dict[str, int | float] | None


def _observe(result: EvolutionResult, reference: PreparedReference) -> Observation:
    from tools.family_pareto_results import density_error

    routing = result.paths.routing
    aggregate = None
    if routing:
        aggregate = {
            "prepare_cpu_s": sum(row.prepare_cpu_s for row in routing.values()),
            "apply_cpu_s": sum(row.apply_cpu_s for row in routing.values()),
            "rows": sum(row.rows for row in routing.values()),
            "cache_hits": sum(row.cache_hits for row in routing.values()),
        }
    return Observation(
        points=result.points.tolist(),
        density_real=result.densities.real.tolist(),
        density_imag=result.densities.imag.tolist(),
        reference_error=density_error(result.densities, reference.reference.density),
        reference_key=reference.key,
        convergence_error=reference.reference.convergence_error,
        routing={name: asdict(report) for name, report in routing.items()},
        aggregate=aggregate,
    )


class SideConfig(BaseModel):
    """One worktree's workload and execution coordinates, serialized to its child.

    worktree contains committed common tools and an already bootstrapped uv
    environment. scenario selects one existing frozen record through select_inputs.
    reference_cache is read-only and cache-only; scene_cache/index override the
    shared defaults, without affecting identity. mode chooses process lifetime;
    cold uses the owner's first mode with a fresh empty Numba parent per child.
    workers fixes Engine width, families fixes its optional candidate whitelist,
    backend selects auto, CPU-only or explicit OpenCL execution, and budget is
    the positive absolute evolve error allowance. No field grants
    CPU/exclusive admission, reference solving, scene creation or cache filling.
    """

    worktree: Path
    scenario: str
    reference_cache: Path
    mode: Literal["first", "cold", "warm_clear_history", "warm_retain_history"]
    scene_cache: Path | None = None
    index: Path | None = None
    workers: int = Field(default=1, ge=1)
    families: tuple[str, ...] | None = None
    backend: Literal["auto", "cpu", "opencl"] = "auto"
    budget: float = Field(default=1e-3, gt=0, allow_inf_nan=False)


def _invoke_engine(
    engine: Engine, case: Scene, config: SideConfig, request: SampleRequest
) -> EvolutionResult:
    return engine.evolve(
        case.source, (case.batch,), budget=config.budget, telemetry=request.telemetry
    )


@contextmanager
def prepared_child(
    config: SideConfig,
    *,
    invoke: Callable[[Engine, Scene, SideConfig, SampleRequest], EvolutionResult]
    | None = None,
) -> Generator[tuple[Ready, SampleRunner[EvolutionResult, Observation]]]:
    """Prepare one selected frozen record and cache-only reference, then own its runner.

    config supplies scene/cache coordinates and Engine settings. invoke defaults
    to one Engine.evolve; hosts may replace that external invocation callback,
    receiving the borrowed Engine, fresh Scene, config and telemetry request.
    Selection uses select_inputs; all reference loading ends before ready or
    any Engine construction. Misses, invalid identities and codec errors fail
    without solving or writing caches. The yielded runner uses only the shared
    owner's modes/clocks. Its source/Engine close on success or failure.
    cold maps to owner first; process/cache lifetime belongs to the controller.
    Imports are lazy so the child can set its Numba parent before initialization.
    """
    from zcu_tools.simulate.evolution import (
        FROZEN_PHYSICAL_VERSION,
        SCENE_CACHE_VERSION,
        SEGMENT_NUMERIC_VERSION,
        Engine,
    )

    from tools.family_pareto import select_inputs, tested_state
    from tools.family_pareto_reference import ReferenceCache, load_references

    (selected,) = select_inputs(
        config.index, config.scenario, scene_cache=config.scene_cache
    )
    (reference,) = load_references(
        [selected.input], ReferenceCache(config.reference_cache)
    )
    invocation = _invoke_engine if invoke is None else invoke

    @contextmanager
    def factory() -> Generator[Session[EvolutionResult, Observation]]:
        case = selected.input.build()
        with (
            closing(case.source),
            closing(
                Engine(config.workers, families=config.families, backend=config.backend)
            ) as engine,
        ):
            yield Session(
                lambda enabled: invocation(
                    engine, case, config, SampleRequest(enabled)
                ),
                lambda result, enabled: _observe(result, reference),
                case.source.history.clear,
            )

    mode = "first" if config.mode == "cold" else config.mode
    ready = Ready(
        protocol_version=PROTOCOL_VERSION,
        physical_version=FROZEN_PHYSICAL_VERSION,
        cache_version=SCENE_CACHE_VERSION,
        records=[selected.identity],
        segment_numeric_version=SEGMENT_NUMERIC_VERSION,
        provenance={
            **tested_state(),
            "interpreter": sys.executable,
            "python": sys.version,
            "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR", ""),
            "process_group": str(os.getpgrp()),
        },
    )
    with SampleRunner(factory, mode=mode) as runner:
        yield ready, runner


@dataclass(frozen=True)
class Launch:
    """A direct child launch, never a relay or a nested resource-lock request.

    config selects its worktree/workload. purpose is probe (ready only), prime
    (discarded same-workload disk-cache filling), or measure. cache_parent is
    None normally, or a new empty Numba directory owned by SideClient for cold.
    The opener borrows that directory and must finish before its context exits.
    """

    config: SideConfig
    purpose: Literal["probe", "prime", "measure"]
    cache_parent: Path | None = None


@dataclass(frozen=True)
class ChildHandle[O]:
    """A synchronous child interface borrowed from a resource-owning context.

    ready is emitted after setup. sample returns one raw observation, or raises
    on child/protocol failure. reset releases retained session state untimed.
    The opener's context, not this handle, owns streams/process shutdown.
    """

    ready: Ready
    sample: Callable[[SampleRequest], Sample[O]]
    reset: Callable[[], None]


class SideClient[O]:
    """Controller-owned direct child lifetime for one side.

    Use as a context manager. The opener maps Launch to an owned ChildHandle
    context; production launches own-worktree uv, synthetic callers can replace
    only that process boundary. Warm retains one child. First/cold probe setup,
    then open one new child per sample, validating every ready before evolving.
    Exceptions propagate without retry; every opened context is closed.
    """

    def __init__(
        self,
        config: SideConfig,
        opener: Callable[[Launch], AbstractContextManager[ChildHandle[O]]],
    ) -> None:
        """Configure workload and borrowed opener; no child is opened until enter."""
        self.config = config
        self._opener = opener
        self._stack = ExitStack()
        self._child: ChildHandle[O] | None = None
        self._ready: Ready | None = None

    @property
    def ready(self) -> Ready:
        """Return initial handshake after enter, otherwise raise RuntimeError."""
        if self._ready is None:
            raise RuntimeError("side has not been opened")
        return self._ready

    def __enter__(self) -> Self:
        """Open warm measurement child or first/cold ready-only probe."""
        if self.config.mode in ("first", "cold"):
            with self._opener(Launch(self.config, "probe")) as child:
                self._ready = child.ready
        else:
            self._child = self._stack.enter_context(
                self._opener(Launch(self.config, "measure"))
            )
            self._ready = self._child.ready
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close retained child resources on success and failure."""
        self._stack.close()

    def prime(self) -> None:
        """For first only, discard off/on samples in a same-workload warmup child.

        Call after validating both sides' initial ready. This is process-level
        disk-cache alignment, not owner warmup, and requires execution approval.
        Cold never primes. New-child handshake failures precede its samples.
        """
        if self.config.mode == "first":
            with self._opener(Launch(self.config, "prime")) as child:
                validate_ready(self.ready, child.ready)
                child.sample(SampleRequest(False))
                child.sample(SampleRequest(True))

    def sample(self, request: SampleRequest) -> Sample[O]:
        """Return one requested sample; cold owns and deletes its empty cache parent.

        Each short-lived child is verified against this side's initial receipt.
        Invalid identity, child failure and cleanup failure propagate without
        another attempt. The caller must have entered this SideClient context.
        """
        if self._child is not None:
            return self._child.sample(request)
        if self.config.mode == "cold":
            with TemporaryDirectory(prefix="paired-cold-") as directory:
                return self._one_sample(request, Path(directory))
        return self._one_sample(request, None)

    def _one_sample(self, request: SampleRequest, parent: Path | None) -> Sample[O]:
        with self._opener(Launch(self.config, "measure", parent)) as child:
            validate_ready(self.ready, child.ready)
            return child.sample(request)

    def reset(self) -> None:
        """Release warm session state before a new calibration/timing group.

        Short-lived modes already release after every sample, so need no reset.
        This does not clear disk caches or change the recorded process policy.
        """
        if self._child is not None:
            self._child.reset()


def collect_abba[O](
    baseline: Callable[[SampleRequest], Sample[O]],
    candidate: Callable[[SampleRequest], Sample[O]],
    *,
    blocks: int,
    telemetry: tuple[bool, bool],
) -> list[list[Sample[O]]]:
    """Request blocks serially in A1,B1,B2,A2 order and retain raw samples.

    baseline/candidate are borrowed synchronous clients, not owned resources.
    blocks must be positive; telemetry gives A and B flags respectively.
    Errors propagate immediately without retry or requests to later clients.
    Zero readings remain unchanged; summarize_blocks validates side totals.
    """
    if blocks < 1:
        raise ValueError("blocks must be positive")
    order = (
        (baseline, telemetry[0]),
        (candidate, telemetry[1]),
        (candidate, telemetry[1]),
        (baseline, telemetry[0]),
    )
    return [
        [client(SampleRequest(enabled)) for client, enabled in order]
        for _ in range(blocks)
    ]


@dataclass(frozen=True)
class CalibrationEvidence[O]:
    """One side's raw off/off and on/off ABBA blocks and their decision inputs."""

    off_off: list[list[Sample[O]]]
    """Noise-floor blocks with both positions telemetry-disabled."""
    on_off: list[list[Sample[O]]]
    """Overhead blocks with A=off and B=on."""
    summary: Calibration
    """Independent wall/process statistics of those same raw blocks."""


@dataclass(frozen=True)
class Comparison[O]:
    """Complete successful comparison; partial or failed runs do not return one."""

    ready: tuple[Ready, Ready]
    """Baseline/candidate initial identities and runtime provenance."""
    calibrations: tuple[CalibrationEvidence[O], CalibrationEvidence[O]]
    """Baseline/candidate noise and overhead measurements in this same execution."""
    telemetry: bool
    """One selected accounting mode shared by both timing sides."""
    blocks: list[list[Sample[O]]]
    """Cross-side A1,B1,B2,A2 timing samples, excluding extra diagnostics."""
    summary: BlockSummary
    """Wall/process statistics calculated only from blocks."""
    diagnostics: tuple[Sample[O], Sample[O]] | None
    """Separate empty-history telemetry-on samples when timing is off; else None."""


def _calibrate[O](side: SideClient[O], blocks: int) -> CalibrationEvidence[O]:
    side.reset()
    noise = collect_abba(
        side.sample, side.sample, blocks=blocks, telemetry=(False, False)
    )
    side.reset()
    overhead = collect_abba(
        side.sample, side.sample, blocks=blocks, telemetry=(False, True)
    )
    side.reset()
    summary = Calibration(summarize_blocks(overhead), summarize_blocks(noise))
    return CalibrationEvidence(noise, overhead, summary)


def run_comparison[O](
    baseline: SideClient[O], candidate: SideClient[O], *, blocks: int
) -> Comparison[O]:
    """Calibrate and compare two entered sides serially, within caller admission.

    blocks is the frozen K for calibration and timing and must exceed two.
    Validate initial handshakes before priming or sampling. Calibrate both
    clocks on each side, select one telemetry flag, then collect cross-side ABBA.
    If timing is off, append one empty-history telemetry-on diagnostic per side.
    The caller owns SideClient contexts and the single exclusive outer runner;
    this function takes no locks. All failures propagate without retry or zeros.
    """
    if blocks <= 2:
        raise ValueError("comparison calibration needs more than two blocks")
    validate_ready(baseline.ready, candidate.ready)
    baseline.prime()
    candidate.prime()
    calibrations = (_calibrate(baseline, blocks), _calibrate(candidate, blocks))
    telemetry = choose_telemetry((calibrations[0].summary, calibrations[1].summary))
    samples = collect_abba(
        baseline.sample,
        candidate.sample,
        blocks=blocks,
        telemetry=(telemetry, telemetry),
    )
    summary = summarize_blocks(samples)
    diagnostics = None
    if not telemetry:
        diagnostics = (
            baseline.sample(SampleRequest(True, empty_history=True)),
            candidate.sample(SampleRequest(True, empty_history=True)),
        )
    return Comparison(
        (baseline.ready, candidate.ready),
        calibrations,
        telemetry,
        samples,
        summary,
        diagnostics,
    )


def _ratio_summary(ratios: list[float]) -> RatioSummary:
    ordered = sorted(ratios)

    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return RatioSummary(ratios, median(ratios), (percentile(0.25), percentile(0.75)))


def summarize_blocks[O](blocks: Sequence[Sequence[Sample[O]]]) -> BlockSummary:
    """Summarize nonempty ABBA blocks, each containing A1, B1, B2, A2.

    Samples are borrowed unchanged. Any incomplete block or nonpositive side
    sum raises ValueError before division. Zero-sum errors identify the one-based
    block, clock and both totals. No epsilon, dropping or replacement is used.
    This pure calculation does not own resources or choose telemetry mode.
    """
    if not blocks:
        raise ValueError("at least one ABBA block is required")
    wall: list[float] = []
    process: list[float] = []
    for number, block in enumerate(blocks, 1):
        if len(block) != 4:
            raise ValueError(f"block {number}: expected four ABBA samples")
        clocks = (
            ("wall", wall, [sample.wall_seconds for sample in block]),
            ("process", process, [sample.process_seconds for sample in block]),
        )
        for clock, ratios, durations in clocks:
            baseline = durations[0] + durations[3]
            candidate = durations[1] + durations[2]
            if baseline <= 0 or candidate <= 0:
                raise ValueError(
                    f"block {number}: {clock}: baseline={baseline}, candidate={candidate}; "
                    "both sums must be positive"
                )
            ratios.append(candidate / baseline)
    return BlockSummary(_ratio_summary(wall), _ratio_summary(process))


class BenchmarkReport(BaseModel):
    """Versioned complete report; no partial run is published as successful.

    coordinates records baseline/candidate workload and execution settings.
    comparison retains ready, raw calibration/timing/diagnostic samples and
    statistics. density_errors gives each block's maximum complete complex
    cross-side density difference, matching point IDs and reference evidence.
    cpu_affinity/native_threads are observed controller settings. exclusive is
    the caller's declaration of outer admission, not another lock acquisition.
    started_utc timestamps this execution outside all evolve timers.
    """

    schema_version: Literal[1] = 1
    coordinates: tuple[SideConfig, SideConfig]
    comparison: Comparison[Observation]
    density_errors: list[float]
    cpu_affinity: list[int]
    native_threads: dict[str, str | None]
    exclusive: bool
    started_utc: str


def _density_errors(comparison: Comparison[Observation]) -> list[float]:
    import numpy as np

    from tools.family_pareto_results import density_error

    errors = []
    for block in comparison.blocks:
        differences = []
        for a in (block[0].observation, block[3].observation):
            for b in (block[1].observation, block[2].observation):
                if a.points != b.points:
                    raise ValueError("cross-side density point order differs")
                actual = np.asarray(a.density_real) + 1j * np.asarray(a.density_imag)
                expected = np.asarray(b.density_real) + 1j * np.asarray(b.density_imag)
                differences.append(density_error(actual, expected))
        errors.append(max(differences))
    return errors


def benchmark(
    baseline: SideConfig,
    candidate: SideConfig,
    *,
    blocks: int,
    exclusive: bool,
    opener: Callable[[Launch], AbstractContextManager[ChildHandle[Observation]]]
    | None = None,
) -> BenchmarkReport:
    """Own both side clients and return a complete same-execution report.

    Scenario, mode, worker width, families and budget must match across the
    two sides. The worktree paths and Engine backend may differ. blocks is the
    frozen K>2. exclusive annotates caller-owned admission and grants no
    resources. opener defaults to launch_child; replacing
    it changes only the external process boundary, not protocol or statistics.
    Errors close both clients and propagate without retries or partial reports.
    Complete cross-density inspection occurs after child shutdown, untimed.
    """
    for field in ("scenario", "mode", "workers", "families", "budget"):
        if getattr(baseline, field) != getattr(candidate, field):
            raise ValueError(f"comparison coordinates differ: {field}")
    started = datetime.now(UTC).isoformat()
    affinity = sorted(os.sched_getaffinity(0))
    native = {
        key: os.environ.get(key)
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        )
    }
    launch = launch_child if opener is None else opener
    with SideClient(baseline, launch) as a, SideClient(candidate, launch) as b:
        comparison = run_comparison(a, b, blocks=blocks)
    return BenchmarkReport(
        coordinates=(baseline, candidate),
        comparison=comparison,
        density_errors=_density_errors(comparison),
        cpu_affinity=affinity,
        native_threads=native,
        exclusive=exclusive,
        started_utc=started,
    )


def _wait_child(child: subprocess.Popen[str]) -> int:
    try:
        return child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.terminate()
    try:
        return child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
        return child.wait(timeout=10)


def _finish_child(child: subprocess.Popen[str]) -> None:
    try:
        if child.stdin is not None and child.poll() is None:
            try:
                child.stdin.write(
                    TypeAdapter(SampleRequest)
                    .dump_json(SampleRequest(False, action="close"))
                    .decode()
                    + "\n"
                )
                child.stdin.flush()
            except BrokenPipeError:
                pass  # The exit status below reports a failed child, never a sample.
        code = _wait_child(child)
    finally:
        if child.stdin is not None:
            child.stdin.close()
        if child.stdout is not None:
            child.stdout.close()
    if code:
        raise RuntimeError(f"measurement child exited with status {code}")


@contextmanager
def launch_child(launch: Launch) -> Generator[ChildHandle[Observation]]:
    """Launch a direct own-worktree uv child, inheriting process group and locks.

    The caller owns exclusive admission; this function never takes a runner lock.
    launch.cache_parent, if supplied, must be a new empty controller-owned Numba
    parent. It is set before child imports, without deleting any existing cache.
    All requests/replies are serial newline JSON. EOF, invalid schema, child exit
    and I/O failures propagate. Exit requests shutdown, then uses 10s watchdogs
    for terminate/kill/reaping; the outer runner also owns whole-group cleanup.
    """
    config = launch.config
    env = dict(os.environ)
    if launch.cache_parent is not None:
        env["NUMBA_CACHE_DIR"] = str(launch.cache_parent)
    command = [
        "uv",
        "run",
        "--directory",
        str(config.worktree),
        "--no-sync",
        "--",
        "python",
        "-m",
        "tools.paired_benchmark",
        "child",
        "--config",
        config.model_dump_json(),
    ]
    child = subprocess.Popen(
        command,
        cwd=config.worktree,
        env=env,
        text=True,
        bufsize=1,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        close_fds=False,  # Preserve the outer runner's inheritable slot descriptors.
    )
    try:
        if child.stdin is None or child.stdout is None:
            raise RuntimeError("measurement child requires stdin/stdout pipes")
        commands, replies = child.stdin, child.stdout

        def read_reply() -> str:
            line = replies.readline()
            if not line:
                raise RuntimeError(
                    f"measurement child closed its reply stream: {child.poll()}"
                )
            return line

        ready = Ready.model_validate_json(read_reply())

        def sample(request: SampleRequest) -> Sample[Observation]:
            commands.write(
                TypeAdapter(SampleRequest).dump_json(request).decode() + "\n"
            )
            commands.flush()
            return Sample[Observation].model_validate_json(read_reply())

        def reset() -> None:
            commands.write(
                TypeAdapter(SampleRequest)
                .dump_json(SampleRequest(False, action="reset"))
                .decode()
                + "\n"
            )
            commands.flush()
            if read_reply().strip() != '{"reset":true}':
                raise ValueError("invalid child reset acknowledgement")

        yield ChildHandle(ready, sample, reset)
    finally:
        _finish_child(child)


def _committed_tools() -> None:
    subprocess.run(
        ["git", "cat-file", "-e", "HEAD:tools/paired_benchmark.py"], check=True
    )
    paths = [
        "tools",
        "lib/zcu_tools/simulate/evolution/frozen.py",
        "lib/zcu_tools/simulate/evolution/scene_cache.py",
        "lib/zcu_tools/simulate/evolution/__init__.py",
    ]
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *paths], check=False
    )
    if result.returncode:
        raise ValueError(
            "measurement tools and codecs must be committed before comparison"
        )


def _child_main(config_json: str) -> int:
    config = SideConfig.model_validate_json(config_json)
    _committed_tools()
    with prepared_child(config) as (ready, runner):
        serve_child(ready, runner, sys.stdin, sys.stdout)
    return 0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    child = commands.add_parser("child", help="internal serial measurement child")
    child.add_argument("--config", required=True)
    run = commands.add_parser("run", help="compare two sides in committed worktrees")
    run.add_argument("--baseline", type=Path, required=True)
    run.add_argument("--candidate", type=Path, required=True)
    for side in ("baseline", "candidate"):
        run.add_argument(
            f"--{side}-backend", choices=("auto", "cpu", "opencl"), default="auto"
        )
    run.add_argument("--scenario", required=True)
    run.add_argument("--index", type=Path)
    run.add_argument("--scene-cache", type=Path)
    run.add_argument("--reference-cache", type=Path, required=True)
    run.add_argument(
        "--mode",
        choices=("first", "cold", "warm_clear_history", "warm_retain_history"),
        default="warm_clear_history",
    )
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--families", nargs="+")
    run.add_argument("--budget", type=float, default=1e-3)
    run.add_argument("--blocks", type=int, required=True)
    run.add_argument(
        "--exclusive",
        action="store_true",
        required=True,
        help="declare caller-owned exclusive runner admission; does not acquire it",
    )
    run.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _run_main(
    args: argparse.Namespace,
    opener: Callable[[Launch], AbstractContextManager[ChildHandle[Observation]]] | None,
) -> int:
    common = dict(
        scenario=args.scenario,
        reference_cache=args.reference_cache,
        scene_cache=args.scene_cache,
        index=args.index,
        mode=args.mode,
        workers=args.workers,
        families=args.families,
        budget=args.budget,
    )
    baseline = SideConfig.model_validate(
        {
            **common,
            "worktree": args.baseline.resolve(),
            "backend": args.baseline_backend,
        }
    )
    candidate = SideConfig.model_validate(
        {
            **common,
            "worktree": args.candidate.resolve(),
            "backend": args.candidate_backend,
        }
    )
    report = benchmark(
        baseline, candidate, blocks=args.blocks, exclusive=args.exclusive, opener=opener
    )
    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", dir=output.parent, delete=False, encoding="utf-8"
    ) as temporary:
        path = Path(temporary.name)
        try:
            temporary.write(report.model_dump_json(indent=2) + "\n")
            temporary.flush()
            path.replace(output)
        finally:
            path.unlink(missing_ok=True)
    return 0


def main(
    argv: list[str] | None = None,
    *,
    opener: Callable[[Launch], AbstractContextManager[ChildHandle[Observation]]]
    | None = None,
) -> int:
    """Parse run/child CLI and publish successful JSON atomically; return 0 or 1.

    argv defaults to process arguments. opener is the benchmark process seam;
    None uses own-worktree uv children and never synchronizes dependencies.
    Expected input/cache/child failures print their cause to stderr and return 1,
    preserving previous output. Argument syntax errors retain argparse's exit 2.
    Callers own compute approval and one outer exclusive runner, not this CLI.
    """
    args = _parse(argv)
    try:
        if args.command == "child":
            return _child_main(args.config)
        return _run_main(args, opener)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"paired benchmark: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
