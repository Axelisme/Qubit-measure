"""Measure Engine.evolve families, not program.acquire latency."""

import argparse
import logging
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, closing, contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import version
from math import isclose
from pathlib import Path
from statistics import median
from time import perf_counter, process_time
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from zcu_tools.simulate.evolution import (
    Engine,
    EvolutionResult,
    FrozenRecord,
    NoValidKernel,
    ReferenceResult,
    digest,
    read_frozen_manifest,
)

from tools.family_pareto_inputs import (
    Input,
    canonical_json,
    input_from_record,
)
from tools.family_pareto_reference import (
    ReferenceCache,
    load_references,
)
from tools.family_pareto_report import render_summary as render_summary
from tools.family_pareto_results import (
    ACCURACY_LIMIT,
    ALL_FAMILIES,
    DIRECTLY_RETAINED,
    FamilyResult,
    InputResult,
    ModeResult,
    Refusal,
    Sample,
    density_error,
    dominance,
    timing_equivalents,
)
from tools.family_pareto_results import mode_error as _mode_error


@dataclass(frozen=True)
class SelectedInput:
    """A selected numeric record, its tools declaration and metadata identity.

    input builds a fresh caller-owned Scene; close each Scene.source after use.
    record is the simulator-owned frozen input, including physical/oracle hashes.
    metadata_sha256 is computed by select_inputs with the public codec digest.
    No source or Engine is created by selection itself.
    """

    input: Input
    record: FrozenRecord
    metadata_sha256: str

    @property
    def identity(self) -> dict[str, str]:
        """Return the storage-independent report/handshake identity for this record."""
        return {
            "name": self.record.name,
            "physical_sha256": self.record.physical_sha256,
            "oracle_sha256": self.record.oracle_sha256,
            "metadata_sha256": self.metadata_sha256,
        }


def select_inputs(
    manifest: Path | None = None,
    scenario: str | None = None,
) -> tuple[SelectedInput, ...]:
    """Select frozen records and their report identity without producer work.

    manifest selects one public codec manifest. None temporarily reads the two
    existing tools/data manifests, only for integration until storage migration.
    scenario=None keeps file/record order; a name requires exactly one match.
    Unknown names raise ValueError listing legal names; ambiguous names list
    their source files. File/codec/metadata failures propagate. No references,
    Engine or source are created here. Each returned input.build() creates a
    fresh Scene whose source the caller must close. identity excludes source
    paths and file hashes; main and paired consumers use the same record fields.
    """
    directory = Path(__file__).parent / "data" / "family_pareto"
    paths = (
        [manifest]
        if manifest is not None
        else [
            directory / "frozen-synthetic.json.gz",
            directory / "frozen-real.json.gz",
        ]
    )
    records = [
        (path, record) for path in paths for record in read_frozen_manifest(path)
    ]
    if scenario is not None:
        matches = [
            (path, record) for path, record in records if record.name == scenario
        ]
        if not matches:
            legal = ", ".join(record.name for _, record in records)
            raise ValueError(f"Unknown scenario: {scenario}. Legal names: {legal}")
        if len(matches) != 1:
            sources = ", ".join(str(path) for path, _ in matches)
            raise ValueError(f"Ambiguous scenario: {scenario}. Sources: {sources}")
        records = matches
    return tuple(
        SelectedInput(input_from_record(record), record, digest(record.metadata))
        for _, record in records
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
    on_sample: Callable[[str, "Sample | None"], None] | None = None
    """Optional untimed observer after each sample; None sample marks a skipped mode."""
    initial_density: NDArray[np.complex128] | None = None
    """DC initial density (M,M); None is timing-only and skips no-evolution checks."""


SessionOpener = Callable[[], AbstractContextManager[EngineSession]]


def _timed_sample(
    session: EngineSession,
    reference: ReferenceResult | None,
    wall_clock: Callable[[], float],
    cpu_clock: Callable[[], float],
) -> Sample:
    wall, cpu = wall_clock(), cpu_clock()
    result = session.evolve()
    cpu_elapsed, wall_elapsed = cpu_clock() - cpu, wall_clock() - wall
    oracle_error = (
        None
        if reference is None
        else density_error(result.densities, reference.density)
    )
    if reference is not None and session.initial_density is not None:
        for point, actual, expected in zip(
            result.points, result.densities, reference.density, strict=True
        ):
            result_motion = density_error(actual, session.initial_density)
            reference_motion = density_error(expected, session.initial_density)
            if result_motion <= 1e-9 and reference_motion > ACCURACY_LIMIT:
                raise RuntimeError(
                    f"point {point}: result moved {result_motion:.1e} from the initial "
                    f"density while the reference moved {reference_motion:.3g}"
                )
    return Sample(
        wall_elapsed,
        cpu_elapsed,
        oracle_error,
        dict(result.paths.segments_by_kernel),
        routing=dict(result.paths.routing),
        family_cpu_s={
            name: report.actual_cpu_s for name, report in result.paths.routing.items()
        },
    )


def _measured_mode(name: str, samples: list[Sample]) -> ModeResult:
    return ModeResult(
        name,
        samples,
        median(s.wall_seconds for s in samples),
        median(s.cpu_seconds for s in samples),
    )


def measure_family(
    open_session: SessionOpener,
    reference: ReferenceResult | None,
    *,
    wall_clock: Callable[[], float] = perf_counter,
    cpu_clock: Callable[[], float] = process_time,
) -> list[ModeResult]:
    """Time public evolves only, with fresh sessions and an optional real oracle.

    The first empty-history evolve is a sample, never a separate probe. Above
    60 wall seconds retain it, immediately time one more evolve in that same
    history, and leave warm_clear_history unmeasured with explicit None times.
    Otherwise collect five samples per mode, with two untimed warmups before
    each warm mode. Setup, cleanup, history clearing and inspection are untimed.
    With a reference and session.initial_density, a point staying within 1e-9
    of its initial density while the reference moves beyond ACCURACY_LIMIT
    raises RuntimeError before recording the sample. This inspection is untimed.
    No process or disk-cache reset occurs here. Failures always propagate.
    """

    def sample(session: EngineSession, mode: str) -> Sample:
        measured = _timed_sample(session, reference, wall_clock, cpu_clock)
        if session.on_sample is not None:
            session.on_sample(mode, measured)
        return measured

    with open_session() as session:
        first = sample(session, "first_evolve")
        if first.wall_seconds > 60:
            if session.on_sample is not None:
                session.on_sample("warm_clear_history", None)
            retained = sample(session, "warm_retain_history")
            return [
                _measured_mode("first_evolve", [first]),
                ModeResult(
                    "warm_clear_history", [], None, None, "slow family not measured"
                ),
                _measured_mode("warm_retain_history", [retained]),
            ]
    samples = [first]
    for _ in range(4):
        with open_session() as session:
            samples.append(sample(session, "first_evolve"))
    modes = [_measured_mode("first_evolve", samples)]
    for name in ("warm_clear_history", "warm_retain_history"):
        with open_session() as session:
            session.evolve()
            session.evolve()
            samples = []
            for _ in range(5):
                if name == "warm_clear_history":
                    session.clear_history()
                samples.append(sample(session, name))
        modes.append(_measured_mode(name, samples))
    return modes


def _policy_whitelist(item: Input, family: str) -> tuple[str, ...] | None:
    """Return None for default, otherwise target and companions deduplicated in order."""
    return (
        None
        if family == "default"
        else tuple(dict.fromkeys((family, *item.companions_for(family))))
    )


@contextmanager
def engine_session(item: Input, family: str) -> Generator[EngineSession]:
    """Own a fresh source/history and single-worker Engine for one family policy.

    'default' uses normal routing. Other names pin the target plus declared
    companions, preserving numerical admission. Construction, cleanup and clear are outside timers.
    """
    families = _policy_whitelist(item, family)
    with whitelist_session(item, families) as session:
        yield session


@contextmanager
def whitelist_session(
    item: Input,
    families: tuple[str, ...] | None,
    *,
    on_sample: Callable[[str, Sample | None], None] | None = None,
) -> Generator[EngineSession]:
    """Own a fresh source/history and Engine with exactly the supplied whitelist.

    None retains default routing. Unknown families and all numeric, evolve or
    cleanup failures propagate. Setup and cleanup remain outside timing.
    """
    case = item.build()
    with closing(case.source), closing(Engine(1, families=families)) as engine:
        yield EngineSession(
            lambda: engine.evolve(
                case.source, (case.batch,), budget=1e-3, telemetry=True
            ),
            case.source.history.clear,
            on_sample,
            initial_density=case.source.initial_density,
        )


class _InitialRefusal(Exception):
    """A structured refusal from the first evolve, distinct from cleanup failures."""

    def __init__(self, refusal: NoValidKernel) -> None:
        super().__init__(str(refusal))
        self.refusal = refusal


def _availability_opener(opener: SessionOpener) -> SessionOpener:
    first = True

    @contextmanager
    def checked_session() -> Generator[EngineSession]:
        with opener() as session:

            def evolve() -> EvolutionResult:
                nonlocal first
                initial, first = first, False
                try:
                    return session.evolve()
                except NoValidKernel as error:
                    if initial:
                        raise _InitialRefusal(error) from error
                    raise

            yield replace(session, evolve=evolve)

    return checked_session


def _check_expected_route(
    sample: Sample, expected: dict[str, tuple[int, float]]
) -> None:
    observed = {
        name: (report.rows, report.duration_ns)
        for name, report in sample.routing.items()
        if report.rows
    }
    counts = {name: rows for name, (rows, _) in expected.items()}
    if sample.paths != counts or observed.keys() != expected.keys():
        raise RuntimeError(
            f"expected route {expected}, got paths {sample.paths} and work {observed}"
        )
    if any(
        observed[name][0] != rows
        or not isclose(observed[name][1], duration, rel_tol=1e-12, abs_tol=1e-9)
        for name, (rows, duration) in expected.items()
    ):
        raise RuntimeError(f"expected route {expected}, got work {observed}")


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
    """Measure a target plus named companions, or unrestricted default routing.

    The first timed evolve tests availability. Only its NoValidKernel becomes
    unavailable, or companion_blocked when attempted contains a companion but
    not the target. Both preserve first-group facts and whitelist classification.
    Neither infers later groups. All other evolve/setup/cleanup failures stop.
    Wall/process CPU describe this entire group, not isolated target speed.
    Oracle errors are measured across all retained samples. With a session's
    initial_density, RuntimeError rejects any point staying within 1e-9 of that
    density when the reference moves beyond ACCURACY_LIMIT. Runtime errors
    retain their identity and gain a note naming the scenario and family.
    """
    whitelist = _policy_whitelist(item, family)
    expected = item.expected_routes.get(family, {}) if family != "default" else {}
    attribution = (
        "default"
        if family == "default"
        else "target_and_companions"
        if item.companions_for(family)
        else "target"
    )
    checked_session = _availability_opener(lambda: opener(item, family))
    try:
        modes = measure_family(
            checked_session, reference, wall_clock=wall_clock, cpu_clock=cpu_clock
        )
    except _InitialRefusal as error:
        refused = error.refusal
        companion_blocked = (
            family != "default"
            and family not in refused.attempted
            and any(name in refused.attempted for name in item.companions_for(family))
        )
        return FamilyResult(
            family,
            None,
            None if companion_blocked else str(refused),
            [],
            companion_blocked=str(refused) if companion_blocked else None,
            expected_route=expected,
            attribution=attribution,
            whitelist=whitelist,
            refusal=Refusal(
                refused.domain,
                refused.points,
                refused.segments,
                refused.attempted,
                {
                    name: "guard_or_precondition"
                    if name in refused.attempted
                    else "not_candidate"
                    for name in whitelist or ()
                },
            ),
        )
    except RuntimeError as error:
        error.add_note(f"scenario {item.name}, family {family}")
        raise
    paths = [sample.paths for mode in modes for sample in mode.samples]
    if family != "default" and any(
        set(path) != set(whitelist or ()) or any(count <= 0 for count in path.values())
        for path in paths
    ):
        raise RuntimeError(f"pinned {family} did not execute exclusively: {paths}")
    if family != "default" and item.expected_routes and not expected:
        raise RuntimeError(f"pinned {family} has no expected route declaration")
    if expected:
        for mode in modes:
            for sample in mode.samples:
                _check_expected_route(sample, expected)
    return FamilyResult(
        family,
        max(_mode_error(mode) for mode in modes if mode.samples),
        None,
        modes,
        expected_route=expected,
        attribution=attribution,
        whitelist=whitelist,
    )


def measure_fixed_group(
    item: Input,
    families: tuple[str, ...],
    reference: ReferenceResult,
    *,
    on_sample: Callable[[str, Sample | None], None] | None = None,
) -> FamilyResult:
    """Time the default route's observed family union as one fixed whitelist.

    The complete input, budget and three history modes match measure_policy.
    This is a group latency, never attributed to any one member or included
    among pinned-family dominance competitors. RuntimeError rejects a point
    staying within 1e-9 of its initial density while the reference moves beyond
    ACCURACY_LIMIT. Runtime errors gain scenario/fixed-group context; all
    failures propagate.
    """
    try:
        modes = measure_family(
            lambda: whitelist_session(item, families, on_sample=on_sample), reference
        )
    except RuntimeError as error:
        error.add_note(f"scenario {item.name}, family fixed_default_group {families}")
        raise
    return FamilyResult(
        "fixed_default_group",
        max(_mode_error(mode) for mode in modes if mode.samples),
        None,
        modes,
        attribution="fixed_family_group",
        whitelist=families,
    )


def _emit(
    progress: Callable[[dict[str, object]], None] | None, **event: object
) -> None:
    if progress is not None:
        progress(event)


def _sample_observer(
    progress: Callable[[dict[str, object]], None] | None,
    input_name: str,
    family: str,
) -> Callable[[str, Sample | None], None] | None:
    if progress is None:
        return None

    def observed(mode: str, sample: Sample | None) -> None:
        _emit(
            progress,
            event="mode_skipped" if sample is None else "sample",
            input=input_name,
            family=family,
            mode=mode,
            sample=None if sample is None else asdict(sample),
            reason="slow family not measured" if sample is None else None,
        )

    return observed


def _measure_policy_observed(
    item: Input,
    family: str,
    reference: ReferenceResult,
    runner: Callable[[Input, str, ReferenceResult], FamilyResult],
    progress: Callable[[dict[str, object]], None] | None,
) -> FamilyResult:
    if progress is None or runner is not measure_policy:
        return runner(item, family, reference)

    @contextmanager
    def open_observed(item: Input, family: str) -> Generator[EngineSession]:
        with engine_session(item, family) as session:
            yield replace(
                session, on_sample=_sample_observer(progress, item.name, family)
            )

    return measure_policy(item, family, reference, opener=open_observed)


def _new_journal(
    path: Path, start: dict[str, object]
) -> Callable[[dict[str, object]], None]:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(start) + "\n")

    def append(event: dict[str, object]) -> None:
        # Close after every event, outside timed evolve. A killed process leaves
        # prior complete lines usable; a torn final line is not a complete event.
        with path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(event) + "\n")

    return append


def run_table(
    inputs: list[Input],
    families: tuple[str, ...] = ALL_FAMILIES,
    *,
    reference_cache: Path | ReferenceCache | None = None,
    policy_runner: Callable[
        [Input, str, ReferenceResult], FamilyResult
    ] = measure_policy,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> list[InputResult]:
    """Measure every family and default for each input, sharing identical oracles.

    Inputs stay in caller order. All references must load from reference_cache
    before any policy or Engine starts. Missing names and simulator cache keys
    are reported together. Equal physics shares one reference object, excluding
    only model identity. No solver runs and no cache entries are written.
    All failures except an initial NoValidKernel stop the whole table.
    Real default paths are also timed with their observed family union pinned,
    separately from dominance. progress receives untimed policy/sample events,
    including explicit skipped modes, before later failures can discard them.
    """
    cache = (
        reference_cache
        if isinstance(reference_cache, ReferenceCache)
        else ReferenceCache(reference_cache)
    )
    prepared = load_references(inputs, cache)
    rows = []
    for index, (item, ready) in enumerate(zip(inputs, prepared, strict=True)):
        _emit(progress, event="input_start", input=item.name)
        key, oracle_key = ready.physical_sha256, ready.oracle_sha256
        oracle_wall, oracle_cpu = ready.wall_seconds, ready.cpu_seconds
        reference = ready.reference
        _emit(
            progress,
            event="reference_ready",
            input=item.name,
            physical_sha256=oracle_key,
            convergence_error=reference.convergence_error,
        )
        policies = []
        for family in (*families, "default"):
            _emit(progress, event="policy_start", input=item.name, family=family)
            measured = _measure_policy_observed(
                item, family, reference, policy_runner, progress
            )
            policies.append(measured)
            _emit(
                progress,
                event="policy_result",
                input=item.name,
                family=family,
                result=asdict(measured),
            )
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
            measure_fixed_group(
                item,
                observed,
                reference,
                on_sample=_sample_observer(progress, item.name, "fixed_default_group"),
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
        _emit(
            progress, event="input_complete", input=item.name, result=asdict(rows[-1])
        )
        print(
            f"{index + 1}/{len(inputs)} {item.name}: "
            f"{sum(p.error is not None for p in policies)} available policies",
            flush=True,
        )
    return rows


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


def local_data_directory() -> Path:
    """Create shared benchmark storage located through Git's absolute common dir.

    All worktrees share the main repository's .agent_state/benchmark/family_pareto.
    Git or filesystem errors propagate; no references are solved here.
    """
    common = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            text=True,
        ).strip()
    )
    directory = common.parent / ".agent_state" / "benchmark" / "family_pareto"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _selected_families(
    names: str | None, *, checking_references: bool
) -> tuple[str, ...]:
    if names is None:
        return ALL_FAMILIES
    if checking_references:
        raise ValueError("--families cannot be combined with --check-references")
    families = tuple(dict.fromkeys(names.split(",")))
    unknown = [name for name in families if name not in ALL_FAMILIES]
    if unknown:
        raise ValueError(
            f"Unknown families: {unknown}. Legal names: {', '.join(ALL_FAMILIES)}"
        )
    return families


def _scenario_paths(args: argparse.Namespace, run_id: str) -> None:
    if args.scenario is None:
        return
    if args.reference_cache is None:
        args.reference_cache = local_data_directory() / "references"
    if args.output is None or (
        not args.check_references
        and (args.summary is None or args.partial_output is None)
    ):
        directory = local_data_directory() / args.scenario / run_id
        directory.mkdir(parents=True)
        args.output = args.output or directory / "matrix.json"
        args.summary = args.summary or directory / "report.md"
        args.partial_output = args.partial_output or directory / "journal.jsonl"
    args.summary_output = args.summary_output or args.output.with_name("summary.json")
    args.compare_summary = args.compare_summary or (
        Path(__file__).parent
        / "data"
        / "family_pareto"
        / "summaries"
        / f"{args.scenario}.json"
    )


def _parse_command(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario")
    parser.add_argument(
        "--exclusive",
        action="store_true",
        help="annotation only; does not acquire a lock",
    )
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--check-references", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--compare-summary", type=Path)
    parser.add_argument("--partial-output", type=Path)
    parser.add_argument("--families", metavar="NAME[,NAME...]")
    return parser.parse_args(argv)


def _write_scenario_summary(
    args: argparse.Namespace,
    rows: list[InputResult],
    state: dict[str, str | bool],
    started_at_utc: str,
    manifest_sha256: str,
) -> None:
    # Summary publication is optional for inventory-wide runs.
    from pydantic import TypeAdapter

    from tools.family_pareto_summary import Provenance, build_summary, publish_summary

    provenance = TypeAdapter(Provenance).validate_python(
        {
            **state,
            "started_at_utc": started_at_utc,
            "exclusive": args.exclusive,
        }
    )
    summary = build_summary(
        rows[0],
        manifest_sha256=manifest_sha256,
        measured_families=_selected_families(args.families, checking_references=False),
        provenance=provenance,
    )
    publish_summary(summary, args.summary_output, args.compare_summary)
    print(f"Scenario summary: {args.summary_output}")


def main(argv: list[str] | None = None) -> int:
    """Check all references or measure prepared inputs; failures return nonzero.

    select_inputs owns frozen source selection and storage-independent record
    identity. --scenario requires exactly one matching record before construction;
    unknown names list legal names and ambiguous names list source files.
    It can be combined with --manifest. Default tools/data sources are temporary
    integration inputs, not the final storage location.

    --check-references loads the complete selection cache-only, writes each
    name/key/convergence error/load time to --output and returns without policies
    or Engine timing. Measurement uses the same preflight and requires --summary.
    Any misses fail the batch before any timing. --reference-cache selects the
    existing directory; this consumer never solves or writes references.
    --families selects pinned families plus default and its fixed observed group.
    --summary-output and --compare-summary retain single-scenario publication
    and comparison; comparison failures do not block current summary writes.
    --exclusive is annotation only. Journals preserve completed samples on later
    failure. Preparation or measurement errors return 1 and preserve existing
    complete reports; write failures may leave partial destination files.
    """
    args = _parse_command(argv)
    progress = None
    try:
        families = _selected_families(
            args.families,
            checking_references=args.check_references,
        )
        started_at_utc = datetime.now(timezone.utc).isoformat()
        if args.scenario is None and (
            args.summary_output is not None or args.compare_summary is not None
        ):
            raise ValueError(
                "--summary-output and --compare-summary require --scenario"
            )
        selected = select_inputs(args.manifest, args.scenario)
        items = [entry.input for entry in selected]
        identities = [entry.identity for entry in selected]
        # Keep the summary field name, but hash record identities, not file paths.
        identity_sha256 = digest(identities)
        run_id = uuid4().hex
        _scenario_paths(args, run_id)
        if args.output is None or (not args.check_references and args.summary is None):
            raise ValueError(
                "requires --output and --summary (except reference checking)"
            )
        state = tested_state()
        if args.check_references:
            prepared = load_references(items, ReferenceCache(args.reference_cache))
            args.output.write_text(
                canonical_json(
                    {
                        "records": identities,
                        "tested_state": state,
                        "inputs": [
                            {
                                **identity,
                                "key": entry.key,
                                "convergence_error": entry.reference.convergence_error,
                                "wall_seconds": entry.wall_seconds,
                                "cpu_seconds": entry.cpu_seconds,
                            }
                            for entry, identity in zip(
                                prepared, identities, strict=True
                            )
                        ],
                        "reference_total_wall_seconds": sum(
                            entry.wall_seconds for entry in prepared
                        ),
                        "reference_total_cpu_seconds": sum(
                            entry.cpu_seconds for entry in prepared
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return 0
        if args.partial_output is not None:
            if args.partial_output.resolve() in {
                args.output.resolve(),
                args.summary.resolve(),
            }:
                raise ValueError(
                    "partial output must be separate from complete outputs"
                )
            progress = _new_journal(
                args.partial_output,
                {
                    "event": "run_start",
                    "status": "partial",
                    "run_id": run_id,
                    "records": identities,
                    "tested_state": state,
                },
            )
        rows = run_table(
            items,
            families,
            reference_cache=ReferenceCache(args.reference_cache),
            progress=progress,
        )
        result = {
            "schema_version": 1,
            "started_at_utc": started_at_utc,
            "exclusive": args.exclusive,
            "run_id": run_id,
            "measured_families": families,
            "records": identities,
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
                "telemetry": True,
                "dominance_cpu_source": "process_time around complete Engine.evolve",
                "family_cpu_source": "FamilyRouting prepare_cpu_s/apply_cpu_s, calling-thread CPU",
                "budget": 1e-3,
                "workers": 1,
                "maximum_samples_per_mode": 5,
                "slow_first_evolve_wall_seconds": 60,
                "slow_clear_history": "unmeasured; n=0; no median or dominance",
                "slow_retain_history": "same session immediately after first evolve; n=1",
                "reference_cache_only": True,
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
                "accuracy_limit": ACCURACY_LIMIT,
                "dominance": (
                    "both policies' maximum complete-sequence final density errors "
                    f"<= accuracy_limit ({ACCURACY_LIMIT:g}); qualified policies "
                    "are equally accurate; neither policy is default/fixed or "
                    f"directly retained ({', '.join(sorted(DIRECTLY_RETAINED))}); "
                    "per shared measured mode and wall/CPU metric, "
                    "competitor_time < .9 * target_time"
                ),
                "unavailable_exception": "NoValidKernel",
            },
            "inputs": [
                {
                    **asdict(row),
                    "dominance_same_companions": row.dominance_same_companions,
                }
                for row in rows
            ],
        }
        summary = render_summary(rows, families)
        args.output.write_text(canonical_json(result) + "\n", encoding="utf-8")
        args.summary.write_text(summary, encoding="utf-8")
        print(
            f"Complete matrix: {args.output}; report: {args.summary}; journal: {args.partial_output}"
        )
        if args.scenario is not None:
            _write_scenario_summary(args, rows, state, started_at_utc, identity_sha256)
        _emit(progress, event="run_complete", run_id=run_id)
    except Exception as error:
        try:
            _emit(
                progress,
                event="run_failed",
                error=str(error),
                error_type=type(error).__name__,
            )
        except Exception:
            logging.exception("Could not append partial failure event")
        # CLI isolation preserves failures; only initial domain refusals are unavailable.
        logging.exception("Pareto table failed; no complete report was written")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
