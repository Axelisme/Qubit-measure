"""Importable Engine-only family-exclusion estimator, not program.acquire.

Call through tools/run_heavy.py --exclusive. No timing assertion belongs in
pytest. This module uses only Engine/ModelSource public execution and history;
it does not assemble benchmark internals or alter numerical admission.
"""

from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from statistics import median
from time import perf_counter, process_time

from zcu_tools.simulate.evolution import Engine

from .accuracy import ALL_FAMILIES, require_families
from .accuracy_cases import AccuracyCase


@dataclass(frozen=True)
class EngineCost:
    """Five raw evolve samples for one family policy and history mode.

    policy is pinned or excluded. mode is first_evolve, warm_clear_history or
    warm_retain_history. Paths are per-sample segment counts. Wall and CPU
    samples and their medians use seconds. Construction, cleanup, history.clear
    and two fixed warmup evolves are untimed. First mode uses five fresh
    source/Engine pairs in this process, not fresh processes or disk caches.
    """

    policy: str
    mode: str
    wall_seconds: tuple[float, ...]
    cpu_seconds: tuple[float, ...]
    paths: tuple[dict[str, int], ...]
    wall_median: float
    cpu_median: float


def measure_engine_cost(
    build_case: Callable[[], AccuracyCase],
) -> tuple[EngineCost, ...]:
    """Time one fixed physical input pinned vs all families except its family.

    build_case must return the same deterministic complete input and a fresh
    owned source/history every time, with exactly one family to pin. Workers=1,
    budget=1e-3, no telemetry, reuse_key or retained numerical preparations.
    Failures propagate. Every pinned sample must exercise its intended family;
    excluded samples must never exercise it. Record actual alternative paths.
    Excluding a whole family only estimates a counterfactual fallback cost,
    not the precise effect of a future narrower guard or program acquire.
    """
    results: list[EngineCost] = []
    for policy in ("pinned", "excluded"):
        for mode in ("first_evolve", "warm_clear_history", "warm_retain_history"):
            walls: list[float] = []
            cpus: list[float] = []
            paths: list[dict[str, int]] = []
            repetitions = 5 if mode == "first_evolve" else 1
            for _ in range(repetitions):
                case = build_case()
                (family,) = case.families
                families = (
                    (family,)
                    if policy == "pinned"
                    else tuple(name for name in ALL_FAMILIES if name != family)
                )
                with (
                    closing(case.source),
                    closing(Engine(1, families=families)) as engine,
                ):
                    if mode != "first_evolve":
                        for _ in range(2):
                            engine.evolve(case.source, (case.batch,), budget=1e-3)
                    count = 1 if mode == "first_evolve" else 5
                    for _ in range(count):
                        if mode == "warm_clear_history":
                            case.source.history.clear()
                        wall, cpu = perf_counter(), process_time()
                        result = engine.evolve(case.source, (case.batch,), budget=1e-3)
                        cpus.append(process_time() - cpu)
                        walls.append(perf_counter() - wall)
                        if policy == "pinned":
                            require_families(result.paths, (family,))
                        else:
                            assert not result.paths.segments_by_kernel.get(family, 0)
                        paths.append(dict(result.paths.segments_by_kernel))
            results.append(
                EngineCost(
                    policy,
                    mode,
                    tuple(walls),
                    tuple(cpus),
                    tuple(paths),
                    median(walls),
                    median(cpus),
                )
            )
    return tuple(results)
