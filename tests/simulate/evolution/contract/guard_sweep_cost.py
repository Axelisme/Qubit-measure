"""Importable Engine-only family-exclusion estimator, not program.acquire.

Call through tools/run_heavy.py --exclusive. No timing assertion belongs in
pytest. This module uses only Engine/ModelSource public execution and history;
it does not assemble benchmark internals or alter numerical admission.
"""

from collections.abc import Callable, Generator
from contextlib import closing, contextmanager
from dataclasses import dataclass

from zcu_tools.simulate.evolution import Engine

from tools.family_pareto import EngineSession, measure_family

from .accuracy import ALL_FAMILIES
from .accuracy_cases import AccuracyCase


@dataclass(frozen=True)
class EngineCost:
    """Raw evolve samples for one family policy and history mode.

    policy is pinned or excluded. mode is first_evolve, warm_clear_history or
    warm_retain_history. Paths are per-sample segment counts. Wall and CPU
    samples and their medians use seconds. Construction, cleanup, history.clear
    and two fixed warmup evolves are untimed. First mode uses five fresh
    source/Engine pairs in this process, not fresh processes or disk caches.
    The shared slow-family protocol instead retains one first/retain sample
    and omits clear-history, whose samples are empty and medians are None.
    """

    policy: str
    mode: str
    wall_seconds: tuple[float, ...]
    cpu_seconds: tuple[float, ...]
    paths: tuple[dict[str, int], ...]
    wall_median: float | None
    cpu_median: float | None


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
    The shared measure_family protocol measures timing/paths without an oracle.
    """
    # Read the pinned policy name outside all timed sessions; no oracle is needed.
    case = build_case()
    with closing(case.source):
        (family,) = case.families
    results: list[EngineCost] = []
    for policy in ("pinned", "excluded"):

        @contextmanager
        def session(policy: str = policy) -> Generator[EngineSession]:
            case = build_case()
            families = (
                (family,)
                if policy == "pinned"
                else tuple(name for name in ALL_FAMILIES if name != family)
            )
            with (
                closing(case.source),
                closing(Engine(1, families=families)) as engine,
            ):
                yield EngineSession(
                    lambda: engine.evolve(case.source, (case.batch,), budget=1e-3),
                    case.source.history.clear,
                )

        modes = measure_family(session, None)
        for mode in modes:
            for sample in mode.samples:
                if policy == "pinned":
                    assert sample.paths.get(family, 0) > 0, (
                        f"expected family not exercised: {family}; got {sample.paths}"
                    )
                else:
                    assert not sample.paths.get(family, 0)
            results.append(
                EngineCost(
                    policy,
                    mode.mode,
                    tuple(s.wall_seconds for s in mode.samples),
                    tuple(s.cpu_seconds for s in mode.samples),
                    tuple(s.paths for s in mode.samples),
                    mode.wall_median,
                    mode.cpu_median,
                )
            )
    return tuple(results)
