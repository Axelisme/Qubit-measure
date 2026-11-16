"""Partial-run progress preserves completed samples without replacing reports."""

import json
from collections.abc import Callable, Generator
from contextlib import closing, contextmanager
from itertools import count
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    Engine,
    EvolutionResult,
    PathDiagnostics,
    ReferenceResult,
    reference_key,
)

from tests.tools.family_pareto_support import (
    make_scene,
    write_test_cache,
    write_test_index,
)
from tools import family_pareto
from tools.family_pareto import EngineSession, main, measure_family
from tools.family_pareto_inputs import Input
from tools.family_pareto_results import Sample


def test_observer_keeps_first_slow_sample_when_the_next_evolve_fails() -> None:
    observed: list[tuple[str, Sample | None]] = []
    ticks = count()
    calls = 0
    density = np.eye(2, dtype=np.complex128)[None] / 2

    @contextmanager
    def session() -> Generator[EngineSession]:
        def evolve() -> EvolutionResult:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("second evolve failed")
            return EvolutionResult(np.array([0]), density, PathDiagnostics())

        yield EngineSession(
            evolve, lambda: None, lambda mode, sample: observed.append((mode, sample))
        )

    with pytest.raises(RuntimeError, match="second evolve failed"):
        measure_family(
            session,
            ReferenceResult(density, 0.0),
            wall_clock=lambda: float(next(ticks)) * 61,
            cpu_clock=lambda: 0.0,
        )
    assert [mode for mode, _ in observed] == ["first_evolve", "warm_clear_history"]
    assert observed[0][1] is not None
    assert observed[1][1] is None


def test_partial_run_retains_samples_and_leaves_old_complete_outputs_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fail_later[**P, R](evolve: Callable[P, R]) -> Callable[P, R]:
        def call(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("Engine collaborator failed")
            return evolve(*args, **kwargs)

        return call

    inputs = [Input("good", lambda: make_scene("idle_coherence"), "small collaborator")]
    manifest, output, summary, journal = (
        tmp_path / name
        for name in ("inputs.json", "matrix.json", "summary.md", "attempt.jsonl")
    )
    write_test_index(inputs, manifest)
    cache = tmp_path / "cache"
    write_test_cache(inputs, cache)
    # main no longer exposes builder injection. Delegate real Engine calls and
    # fail only after earlier samples have reached the public progress journal.
    monkeypatch.setattr(Engine, "evolve", fail_later(Engine.evolve))
    output.write_text("previous complete output")
    summary.write_text("previous complete summary")
    assert (
        main(
            [
                "--index",
                str(manifest),
                "--scene-cache",
                str(tmp_path / "scenes"),
                "--reference-cache",
                str(cache),
                "--output",
                str(output),
                "--summary",
                str(summary),
                "--partial-output",
                str(journal),
            ],
        )
        == 1
    )
    assert output.read_text() == "previous complete output"
    assert summary.read_text() == "previous complete summary"
    assert journal.exists(), "partial progress must survive a later input failure"
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert events[0]["event"] == "run_start"
    assert events[0]["status"] == "partial"
    assert any(
        event["event"] == "sample" and event["input"] == "good" for event in events
    )
    assert events[-1]["event"] == "run_failed"
    assert events[-1]["error"] == "Engine collaborator failed"


def test_cli_checks_references_without_policies_engine_or_a_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = [
        Input("idle", lambda: make_scene("idle_coherence"), "small check collaborator"),
        Input("alias", lambda: make_scene("idle_coherence"), "same reference"),
    ]
    manifest, output, cache = (
        tmp_path / name for name in ("inputs.json", "references.json", "cache")
    )
    write_test_index(inputs, manifest)
    write_test_cache(inputs, cache)
    scene = inputs[0].build()
    with closing(scene.source):
        key = reference_key(scene.source, scene.batch)

    def forbidden_measurement(*args: object, **kwargs: object) -> None:
        raise AssertionError("check must not start measurement or Engine")

    # main has no policy/session injection; trap the external measurement effects.
    monkeypatch.setattr(family_pareto, "run_table", forbidden_measurement)
    monkeypatch.setattr(family_pareto, "Engine", forbidden_measurement)
    assert (
        main(
            [
                "--index",
                str(manifest),
                "--scene-cache",
                str(tmp_path / "scenes"),
                "--check-references",
                "--reference-cache",
                str(cache),
                "--output",
                str(output),
            ],
        )
        == 0
    )
    entries = json.loads(output.read_text())["inputs"]
    assert [entry["name"] for entry in entries] == ["idle", "alias"]
    assert [entry["key"] for entry in entries] == [key, key]
    assert [entry["convergence_error"] for entry in entries] == [0.0, 0.0]
    assert all(
        entry["wall_seconds"] >= 0 and entry["cpu_seconds"] >= 0 for entry in entries
    )


def test_cli_reads_a_frozen_scenario_for_reference_check(tmp_path: Path) -> None:
    inputs = [Input("idle", lambda: make_scene("idle_coherence"), "frozen check")]
    manifest, output, cache = (
        tmp_path / name for name in ("inputs.json.gz", "references.json", "cache")
    )
    write_test_index(inputs, manifest)
    write_test_cache(inputs, cache)

    assert (
        main(
            [
                "--index",
                str(manifest),
                "--scene-cache",
                str(tmp_path / "scenes"),
                "--scenario",
                "idle",
                "--check-references",
                "--reference-cache",
                str(cache),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert [entry["name"] for entry in report["inputs"]] == ["idle"]
    assert report["inputs"][0]["convergence_error"] == 0.0


def test_partial_attempt_never_reuses_an_old_journal(tmp_path: Path) -> None:
    inputs = [Input("idle", lambda: make_scene("idle_coherence"), "small collaborator")]
    manifest, output, summary, journal = (
        tmp_path / name
        for name in ("inputs.json", "matrix.json", "summary.md", "attempt.jsonl")
    )
    write_test_index(inputs, manifest)
    journal.write_text("earlier attempt")
    assert (
        main(
            [
                "--index",
                str(manifest),
                "--scene-cache",
                str(tmp_path / "scenes"),
                "--output",
                str(output),
                "--summary",
                str(summary),
                "--partial-output",
                str(journal),
            ],
        )
        == 1
    )
    assert journal.read_text() == "earlier attempt"
    assert not output.exists()
