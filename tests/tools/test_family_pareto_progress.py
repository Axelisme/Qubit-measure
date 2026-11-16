"""Partial-run progress preserves completed samples without replacing reports."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from itertools import count
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EvolutionResult, PathDiagnostics

from tests.tools.family_pareto_support import make_scene
from tools.family_pareto import EngineSession, main, measure_family
from tools.family_pareto_inputs import Input, Scene
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
            None,
            wall_clock=lambda: float(next(ticks)) * 61,
            cpu_clock=lambda: 0.0,
        )
    assert [mode for mode, _ in observed] == ["first_evolve", "warm_clear_history"]
    assert observed[0][1] is not None
    assert observed[1][1] is None


def test_partial_run_retains_samples_and_leaves_old_complete_outputs_untouched(
    tmp_path: Path,
) -> None:
    builds = 0

    def failing_later() -> Scene:
        nonlocal builds
        builds += 1
        if builds == 3:
            raise RuntimeError("input collaborator failed")
        return make_scene("idle_coherence")

    inputs = [
        Input("good", lambda: make_scene("idle_coherence"), "small collaborator"),
        Input("bad", failing_later, "controlled input failure"),
    ]
    manifest, output, summary, journal = (
        tmp_path / name
        for name in ("inputs.json", "matrix.json", "summary.md", "attempt.jsonl")
    )
    assert main(["--freeze-inputs", str(manifest)], inputs=inputs) == 0
    output.write_text("previous complete output")
    summary.write_text("previous complete summary")
    assert (
        main(
            [
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--summary",
                str(summary),
                "--partial-output",
                str(journal),
            ],
            inputs=inputs,
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
    assert events[-1]["error"] == "input collaborator failed"


def test_cli_prepares_references_without_timing_or_a_summary(tmp_path: Path) -> None:
    builds = 0

    def build() -> Scene:
        nonlocal builds
        builds += 1
        if builds > 3:
            raise RuntimeError("no policy sessions allowed during preparation")
        return make_scene("idle_coherence")

    inputs = [Input("idle", build, "small preparation collaborator")]
    manifest, output, cache = (
        tmp_path / name for name in ("inputs.json", "references.json", "cache")
    )
    assert main(["--freeze-inputs", str(manifest)], inputs=inputs) == 0
    assert (
        main(
            [
                "--manifest",
                str(manifest),
                "--prepare-references",
                "--reference-cache",
                str(cache),
                "--output",
                str(output),
            ],
            inputs=inputs,
        )
        == 0
    )
    prepared = json.loads(output.read_text())
    assert prepared["inputs"][0]["convergence_error"] <= 2e-6
    assert prepared["inputs"][0]["density_shape"] == [2, 3, 3]
    assert prepared["inputs"][0]["physical_sha256"]


def test_partial_attempt_never_reuses_an_old_journal(tmp_path: Path) -> None:
    inputs = [Input("idle", lambda: make_scene("idle_coherence"), "small collaborator")]
    manifest, output, summary, journal = (
        tmp_path / name
        for name in ("inputs.json", "matrix.json", "summary.md", "attempt.jsonl")
    )
    assert main(["--freeze-inputs", str(manifest)], inputs=inputs) == 0
    journal.write_text("earlier attempt")
    assert (
        main(
            [
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--summary",
                str(summary),
                "--partial-output",
                str(journal),
            ],
            inputs=inputs,
        )
        == 1
    )
    assert journal.read_text() == "earlier attempt"
    assert not output.exists()
