"""Cache-only batch preflight at the public measurement seam."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.simulate.evolution import ReferenceResult, reference_key, write_reference

from tests.tools.family_pareto_support import make_scene
from tools import family_pareto
from tools.family_pareto import run_table
from tools.family_pareto_inputs import Input, Scene
from tools.family_pareto_reference import ReferenceCache
from tools.family_pareto_results import FamilyResult


def test_table_reports_all_misses_before_any_policy_or_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def build(offset: float) -> Scene:
        scene = make_scene("idle_coherence")
        return replace(
            scene, batch=replace(scene.batch, end_ns=scene.batch.end_ns + offset)
        )

    inputs = [
        Input("hit", lambda: build(0), "cached"),
        Input("missing-second", lambda: build(1), "missing"),
        Input("missing-third", lambda: build(2), "missing"),
    ]
    scene = inputs[0].build()
    with closing(scene.source):
        write_reference(
            tmp_path,
            scene.source,
            scene.batch,
            ReferenceResult(
                np.tile(scene.source.initial_density, (len(scene.batch.points), 1, 1)),
                1e-9,
            ),
        )
    missing_keys = []
    for item in inputs[1:]:
        scene = item.build()
        with closing(scene.source):
            missing_keys.append(reference_key(scene.source, scene.batch))

    def policy(item: Input, family: str, reference: ReferenceResult) -> FamilyResult:
        calls.append("policy")
        return FamilyResult(family, None, "unused", [])

    def engine(*args: object, **kwargs: object) -> None:
        calls.append("Engine")
        raise AssertionError("Engine must not start before complete cache preflight")

    # No public observer covers Engine construction; trap this external effect.
    monkeypatch.setattr(family_pareto, "Engine", engine)
    with pytest.raises(FileNotFoundError, match="reference cache miss") as failure:
        run_table(
            inputs,
            ("t4",),
            reference_cache=ReferenceCache(tmp_path),
            policy_runner=policy,
        )
    assert calls == []
    for item, key in zip(inputs[1:], missing_keys, strict=True):
        assert item.name in str(failure.value)
        assert key in str(failure.value)
