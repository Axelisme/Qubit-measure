"""Public Engine Floquet admission, fallback routing and full-density checks."""

import json
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, NoValidKernel

from .contract.floquet_cases import floquet_case, prefix_deferred_case
from .contract.reference import converged_reference, density_error

_EXPECTED_ROWS = json.loads(
    Path(__file__).with_name("contract").joinpath("floquet_admission.json").read_text()
)["rows"]
_NAMES = (
    "legacy",
    "sideband",
    "bound-refusal",
    "dynamic-range",
    "invalid-frame",
    "mixed-duration",
    "mixed-gain",
)
# ADR-0078: short CPU fallbacks use the fixed physical accuracy budget.
# The long mixed-duration control keeps its pre-staging snapshot.
_CONVERGED_REFERENCE_CASES = {
    "bound-refusal",
    "dynamic-range",
    "invalid-frame",
    "mixed-gain",
}


@pytest.mark.parametrize(("index", "name"), enumerate(_NAMES))
def test_floquet_admission_and_density(index: int, name: str) -> None:
    expected = _EXPECTED_ROWS[index]
    case = floquet_case(name)
    with closing(case.source), closing(Engine(1, families=("floquet",))) as pinned:
        if expected["accepted"]:
            admitted = pinned.evolve(case.source, (case.batch,))
            assert admitted.paths.segments_by_kernel == {
                "floquet": len(case.batch.points)
            }
        else:
            with pytest.raises(NoValidKernel) as failure:
                pinned.evolve(case.source, (case.batch,))
            refusal = expected["refusal"]
            assert failure.value.points == tuple(refusal["points"])
            assert failure.value.segments == tuple(refusal["segments"])
            assert failure.value.attempted == tuple(refusal["attempted"])
        with closing(Engine(1, families=("floquet", "dop853"))) as engine:
            result = engine.evolve(case.source, (case.batch,), telemetry=True)
        assert result.paths.segments_by_kernel == expected["families"]
        routing = result.paths.routing.get("floquet")
        assert (0 if routing is None else routing.rejected_rows) == expected[
            "rejected_rows"
        ]
        if name in _CONVERGED_REFERENCE_CASES:
            reference = converged_reference(case.source, case.batch)
            assert density_error(result.densities, reference.density) <= 2e-3
        else:
            density = np.array(expected["density_real"]) + 1j * np.array(
                expected["density_imag"]
            )
            np.testing.assert_allclose(result.densities, density, rtol=0, atol=2e-12)


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("name", ["sideband", "bound-refusal"])
def test_prefix_and_deferred_rows_share_frames_within_each_evolve(
    workers: int, name: str
) -> None:
    case = prefix_deferred_case(name)
    with (
        closing(case.source),
        closing(Engine(workers, families=("floquet", "t6", "dop853"))) as engine,
    ):
        first = engine.evolve(case.source, (case.batch,), telemetry=True)
        second = engine.evolve(case.source, (case.batch,), telemetry=True)
    for result in (first, second):
        assert result.paths.routing["floquet"].prepare_builds == 2
        if name == "sideband":
            assert result.paths.routing["floquet"].rejected_rows == 0
            assert result.paths.segments_by_kernel["floquet"] == 4
        else:
            assert result.paths.routing["floquet"].rejected_rows == 3
            assert "floquet" not in result.paths.segments_by_kernel
    assert first.paths.segments_by_kernel == second.paths.segments_by_kernel
    np.testing.assert_array_equal(first.densities, second.densities)


def test_refused_prefix_frame_keeps_deferred_sideband_candidates() -> None:
    case = prefix_deferred_case("sideband", prefix_duration_ns=16000.0)
    with (
        closing(case.source),
        closing(Engine(1, families=("floquet", "t6", "dop853"))) as engine,
    ):
        result = engine.evolve(case.source, (case.batch,), telemetry=True)
    assert result.paths.routing["floquet"].prepare_builds == 2
    assert result.paths.routing["floquet"].rejected_rows == 1
    assert result.paths.segments_by_kernel["floquet"] == 2
