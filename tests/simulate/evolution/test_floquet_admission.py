"""Public Engine admission, routing and density against the pre-staging oracle."""

import json
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, NoValidKernel

from .contract.floquet_cases import floquet_case

_ORACLE = json.loads(
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


@pytest.mark.parametrize(("index", "name"), enumerate(_NAMES))
def test_floquet_admission_and_density_match_base(index: int, name: str) -> None:
    expected = _ORACLE[index]
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
        density = np.array(expected["density_real"]) + 1j * np.array(
            expected["density_imag"]
        )
        np.testing.assert_allclose(result.densities, density, rtol=0, atol=2e-12)
