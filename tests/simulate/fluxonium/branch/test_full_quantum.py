from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
import pytest
from zcu_tools.simulate.fluxonium.branch.full_quantum import (
    calc_branch_population,
    calc_branch_population_over_flux,
    make_hilbertspace,
)


def test_full_quantum_upto_is_required() -> None:
    assert (
        inspect.signature(calc_branch_population).parameters["upto"].default
        is inspect.Parameter.empty
    )
    assert (
        inspect.signature(calc_branch_population_over_flux).parameters["upto"].default
        is inspect.Parameter.empty
    )


def test_calc_branch_population_rejects_non_positive_upto() -> None:
    with pytest.raises(ValueError, match="upto must be positive"):
        calc_branch_population(cast(Any, object()), [0], upto=0)


def test_calc_branch_population_over_flux_rejects_non_positive_upto() -> None:
    with pytest.raises(ValueError, match="upto must be positive"):
        calc_branch_population_over_flux(
            np.array([0.5]),
            params=(4.0, 1.0, 0.5),
            r_f=5.0,
            qub_dim=2,
            qub_cutoff=10,
            res_dim=2,
            g=0.05,
            upto=0,
        )


def test_branch_population_matches_uncoupled_bare_state_occupations() -> None:
    hilbertspace = make_hilbertspace(
        params=(4.0, 1.0, 0.5),
        r_f=5.0,
        qub_dim=3,
        qub_cutoff=10,
        res_dim=3,
        g=0.0,
        flux=0.35,
    )

    populations = calc_branch_population(hilbertspace, branchs=[2, 0, 1], upto=2)

    assert list(populations) == [2, 0, 1]
    for branch, expected in [(2, [2.0, 2.0]), (0, [0.0, 0.0]), (1, [1.0, 1.0])]:
        assert populations[branch].shape == (2,)
        np.testing.assert_allclose(populations[branch], expected, rtol=0.0, atol=1e-12)


def test_branch_population_over_flux_preserves_uncoupled_occupations() -> None:
    populations = calc_branch_population_over_flux(
        np.array([0.2, 0.35, 0.5]),
        params=(4.0, 1.0, 0.5),
        r_f=5.0,
        qub_dim=3,
        qub_cutoff=10,
        res_dim=3,
        g=0.0,
        upto=2,
        branchs=[2, 0, 1],
        batch_size=2,
    )

    assert populations.shape == (3, 3, 2)
    np.testing.assert_allclose(
        populations,
        [
            [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]],
            [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]],
            [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]],
        ],
        rtol=0.0,
        atol=1e-12,
    )
