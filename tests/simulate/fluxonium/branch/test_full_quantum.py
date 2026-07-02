from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from zcu_tools.simulate.fluxonium.branch.full_quantum import (
    calc_branch_population,
    calc_branch_population_over_flux,
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
