import matplotlib.pyplot as plt
import numpy as np
from zcu_tools.experiment.utils.single_shot import fit_singleshot2d


def test_singleshot2d_fit_repeats_exactly_above_the_subsample_limit() -> None:
    rng = np.random.default_rng(5)
    blobs = [complex(-1.0, 0.0), complex(1.0, 0.5)]
    signals = np.concatenate(
        [
            center + rng.normal(0.0, 0.4, 7000) + 1j * rng.normal(0.0, 0.4, 7000)
            for center in blobs
        ]
    )
    open_before = set(plt.get_fignums())
    state = np.random.get_state()
    try:
        np.random.seed(0)
        first = fit_singleshot2d(signals, num_gauss=2, method="standard")
        np.random.seed(1)  # the global RNG must not change the subsample
        second = fit_singleshot2d(signals, num_gauss=2, method="standard")
    finally:
        np.random.set_state(state)
        for number in set(plt.get_fignums()) - open_before:
            plt.close(number)

    np.testing.assert_array_equal(first, second)
