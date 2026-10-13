"""Classification boundary rendering through the public plotting helper."""

import matplotlib.pyplot as plt
import numpy as np
from zcu_tools.plotting.singleshot import plot_with_classified


def test_plot_boundaries_stay_in_assigned_half_planes() -> None:
    fig, ax = plt.subplots()
    try:
        plot_with_classified(ax, np.array([-1, 1, 0j]), -1, 1, 3)
        assert len(ax.patches) == 2
        assert np.asarray(ax.patches[0].get_path().vertices)[:, 0].max() <= 1e-12
        assert np.asarray(ax.patches[1].get_path().vertices)[:, 0].min() >= -1e-12
    finally:
        plt.close(fig)
