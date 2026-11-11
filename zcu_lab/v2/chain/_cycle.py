"""Pieces shared by the K-cycle readout-chain experiments of this family.

One cycle is one shot (one rep). It starts after the thermalisation wait K,
which is the program's relax_delay, and then plays eight blocks of
(D(x), read, read): the experiment's operation D at the sweep point x, then
two reads of the same pulse readout. Read 2k follows D_k and read 2k+1 follows
a read; chain_cycle_sums groups its 65 numbers by this order. Each experiment
writes this loop in its own run; every read is followed by delay_auto, so the
second read starts after the first ADC window and the next D after the second.
"""

from collections.abc import Callable

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from zcu_tools.analysis.readout_chain import (
    CYCLE_FEATURE_COUNT,
    CycleFit,
    CycleSums,
    cycle_curves,
)
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.plotting.plots import Plots

# Eight (D, read, read) blocks give the 16 reads per shot of chain_cycle_sums.
BLOCKS_PER_CYCLE = 8
READS_PER_CYCLE = 2 * BLOCKS_PER_CYCLE

# The 65 numbers of one cycle, as the feature axis of the saved sums.
FEATURES = tuple(range(CYCLE_FEATURE_COUNT))
# Running sums per point are z zᵀ with z = (1, the 65 numbers of a cycle).
AUGMENTED_SIZE = CYCLE_FEATURE_COUNT + 1


def cycle_sums_variables(axis: VariableAxisSpec) -> tuple[VariableSpec, ...]:
    """Return the grouped variables that save CycleSums along the sweep axis.

    axis declares the sweep coordinate with field_name="axis". The record keeps
    per point the cycle count and the 65 feature sums, and one 65 × 65 product
    sum added over all points, which only the live plot uses.
    """
    return (
        VariableSpec(
            "cycle_count",
            (axis,),
            VariableZSpec("cycle_count", label="Cycles", unit=""),
        ),
        VariableSpec(
            "feature_sums",
            (VariableAxisSpec(label="Feature", unit="", values=FEATURES), axis),
            VariableZSpec("feature_sums", label="Feature sum", unit="a.u."),
        ),
        VariableSpec(
            "product_sum",
            (
                VariableAxisSpec(label="Column feature", unit="", values=FEATURES),
                VariableAxisSpec(label="Row feature", unit="", values=FEATURES),
            ),
            VariableZSpec("product_sum", label="Product sum", unit="a.u."),
        ),
    )


def restore_cycle_sums[C: ConfigBase](data: GroupedLoadData[C]) -> CycleSums:
    """Rebuild the saved CycleSums from the variables of cycle_sums_variables."""
    cycle_count = data.variable("cycle_count")
    return CycleSums(
        axis=cycle_count.axes[0],
        cycle_count=cycle_count.z,
        feature_sums=data.variable("feature_sums").z,
        product_sum=data.variable("product_sum").z,
    )


def live_cycle_curves(
    plots: Plots, axis: NDArray[np.float64], *, xlabel: str
) -> Callable[[NDArray[np.float64]], None]:
    """Show the all-number and first-read curves of cycle_curves on one axes.

    Create the "measurement" figure with both curves in sigma units and return
    the SignalBuffer update. The update receives the (points, 66, 66) running
    sums of all completed rounds, pools them into CycleSums along axis and
    redraws both curves. cycle_curves errors propagate.
    """
    fig, ax = plots.subplots(num="measurement")
    ax.set(xlabel=xlabel, ylabel="Projected signal (sigma)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid()
    (all_line,) = ax.plot([], [], marker=".", label="all 65 numbers")
    (first_line,) = ax.plot([], [], marker=".", label="first read")
    ax.legend()

    @plots.on_plot_thread
    def update(augmented_sums: NDArray[np.float64]) -> tuple[Line2D, Line2D]:
        curves = cycle_curves(CycleSums.from_augmented(axis, augmented_sums))
        all_line.set_data(axis, curves.main_sigma)
        first_line.set_data(axis, curves.first_read_sigma)
        ax.relim()
        ax.autoscale_view()
        return all_line, first_line

    fig.show()
    return update


def first_read_iq(sums: CycleSums) -> NDArray[np.complex128]:
    """Return the mean complex IQ of read 0 at each point, in ADC units."""
    return (sums.feature_sums[:, 0] + 1j * sums.feature_sums[:, 1]) / sums.cycle_count


def checks_text(fit: CycleFit) -> str:
    """Return the three lack-of-fit checks as J/df, with p where fit gives one."""
    parts = []
    for name, check in (
        ("within-point", fit.within_point),
        ("sharing", fit.sharing),
        ("curve", fit.curve),
    ):
        p = "" if check.p is None else f", p={check.p:.2g}"
        parts.append(f"{name} J/df={check.J:.0f}/{check.df}{p}")
    return "; ".join(parts)
