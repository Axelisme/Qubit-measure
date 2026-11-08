"""Pieces shared by the K-cycle readout-chain experiments of this family.

One cycle is one shot (one rep). It starts after the thermalisation wait K,
which is the program's relax_delay, and then plays eight blocks of
(D(x), read, read): the experiment's operation D at the sweep point x, then
two reads of the same pulse readout. Read 2k follows D_k and read 2k+1 follows
a read; chain_cycle_sums groups its 65 numbers by this order.
"""

from collections.abc import Callable

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from zcu_tools.analysis.readout_chain import (
    CYCLE_FEATURE_COUNT,
    CycleFit,
    CycleModel,
    CycleSums,
    cycle_curves,
)
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
    config,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import PulseReadout, TimelineBuilder

# Eight (D, read, read) blocks give the 16 reads per shot of chain_cycle_sums.
BLOCKS_PER_CYCLE = 8
READS_PER_CYCLE = 2 * BLOCKS_PER_CYCLE

# The 65 numbers of one cycle, as the feature axis of the saved sums.
FEATURES = tuple(range(CYCLE_FEATURE_COUNT))
# Running sums per point are z zᵀ with z = (1, the 65 numbers of a cycle).
AUGMENTED_SIZE = CYCLE_FEATURE_COUNT + 1


def emit_cycle(
    builder: TimelineBuilder,
    readout: PulseReadout,
    emit_operation: Callable[[], None],
) -> None:
    """Record the eight (D, read, read) blocks of one cycle on builder.

    emit_operation records the operation D for the current sweep point, for
    example builder.emit of a swept drive pulse. readout is the one pulse
    readout whose two emissions per block make the 16 reads. Every read is
    followed by a wait for its ADC window to close, so the second read starts
    after the first, and the next D starts after the second. The wait K is the
    builder's relax_delay, which follows every cycle, so each cycle starts K
    after the previous one ends.
    """
    with builder.repeat("cycle", BLOCKS_PER_CYCLE):
        emit_operation()
        builder.emit(readout)
        builder.delay_auto()
        builder.emit(readout)
        builder.delay_auto()


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


def g_and_third_decay_times(
    fit: CycleFit, g_name: str, third_name: str
) -> tuple[float, float, float, float]:
    """Return the g and third-state decay times of a T1 or echo fit, with errors.

    g_name is the time constant of the model's g and e rows (rows 0 and 1)
    and third_name that of the third row (row 2), for example "t1" and "tf".
    Return (g time, its Wald error, third-state time, its Wald error) in the
    fit's units.
    """
    # In t1_model and t2_echo_model the third row's curve is the same family
    # as the g row's, a free affine function of exp(-x / T). Exchanging the
    # two rows together with their time constants fits the data exactly as
    # well for T1, whose e row is flat, and nearly as well for echo when the
    # third state is rarely populated. So the fit can land in either labelling
    # and the checks cannot tell them apart. In thermal equilibrium g is the
    # most populated state. The fit may also put g in row 1, which shares the
    # g row's time constant, so the two times are exchanged only when the third
    # row has the largest fitted thermal population.
    thermal = fit.nuisances.thermal
    if thermal[2] > max(thermal[0], thermal[1]):
        g_name, third_name = third_name, g_name
    return (
        fit.parameters[g_name],
        fit.errors[g_name],
        fit.parameters[third_name],
        fit.errors[third_name],
    )


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


def plot_cycle_fit(
    plots: Plots,
    model: CycleModel,
    fit: CycleFit,
    axis: NDArray[np.float64],
    *,
    xlabel: str,
    result: str,
) -> None:
    """Publish "populations": per-point populations and the fitted curves.

    Points are fit.populations with their Wald error bars; lines are the
    model's populations at the fitted values on a dense axis, one color per
    state. The legend gives each state's fitted thermal population, since
    state labels are arbitrary.
    result is the first title line, for example the fitted parameter.
    """
    dense = np.linspace(axis.min(), axis.max(), 400)
    fitted = model.populations(dense, fit.parameters, fit.nuisances)

    fig, ax = plots.subplots(num="populations", figsize=config.figsize)
    for state in range(3):
        thermal = fit.nuisances.thermal[state]
        ax.errorbar(
            axis,
            fit.populations[:, state],
            yerr=fit.population_errors[:, state],
            fmt="o",
            markersize=3,
            color=f"C{state}",
            label=f"state {state}, thermal {thermal:.2f}",
        )
        ax.plot(dense, fitted[:, state], color=f"C{state}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("First-read population")
    ax.set_title(f"{result}\n{checks_text(fit)}", fontsize=9)
    # Outside the axes: the curves fill the whole population range.
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(True)
    fig.tight_layout()
