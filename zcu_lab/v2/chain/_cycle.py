"""Pieces shared by the K-cycle readout-chain experiments of this family.

One cycle is one shot (one rep). It starts after the thermalisation wait K,
which is the program's relax_delay, and then plays eight blocks of
(D(x), read, read): the experiment's operation D at the sweep point x, then
two reads of the same pulse readout. Read 2k follows D_k and read 2k+1 follows
a read; chain_cycle_sums groups its 65 numbers by this order. Each experiment
writes this loop in its own run; every read is followed by delay_auto, so the
second read starts after the first ADC window and the next D after the second.
"""

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.readout_chain import (
    CYCLE_FEATURE_COUNT,
    CycleFit,
    CycleSums,
)
from zcu_tools.analysis.shot_uncertainty import iq_shot_sem
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.utils.process import find_rotate_angle

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


def first_read_iq(sums: CycleSums) -> NDArray[np.complex128]:
    """Return the mean complex IQ of read 0 at each point, in ADC units."""
    return (sums.feature_sums[:, 0] + 1j * sums.feature_sums[:, 1]) / sums.cycle_count


def first_read_projection(
    sums: CycleSums, *, skip: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return first-read mean IQ and its shot SEM on the same axis, in a.u.

    Use the dominant axis of the sweep-point means after omitting the first
    skip points from the axis estimate, with the same sign as rotate2real.
    Project all points, using I if fewer than two points define the axis.
    CycleSums already preserves the first read's raw per-point second moments.
    """
    signals = first_read_iq(sums)
    axis_samples = signals[skip:]
    angle = find_rotate_angle(axis_samples) if len(axis_samples) >= 2 else 0.0
    moments = np.column_stack(
        (sums.cycle_count, sums.feature_sums[:, :2], sums.feature_sums[:, 32:35])
    )
    return (signals * np.exp(-1j * angle)).real, iq_shot_sem(
        moments, direction=(np.cos(angle), np.sin(angle))
    )


def checks_text(fit: CycleFit) -> str:
    """Return one line per lack-of-fit check, with p where fit gives one."""
    parts = []
    for name, check in (
        ("within-point", fit.within_point),
        ("sharing", fit.sharing),
        ("curve", fit.curve),
    ):
        if not np.isfinite(check.J):
            parts.append(f"{name}: J unavailable (df={check.df})")
            continue
        p = "" if check.p is None else f", p={check.p:.2g}"
        parts.append(f"{name} J/df={check.J:.3g}/{check.df}{p}")
    return "\n".join(parts)
