"""Saved-sum mapping and best-setting summary for thermal-chain sweeps."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.readout_chain import (
    THERMAL_MOMENT_COUNT,
    SettingScore,
    ThermalChainSums,
)
from zcu_tools.experiment import (
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)

READS_PER_SHOT = 16


def readout_sums_variables(setting: VariableAxisSpec) -> tuple[VariableSpec, ...]:
    """Map the sweep axis and all ThermalChainSums fields to saved variables.

    setting supplies the label, disk unit and scale of the result's axis.
    All 107 sums remain available, including every adjacent pair component.
    """
    return (
        VariableSpec(
            "shot_count",
            (setting,),
            VariableZSpec("shot_count", label="Shots", unit=""),
        ),
        VariableSpec(
            "feature_sums",
            (
                VariableAxisSpec(
                    label="Moment", unit="", values=tuple(range(THERMAL_MOMENT_COUNT))
                ),
                setting,
            ),
            VariableZSpec("feature_sums", label="Moment sum", unit="a.u."),
        ),
    )


def restore_readout_sums(data: GroupedLoadData[Any]) -> ThermalChainSums:
    """Restore shot counts, axis and 107 sums from their saved variables."""
    shot_count = data.variable("shot_count")
    return ThermalChainSums(
        axis=shot_count.axes[0],
        shot_count=shot_count.z,
        feature_sums=data.variable("feature_sums").z,
    )


def finished_sums(
    axis: NDArray[np.float64], sums: NDArray[np.float64]
) -> ThermalChainSums:
    """Keep settings with a completed round from a (settings,108) buffer.

    axis uses the experiment's units. Unfinished settings are NaN rows.
    Raise RuntimeError if no setting has completed a round.
    """
    finished = ~np.isnan(sums[:, 0])
    if not finished.any():
        raise RuntimeError(
            "Stopped before any setting finished a round; no sums to keep"
        )
    return ThermalChainSums.from_processor(axis[finished], sums[finished])


@dataclass(frozen=True)
class ChainAnalysis:
    """Best setting by descending thermal-chain score; no writeback proposed.

    best: SettingScore in the sweep's units (gain a.u., frequency MHz or
        ADC window us). Errors are model-conditional approximate scales.
        M diagnoses correlation growth, not identifiable physical leakage.
        QND is reported without clipping. Exact score ties keep acquisition order.
    """

    best: SettingScore
