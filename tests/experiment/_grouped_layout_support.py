"""Synthetic mixed-rank grouped data shared by the public save contracts."""

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pytest
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import DataVariable, LabberPayload
from zcu_tools.experiment import (
    GroupedAxesSpec,
    GroupedLoadData,
    RunRecord,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)


class MixedCfg(ConfigBase):
    name: str = "mixed grids"


@dataclass(frozen=True)
class MixedResult:
    delays: np.ndarray
    tones: np.ndarray
    signals: np.ndarray
    detunes: np.ndarray


def build_mixed(data: GroupedLoadData[MixedCfg]) -> MixedResult:
    signal = data.variable("signals")
    return MixedResult(
        signal.axes[0], signal.axes[1], signal.z, data.variable("detunes").z
    )


MIXED_COMMON = GroupedAxesSpec(
    variables=(
        VariableSpec(
            "signals",
            (
                VariableAxisSpec("Delay", "s", "delays", scale=1e-6),
                VariableAxisSpec("Tone", "", "tones", dtype=np.int64),
            ),
            VariableZSpec("signals", "Signal", "V", scale=1e-3, dtype=np.complex128),
        ),
        VariableSpec(
            "detunes",
            (VariableAxisSpec("Tone", "", "tones", dtype=np.int64),),
            VariableZSpec("detunes", "Detune", "Hz", scale=1e6, dtype=np.float64),
        ),
    ),
    result_type=MixedResult,
    cfg_type=MixedCfg,
    tag="test/mixed-grids",
    result_builder=build_mixed,
)


def mixed_spec(*, independent: bool) -> GroupedAxesSpec[MixedResult, MixedCfg]:
    if not independent:
        return MIXED_COMMON
    try:
        return replace(MIXED_COMMON, labber_layout="independent")
    except TypeError as error:
        if "labber_layout" not in str(error):
            raise
        pytest.fail(
            "GroupedAxesSpec must accept explicitly declared independent layout"
        )


def mixed_record() -> RunRecord[MixedCfg, MixedResult]:
    return RunRecord(
        cfg=MixedCfg(),
        result=MixedResult(
            delays=np.array([1.0, 2.0, 4.0]),
            tones=np.array([2, 5]),
            signals=np.array([[1 + 2j, 3 - 4j, 5 + 6j], [7 - 8j, 9 + 10j, 11 - 12j]]),
            detunes=np.array([0.25, -0.5]),
        ),
    )


def assert_mixed_disk_values(variables: Mapping[DataVariable, LabberPayload]) -> None:
    assert tuple(variables) == (DataVariable("signals"), DataVariable("detunes"))
    signal, detune = (
        variables[DataVariable("signals")],
        variables[DataVariable("detunes")],
    )
    assert [(axis.name, axis.unit) for axis in signal.axes] == [
        ("Delay", "s"),
        ("Tone", ""),
    ]
    assert [(axis.name, axis.unit) for axis in detune.axes] == [("Tone", "")]
    np.testing.assert_allclose(signal.axes[0].values, [1e-6, 2e-6, 4e-6])
    np.testing.assert_array_equal(signal.axes[1].values, [2, 5])
    np.testing.assert_array_equal(detune.axes[0].values, [2, 5])
    np.testing.assert_allclose(
        signal.z,
        [
            [0.001 + 0.002j, 0.003 - 0.004j, 0.005 + 0.006j],
            [0.007 - 0.008j, 0.009 + 0.010j, 0.011 - 0.012j],
        ],
    )
    np.testing.assert_allclose(detune.z, [250_000.0, -500_000.0])
