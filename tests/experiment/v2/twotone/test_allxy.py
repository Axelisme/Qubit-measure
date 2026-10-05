"""AllXY analysis reports the amplitude error as a relative drive amplitude."""

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.twotone.allxy import (
    ALLXY_SEQUENCE,
    AllXY_Exp,
    AllXY_Result,
    AllXYAnalyzeOptions,
    AllXYCfg,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots

_ANGLES = {"I": 0.0, "X90": np.pi / 2, "Y90": np.pi / 2, "X180": np.pi, "Y180": np.pi}


def _rotation(gate: str, scale: float) -> np.ndarray:
    theta = _ANGLES[gate] * scale
    c, s = np.cos(theta), np.sin(theta)
    if gate.startswith("Y"):
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _exact_signals(amplitude_error: float) -> np.ndarray:
    """Bloch z after each pair of resonant rotations scaled by 1 + error."""
    scale = 1.0 + amplitude_error
    z = []
    for first, second in ALLXY_SEQUENCE:
        bloch = _rotation(second, scale) @ _rotation(first, scale) @ [0.0, 0.0, 1.0]
        z.append(bloch[2])
    return 1.0 + 0.5 * np.asarray(z)


@pytest.mark.parametrize("amplitude_error", [0.02, -0.03])
def test_analysis_recovers_relative_amplitude_error(amplitude_error: float) -> None:
    record: RunRecord[AllXYCfg, AllXY_Result] = RunRecord(
        cfg=None,
        result=AllXY_Result(
            gate_idxs=np.arange(len(ALLXY_SEQUENCE), dtype=np.int64),
            signals=_exact_signals(amplitude_error).astype(np.complex128),
        ),
    )
    plots = Plots(NonPresentingHost())
    try:
        analysis = AllXY_Exp().analyze(record, AllXYAnalyzeOptions(), plots=plots)
    finally:
        plots.finish(present=False)
        plots.release()

    assert analysis.amplitude_error == pytest.approx(amplitude_error, abs=0.002)
    assert analysis.detune_param == pytest.approx(0.0, abs=0.005)
    assert analysis.residual_rms < 0.01
