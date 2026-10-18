"""Logical phase from independently acquired X90/Y90 projection shots."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .joint_readout.core import GECenters


@dataclass(frozen=True)
class QuadraturePhase:
    """GE-normalized contrast and logical Bloch phase in degrees, wrapped [-180,180).

    Unidentifiable phase is None with a reason, never a fabricated zero.
    Contrast describes thermal ensemble coherence, not pure-state fidelity.
    """

    contrast: float
    logical_phase_deg: float | None
    reason: str | None = None


def wrap_phase(degrees: float) -> float:
    """Wrap a finite logical angle in degrees to [-180,180)."""
    if not np.isfinite(degrees):
        raise ValueError("Logical phase must be finite")
    return float((degrees + 180) % 360 - 180)


def quadrature_phase(
    x_iq: NDArray[np.complex128],
    y_iq: NDArray[np.complex128],
    centers: GECenters,
) -> QuadraturePhase:
    """Return ensemble phase accrued after a calibrated target X90 preparation.

    x_iq/y_iq are nonempty one-dimensional independent projection shot arrays,
    not same-shot X/Y observations. Both use the saved source's GE centers and
    the same target tone. X projection is another X90 at preparation phase;
    Y projection is X90 with +90 degree pulse phase. For R_axis(+pi/2),
    q_X=1-2*P0_X=2*P1_X-1=contrast*cos(theta), and
    q_Y=1-2*P0_Y=2*P1_Y-1=contrast*sin(theta), where theta is
    logical Bloch Z accrual relative to the preparation frame. GE projection is
    linear along e-g, not nearest-center classification, clipping or inversion.
    Arrays may have different shot counts. Invalid shape/nonfinite IQ raises
    ValueError; GECenters owns finite/distinct center validation. Exactly zero
    contrast returns phase=None with a reason. No scientific contrast gate or
    thermal-state purification is applied; operators must inspect weak contrast.
    """
    values: list[float] = []
    delta = centers.e - centers.g
    for iq in (x_iq, y_iq):
        if iq.ndim != 1 or not iq.size or not np.all(np.isfinite(iq)):
            raise ValueError("Projection IQ must be a nonempty finite shot vector")
        population = ((iq.mean() - centers.g) / delta).real
        values.append(float(2 * population - 1))
    x, y = values
    contrast = float(np.hypot(x, y))
    if not np.isfinite(contrast):
        raise ValueError("GE-normalized quadratures must be finite")
    if contrast == 0:
        return QuadraturePhase(contrast, None, "Zero quadrature contrast")
    return QuadraturePhase(contrast, wrap_phase(float(np.degrees(np.arctan2(y, x)))))
