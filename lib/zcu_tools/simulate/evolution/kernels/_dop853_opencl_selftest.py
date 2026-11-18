"""Fixed mixed-precision check before automatic DOP853 GPU admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..secular import SecularDrive, SecularModel
from ._dop853_data import Segments, pack_models
from .dop853 import DOP853Plan, EndpointData, advance_endpoints

if TYPE_CHECKING:
    from ._dop853_opencl import OpenCLDOP853

PROBE_DIMENSION = 3
PROBE_REVISION = "1"


def _probe_input() -> EndpointData:
    # Noncommuting quadratures, nonzero phase, complex sampled playback and
    # lossy dynamics expose a wrong phasor, lane write or endpoint handoff.
    drive = SecularDrive(
        "probe",
        np.array([[0, 0.04, 0], [0.04, 0, 0.02], [0, 0.02, 0]], np.complex128),
        np.array([[0, 0.025j, 0], [-0.025j, 0, 0.03j], [0, -0.03j, 0]], np.complex128),
    )
    model = SecularModel(
        (0,),
        "opencl-self-test",
        np.array([0.0, 0.211, 0.437]),
        np.eye(PROBE_DIMENSION, dtype=np.complex128),
        (drive,),
        np.array([[0, 0.001, 0], [0, 0, 0.002], [0, 0, 0]]),
        0.0002 * (1 - np.eye(PROBE_DIMENSION)),
    )
    knots = np.array([0.0, 21.0, 53.0, 81.0, 100.0])
    segments = Segments(
        np.array([0], dtype=np.int64),
        np.array([0.0]),
        knots[-1:],
        np.array([2], dtype=np.int64),
        np.array([[0, 0]], dtype=np.int64),
        np.array([[0.173, -0.229]]),
        np.array([[0.17, -0.13]]),
        np.array([[0.19, -0.31]]),
        np.array([[0, 0]], dtype=np.int64),
        np.zeros((1, 2)),
        np.array([0, len(knots)], dtype=np.int64),
        knots,
        np.array([0, 0.6 + 0.3j, 1 - 0.1j, 0.4 - 0.4j, 0]),
        np.empty(0, dtype=np.int64),
        np.empty(0),
    )
    return EndpointData(pack_models([model]), segments)


def endpoint_error(runner: OpenCLDOP853) -> float:
    """Run one fixed 100 ns sampled endpoint; return max CPU–GPU entry error.

    Compare production FP64 CPU and mixed-precision GPU DOP853 at the same
    budget; propagate build, driver and arithmetic failures to detection.
    """
    data = _probe_input()
    psi = np.array([2, 1j, 1 - 1j]) / np.sqrt(7)
    initial = (psi[:, None] * psi.conj())[None, :, :]
    cpu = initial.copy()
    row = np.zeros(1, dtype=np.int64)
    budget = 1e-3
    gpu = runner.advance(data, (row, row), initial, budget)
    advance_endpoints(
        data, (row, row), cpu, budget, DOP853Plan(np.array(["interaction"]), row - 1)
    )
    return float(np.max(np.abs(cpu - gpu)))
