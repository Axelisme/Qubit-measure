"""Driven cavity photons, dispersive Stark shifts and branch dephasing."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.adc import ReadoutPulse, cavity_fields

from .model import JointModel, readout_model
from .parameters import Resonator

RealArray = NDArray[np.float64]


@dataclass(frozen=True)
class CavityResponse:
    """Owned numeric response in DC joint energy-state order, with two cavities.

    chi_ghz: (state,2) state-conditioned cavity frequency minus bare frequency.
    photons: (...,state,2) dimensionless |alpha|², including coherent interference.
    stark_shift_ghz: (...,state) sum of chi_ghz * photons, in energy/h GHz.
    dephasing_per_ns: (...,state,state) symmetric coherence decay rates, with zero
        diagonal. The shared-line term is half the squared difference of the
        summed outgoing branch fields. Independent internal losses add half
        kappa_internal times each branch-field difference squared.
    response() has a leading time axis; average() has no leading axis. These are
    driven contributions only, not residual thermal-photon dephasing or T1/2.
    """

    chi_ghz: RealArray
    photons: RealArray
    stark_shift_ghz: RealArray
    dephasing_per_ns: RealArray


class CavityDrive:
    """Fixed-DC cavity response for one point's incident ReadoutPulse tuple.

    joint and resonators supply the existing joint-state pulls and shared-line
    linewidths. pulses use absolute ns/GHz and pulse-local phases in the neutral
    readout model's zero-GHz reference, as in FTFTemplatePlan.cavity_drives.
    No gain conversion occurs here. Empty pulses mean vacuum; multiple pulses
    add fields coherently, including ringdown. Population rates never mix the
    fixed state branches. This object performs no density evolution or sampling.
    """

    def __init__(
        self,
        joint: JointModel,
        resonators: tuple[Resonator, Resonator],
        pulses: tuple[ReadoutPulse, ...],
    ) -> None:
        self._model = readout_model(joint, resonators)
        self._chi = self._model.frequencies_ghz - [r.frequency_ghz for r in resonators]
        self._pulses = pulses

    def response(self, times_ns: RealArray) -> CavityResponse:
        """Evaluate at a finite nonempty increasing emitted-ns vector.

        Returns CavityResponse arrays with a leading time axis. Invalid times
        or envelope values raise ValueError; ADC field failures raise RuntimeError.
        Receiver calibration and delay do not enter the intracavity response.
        """
        fields = cavity_fields(self._model, self._pulses, times_ns)
        photons = np.abs(fields) ** 2
        delta = fields[:, :, None, :] - fields[:, None, :, :]
        outgoing = delta @ np.sqrt(self._model.kappa_external_per_ns / 2)
        dephasing = np.abs(outgoing) ** 2 / 2
        dephasing += np.sum(
            np.abs(delta) ** 2 * (self._model.kappa_internal_per_ns / 2), axis=-1
        )
        return CavityResponse(
            self._chi.copy(),
            photons,
            np.sum(self._chi * photons, axis=-1),
            dephasing,
        )

    def average(self, start_ns: float, end_ns: float) -> CavityResponse:
        """Average photons, Stark shifts and decay rates over a positive interval.

        Endpoints are finite absolute emitted ns, with end_ns > start_ns. Fields
        retain their full pulse history, even for an interval after switch-off.
        Integrates the rates themselves, not squared time-averaged fields.
        Gaussian quadrature splits at pulse supports and declared envelope corners
        and refines to rtol=1e-7, atol=1e-13. Nonconvergence raises RuntimeError;
        invalid intervals/envelopes raise ValueError. Output has no time axis.
        """
        if not np.isfinite([start_ns, end_ns]).all() or end_ns <= start_ns:
            raise ValueError("cavity average interval must be finite and positive")
        corners = {start_ns, end_ns} | {
            t
            for pulse in self._pulses
            for t in (pulse.start_ns, pulse.end_ns, *pulse.breakpoints_ns)
            if start_ns < t < end_ns
        }
        boundaries = np.array(sorted(corners))
        previous = None
        for order in (16, 32, 64, 128, 256, 512, 1024):
            nodes, weights = _gauss_rule(order)
            half = np.diff(boundaries) / 2
            mid = (boundaries[1:] + boundaries[:-1]) / 2
            times = (mid[:, None] + half[:, None] * nodes).ravel()
            scaled = (half[:, None] * weights).ravel() / (end_ns - start_ns)
            response = self.response(times)
            values = tuple(
                np.einsum("t,t...->...", scaled, value)
                for value in (
                    response.photons,
                    response.stark_shift_ghz,
                    response.dephasing_per_ns,
                )
            )
            if previous is not None and all(
                np.allclose(a, b, rtol=1e-7, atol=1e-13)
                for a, b in zip(values, previous, strict=True)
            ):
                return CavityResponse(self._chi.copy(), *values)
            previous = values
        raise RuntimeError("cavity segment average did not converge")


@lru_cache(maxsize=7)
def _gauss_rule(order: int) -> tuple[RealArray, RealArray]:
    """Share immutable quadrature coordinates, never physical response data."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights
