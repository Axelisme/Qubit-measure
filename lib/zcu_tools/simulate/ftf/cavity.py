"""Driven cavity photons, dispersive Stark shifts and branch dephasing."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.adc import ReadoutPulse, cavity_field_average, cavity_fields

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
    No gain conversion occurs here. Empty pulses with omitted initial inputs
    mean vacuum; explicit initial memory retains ringdown. Multiple pulses
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

    def advance(
        self,
        start_ns: float,
        end_ns: float,
        populations: RealArray,
        initial_field: NDArray[np.complex128],
    ) -> tuple[CavityResponse, NDArray[np.complex128]]:
        """Average one span and return its selected branch's final complex field.

        start_ns/end_ns are finite absolute emitted ns with end > start.
        populations is a finite real (state,) vector in increasing DC-energy
        order at the span start. Its maximum selects the main branch; exact ties
        select the lowest energy (first state). No normalization or clipping
        occurs. initial_field is a finite complex (2,) shared by all branches at
        start_ns, in sqrt(photons) in the zero-GHz readout reference. Pulse-local
        phases/envelopes retain their original absolute supports. Only this span
        is propagated, including ringdown after pulse switch-off.

        Returns the span-averaged CavityResponse and an owned complex (2,) final
        main-branch field for the next span. This dominant-branch construction is
        semiclassical, not a representation of quantum cavity-state entanglement.
        Invalid coordinates, shapes, populations or fields raise ValueError;
        ADC propagation/convergence failures raise RuntimeError.
        """
        values = np.asarray(populations, dtype=np.float64)
        if values.shape != (len(self._chi),) or not np.isfinite(values).all():
            raise ValueError("cavity populations must be a finite (state,) vector")
        average = self.average(
            start_ns,
            end_ns,
            initial_field=initial_field,
            initial_time_ns=start_ns,
        )
        fields = cavity_fields(
            self._model,
            self._pulses,
            np.array([start_ns, end_ns]),
            initial_field=initial_field,
            initial_time_ns=start_ns,
        )
        return average, fields[-1, int(np.argmax(values))].copy()

    def average(
        self,
        start_ns: float,
        end_ns: float,
        *,
        initial_field: NDArray[np.complex128] | None = None,
        initial_time_ns: float = 0.0,
    ) -> CavityResponse:
        """Average photons, Stark shifts and decay rates over a positive interval.

        Endpoints are finite absolute emitted ns, with end_ns > start_ns. ADC
        retains full pulse history, including ringdown, and integrates same-time
        quadratic fields, not squared time-averaged fields. initial_field is an
        optional finite complex (2,) shared by all branches at initial_time_ns
        in sqrt(photons), in the zero-GHz readout reference. None means zero.
        initial_time_ns is finite nonnegative absolute ns, default 0, in the
        pulse's original time reference. Only history after that time propagates;
        pulse-local phases/envelopes remain unchanged. start_ns before this time
        raises ValueError naming both times. Omitting both initial inputs retains
        vacuum history from t=0. Flat inputs use
        analytic integrals; shaped inputs retain rtol=1e-7, atol=1e-13 quadrature.
        Invalid intervals/envelopes raise ValueError; solver or convergence
        failures raise RuntimeError. Output has no time axis.
        """
        average = cavity_field_average(
            self._model,
            self._pulses,
            start_ns,
            end_ns,
            initial_field=initial_field,
            initial_time_ns=initial_time_ns,
        )
        products = average.field_products
        n = len(self._chi)
        same = products[np.arange(n), :, np.arange(n), :]
        differences = (
            same[:, None, :, :]
            + same[None, :, :, :]
            - products.transpose(0, 2, 1, 3)
            - products.transpose(2, 0, 1, 3)
        )
        v = np.sqrt(self._model.kappa_external_per_ns / 2)
        dephasing = np.einsum("ijrq,r,q->ij", differences, v, v).real / 2
        dephasing += (
            np.einsum("ijrr,r->ij", differences, self._model.kappa_internal_per_ns).real
            / 2
        )
        return CavityResponse(
            self._chi.copy(),
            average.photons,
            np.sum(self._chi * average.photons, axis=-1),
            dephasing,
        )
