"""Explicit initial-state, readout-path and receiver sampling."""

import numpy as np

from zcu_tools.core.simulation import SamplingIdentity

from ._trajectories import PathFields
from .request import (
    ADCSamples,
    PopulationInput,
    ReadoutPulse,
    SampleGrid,
)
from .response import ConditionalReadout


class ADC(ConditionalReadout):
    """Independent ADC tool combining conditional means with explicit sampling.

    model: shared-line/rates/receiver model, independent of an evolution instance.
    rtol/atol: positive finite internal readout solver tolerances. See response
        for deterministic first-moment semantics. The instance only reuses
        matrices; calls never inherit a previous shot's state, fields or noise.
    """

    def sample(
        self,
        populations: PopulationInput,
        pulses: tuple[ReadoutPulse, ...],
        grid: SampleGrid,
        identity: SamplingIdentity,
        *,
        shots: int,
    ) -> ADCSamples:
        """Draw initial states and readout jump paths, then independent I/Q noise.

        populations must match model.state_labels exactly. pulses/grid describe
        one readout repeated independently shots times. identity seeds PCG64
        locally, without touching global RNG. shots is a nonnegative integer.
        Single-pulse traces follow continuous-time directional transitions and
        retain cavity memory across jumps. Vacuum fields are prepared once, not
        solved per shot. Multi-pulse contents retain conditional-mean sampling.
        Return one realization with float64 traces and exact
        discrete sample sums; raw/averaged consumers must share these arrays.
        ValueError reports invalid state order or shot count; response propagates
        solver/callback failures. No QICK normalization, rounding or cross-shot
        memory is modeled.
        """
        return self.sample_windows(populations, pulses, (grid,), identity, shots=shots)[
            0
        ]

    def sample_windows(
        self,
        populations: PopulationInput,
        pulses: tuple[ReadoutPulse, ...],
        grids: tuple[SampleGrid, ...],
        identity: SamplingIdentity,
        *,
        shots: int,
    ) -> tuple[ADCSamples, ...]:
        """Sample several grids with one joint initial-state draw per shot.

        Inputs have the same units/order as sample. Each grid may have independent
        timing/demodulation, but uses the same physical pulse/rates and origin.
        Receiver noises are independent across windows/samples/shots. Return a
        tuple in grids order; each owns (shot,sample,2) traces, exact sums and the
        common initial-state indices. Invalid population order/count raises.
        Single-pulse grids share one continuous-time path per shot, including
        jumps during ringdown. The population origin is pulse start, not the
        first sample. Multi-pulse contents retain conditional-mean sampling.
        """
        if populations.state_labels != self.model.state_labels:
            raise ValueError("population state order must match readout model")
        if type(shots) is not int or shots < 0:
            raise ValueError("shots must be a nonnegative integer")
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence((identity.seed, *identity.stream)))
        )
        states = rng.choice(len(populations.values), size=shots, p=populations.values)
        if grids and len(pulses) == 1 and np.any(self.model.rates_per_ns):
            times = np.unique(np.concatenate([grid.times_ns for grid in grids]))
            delay = self.model.receiver.time_of_flight_ns
            paths = PathFields(
                self.model, pulses[0], times - delay, self.rtol, self.atol
            )
            outgoing = paths.draw(states, rng)
            outgoing *= np.exp(2j * np.pi * self.model.reference_frequency_ghz * delay)
            return tuple(
                self._draw(
                    self._calibrate(
                        outgoing[np.searchsorted(times, grid.times_ns)], grid
                    ),
                    states,
                    rng,
                    identity,
                )
                for grid in grids
            )
        return tuple(
            self._draw(
                self.response(pulses, grid).traces[states].copy(), states, rng, identity
            )
            for grid in grids
        )

    def _draw(self, traces, states, rng, identity: SamplingIdentity) -> ADCSamples:
        covariance = self.model.receiver.noise_covariance
        if np.any(covariance):
            eigenvalues, vectors = np.linalg.eigh(covariance)
            # A PSD receiver covariance can have a zero eigenvalue with a tiny
            # roundoff residual. This factorization does not alter solver states.
            root = vectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
            noise = rng.standard_normal(traces.shape) @ root.T
            traces += noise
        return ADCSamples(traces, traces.sum(axis=1), states, identity)
