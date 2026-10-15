"""Explicit joint-initial-state and receiver sampling, no trajectories."""

import numpy as np

from .request import (
    ADCSamples,
    PopulationInput,
    ReadoutPulse,
    SampleGrid,
    SamplingIdentity,
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
        """Draw joint initial state once per shot, then independent I/Q noise.

        populations must match model.state_labels exactly. pulses/grid describe
        one readout repeated independently shots times. identity seeds PCG64
        locally, without touching global RNG. shots is a nonnegative integer.
        The deterministic response is solved for all initial states once, not
        once per shot. Return one realization with float64 traces and exact
        discrete sample sums; raw/averaged consumers must share these arrays.
        ValueError reports invalid state order or shot count; response propagates
        solver/callback failures. No QICK normalization, rounding, transition
        trajectory variance or cross-shot memory is modeled.
        """
        if populations.state_labels != self.model.state_labels:
            raise ValueError("population state order must match readout model")
        if type(shots) is not int or shots < 0:
            raise ValueError("shots must be a nonnegative integer")
        conditional = self.response(pulses, grid)
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence((identity.seed, *identity.stream)))
        )
        states = rng.choice(len(populations.values), size=shots, p=populations.values)
        return self._draw(conditional, states, rng, identity)

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
        Conditional responses are solved once per distinct grid, not per shot.
        """
        if populations.state_labels != self.model.state_labels:
            raise ValueError("population state order must match readout model")
        if type(shots) is not int or shots < 0:
            raise ValueError("shots must be a nonnegative integer")
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence((identity.seed, *identity.stream)))
        )
        states = rng.choice(len(populations.values), size=shots, p=populations.values)
        return tuple(
            self._draw(self.response(pulses, grid), states, rng, identity)
            for grid in grids
        )

    def _draw(self, conditional, states, rng, identity: SamplingIdentity) -> ADCSamples:
        traces = conditional.traces[states].copy()
        covariance = self.model.receiver.noise_covariance
        if np.any(covariance):
            eigenvalues, vectors = np.linalg.eigh(covariance)
            # A PSD receiver covariance can have a zero eigenvalue with a tiny
            # roundoff residual. This factorization does not alter solver states.
            root = vectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
            noise = rng.standard_normal(traces.shape) @ root.T
            traces += noise
        return ADCSamples(traces, traces.sum(axis=1), states, identity)
