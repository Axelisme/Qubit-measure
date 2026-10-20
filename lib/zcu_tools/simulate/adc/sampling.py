"""Explicit initial-state, readout-path and receiver sampling."""

from collections.abc import Mapping

import numpy as np

from zcu_tools.core.simulation import ChannelSignals, RoundSignals, SamplingIdentity

from ._integrated_paths import IntegratedPaths
from ._jump_paths import JumpPaths
from ._path_sampling import physical_shots
from .model import RealArray
from .raw import PreparedRawShots
from .readout_batch import ReadoutBatch
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
        Traces follow continuous-time directional transitions and
        retain cavity memory across jumps. Vacuum fields are prepared once, not
        solved per shot. All coherent carriers share the same physical path.
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
        All carriers and grids share one continuous-time path per shot,
        including jumps during ringdown and gaps. The population origin is the
        earliest pulse start, not the first sample.
        """
        if populations.state_labels != self.model.state_labels:
            raise ValueError("population state order must match readout model")
        if type(shots) is not int or shots < 0:
            raise ValueError("shots must be a nonnegative integer")
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence((identity.seed, *identity.stream)))
        )
        states = rng.choice(len(populations.values), size=shots, p=populations.values)
        if grids and pulses and np.any(self.model.rates_per_ns):
            observer = IntegratedPaths(self.model, pulses, grids, self.rtol, self.atol)
            paths = JumpPaths(
                states, observer.start, observer.end, self.model.rates_per_ns, rng
            )
            return tuple(
                self._draw(values, states, rng, identity)
                for values in observer.evaluate(paths, traces=True)
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


def sample_traces(
    populations: RealArray,
    batches: Mapping[int, ReadoutBatch],
    *,
    reps: int,
    identity: SamplingIdentity,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> RoundSignals:
    """Draw complete physical per-execution traces for every declared ADC.

    populations is finite nonnegative (*point_dims,states), normalized within
    1e-6, in all batch models' identical state order. batches maps nonnegative
    ADC indices in declaration order to physical pulse/grid contents. Each
    content_index is flat C-order (prod(point_dims),); all ADCs must use the
    same directional rates. Each ADC has equal positive sample counts across
    reads and contents; different ADCs may have different grids/read counts.
    reps is a nonnegative integer. identity is the complete validated local
    SamplingIdentity; its seed and full stream seed PCG64 without appending any
    suffix. FTF supplies stream (acquire_sequence,round_index,0). rtol/atol are
    positive finite vacuum-baseline solver tolerances.

    Return RoundSignals in mapping order with owned float64 IQ arrays shaped
    (prod(point_dims)*reps,reads,samples,2), points then shots in C order.
    One full CTMC path per point/shot is shared across all carriers, ADCs and
    windows from the earliest pulse to the latest emitted sample, including
    gaps and ringdown. Receivers add independent per-sample noise and offset
    once after coherent superposition. There is no rounding or normalization.
    Calls reproduce within the same numeric/RNG version without retaining
    paths/noisy results, mutating inputs or touching global RNG.

    Invalid shapes, rates, populations, sample counts, reps, identity or solver
    tolerances raise ValueError. Baseline/field failures raise RuntimeError;
    envelope errors propagate. No partial results are returned.
    """
    prepared = PreparedRawShots(populations, batches, reps=reps, rtol=rtol, atol=atol)
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence((identity.seed, *identity.stream)))
    )
    values = physical_shots(prepared, rng, traces=True)
    return RoundSignals(
        tuple(
            ChannelSignals(channel, iq.reshape((-1, *iq.shape[2:])))
            for channel, iq in values.items()
        )
    )
