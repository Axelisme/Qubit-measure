"""One single-fluxonium acquire lifetime and ADC-owned round sampling."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import (
    AcquireRequest,
    CancellationQuery,
    ChannelSignals,
    EnvironmentSnapshot,
    RoundSignals,
    SamplingIdentity,
)
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.resources.simulation_replay import ReplayAcquire, save_replay
from zcu_tools.simulate.adc import (
    ChannelReadout,
    PreparedRawShots,
    round_statistics,
    sample_traces,
)
from zcu_tools.simulate.evolution import PathDiagnostics

from . import replay

if TYPE_CHECKING:
    from .runtime import SingleFluxoniumSource


@dataclass(frozen=True)
class SingleFluxoniumPreparation:
    """Noiseless acquisition inputs retained across rounds, never measured IQ.

    key: content digest including settings, snapshot, controls and layout.
    populations: read-only (*point_dims,states) DC-state probabilities after ADC
        normalization. channel: ADC conditional sums with matching state order.
    raw: immutable deferred physical-shot inputs, without a round RNG identity.
    paths: evolution-owned diagnostics, repeated on valid noiseless reuse.
    This record keeps physical and ADC state order together for Source/Acquire.
    """

    key: bytes
    populations: NDArray[np.float64]
    channel: ChannelReadout
    raw: PreparedRawShots
    paths: PathDiagnostics


class SingleFluxoniumAcquire:
    """One fixed request, invalidated by the next acquire or source close.

    Construct through SingleFluxoniumSource.open_acquire. Calls must serialize.
    Numeric preparation rereads controls and live snapshot each round; only
    noiseless inputs may be reused. Fresh RNG identities select each round.
    Returned raw sources own their retained inputs independently of this session.
    """

    def __init__(
        self, source: "SingleFluxoniumSource", request: AcquireRequest
    ) -> None:
        self._source = source
        self.request = request
        self._active = True

    def invalidate(self) -> None:
        """Reject later rounds; idempotent, without invalidating delivered raw."""
        self._active = False

    def sample_round(
        self,
        snapshot: EnvironmentSnapshot,
        index: int,
        *,
        cancel: CancellationQuery | None = None,
    ) -> IntegratedRound | RoundSignals:
        """Evolve/read/sample this zero-based round, or propagate numeric errors.

        snapshot supplies detached native DC values. Negative indices raise
        ValueError. An invalidated session raises RuntimeError. Private replay
        writes finish before return; I/O failures propagate. No partial result
        or failed preparation is retained. cancel is forwarded to evolution's
        chunk checkpoints; None disables cancellation. Pacing, completed-round
        retention and QICK normalization belong to the acquisition boundary.
        """
        if not self._active:
            raise RuntimeError("single-fluxonium acquire session is no longer active")
        if type(index) is not int or index < 0:
            raise ValueError("round index must be a nonnegative integer")
        owner, acquire = self._source, self.request
        prepared = owner.prepare(acquire, snapshot, cancel=cancel)
        if acquire.mode == "integrated":
            statistics = round_statistics(
                prepared.populations,
                (prepared.channel,),
                reps=acquire.reps,
                seed=acquire.root_seed,
                acquire_sequence=acquire.sequence,
                round_index=index,
            )
            result = IntegratedRound(
                statistics, prepared.raw.bind(statistics.raw_identity)
            )
        else:
            traces = sample_traces(
                prepared.populations,
                dict(prepared.raw.batches),
                reps=acquire.reps,
                identity=SamplingIdentity(
                    acquire.root_seed, (acquire.sequence, index, 0)
                ),
                rtol=owner.settings.numerics.solver.rtol,
                atol=owner.settings.numerics.solver.atol,
            )
            channels = []
            for channel in traces.channels:
                shaped = channel.iq.reshape(
                    (*acquire.point_dims, acquire.reps, *channel.iq.shape[1:])
                )
                ordered = np.moveaxis(
                    shaped, len(acquire.point_dims), acquire.avg_level
                )
                channels.append(
                    ChannelSignals(channel.channel, ordered.reshape(channel.iq.shape))
                )
            result = RoundSignals(tuple(channels))
        if owner.private_replay is not None:
            save_replay(
                owner.private_replay,
                ReplayAcquire(
                    settings=owner.effective_settings,
                    request=acquire,
                    versions=replay.replay_versions(),
                ),
                replace(snapshot, model_identity=owner.identity)
                if owner.settings.drift is not None
                else snapshot,
                index,
            )
        return result
