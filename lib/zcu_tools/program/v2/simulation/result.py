"""Integrated environment delivery: pure statistics plus environment-owned raw."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.boundary import CancellationQuery, RoundStatistics


class RawShots(Protocol):
    """Environment-owned deferred shot source, used by processing and getters.

    Implementations retain their noiseless inputs and round-local sampling
    identity for as long as this source is referenced, independently of later
    rounds, cache eviction or binding release. They must not borrow mutable
    backend buffers. ADC and white_noise implement this structurally; numeric
    packages do not import this protocol. Calls are synchronous and serialized
    by the acquiring caller, not required to be thread-safe.
    """

    def materialize(self, reps: int, avg_level: int) -> dict[int, NDArray[np.int64]]:
        """Own full per-shot ADC sums, without rerunning the backend.

        reps is the nonnegative AcquireRequest.reps. avg_level inserts the shots
        axis in the C-order point dimensions, in [0,len(point_dims)]. Return one
        int64 array per requested ADC, keyed by its channel index, with shape
        (*loop_dims,reads,2). Last axis is [I,Q]; reads follow trigger order.
        Values include the environment's physical signal, receiver noise and
        nearest-even rounding. There is no normalization or enforced equality
        to the independently sampled direct-mean totals.

        Every call owns fresh writable arrays, including for zero reps (empty
        shots axis). Equal identity, inputs, reps/layout and numeric/RNG versions
        reproduce identical values; calls never advance the mean stream. Wrong
        reps/layout, invalid data and int64 overflow raise ValueError. Numeric
        solver failures may raise RuntimeError. No partial mapping is returned.
        """
        ...


@runtime_checkable
class CancellableRawShots(RawShots, Protocol):
    """Optional cooperative cancellation for shot work required by this round."""

    def materialize_cancellable(
        self, reps: int, avg_level: int, *, cancel: CancellationQuery
    ) -> dict[int, NDArray[np.int64]]:
        """Follow materialize's layout and RNG contract, or raise Cancelled.

        Borrow the read-only query for this call. No partial arrays escape on
        cancellation; later materialize calls must not read this query.
        """
        ...


@dataclass(frozen=True)
class IntegratedRound:
    """One integrated delivery from an environment, not a second raw generator.

    statistics: pure core summary with arrays borrowed until SimulationSoc copies
        them. It contains no materializer, callback or shot cache.
    raw: environment-owned source with the independent lifetime defined by
        RawShots. The program retains it only for the last completed round, and
        calls it only when shot processing or a getter needs raw. The source is
        not executed during direct averaging. Binding mode validation requires
        this record for integrated mode and RoundSignals for decimated mode.
    """

    statistics: RoundStatistics
    raw: RawShots
