"""Independent Gaussian numeric environment; no program or device dependency."""

from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import (
    AcquireRequest,
    ChannelSignals,
    ChannelStatistics,
    EnvironmentSnapshot,
    RoundRequest,
    RoundSignals,
    RoundStatistics,
    SamplingIdentity,
)

from .rounding import shot_layout


@dataclass(frozen=True)
class WhiteNoiseRawShots:
    """Complete deferred Gaussian ADC sums for one round.

    std_adc is finite nonnegative per-sample standard deviation. acquire retains
    the immutable point/repetition/read/sample layout; identity is a local PCG64
    stream normally ending in 1. No totals or noisy samples are retained. Sources
    survive later rounds and return independently owned arrays on every call.
    Constructed by WhiteNoise.execute after model/request validation.
    """

    std_adc: float
    acquire: AcquireRequest
    identity: SamplingIdentity

    def materialize(self, reps: int, avg_level: int) -> dict[int, NDArray[np.int64]]:
        """Generate reproducible int64 (*loop_dims,reads,2) Gaussian shot sums.

        reps must equal acquire.reps and avg_level inserts shots among the C-order
        point dimensions. Invalid reps/layout or unsafe int64 sums raise
        ValueError. Noise, nearest-even rounding and layout are owned here; raw
        is not shifted to independently drawn round totals. Zero noise gives
        exact zero and zero reps gives empty shots. Repeated calls restart the
        identity, never changing global or average RNG streams.
        """
        if type(reps) is not int or reps != self.acquire.reps:
            raise ValueError("reps must match the raw preparation")
        point_dims = tuple(
            n
            for axis, n in enumerate(self.acquire.loop_dims)
            if axis != self.acquire.avg_level
        )
        if type(avg_level) is not int or not 0 <= avg_level <= len(point_dims):
            raise ValueError("avg_level is outside the point-axis layout")
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence((self.identity.seed, *self.identity.stream))
            )
        )
        raw = {}
        for window in self.acquire.readouts:
            values = rng.normal(
                0,
                self.std_adc * np.sqrt(window.samples),
                (prod(point_dims), reps, window.reads, 2),
            )
            raw[window.channel] = shot_layout(values, point_dims, reps, avg_level)
        return raw


@dataclass(frozen=True)
class WhiteNoise:
    """Independent Gaussian numeric source used by the white_noise environment.

    std_adc: finite nonnegative per-sample I/Q standard deviation in ADC units.
    Invalid amplitudes raise ValueError. The source owns no drivers or RNG state;
    each detached RoundRequest supplies its own reproducible stream identity.
    """

    std_adc: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.std_adc) or self.std_adc < 0:
            raise ValueError("std_adc must be finite and nonnegative")

    def snapshot(self) -> EnvironmentSnapshot:
        """Return detached numeric noise identity, without sampling or devices."""
        return EnvironmentSnapshot("white_noise-v1", f"std_adc={self.std_adc!r}")

    def execute(
        self, request: RoundRequest
    ) -> tuple[RoundStatistics, WhiteNoiseRawShots] | RoundSignals:
        """Return owned numeric data for a detached core RoundRequest.

        Integrated mode returns (RoundStatistics, WhiteNoiseRawShots). Totals
        have shape (*point_dims,reads,2), with point_dims taken from loop_dims
        minus avg_level. They include receiver and reps/12 rounding variance.
        The raw source is not called here and retains its layout independently
        of subsequent rounds. The composition root wraps the pair for program.

        Decimated mode returns RoundSignals with one float64 array per requested
        ADC, shaped (prod(loop_dims),reads,samples,2). Zero noise stays exactly
        zero in either mode. Request seed/sequence/index select local PCG64;
        raw uses an independent suffix-1 stream. NumPy rejects invalid seed or
        array dimensions. Neither input arrays nor global RNG state are changed.
        """
        if request.acquire.mode == "integrated":
            statistics = self._integrated_statistics(request)
            return statistics, WhiteNoiseRawShots(
                self.std_adc, request.acquire, statistics.raw_identity
            )
        acquire = request.acquire
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(
                    [acquire.root_seed, acquire.sequence, request.index]
                )
            )
        )
        channels = []
        for window in acquire.readouts:
            shape = (prod(acquire.loop_dims), window.reads, window.samples, 2)
            channels.append(
                ChannelSignals(window.channel, rng.normal(0, self.std_adc, shape))
            )
        return RoundSignals(tuple(channels))

    def _integrated_statistics(self, request: RoundRequest) -> RoundStatistics:
        acquire = request.acquire
        point_dims = tuple(
            n for axis, n in enumerate(acquire.loop_dims) if axis != acquire.avg_level
        )
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(
                    [acquire.root_seed, acquire.sequence, request.index, 0]
                )
            )
        )
        channels = []
        for window in acquire.readouts:
            variance = self.std_adc**2 * window.samples
            # Zero-noise environments promise exact zero, without rounding noise.
            total_variance = acquire.reps * (variance + 1 / 12) if self.std_adc else 0
            shape = (*point_dims, window.reads, 2)
            channels.append(
                ChannelStatistics(
                    window.channel,
                    rng.normal(0, np.sqrt(total_variance), shape),
                    np.zeros((*point_dims, window.reads, 1, 2)),
                    np.eye(2) * variance,
                )
            )
        return RoundStatistics(
            np.full((*point_dims, 1), acquire.reps, dtype=np.int64),
            tuple(channels),
            SamplingIdentity(acquire.root_seed, (acquire.sequence, request.index, 1)),
        )
