"""Immutable numeric acquisition boundary shared by frontend and environments.

Times are us, frequencies MHz, phases degrees, gain normalized DAC amplitude.
Execution is C-order over loop_dims (including reps); read order is trigger order.
Arrays in RoundSignals are float64 ADC sample units, last axis [I, Q]. Backend
sums are not normalized or rounded. No driver, Module or callback crosses here.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

AcquisitionMode = Literal["integrated", "decimated"]


@dataclass(frozen=True)
class PulseEvent:
    gen_ch: int
    start_us: float
    duration_us: float
    frequency_mhz: float
    phase_deg: float
    gain: float


@dataclass(frozen=True)
class ReadoutWindow:
    channel: int
    gen_ch: int
    samples: int
    sample_rate_mhz: float
    trigger_us: tuple[float, ...]

    @property
    def reads(self) -> int:
        return len(self.trigger_us)


@dataclass(frozen=True)
class AcquireRequest:
    """Fixed compiled controls/layout and private sampling identity for one call.

    controls contains one timeline per execution. root_seed and sequence permit
    independent round replay; they are not normal GUI metadata. rng_version
    identifies the seed derivation/algorithm, including the NumPy version.
    """

    mode: AcquisitionMode
    loop_dims: tuple[int, ...]
    avg_level: int
    readouts: tuple[ReadoutWindow, ...]
    controls: tuple[tuple[PulseEvent, ...], ...]
    root_seed: int
    sequence: int
    rng_version: str

    @property
    def executions(self) -> int:
        return len(self.controls)


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Frozen numeric device values and resolvable model/calibration identity.

    Authors retain referenced model content with their environment. values uses
    author-defined names and native device units; version names the configuration
    schema. Snapshot capture belongs to the environment, not the numeric backend.
    """

    version: str
    model_identity: str
    values: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class RoundRequest:
    acquire: AcquireRequest
    snapshot: EnvironmentSnapshot
    index: int


@dataclass(frozen=True)
class ChannelSignals:
    channel: int
    iq: NDArray[np.float64]


@dataclass(frozen=True)
class RoundSignals:
    """One joint realization, all ADCs; arrays are borrowed until adapter copies.

    Integrated IQ has shape (execution, read, 2). Decimated IQ has shape
    (execution, read, sample, 2). Channels may have different sample counts.
    Missing/extra channels, wrong shapes and nonfinite values fail acquisition.
    """

    channels: tuple[ChannelSignals, ...]
