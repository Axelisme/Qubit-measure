"""Immutable numeric acquisition boundary shared by frontend and environments.

Times are us, frequencies MHz, phases degrees, gain normalized DAC amplitude.
Execution is C-order over loop_dims (including reps); read order is trigger order.
Arrays in RoundSignals are float64 ADC sample units, last axis [I, Q]. Backend
sums are not normalized or rounded. No driver, Module or callback crosses here.
These frozen records describe the boundary; their constructors do not validate
field ranges, layouts or array contents. Frontend and adapter checks are separate.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

AcquisitionMode = Literal["integrated", "decimated"]


@dataclass(frozen=True)
class PulseEnvelope:
    """Owned immutable signed complex envelope, continuously linearly interpolated.

    Attributes:
        sample_us: Strictly increasing local times in us from pulse start. These
            are the actual waveform sample axis, not solver integration steps.
        values: Dimensionless complex DAC amplitudes at sample_us, preserving
            original scale and sign. No peak normalization is applied. Interpolate
            real and imaginary parts linearly; hold the final sample through pulse
            end and use zero outside the event. Gain is applied separately.
    """

    sample_us: tuple[float, ...]
    values: tuple[complex, ...]


@dataclass(frozen=True)
class PulseEvent:
    """One generator pulse in a single execution's numeric timeline.

    Attributes:
        gen_ch: Zero-based generator index in the bound board configuration.
        start_us: Pulse start in us from this execution's common time origin,
            also used by ReadoutWindow.trigger_us. Not a wall-clock timestamp.
        duration_us: Pulse duration in us, quantized on the generator clock by
            the frontend; independent of the ADC duration.
        frequency_mhz: Generator frequency in MHz after QICK DDS quantization,
            matched to the readout channel by the current frontend.
        phase_deg: Generator phase in degrees after QICK phase-word conversion.
            This record imposes no additional wrapping or range check.
        gain: Signed normalized DAC amplitude after gain-word quantization,
            relative to the generator's maximum gain word. Nominal full scale
            is -1 to 1; this record does not enforce that range.
        envelope: Owned immutable continuous complex amplitude, or None for
            unity throughout duration_us. Never includes gain or carrier phase.
        phase_reference_us: Common carrier phase origin in execution us. Phase at
            time t is phase_deg + 360 * frequency_mhz * (t-phase_reference_us).
            Pulses do not independently reset their oscillator phase.
        source: Originating Module name for diagnostics, detached from the Module.
        iterations: Enclosing semantic loop (name, zero-based index) coordinates.
    """

    gen_ch: int
    start_us: float
    duration_us: float
    frequency_mhz: float
    phase_deg: float
    gain: float
    envelope: PulseEnvelope | None = None
    phase_reference_us: float = 0.0
    source: str = ""
    iterations: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class ReadoutWindow:
    """One ADC's fixed sample grid and read triggers for each execution.

    Attributes:
        channel: Zero-based ADC/readout index in the bound board configuration.
        gen_ch: Zero-based generator index routed to this ADC. The current
            frontend uses the last PulseReadout route declared for this ADC.
        samples: Compiled integer sample count per trigger, shared by all reads
            on this ADC. It is a count, not a duration or an integration weight.
        sample_rate_mhz: ADC output sample rate in MHz, or samples per us.
            The sample period is 1 / sample_rate_mhz us.
        trigger_us: Read start times in us from the same per-execution origin
            as PulseEvent.start_us. Tuple order is trigger emission order,
            not a separate chronological sort. Read r uses trigger_us[r] and
            samples equally spaced samples on this ADC's grid.
        frequency_mhz: Demodulation frequencies in MHz, one per trigger. Empty
            for legacy shape-only windows; populated in every PointTimeline.
        phase_deg: Demodulation phases in degrees, one per trigger; empty for
            shape-only windows. The common phase origin is execution time zero.
        generator_channels: Generator index per trigger in the same order as
            trigger_us. Empty in shape-only records, whose gen_ch is the fallback.
            PointTimeline windows populate this even when all reads use one route.
            gen_ch retains the final route for the fixed ADC shape descriptor.
    """

    channel: int
    gen_ch: int
    samples: int
    sample_rate_mhz: float
    trigger_us: tuple[float, ...]
    frequency_mhz: tuple[float, ...] = ()
    phase_deg: tuple[float, ...] = ()
    generator_channels: tuple[int, ...] = ()

    @property
    def reads(self) -> int:
        """Return the number of reads per execution, len(trigger_us)."""
        return len(self.trigger_us)


@dataclass(frozen=True)
class PointTimeline:
    """Detached numeric controls for one execution with time origin zero.

    Attributes:
        coordinates: Quantized hardware-loop (name, zero-based index) pairs in
            outermost-first order, including reps. Control values in events are
            already rounded by the board's clocks/DDS; indices are not voltages.
        events: Generator events in semantic emission order, not time-sorted.
        readouts: ADC declaration order; each window retains trigger emission
            order and this execution's trigger/demod values.
        duration_us: End of the body in execution us, including blocking waits
            and pending channel activity but excluding inter-shot relax_delay.
    """

    coordinates: tuple[tuple[str, int], ...]
    events: tuple[PulseEvent, ...]
    readouts: tuple[ReadoutWindow, ...]
    duration_us: float


@dataclass(frozen=True)
class AcquireRequest:
    """Fixed compiled controls/layout and private sampling identity for one call.

    Attributes:
        mode: "integrated" requests unnormalized ADC sample sums; "decimated"
            requests individual ADC samples. Neither mode asks for averaging.
        loop_dims: Compiled loop lengths, including reps, ordered outermost to
            innermost. Executions flatten these axes in C order, with the last
            loop varying fastest. Their product is the control timeline count.
        avg_level: Zero-based axis of loop_dims that QICK averages for integrated
            acquisition, normally the reps axis. It does not request backend
            averaging. Decimated acquisition retains executions and averages
            only software rounds in the existing acquisition pipeline.
        readouts: One window per declared ADC, in QICK ADC declaration order,
            not sorted by channel number. Each window defines its read order.
        controls: One tuple of PulseEvent objects per flattened execution in
            loop_dims C order. Events retain frontend emission order and share
            that execution's time origin; identical to timelines[i].events.
        timelines: Detached per-execution controls, coordinates and ADC timing.
            Empty only for callers constructing legacy shape-only records.
            readouts above supplies fixed shapes; timelines carries dynamic
            triggers and demodulation values without changing those shapes.
        root_seed: Nonnegative integer entropy supplied by the environment
            author or generated once by the binding when seed is omitted.
        sequence: Zero-based, monotonically increasing acquire identity from
            the binding. An admitted failed or cancelled call consumes it;
            rejected reentry does not. It is not a software round index.
        rng_version: Binding-defined RNG/seed-derivation version string,
            currently "PCG64/SeedSequence-v1/numpy-<version>". White noise uses
            SeedSequence([root_seed, sequence, round_index]) with PCG64.

    root_seed, sequence and rng_version are private replay metadata, not normal
    GUI/MCP metadata. Replay also requires the round snapshot and backend version;
    equal seeds alone do not promise bit-exact results across library versions.
    """

    mode: AcquisitionMode
    loop_dims: tuple[int, ...]
    avg_level: int
    readouts: tuple[ReadoutWindow, ...]
    controls: tuple[tuple[PulseEvent, ...], ...]
    root_seed: int
    sequence: int
    rng_version: str
    timelines: tuple[PointTimeline, ...] = ()

    @property
    def executions(self) -> int:
        """Return len(controls), the number of flattened sequence executions.

        The frontend makes this equal to the product of loop_dims; this property
        counts timelines without checking that relationship.
        """
        return len(self.controls)


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Frozen numeric device values and resolvable model/calibration identity.

    Attributes:
        version: Environment-author-defined configuration/schema version string,
            for example "white_noise-v1". It is not the binding's backend or RNG
            version and has no shared format enforced by this record.
        model_identity: Environment-author-defined string identifying the model
            and calibration content for this round. For white noise it is
            "std_adc=<value>". It need not be a path, digest or public identifier.
        values: Tuple of (name, numeric value) pairs captured for this round.
            Names and native device units are defined by the environment author,
            not by this record. Empty means no numeric device inputs. Pair order
            is author-defined; the record does not sort or check unique names.

    The environment captures this snapshot immediately before round execution.
    Authors retain any referenced model/calibration content for replay; this
    record neither resolves references nor retains that content itself. All fields
    are private replay inputs, not normal GUI/MCP metadata or public model truth.
    """

    version: str
    model_identity: str
    values: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class RoundRequest:
    """Private numeric inputs for one synchronous backend round.

    Attributes:
        acquire: The fixed AcquireRequest shared by every round of this call,
            including controls, layout, mode and root/acquire sampling identity.
        snapshot: Environment inputs captured immediately before this round,
            after any successful preceding round hook. Later device changes do
            not alter the snapshot or this acquire's controls/layout.
        index: Zero-based software round index within this acquire, not an
            execution/shot index. White noise combines it with acquire.root_seed
            and acquire.sequence to select this round's random stream.
    """

    acquire: AcquireRequest
    snapshot: EnvironmentSnapshot
    index: int


@dataclass(frozen=True)
class ChannelSignals:
    """One ADC's unrounded float64 I/Q data from a joint round realization.

    Attributes:
        channel: Zero-based ADC/readout index matching a ReadoutWindow.channel
            in the request. It is not the position in the returned channel tuple.
        iq: Finite float64 array with last axis [I, Q]. Integrated mode uses
            shape (execution, read, 2) and unnormalized discrete ADC sample sums.
            Decimated mode uses (execution, read, sample, 2) and individual ADC
            sample units, with sample length equal to the window's samples.
            Executions follow loop_dims C order; reads follow trigger_us order;
            samples follow increasing time at 1 / sample_rate_mhz us intervals.
            No reps/rounds averaging or integer rounding has been applied.

    The backend owns iq and lends it until the adapter finishes copying the round.
    It must not mutate or reuse that storage during the borrow. A frozen record
    does not make the array read-only. The adapter checks shape and finiteness.
    """

    channel: int
    iq: NDArray[np.float64]


@dataclass(frozen=True)
class RoundSignals:
    """One joint realization for every requested ADC in a software round.

    Attributes:
        channels: One ChannelSignals per requested ADC. Tuple order is arbitrary;
            the adapter restores AcquireRequest.readouts order by channel index.
            All entries belong to the same round realization. ADCs may have
            different sample counts and are not padded to a common length.

    ChannelSignals defines the per-mode array layout, units and borrow lifetime.
    The backend lends arrays until the adapter copies them; repeated getters use
    the stored round buffers and do not execute the backend again. The adapter
    rejects missing, duplicate or extra ADCs, wrong shapes and nonfinite values.
    It rounds integrated sums to int64 with nearest-even rounding and rejects
    overflow. In decimated mode it copies the traces and sums those same samples
    for companion raw data. QICK owns normalization and averaging afterward.
    """

    channels: tuple[ChannelSignals, ...]
