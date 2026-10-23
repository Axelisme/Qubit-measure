# Six independent fluxonium cavities

**Last updated:** 2026-10-24, physical-time pacing

**Readers:** framework-dev, lab-author. Build and acquire six independent cavities through the normal simulation SoC.

`six_fluxonium(configuration)` accepts `SixFluxoniumSettings` or a YAML path and
returns `SimulationEnvironment`. The builder creates seven addressed FakeDevices
in A, `Q1_bias` through `Q6_bias` and `coil_bias`. Each local line affects only
its own circuit. The shared coil affects all six, with each circuit's own
current offsets and periods. There is no local-line crosstalk, joint Hilbert
space or fast-flux input. Construction does not evolve states, warm numeric
caches or open hardware.

Every round captures all seven native currents without changing the devices.
The six owner computes `(I_coil - coil_offset)/coil_period +
(I_local - offset)/period` for each circuit. It passes that fixed reduced flux
as the corresponding `SingleFluxoniumSource`'s `offset_phi0`, with zero
`flux_per_unit` and a single-value local-current snapshot. The single backend's
schema and physics stay unchanged. Its preparation keys include the resulting
settings and local snapshot, so changing either contribution refreshes physics.
The outer `EnvironmentSnapshot` retains all seven raw currents and the full
six-system settings identity. Bound drivers survive registry removal or rename;
no acquisition restores startup currents.

The host calls `close_backend()` after acquisitions drain, then closes its devices.
Backend close leaves all seven DC drivers untouched. Delivered raw sources retain their
noiseless ADC inputs after later acquisitions and backend close.

## Cache budgets

Top-level `numerics` sets residency caps in bytes for the whole environment.
The demo's `default.yaml` sets `prepare_cache_bytes` to 768 MiB,
`result_memo_bytes` to 64 MiB and `routing_cache_bytes` to 4 MiB.
Each of the six independent Engines receives one sixth of each total, rounded
down, giving each Engine 128 MiB for prepare. At most five bytes per total
remain unused.
Prepare and memo keys include the complete physical model identity and numeric
budget. Routing reuse also requires the same source instance.

`adc_cache_bytes` defaults to 8 MiB per source and is not divided.
The demo settings therefore allow at most 884 MiB of cache residency across six
sources, including estimated metadata. Zero disables the corresponding cache.
All fields accept nonnegative integers, with no fixed upper bound.
Oversized products still compute but do not remain resident. Cache eviction
changes repeated work, not physical inputs, results or random streams.
Models, active preparations, scratch and GUI/native allocations are outside
these caps. They are not a process RSS limit.
The single-fluxonium environment keeps its original per-source defaults.

## Routing and acquisition

Mux generator 0 binds ADC0..5. Microwave generators 1..6 control Q1..Q6.
Pulse generator 13 binds ADC6 for single-tone sweeps. Generators 7..12 are unbound.
At every point, each incident tone selects the nearest bare cavity by frequency.
Each ADC demodulation frequency selects the observed cavity. Distances tied within
1e-9 MHz raise `ValueError`; the tolerance covers floating subtraction only.

Routing follows frequency, not ADC index or tone declaration order. A pulse sweep
can cross all six cavity partitions. Multiple coherent tones assigned to one
cavity enter its existing single-cavity ADC model together. ADCs observing the
same cavity at the same point share state populations and jump histories.
Independent cavities draw separate streams. QICK retains ADC declaration order,
normalization, software-round averaging and companion sums from decimated traces.

A request uses either the pulse route or mux routes. All reads of one ADC at one
point must select the same cavity. Earlier cavity pulses, missing incident tones,
unbound controls and ambiguous frequency assignments fail before evolution.
The backend supports terminal readout only, as the single-model source does.
Pulse ADC6 uses Q1's receiver gain, offset, phase, time of flight and noise PSD.
Its physical cavity and incident-field calibration follow the frequency route.
Q1's seed selects the environment root stream. The largest per-qubit
`numerics.time_scale` multiplies the round's physical duration, computed from reps
and every point's timeline plus relaxation. All six default to 1.0 (physical
time); all must be zero to disable waiting. Stop interrupts pacing at 10 ms checkpoints.

## Bounded statistics

The backend partitions only routing patterns present in the request. It prepares
independent cavity populations and ADC responses on those point subsets.
Integrated mode preserves every sampled marginal count. Sorted cumulative-count
boundaries encode the common summary in at most `1 + sum(levels - 1)` bins.
For six 21-level circuits this is 121 bins, rather than `21**6` joint states.
This summary represents marginal moments, not a joint quantum state.
Raw and decimated draws use the ADC owner's physical samplers, not that histogram.
