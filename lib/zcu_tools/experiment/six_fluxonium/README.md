# Six independent fluxonium cavities

**Last updated:** 2026-10-27 (UTC), routing-only residency allowance

**Readers:** framework-dev, lab-author, agent. Build and acquire six independent cavities through the normal simulation SoC.

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

Top-level `numerics.routing_cache_bytes` sets the environment-wide routing-facts
allowance, default 4 MiB. Each Engine receives one sixth, rounded down, with at
most five bytes unused. Zero disables routing retention. The value is a
nonnegative integer without a fixed upper bound. Routing reuse also requires
the same source instance.

Engine numerical preparations and ADC solvers are local to their current work.
Single sources retain loss-independent structure and their complete preparation
under the [single-source lifetime contract](../single_fluxonium/README.md#reuse-and-lifetime).
Models, complete preparations, scratch and GUI/native allocations are outside
the routing cap. It is not a process RSS limit.

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

## Independent coherence drift

Each `qubits.Q1..Q6.drift` section accepts the single-fluxonium log-OU settings.
For example, `qubits.Q2.drift.q_cap: {sigma: 0.15, tau_s: 3600.0}` varies only
Q2's capacitive quality factor. Sigma is the stationary standard deviation of
ln(loss); tau_s is correlation time in seconds. Selected YAML losses must be
positive. Omit a parameter to keep it constant, or omit/null the section to
disable that qubit's drift. Drift-off preserves the existing sampling streams.

The environment derives separate drift streams from its Q1 root seed and stable
Q1..Q6 identities, regardless of YAML mapping order or ADC routing. Per-qubit
seeds do not override that root. Each environment starts at the YAML losses.
Admission samples every enabled qubit at the same simulated time, before request
construction. All points, routed receivers and software rounds then use that
acquisition's frozen losses. The public `SixFluxoniumSource.sources` tuple follows
Q1..Q6 order and exposes the effective settings sent to each physical source.

One `SimulationClock` serves the environment and all six sources. Only the outer
binding credits a completed round's unscaled physical duration, once per round.
The credit is `max(0, physical seconds - elapsed wall seconds)`, including pacing
in elapsed time. Idle wall time advances naturally. Zero time scale skips waiting
without losing physical time. Cancelled or failed rounds add no credit.
Builder `clock=SimulationClock(monotonic=...)` injects wall seconds for offline use.

Enabled qubits append one truth row each per acquisition to
`~/.local/state/zcu-tools/simulation/six-fluxonium/<session>/truth.jsonl`.
Builder `truth_log=Path(...)` selects another file. Rows include the shared session,
`qubit` identity, `sim_now` in seconds, UTC `wall_time`, and effective losses.
Truth I/O errors propagate at admission. No truth enters Run, GUI, MCP or Labber
metadata. Drift-off writes nothing. A new environment resets its clock offset and
drift state. Equal roots and acquisition time sequences reconstruct each qubit's
trajectory; six-fluxonium does not provide private acquisition replay.
Single-fluxonium's pulse-only private replay contract is unchanged.

## Bounded statistics

The backend partitions only routing patterns present in the request. It prepares
independent cavity populations and ADC responses on those point subsets.
Integrated mode preserves every sampled marginal count. Sorted cumulative-count
boundaries encode the common summary in at most `1 + sum(levels - 1)` bins.
For six 21-level circuits this is 121 bins, rather than `21**6` joint states.
This summary represents marginal moments, not a joint quantum state.
Raw and decimated draws use the ADC owner's physical samplers, not that histogram.
