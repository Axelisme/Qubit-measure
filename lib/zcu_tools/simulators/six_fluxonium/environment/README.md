# Six independent fluxonium cavities

**Last updated:** 2026-11-17，per-point projection continuation and integrated categories

**Readers:** framework-dev, lab-author, agent. Build and acquire six independent cavities through the normal simulation SoC.

[ADR-0072](../../../../../docs/adr/0072-module-boundaries.md) defines model/framework ownership and physical charge admission.
[ADR-0078](../../../../../docs/adr/0078-evolution-module-boundary.md) governs routing history, preparation lifetimes and raw sampling ownership.
[ADR-0084](../../../../../docs/adr/0084-simulation-integer-time-base.md) defines integer request timing and physical-time conversion.

`six_fluxonium(configuration, *, soccfg, workers=1, clock=None, truth_log=None, history=None, families=None)` accepts `SixFluxoniumSettings` or a YAML path and
returns `SimulationEnvironment`. The builder creates seven addressed FakeDevices
in A, `Q1_bias` through `Q6_bias` and `coil_bias`. Each local line affects only
its own circuit. The shared coil affects all six, with each circuit's own
current offsets and periods. There is no local-line crosstalk, joint Hilbert
space or fast-flux input. Construction does not evolve states, warm numeric
caches or open hardware. Required `soccfg` comes from caller-owned board YAML
through `program.build_soccfg`. It supplies the mux and ordinary channel
capabilities independently of the physical model settings.

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

Pass `history=RoutingHistory()` from `zcu_tools.simulate.evolution` to lend one
history to all six model sources, including rebuilt sources. Closing the backend
does not clear borrowed history. With `history=None`, each source owns a separate
history. After acquires drain, `history.clear()` forces fresh solves without
resetting sampling streams, simulated time or model caches.
`families={"t4", "dop853"}` restricts every cavity's evolution spans, including
projective continuations. Unknown names raise at construction; an unavailable
domain raises at acquire without bypassing guards or falling back outside the
whitelist. Omitting both keywords preserves default routing. Neither control
belongs in physical YAML or program cfg.

`workers` selects each cavity Engine's parallel width across timelines and
keeps the direct-call default of 1. Values below 1 raise `ValueError` through
Engine. The six sources prepare one after another, not in parallel; worker pools
exist only during multi-timeline evolution. No sixfold worker pool runs at once.
GUI startup reserves one host CPU, with at least one worker, and does no warmup.
Tests, benchmarks and research keep the serial factory default. The startup
policy is defined in [ADR-0069](../../../../../docs/adr/0069-experiment-core-and-frontends.md#顯式-cfg-與裝置環境).

## Cache budgets

Top-level `numerics.routing_cache_bytes` sets the environment-wide routing-facts
allowance, required in YAML. The example uses 4 MiB. Each Engine receives one sixth, rounded down, with at
most five bytes unused. Zero disables routing retention. The value is a
nonnegative integer without a fixed upper bound. Routing reuse also requires
the same source instance.

Engine numerical preparations and ADC solvers are local to their current work.
Single sources retain loss-independent structure and their complete preparation
under the [single-source lifetime contract](../../single_fluxonium/environment/README.md#reuse-and-lifetime).
Models, complete preparations, scratch and GUI/native allocations are outside
the routing cap. It is not a process RSS limit.

## Routing and acquisition

The YAML mux generator binds the six configured mux ADCs. Each qubit's
microwave generator controls only that qubit. The YAML pulse generator binds
its separate pulse ADC for single-tone sweeps. All other channels stay unbound.
At every point, each incident tone selects the nearest bare cavity by frequency.
Each ADC demodulation frequency selects the observed cavity. Distances tied within
1e-9 MHz raise `ValueError`; the tolerance covers floating subtraction only.

Routing follows frequency, not ADC index or tone declaration order. A pulse sweep
can cross all six cavity partitions. Multiple coherent tones assigned to one
cavity enter its existing single-cavity ADC model together. ADCs observing the
same cavity at the same point share state populations and jump histories.
Independent cavities draw separate streams. QICK retains ADC declaration order,
normalization, software-round averaging and companion sums from decimated traces.

Nonempty `AcquireRequest.charge_requirements` raises `ValueError` during routing,
before projection or evolution. This environment does not yet support explicit
charge admission; it does not discard requirements when projecting local requests.
Empty requirements preserve the existing routing and acquisition behavior.

A request uses either the pulse route or mux routes. All reads of one ADC at one
point must select the same cavity. Untriggered cavity drives, missing terminal
tones, unbound controls and ambiguous frequency assignments fail before evolution.
Triggered nonterminal blocks follow their tones into each cavity's complete
observation set. Single's public `preflight` owns classification and validates
every ADC/template layout. Six completes preflight for all routed sources
before any density evolution or sampling. A source requires equal projective
counts and per-ADC/event grid counts across its templates. Conflicts name the
ADC, templates and layouts. Different sources may have different topologies.

Each cavity prepares one complete projection history through single's public
`prepare_readouts`. An ADC may omit herald or terminal observations without
losing their logical event identity. ADCs observing the same event share its
CTMC path. Independent cavities keep separate histories and sampling streams.
Projection uses rank-1 DC energy states, not groups inferred from IQ separation.
ADC jumps during herald affect IQ but do not feed back into later evolution.
This does not add MIST or data-dependent feedback for active reset. Trigger
pairing and nonterminal sample-window rejection follow the shared
[readout planner](../../../experiment/README.md#explicit-simulation-environments).
Each qubit's `numerics.projection_prune_floor` controls continuation at each point.
It requires a finite `0 <= floor < 1`; the example supplies `1e-9`.
A source state below the floor at a point uses an identity transition row there,
approximating it as staying in that state. Zero gives the exact chain.
No probability is discarded or renormalized. Path total variation from the exact
chain is at most the sum of that point's `uncontinued_probability` across
transitions, reported by single's projection diagnostics.
The pulse ADC uses Q1's receiver gain, offset, phase, time of flight and noise PSD.
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
independent cavity histories and ADC responses on those point subsets.
Integrated mode represents each cavity with a marginal category axis whose
counts sum to reps. Categories need not represent physical shot paths.
The ADC owner's `cumulative_count_bins` preserves each sampled marginal count
on shared sorted cumulative-count boundaries in at most
`1 + sum(axis_sizes - 1)` bins. Each axis size is the maximum category count
width for that cavity, with its retained level count as the minimum.
Without projections, six 21-level circuits still use 121 bins, rather than
`21**6` joint states. This summary represents marginal moments,
not a cross-cavity joint quantum state or a physical joint-shot sample.
Raw retains the complete independent sources through `IndependentRawShots`.
Raw and decimated draws use the ADC owner's physical samplers, not that histogram.
Terminal-only raw, integrated and decimated numeric outputs remain unchanged.
