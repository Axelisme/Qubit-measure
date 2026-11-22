# FTF environment composition

**Last updated:** 2026-11-22, acquire-local raw plans

**讀者：** framework-dev、agent。FTF 環境的組裝接縫、數值接線與 acquisition 契約。

`ftf(configuration, *, soccfg, acquisition=None, workers=1, warmup=False, time_scale=None, history=None, families=None)`
builds the same `SimulationEnvironment` for GUI startup and Notebook callers. It creates fixed-unit V/A `FakeDevice` sources in one
`DeviceManager`, uses each source name as its explicit FakeDevice address,
applies initial values at environment construction and binds the caller's required
`soccfg`. The caller builds that board from independent YAML with `program.build_soccfg`.
The builder does not infer channel counts from wiring. `FTFSource` and
`plan_template` require the same acquisition profile for numeric PFB carrier selection. The returned manager owns cleanup. A
builder failure closes its resources and preserves both startup and close errors.
Startup, Notebook calls and caller-owned YAML setup belong to
[simulation environments](../../../../../zcu_lab/simulation_envs/README.md#ftf-startup-與-yaml-設定).
Field units and entry wiring belong to the [FTF field guide](../../../../../zcu_lab/simulation_envs/FTF/field-guide.md);
schema and validation belong to [FTF config](../config/README.md).

Pass `history=RoutingHistory()` from `zcu_tools.simulate.evolution` to retain
caller-controlled routing history across acquires and model rebuilds. The source
borrows it and does not clear it on close. After acquires drain, `history.clear()`
forces the next acquire to solve again without resetting sampling streams,
simulated time or model caches. `families={"t4", "dop853"}` restricts every
span, including projective continuations. Unknown names raise at construction;
an unavailable domain raises at acquire without bypassing guards or falling
back outside the whitelist. Omitting both keywords preserves default routing
and source-owned history. These controls are not physical YAML or program cfg.

## Contents

- [Round pacing](#round-pacing)
- [Cooperative Stop](#cooperative-stop)
- [Debug capture](#debug-capture)
- [Routing telemetry](#routing-telemetry)
- [Numeric source](#numeric-source)
  - [Acquisition clock and loss drift](#acquisition-clock-and-loss-drift)
  - [Workers and engine budget](#workers-and-engine-budget)
  - [Physical facts and conditional readout](#physical-facts-and-conditional-readout)
- [Startup preparation](#startup-preparation)
- [Acquisition and numeric composition](#acquisition-and-numeric-composition)
  - [Dimension selection](#dimension-selection)
  - [Preparation reuse](#preparation-reuse)
  - [Control conversion](#control-conversion)
  - [Flux pulses](#flux-pulses)
  - [Cavity drives](#cavity-drives)
  - [Readout and sampling](#readout-and-sampling)
  - [Known approximations](#known-approximations)
  - [Physical applicability](#physical-applicability)

## Round pacing

`numerics.time_scale` is a finite nonnegative physical-time multiplier in both
acquisition modes. It defaults to 1.0; 0.1 runs ten times faster and zero skips
waiting. Each round targets `time_scale * reps * sum(timeline duration + relax_delay)`,
with timeline us converted to seconds. Readout and pre/post delays are included.
Snapshot capture and numeric execution consume this budget; only the positive
remainder sleeps. Slow rounds and failures during numeric execution add no wait.
Raw materialization is outside the pacing budget. Stop interrupts pacing
at checkpoints at most 10 ms apart. Integrated acquisition discards that round;
decimated acquisition keeps the completed trace and stops before the next round.
Startup preparation, program construction and round hooks are outside this
budget. No GUI field is added; final plot-buffer flushing is unchanged.

Profiling overrides and the caller's board-loading example belong to
[environment setup](../../../../../zcu_lab/simulation_envs/README.md#關閉補等與-telemetry).

## Cooperative Stop

FTF reads the acquire's Stop query before advancing each projective outcome.
[Evolution](../../../simulate/evolution/README.md#accuracy-failures-and-lifecycle)
checks during geometry and WorkPlan construction and at execution boundaries.
When the current integrated round requires raw shots, [ADC](../../../simulate/adc/README.md)
checks while sampling and before handing the shots back. The
[program adapter](../../../program/v2/simulation/README.md#numeric-contract-and-extension-points)
handles publication and retains completed rounds on Stop.

[ADR-0078](../../../../../docs/adr/0078-evolution-module-boundary.md#ftf-stop-用合作式檢查點設計上-5-秒內)
sets five seconds as the design target between checks under normal parameters,
not a deadline or a gate. Waiting for startup warmup does not check Stop and
can take longer with empty caches. Cold raw sampling can also exceed the target
between the first event check and the first point-group check. A single-point
sample does not cover larger sweeps or more retained projection outcomes.

## Debug capture

`numerics.debug_capture` defaults to false. Enable it to publish one frozen
Engine-input and route manifest after each admitted acquire, including failure.
Every evolve in that acquire shares the same buffer. Telemetry can remain off.
Open the `capture-*.json.gz` manifest to read `inputs` for Engine inputs and per-row
routes. In `metadata`, use a round's `model_identity` to look up its settings
in `effective_settings`. Each attempted round retains its DC values, even if
execution failed after its snapshot. If no snapshot was taken, read
`admission_settings` instead. `sampling` shows the actual seed and sequence
when a round snapshot exists.

FTF owns `metadata.ftf.projective_spans`. Join an input record's Engine
`capture_id` to this map, then use its Engine point ID (a decimal-string key in
`histories`) to find the ordered projected states. Each entry gives the original
C-order acquisition `point`, absolute `span_ns`, and software `round_index`.
Use the capture ID rather than the Engine point ID alone: the same row identity
can appear in several spans and rounds. Recorded failed calls have the same
lookup. A reused preparation issues no new evolve and adds no entry.

When telemetry is enabled, use `capture_file` in `routing-*.jsonl` to match the
manifest to its acquire row. The row shows family totals, stage timings and
projection outcomes and pruned weights. Capture writes run before summary
construction and outside binding's measured wall time.
See the [shared switch contract](../../README.md#debug-capture) for empty
captures, retention and the cache directory.

## Routing telemetry

`numerics.telemetry.enabled` defaults to **True**. Each admitted acquire writes
one main JSON line, including all software rounds and partial work after a failure.
`numerics.telemetry.directory` defaults to `null`, selecting
`<project root>/.sim_telemetry/routing-YYYYMMDD.jsonl`. Dates and timestamps use
UTC. Relative explicit directories resolve at environment startup. The GUI and
experiment acquisition calls need no changes.

YAML switches and `settings.with_telemetry(...)` setup belong to
[environment configuration](../../../../../zcu_lab/simulation_envs/README.md#關閉補等與-telemetry).

Each main line contains `timestamp`, `points` (request grid, excluding shots),
`rounds` (captured attempts), `wall_s` (admitted acquire time including pacing,
excluding telemetry I/O), `frontend_cpu_s` (geometry and WorkPlan calling-thread
CPU), `estimated_cost_unit` and `families`. Failed calls
before a captured request have zero points and empty families.
Rows and hits sum across rounds. Noiseless reuse contributes logical rows and
cache hits but zero kernel CPU, rejections and estimated work. Actual CPU excludes
ADC, sampling and routing bookkeeping; wall includes those operations.
Numeric templates contain no program/experiment identifier, so `experiment` is
omitted rather than guessed from controls or caller frames.

The [shared telemetry contract](../../README.md#acquire-telemetry) lists the
persisted family fields and defines the common summary and directory settings.
`FTFSource.finish_acquire(outcome)` receives the binding's `AcquireOutcome` and
projects its builder-only `compile_wall_s`, in-acquire raw-shot
`materialize_wall_s` and `complete`.
`complete` means the acquire body and cleanup returned normally, including partial
Stop results. An escaping error is false.
`evolve_wall_s` sums each `Engine.evolve` call, including dimension pilots,
cavity-memory spans and projective outcome continuations across software rounds.
`readout_wall_s` sums cavity field propagation, population conversion, conditional
ADC preparation and integrated or decimated sampling. These scopes do not overlap
evolution. A failing call contributes time through its exception; a stage never
entered contributes `0.0`. Reusing a preparation adds no evolution wall, but still
counts fresh round sampling. Preparation and pacing remain in total wall's
remainder. If a later getter materializes retained raw and returns successfully,
the writer appends a [supplemental timing record](../../README.md#acquire-telemetry).
The supplement carries the original `acquire_id` and materialization time, not
`points` or `families`.
FTF records one dimension decision per acquire in `extras`, not per round.
See [Dimension selection](#dimension-selection) for the cases that skip the pilot.

| Field | Meaning |
| --- | --- |
| `ftf.selected_dimension` | Selected dimension N, or `null` if selection did not finish. |
| `ftf.dimension_reused` | `true` when N comes from an earlier acquire of the same content, even if this acquire needs to evolve again. Otherwise `false`. |
| `ftf.pilot_point` | Pilot timeline's zero-based index in the whole acquire grid flattened in C order, not its position in a template. Later rounds keep this index. It is `null` if no pilot completed. |
| `ftf.pilot_drop` | `drop(N)`, the largest magnitude of an upper-reference density entry outside the selected block. It is zero when a pilot selects the upper basis, or `null` if no pilot completed. |
| `ftf.pilot_error` | Maximum absolute density difference between the upper reference and the zero-padded verification, even if verification rejects the smaller basis. It is `null` if no verification completed. |
| `ftf.pilot_wall_s` | Pilot selection wall time in seconds, including elapsed pilot work on failure. It overlaps `evolve_wall_s` and is `0.0` when the pilot does not run. |
| `ftf.pilot_verify_price_skipped` | `true` when the estimate ruled out a smaller-basis check, including break-even. Otherwise `false`. |
| `ftf.pilot_candidate_dimension`, `ftf.pilot_candidate_drop` | Smaller basis k and `drop(k)` before verification. These describe the candidate even when N falls back to the upper basis. |
| `ftf.pilot_upper_dimension`, `ftf.pilot_upper_wall_s` | Upper basis size and measured wall seconds c for the isolated upper solve. c excludes locating the pilot and checking discarded density entries. |
| `ftf.pilot_timelines`, `ftf.pilot_workers`, `ftf.pilot_remaining_waves` | First-round timeline count T, configured host worker width W, and remaining waves R. |
| `ftf.pilot_estimated_verify_s`, `ftf.pilot_estimated_savings_s` | Estimated wall seconds for one smaller solve and the wall seconds it could save over the remaining waves. |

The estimate fields are `null` if no smaller candidate was priced and reported.
An upper-only candidate has a pilot point but no estimate.
A price skip has an estimate and no error. An accuracy rejection has an estimate
and a completed verification error above `1e-3`. A reused N carries no pilot estimate
from the earlier acquire.

Projection values live only in `extras`:
`ftf.projection_outcomes` counts retained outcome histories per C-order point in the
latest solved round. `ftf.projection_pruned_weight` gives each point's cumulative
removed probability before redistribution. Both arrays are empty for
terminal-only templates or when no solve completed. Reused preparations retain
these values. They do not count the semiclassical cavity main state.

The writer caps its `routing-*.jsonl` files at **50 MiB**. Before each append it
deletes oldest files by modification time, breaking ties by filename, until the
line fits. Today's file can be removed and recreated. Unrelated files remain.
A filesystem or serialization failure emits at most one warning per environment
and never aborts acquire, even with warnings configured as errors. Local files
are gitignored. `tools/analyze_routing_telemetry.py <directory or files...>` reports
wall stages and their remainder by normal return, escaping error or unknown
outcome, family usage shares, rejection counts, CPU per relative cost unit, and
weighted `(actual - estimated) / estimated` error where CPU estimates exist.
These statistics use main rows only.

## Numeric source

`FTFSource.snapshot` reads live DC values before each round. It checks units and
limits without restoring startup values. Source identities survive session
registry disconnects and name changes. Reconnection retrieves the original
class-retained driver, preserving V/A units and current values.
`execute` consumes detached templates.
`open_acquire` rejects nonempty `AcquireRequest.charge_requirements` with
`ValueError` before evolution. This environment does not yet support explicit
charge admission. Empty requirements preserve ordinary acquisition.

### Acquisition clock and loss drift

`FTFAcquisitionOptions(clock=None, truth_log=None)` selects
one environment clock and the private drift truth log. Pass it as `acquisition=` to
`ftf` or `FTFSource`, not as separate builder/source keywords. An omitted clock
creates one per environment. A caller-owned clock must not be shared between
independent environments. The builder shares it between acquisition admission and
completed-round duration accounting; no hardware clock is read or changed.

`SimulationClock` returns monotonic wall seconds plus a nondecreasing offset.
After each successful round it adds `max(0, physical_s - elapsed_s)` to the offset.
`physical_s` is unscaled; `elapsed_s` includes snapshot, execution and pacing sleep.
Failed or cancelled rounds add no credit. Even with `time_scale=0`, successful
rounds account for unscaled simulated physical time. Idle time between acquisitions
advances the wall-clock part naturally.

Optional `settings.drift` freezes independent f1, coupler and f2 log-OU losses
at `begin_acquire(seed=...)`. Root seed and loop key derive separate streams.
F1/f2 select q_cap, xqp and aphi; the coupler selects only q_cap and aphi.
Every selected chip loss must be positive. `sigma` is the stationary standard
deviation in ln(loss), with log mean ln(the configured chip loss). `tau_s` is the
positive correlation time in seconds. The first acquisition uses chip initial
values. Later begins transition by elapsed simulated seconds. All points and
rounds use the same effective settings, with `drift=None`. Omitted or null loops
and targets stay constant; q_ind never drifts. A seed alone does not fix the
wall-clock time sequence. The commented YAML example and configuration limits
are in the [Drift guide](../../../../../zcu_lab/simulation_envs/FTF/field-guide.md#drift跨-acquisition-損耗漂移).
Direct `open_acquire` callers need not call begin; open samples once when no
admission is pending. A new begin invalidates the previous session. Enabled drift
rebuilds physical facts and clears complete preparations each acquisition.
Conditional ADC solvers are request-local and need no source-level invalidation.
Omitted/null drift leaves sampling streams and numerical outputs unchanged.

Each enabled acquisition appends three loop rows to the private truth log.
`truth_log=None` selects `~/.local/state/zcu-tools/simulation/ftf/<session>/truth.jsonl`.
Each begin writes rows with `qubit` keys `f1`, `coupler`, `f2`, a session UUID,
simulated seconds, UTC wall time and effective q_cap/q_ind/xqp/aphi.
The coupler has JSON null q_ind/xqp. An explicit `truth_log=Path(...)` selects
another file; relative paths resolve from cwd. Enabled drift creates parent
directories and appends; callers own truth-log retention.
A new source or environment restart creates a new session starting at chip values.
Disabled drift writes nothing. Truth-log serialization and filesystem failures
propagate before request construction.
Routing telemetry remains a separate YAML-controlled policy. Keep truth logs
and debug captures out of ordinary Run, GUI and MCP metadata.

### Workers and engine budget

The segment engine's fast algorithms are the default numeric path.
One source owns one `Engine` with a 4 MiB routing-facts allowance.
Numerical preparations live within one evolve. The source's complete preparation
LRU remains separate from Engine and does not retain noisy outputs.
The keyword `workers` selects the engine's sole parallel width; direct callers
use one worker by default, including tests, benchmarks and research.
GUI startup reserves one host CPU, with at least one worker. The Notebook
reserves two CPUs. Both pass `warmup=True` to start CPU and GPU preparation.
The shared GUI worker policy is
`gui.app.measure.startup_context.simulation_workers(cpu_count)`, defined by
[ADR-0069](../../../../../docs/adr/0069-experiment-core-and-frontends.md#顯式-cfg-與裝置環境).
Startup instructions belong to
[FTF startup](../../../../../zcu_lab/simulation_envs/FTF/README.md#演化與除錯).
Evolution limits
native BLAS to one thread for the entire call. Overlapping evolves share a
process-level reference count: the first sets the limit and the last restores
it. Do not independently change threadpoolctl limits while an evolve is active.
Empty work, one timeline, and `workers=1` run serially without a joblib pool.
Other calls own a threading pool only during `evolve`; no idle pool survives.
`FTFSource.close()` clears preparations, remembered dimensions and the engine's routing facts.
It clears source-owned routing history but preserves caller-borrowed history.
Dropping the environment also releases this in-memory state; no disk cache is involved.

### Physical facts and conditional readout

The source retains at most one snapshot's `FTFModelFacts` across acquires.
The exact key includes full settings, the entire snapshot and backend/numeric
revisions. Facts own the joint projection, Gibbs/source, DC eigen/bath memo and
neutral readout model. All templates share these physical inputs.
Evolved densities remain acquire/template data. Each preparation constructs
`ConditionalReadout` for its current numeric model, receiver and tolerances.
The complete preparation LRU retains conditional sums, not the ADC solver.
Pulse/grid timing, carrier phases, demodulation and opaque callback identity
keep their existing meanings. Every round still samples IQ.
Settings, snapshot or numeric changes replace physical facts and invalidate
complete preparations. No projected operators or rates cross flux points.

## Startup preparation

Callers select startup work with `ftf(..., warmup=...)`:

- `False` (default) starts no background work.
- `True` prepares common CPU paths, then the OpenCL DOP853 program for the model's
  `retained_levels`. It skips the GPU item if pyopencl is absent or `families`
  excludes `dop853`. If detection finds no usable GPU, the item completes
  without a solve. Otherwise the disposable Engine runs a synthetic solve.
  Its automatic routing selects CPU for unsupported dimensions. If the solve
  uses OpenCL, preparation leaves the program in the disk cache.
- An explicit `EnvironmentWarmup` runs only the caller's items.

GUI and Notebook policy belongs to
[environment setup](../../../../../zcu_lab/simulation_envs/README.md#workers-與-startup-preparation).
Environments requesting identical items share a future. Within one process,
each CPU item runs once and each dimension gets at most one GPU warmup.
The GPU item uses a disposable Engine and fills the existing device-check and
program disk caches. Preparation never samples
or changes the live devices, binding seed, acquire sequence or resident Engine.
Startup overlaps compilation with other work; it does not reduce its total cost.

`env.warmup` is an `EnvironmentWarmup`, or None when disabled. Its immutable
`status` reports `state` (pending/running/ready/failed), `completed` item names,
`current`, `total` and an optional `error`. Poll it for progress.
`env.warmup.start()` returns the same non-cancellable future on every call;
callbacks receive the terminal status and must marshal GUI updates to the owner
thread. `env.wait_ready(timeout=None)` waits for completion and returns the terminal
status, or None when disabled. A timeout raises `TimeoutError` without cancelling
work. A CPU preparation error stops the manifest before the GPU item. An error
after successful GPU detection stops the GPU item. Both errors appear in the
failed status. Acquire still uses on-demand compilation or CPU fallback when
needed. An eligible acquire reports detection and build failures through routing
diagnostics.
The worker exits after completion. Call `env.close_backend()` after acquisitions drain to wait
for preparation and release the Engine. Device cleanup is separate; use
`closing_environment(env)` to release both. Closing devices alone does not wait
for independent numeric preparation.

Acquire automatically waits after binding admission and before consuming its RNG
sequence or constructing its request. Call `env.wait_ready()` **before starting a
caller-owned timer** to exclude startup work from that timer. A timer wrapped
around an immediately invoked blocking acquire necessarily includes its wait.
No frontend timer or status display is installed by this backend API.

The coverage manifest in `warmup.py` covers shaped carrier families (>9 carriers),
shaped direct integration, covariant single-tone (including restored-mask
coefficients) and simultaneous gate playback,
Dyson gain batches, constant-tone block RWA, sampled and dual-constant multitone
RWA, dense Floquet periods, t4, ADC integrated/decimated/mux responses and compact
statistics validation for writable and readonly borrowed arrays.
Multitone inputs exercise immutable model data and owned mutable Fourier/mask
working arrays, so admission coverage follows array signatures as well as routes.
It does not promise every possible signature or kernel. New routes require
updating the representative inputs and checking
real acquire compile events with an empty scratch `NUMBA_CACHE_DIR`.

## Acquisition and numeric composition

### Dimension selection

`numerics.truncation.retained_levels` sets the upper joint basis. Acquires with
fewer than eight timelines or an upper basis of at most twelve states use that
basis directly. A projective read or an early cavity pulse also uses the upper
basis: its continuation is not one terminal-only pilot evolution.

For an eligible acquire, FTF selects one pilot from the first round. It takes
the largest sum of absolute in-play drive areas scaled by the port matrix
elements. Ties take the largest carrier detuning from the active port's
strongest ground transition, then the first C-order point. The upper model
evolves that timeline once. FTF finds the smallest integer dimension of at least
twelve for which the largest magnitude of any discarded row or column entry is
at most `5e-4`. If that dimension k is smaller than the upper basis, FTF estimates
the cost of checking it. Let c be the isolated upper solve's wall time, r be
k divided by the upper basis size, and R be `ceil((T - 1) / W)` for T timelines
and configured worker width W. FTF uses linear scaling: the check costs `r * c`
and could save `R * (1 - r) * c` over the remaining timelines. FTF skips the check
and keeps the upper basis unless the expected savings strictly exceed the cost.
The decision does not require telemetry to be enabled.

If the check pays back, FTF evolves the pilot once at k. It accepts k only when
the zero-padded density differs from the upper reference by at most `1e-3` in
every entry. This checks accuracy on the pilot, not on untested sweep points.

The reduced model projects the initial state onto the energy prefix and
renormalizes it. The first successful selection fixes the batch dimension across
software rounds. Each round builds its model from that round's live DC snapshot.
In the pilot round, FTF uses the isolated upper solve for the pilot point and
solves the rest as one batch at the selected dimension. The smaller verification
never supplies output. Later rounds solve all points together at the selected
dimension. FTF pads smaller densities with zero rows and columns before ADC.
Isolated pilot solves do not add to the production source's routing history.

### Preparation reuse

`open_acquire(request)` starts a session for a fixed request. Its
`sample_round(snapshot, index)` reuses the preceding round's noiseless event
populations and conditional response when the snapshot and solve shape match.
The split pilot round does not supply a preparation for the next full batch.
Every round draws a new stream.
Opening another acquire invalidates the old session and drops its references.
Projection continuation densities exist only during span preparation. A later shot, round or acquire does not inherit a preceding
outcome's terminal state or sampled herald result. Immutable outcome weights,
event populations and conditional sums follow the complete-preparation reuse
rules below. See [ADR-0078](../../../../../docs/adr/0078-evolution-module-boundary.md#損耗漂移下的結構與-preparation-lifetime).
A source-owned 4 MiB LRU retains owned, read-only probabilities, conditional sums,
content mappings, path diagnostics and deferred raw inputs across acquires.
The acquire session owns the [ADC raw plan memo](../../../simulate/adc/README.md#integrated-raw-shots).
It retires that memo on batch invalidation, managed acquire exit, matching
direct finish, a new acquire or close. The binding's completion callback
retires the memo even when telemetry and debug capture are both disabled.
It runs finish diagnostics only if either is enabled. A preparation cache hit
in a new acquire starts without a raw plan.
A later acquire can reuse either a pilot round's split preparation or a full-batch
preparation. Both retain the selected N. A separate 256-entry LRU remembers N for
that same content after the preparation is evicted. On a preparation miss with a
remembered N, FTF skips the pilot and evolves the full batch at N.
The key still reads full current C1 content, layout, mode, reps and readout
values, including absolute phase origins. Physical settings, snapshot or numeric
changes, engine replacement, failed preparation and close clear both caches.
A later round at a changed snapshot keeps its session's N and can reuse that
round's preparation locally. A new acquire at that snapshot runs its own pilot. These caches never retain request seeds,
acquire sequences, noisy IQ or round results. Hits update
numeric-use identity and repeat report-only accuracy warnings through evolution's
`PathDiagnostics.warn_exceeded()`. Hits ask `Engine.reuse` to resolve the current
selection from immutable inventory and pricing facts. FTF neither inspects
cumulative source counters nor infers history sensitivity from count increments. An unchanged
selection returns the saved preparation without template conversion, geometry,
WorkPlan construction, numerical preparation, population conversion or cache put.
A changed route or evicted routing facts calls `Engine.evolve` and refreshes
populations and paths while retaining ADC channels. A prior N still applies, so
this refresh evolves the full batch without a pilot. The split and full shapes
register separate routing facts against the stable model source for their N. Content keys exclude
cumulative counts, so growing usage does not force a preparation-cache miss.
Evolution counts each distinct acquire content's complete row inventory once;
identical repeats do not invent additional numerical playback to amortize a gate.
Equal consecutive full-batch software rounds remain one preparation. The public
`source.preparation_cache` reports cumulative `builds`, `hits` and `resident_bytes`.
Its `max_bytes` can be adjusted between calls; zero disables preparation
residency without discarding the separate N records. The budget
counts Python containers/keys and whole ndarray allocations, including bases of
views. Active/caller references are separate from resident budgets.
Conditional ADC solvers are request-local. A separate source-owned
32 MiB `envelope_encoder` retains a domain-separated SHA256 fingerprint for each
resident immutable `PulseEnvelope`. The full binary content (types, dtypes,
shapes, sample times and complex values) is hashed once on a miss. Each outer key
occurrence hashes only a domain tag and the 32-byte fingerprint. Equal content
produces equal keys with or without an encoder; identities only select resident
reuse. These keys stay in process-local caches. The budget
includes the full retained envelope, numeric objects, fingerprints and Python
containers. It reports cumulative `builds`/`hits` and `resident_bytes`;
`max_bytes` is adjustable between calls and zero disables
residency. The capacity covers the full 101-length Gaussian host-scan working set,
not just a short scan. Borrowed request arrays are reread on every key lookup.
Generation changes still invalidate preparations; immutable fingerprints contain
no generation data. Eviction or disabled residency recomputes the complete
fingerprint without changing keys. Source close releases the fingerprints and
envelopes too. Every round still captures a live snapshot and samples the
original stream. The binding serializes calls;
direct author callers must do the same.

### Control conversion

`plan_template` accepts optional matching `FTFModelFacts` for shared physical
inputs; standalone calls build their own facts. `FTFSource.model_facts` supplies
the source-owned latest facts and rejects calls after close.
`plan_template` requires the originating request's `time_base` keyword.
It converts QICK ticks/MHz/degrees into ns/GHz/radians with point arrays.
`TimeBase.ns` owns exact nearest-2-ns rounding and converts complete integer ADC
sample grids. Preparation keys include the clock. It preserves signed gains, complex sampled envelopes and the common
carrier origin for evolution. Readout converts the program's common-origin
phase to ADC's pulse-local phase by subtracting
`2*pi*(f-f_ref)*pulse_start_ns`, with frequencies in GHz. When pulses are
present, the first terminal carrier sets the model reference and its phase
stays unchanged. The existing
`phase_reference_ticks` contribution, converted through `TimeBase.ns`, stays part of the common-origin phase.
ADC demodulation keeps its absolute-time grid and phase at shot time zero.
Each ADC trigger uses the physical frequency from its named config, including
static PFB configs.
An isolated addressed carrier's conditional IQ center is invariant under a
common pulse/grid time shift. Mux ADCs select incident carriers by the supplied board's PFB channel
before demodulation. The caller's board YAML determines Nyquist folding,
PFBv4 channel spacing, offset and passband. Changing its ADC clock ratios or
decimation changes carrier selection.
Different-channel carriers are excluded, including when only one carrier is
incident. If an observed event has no in-channel carrier, planning retains its
grids with empty pulses. Acquisition emits receiver background offset and noise
without a cavity response.
An addressed single carrier retains its previous output bits. Carriers
in the same channel still superpose, so their finite-window leakage can depend
on readout start. This ideal selection does not model FIR response, filter
transients or anti-alias filtering. Pulse supports use absolute nearest-2-ns rounding, ties even.
The model owner supplies a round-DC `ModelSource`. Charge and small-signal flux
controls evolve to the terminal pulse start, or to the receiver's first trigger
when the template has no active readout pulse. Each drive line selects full
charge n or the derivative of the existing
fixed-coordinate Hamiltonian at that round's DC point. Omitted operator means
charge. Charge calibration uses GHz/gain; flux calibration uses Phi0/gain and
multiplies the derivative in GHz/Phi0. Signed complex crosstalk uses
Re[c*z] = Re(c)*Re(z) - Im(c)*Im(z) for either operator.
Flux drives keep that linearized DC operator throughout the round, including
secular energy-basis rotations. They do not activate the shared engine's
large-amplitude flux controls, filtering or flux-tail paths. Version 2 settings
group physical values, named DC/drive lines, readout and numerics. Version 1 is
not accepted.

### Flux pulses

Optional `flux_lines.f1`, `flux_lines.coupler` and `flux_lines.f2` each bind one
generator to one loop. A pulse on that generator contributes signed gain times
`flux_phi0_per_gain` in Phi0. The caller's board profile must include those
generator channels. Line display names may repeat across loops. Pulses on
unbound generators still fail. With no flux lines, the existing fixed-DC batch
and microwave controls remain unchanged.

`plan_template` converts flux slots to `FluxTrack` inputs for the FTF numeric
owner's `schedule` and `flux_waveform`. Each source passes through its own
`delay_ns`, base `tau_ns` and signed `distortion_terms` before entering its loop.
Complex envelopes and carrier phases form the real lab-frame signal before
filtering. Nonzero flux carriers also select the continuous waveform.
The numeric owner samples continuous waveforms every 2 ns and interpolates
linearly. Pulse endpoints round independently to nearest 2 ns, ties even.
Delayed pulse endpoints, sampled-envelope knots and actual interpolated plateau
edges join cavity boundaries in `segment_boundaries`. Plateau edges follow the
sampled waveform even when a line delay falls between DAC nodes.
The snapshot still supplies the round's DC flux and Gibbs
state. Pulses never change the snapshot or rebuild the fixed-DC projection.

Bath levels round midpoint offsets from that DC to integer multiples of
1e-3 Phi0, ties even. Offsets smaller than 5e-4 Phi0 snap to DC. Equal adjacent
levels merge. Filtering retains tails until their conservative absolute sum is
below 5e-4 Phi0 per loop, then appends DC. Signed cancellation does not shorten
that bound. Evolution stops at the readout endpoint even if a tail continues.
The terminal ADC model always uses the snapshot DC, including when the pulse
has not settled. Sequences that need DC readout must allow settling first.

Ideal constant baseband platforms with zero tau and no distortion omit the
continuous waveform. Without microwaves they use `t4`, with sudden basis changes
between levels. Filtered or varying sampled envelopes provide a continuous
waveform. Without microwaves, continuous varying spans use `adiabatic_map`.
It preserves instantaneous eigenstate populations even when the row bath has
nonzero rates. Those rates affect coherence decay only; this kernel models no
interlevel transitions. The engine does not check the adiabatic condition.
Neglecting population relaxation on a ramp is reasonable only when the ramp is
much shorter than T1. A ramp near T1 underestimates relaxation on this route.
With microwaves, continuous flux rows use `flux_dop853`.
There is no separate sudden/adiabatic switch.

The Hamiltonian uses the existing exact flux expansion inside the retained
fixed-DC projection. Large excursions can leave that subspace. Users must check
local and retained-level convergence for their excursion and ramp; these routes
do not certify reset fidelity or nonadiabatic accuracy. Small-signal
`DriveLine.operator: flux` remains a separate fixed-DC derivative control.
The [fixed-DC flux benchmark](../../../../../docs/discussions/2026-10-28-ftf-fixed-dc-flux-benchmark.md)
provides bounds only, not certification. M16 is the largest successful run, not
a converged reference; M24's first point and the M32 reference fail phase/gauge
quadrature. The other five M24 candidates were not run. Neither the excursion
range nor device reset fidelity is certified.

#### Reset interpretation

A reset sequence ramps to a park flux, holds there, returns to the snapshot DC,
and starts terminal readout after the return. Disable drift when comparing hold
times. Compute park Gibbs weights from `JointModel.eigensystem(park_flux)` at the
configured temperature. Adiabatic return maps those energy-index populations
back to DC. It does not rethermalize them to the DC Gibbs state.

In the no-microwave sequence here, population relaxation occurs during a truly
constant park span routed to `t4`, not during the continuous `adiabatic_map`
ramps. In a two-state model, that constant hold approaches park equilibrium with
`T1 = 1 / (rates[0,1] + rates[1,0])`, where `rates = joint.rates(park_flux)`.
A larger park gap alone does not guarantee a shorter T1. Check the rates too.
A filtered waveform routed entirely through `adiabatic_map` is not a T1 reset.
A filter tail that has not formed a constant plateau does not guarantee reset.
Callers must check the actual routing and allow settling into a constant park
span, as well as settling back to DC before readout. The planner separates
constant plateaus at the sampled waveform's motion boundaries. A merged row
that still contains motion can remain `adiabatic_map` even if its quantized
level is constant.

`tests/simulators/ftf/environment/test_flux_reset.py` checks synthetic templates through
`plan_template` and `Engine.evolve`, observing the density at terminal readout
start. Long constant holds approach park Gibbs populations after adiabatic
return. The constant-envelope control instead returns a basis-rotated park
density through sudden steps. These checks verify the mapping semantics and
constant-hold relaxation, including line delays between waveform samples.
The short ramps do not establish physical adiabaticity. These small-model checks
also do not certify local/retained-level convergence, ADC reset fidelity, or
large-excursion device accuracy.

### Cavity drives

At each point, the latest active readout-channel block is terminal. The planner
classifies earlier blocks with ADC triggers as projective reads and earlier blocks
without owned windows as cavity drives. For pairing, the shared planner uses
ADC's `adc_to_device_time_ns` to subtract
[`Receiver.time_of_flight_ns`](../../../simulate/adc/README.md) from the window
endpoints. The ADC keeps the original grid for demodulation.
A block's pairing interval includes ringdown up to the next block or the point end.
The planner records each window's intersections. The last intersecting block
owns that window. Neither pairing nor sharing pulse history moves pulse emissions,
projections or ADC grids.
The timeline compiler still rejects overlapping windows for successive
timeline triggers on the same ADC
([ADR-0083](../../../../../docs/adr/0083-program-macros-and-timelines.md#timeline-只存結尾與-reference-分開)).

`FTFTemplatePlan.projective_reads` retains each point's ordered projection times,
incident pulses and bound triggers. The numeric batch includes projection starts
as segment boundaries. Planning does not project or evolve densities.
Projective block counts and each ADC's per-event grid counts must match across
points. Planning rejects a difference with the point and differing counts.
Before evolution or sampling, request preparation also compares all templates'
projective counts and each ADC's per-event grid counts. A difference raises
`ValueError` naming both templates and layouts, even when total reads match.
This check belongs to environment preparation, not the program frontend.

At each projective start, runtime replaces the DC density with rank-1 energy
outcomes and weights them by its normalized diagonal. It retains the complete
outcome history across later reads. Each projection discards the incoming
cavity field and starts every new outcome from vacuum, as ADC does at event
start. The projective pulse drives that outcome's cavity during continuation.
Projection uses all retained energy levels, not IQ-resolvability clusters.
ADC CTMC transitions during a herald affect its IQ only. They do not change the
outcome's later quantum evolution. This approximation excludes active reset and
measurement-induced state transitions.

`numerics.projection_weight_floor` defaults to `1e-9`. Runtime removes global
history weights below this value and renormalizes surviving weights proportionally.
Zero disables pruning, including zero-weight histories. Removing all histories
raises `ValueError` naming the point and floor. The schema accepts finite values
in `[0, 1)`. Telemetry records retained counts and removed probability.

`simulate.adc` solves the linear, state-conditioned cavity fields. The incident
amplitude uses the same `input_per_gain` conversion as terminal readout.
For state s and resonator r, the photon number is `n[s,r] = |alpha[s,r]|²`.
The Stark energy shift in GHz is `sum_r chi[s,r] * n[s,r]`, where chi is the
joint model's state-dependent cavity frequency minus its bare frequency.
Differences between state-conditioned outgoing fields cause photon-shot-noise
dephasing, including shared-line interference and independent internal losses.
These driven contributions are separate from residual thermal-photon dephasing.

FTF queries evolution's final spans, then advances density and cavity memory
sequentially. The first span and each projection start from vacuum. Otherwise,
all state-conditioned fields start from the previous main branch's final
field. The main branch has the largest population in the span-start DC density;
exact ties select the lowest energy. Each span supplies ADC's `initial_field`
and absolute `initial_time_ns`. Fields retain ringdown after pulse switch-off.
Selecting one
branch for a superposition is a semiclassical approximation, not a description
of quantum cavity-state entanglement.

FTF averages Stark energy shifts and driven dephasing over each span and passes
neutral `SegmentShift` values to evolution. Evolution propagates only that span
from the preceding output density, without normalizing or clipping it. Absolute
control coordinates, carrier phases and pulse-local envelopes keep their common
origin. Within a span, evolution uses the averages rather than instantaneous
fields. Templates with neither cavity drives nor projective reads retain their
original single-evolve path.

Projective continuation sends one point's ready histories through its current
absolute span in one `Engine.evolve` call. FTF derives a nonnegative 63-bit Engine
row ID from the SHA-256 hash of the original point and its ordered history. The
same pair keeps its ID across spans, rounds and preparations. Within one
preparation, FTF checks for collisions across all points and spans, including
points with identical numerical inputs. If two different pairs produce the same
ID, FTF raises `ValueError` naming the colliding pair, without merging the rows
or returning a partial result.

FTF does not check for collisions across preparations. Such a collision can
cause Engine to skip counting a batch's routing usage only if two different
pairs share an ID and the rest of their batches' Engine inputs match bit for bit.
A lower usage count can change the selected kernel. The same density accuracy
budget still applies.

The result returns to its history by ID before the next span starts. Engine row
order does not change the original order of histories used for sampling.

`FTFNumericUse.projective_spans` gives each solved span's row IDs and histories,
Engine path class membership, and Stark inversions. Aggregate path classes and
Stark warnings still refer to the original point. Each point counts once per
class across the acquire. If Engine rejects rows with `NoValidKernel`, FTF reports
the original point, absolute span, and rejected histories, with the Engine
exception as the cause. Other Engine exceptions retain their type and an FTF
note for the batch; cancellation is unchanged. See [ADR-0078](../../../../../docs/adr/0078-evolution-module-boundary.md#黑盒邊界是-simulateevolution)
and [Debug capture](#debug-capture).

Each cavity-driven preparation retains all span reuse keys. Cache hits check
every key through `Engine.reuse`. Projective continuation keys identify the
original point, ordered histories, initial densities and span shifts. A failed
or cancelled batch publishes no partial preparation.

One ADC `CavityAverageCache` serves each cavity span composition. FTF sets a
512 MiB budget per composition. Its g/e points share the owner; different bias
models do not. ADC's library default remains 64 MiB.
Both projective and non-projective paths call `CavityAverageCache.span`.
An exact key match reuses flat second moments and all-state endpoint fields.
FTF selects the main branch, builds each `SegmentShift`, and passes the selected
field to the next span. Projective continuation does these steps separately for
each history, without a second memo. Neither path caches shaped inputs. ADC owns
keys, arrays and admission. Composition clears all entries on return or
exception; Engine and the long-lived source do not retain the owner.

In the fixed projective prefixes, `chain/amp_rabi` and `chain/len_rabi` gain no
hits above 64 MiB. For `chain_length` at 2.2 µs, the 64, 128, 256 and 512 MiB
budgets give 5,686, 8,284, 14,392 and 19,346 exact hits in 29,921 spans. Each
doubling adds more than 10% of the 19,346 cross-history repeat opportunities.
At 512 MiB the memo admits every distinct key in the prefix, so a larger budget
adds no hits there. The peak memo holds 475,546,067 B. The process peak RSS is
385 MiB higher than at 64 MiB, the largest increase of the three prefixes.

A complete `mist/flux_dep` Run gives 20,200 exact hits and no declines at either
64 or 512 MiB. The largest owner holds 8,991,628 B in both runs. The observed
process peak RSS is 3.49 MiB lower at 512 MiB, within the 512 MiB increase limit
for sharing the budget across both FTF paths.

When a cavity-drive Stark shift reverses two adjacent state energies, FTF
continues with the same dispersive calculation and keeps the original state order.
At acquire completion, it emits one `UserWarning` per reversed pair, naming the
states, first point and segment, and total affected points and segments. Repeated
observations of the same segment across rounds or projection histories count once.

The warning gives the tightest upper limit on a common multiplier of the configured
cavity-drive gain magnitudes. It holds gain ratios and the observed cavity histories
fixed, including ringdown. Results above the limit are outside the dispersive
model's validity. The model has no measurement-induced state transitions and does
not model MIST onset.

### Readout and sampling

Readout uses the DC model and populations in its energy basis. Terminal-only
preparations call ADC's `normalize_populations` once after `batch_populations`,
before conditional responses or sampling. Projective preparations normalize the
diagonal at each projection and each retained terminal outcome. ADC owns the
1e-6 row-sum tolerance, clipping of all negative entries and row normalization.
Invalid rows fail without sampling. ADC raises `PopulationValidationError` with `rejected_rows` for the
reported validation rule. FTF catches only that type, retains its count in
`FTFNumericUse.rejected_rows`, then re-raises without sampling. Successful
preparations retain zero rejected rows. The source retains the latest
evolve diagnostics even when this preparation fails, including the engine's
report-only maximum trace error. Raw evolution densities remain unchanged.
Drive-line controls evolve up to the terminal pulse start, including earlier
control portions after a herald. With active pulses, the pulse path requires
exactly one terminal pulse per point, including mutually exclusive slots. If
its readout generator has no active pulse anywhere in the template, the bound
ADC's first trigger sets the evolution endpoint at each point. This path uses
the neutral 0 GHz model reference and retains the ADC grids without creating a
carrier. Missing terminal pulses at only some points still raise `ValueError`.

Mux readout permits 1..8 coherent incident tones with a common pulse start.
Each mux ADC observes only the shared-line carriers in its PFB channel through
its own demodulation, calibration, delay and sample grid. The first terminal
carrier before PFB selection sets the rotating reference for every ADC. Logical
events select independently when an ADC retunes between reads. Multiple grids
within one event must address the same PFB channel; planning raises `ValueError`
if they span different channels. The tones are not assigned to individual
resonators. Multiple terminal readout blocks are not supported.

Both an observed mux channel with no in-channel carrier and a receiver-only
pulse-path event retain their read columns. With no incident field, the ADC
emits only the receiver's configured offset and noise, without CTMC or cavity
propagation.
Constant readout envelope values
fold into amplitude with `envelope=None`, enabling the exact field solution.
Nonconstant samples become ADC `PiecewiseLinearEnvelope` inputs without changing
knots, interpolation, final-value hold, gain or phase. Conditional responses are
solved once per distinct content, not per shot.

`plan_template` passes `readout.adc.time_of_flight_ns` to the numeric receiver
and the ordered r1/r2 `mismatch_phase_rad` values to `ReadoutModel`. All default
to zero. Delay does not move evolution's endpoint, pulse support or ADC grids.
ADC owns the causal output shift, absolute carrier delay phase, per-resonator
scattering phase and conjugation to microwave I+iQ before calibration/noise.

`FTFReadoutConfig(pulse, mux=None)` is the detached author configuration below
settings. `pulse` is one `FTFReceiverConfig`; its generator and incident amplitude
calibration remain in `settings.readout.drive`. The optional
`FTFMuxReadoutConfig(gen_ch, input_per_gain, receivers)` binds a separate generator
and 1..8 unique receivers, disjoint from the pulse ADC. `readout_configuration`
adapts the single `readout.adc` object and optional `readout.mux` section without
changing calibrations. Omitting mux disables only that path. The removed
`readout.drive.kind` field raises a field-path error at YAML loading. Generator
and receiver indices are nonnegative integers with no fixed upper bound.
Active readout generators select the pulse or mux path. With neither active,
the planner uses the pulse path. Each readout window must use a receiver and
generator from that path. `FTFSource.open_acquire` rejects
pulse and mux generators anywhere in one request's templates and names both
indices. Standalone `plan_template` rejects mixing within its template too.
Separate acquires can select different paths in the same environment.
The caller supplies the board profile independently of model wiring. Runtime
uses that profile without inferring channel counts or selecting a mock profile;
only the configured pulse and mux readout paths route to ADCs, while drive and fast-flux lines are declared as DACs only.
`plan_template` and `FTFSource` accept an explicit `readout=` override. Plans expose
`readouts`, an ordered tuple of `FTFReadoutPlan(channel, batch)`, not a singleton.
Each batch retains the same chronological logical-event axis. At one point,
grids with the same ADC and owner share the union of their intersecting pulse
blocks, once per block, subject to mux PFB selection. An early member pulse
can leave a field in a later grid that intersects only the owner block. A
different ADC can have different pulse members for that event. The ADCs share
the event's CTMC path, not each other's pulse lists or noise.

A window entirely before the first block can use an existing event without
field grids for that ADC at that point. The planner places baseline grids where
the per-event grid counts can match across points. The baseline produces only
receiver offset and noise without adding a projection or solving a cavity field.
If the layout cannot represent that window, planning reports the ADC, trigger
and reason.

Empty grids mean the ADC does not observe the event or emit a read column. All ADCs may omit
terminal observations; the terminal event and evolution endpoint remain in the
plan. Each ADC must still observe at least one event. Projective plans retain
point-local event contents. Terminal-only plans deduplicate when every point
has just its owner block or every point has empty pulses. Earlier members or
mixed field and baseline membership require point-local rows. Integrated flat,
common-start mux batches use ADC's carrier-wise linear sums. Each carrier keeps
the guarded direct-sum or constant-field/full-exponential route. Delay, mismatch,
ringdown and per-receiver offset remain part of ADC's response contract. Shaped
mux batches use ADC affine-input convolution propagation for known numeric
envelopes. Opaque callbacks and unsupported timing retain the field solver. Flat common-start
decimated batches superpose vacuum-start carrier fields through the same line,
then calibrate once per receiver. Single-carrier fast paths and evolution are unchanged.

The receiver noise PSD is two-sided, in ADC sample units squared times ns.
Discrete sample covariance is PSD/dt_ns using the ADC output rate. Receiver gain
does not scale noise. Integrated acquisition returns `IntegratedRound` with pure
`RoundStatistics` and an ADC-owned RawShots source. Counts remain shared across
all ADCs/reads, with separate direct totals. ADC retains noiseless model/pulse/grid
inputs for delayed raw; no shot paths or noisy outputs enter retained preparations.
The raw source retains its round's physical inputs and sampling identity after
later rounds, preparation eviction and `FTFSource.close()`. A program getter
uses that source without capturing devices or rerunning evolution/readout
preparation. It cannot switch to another program's more recent round.
The same physical inputs feed ADC's required batch-based trace interface in
decimated mode. Each shot selects one complete projection history. Integrated
counts flatten its outcome and terminal-state axes. Herald conditional sums use
the corresponding history state; terminal sums use the terminal state. Each
logical event has its own CTMC path, shared across its carriers, ADCs and grids.
An ADC that omits a herald keeps the terminal's original logical event index. Decimated acquisition returns `RoundSignals`,
reordering point-major samples into QICK loop order. ADC owns integrated raw generation, noise, rounding and loop layout. Program
owns detached buffers, declaration order and acquisition averaging. All channels at the same shot
index share one sampled joint state; receiver noise is independent. There is no
extra tone axis: each declared ADC has its own raw-shot array. Output follows
program ADC declaration order, which is mux tone order, not sorted channel index.

### Known approximations

- **Vacuum at event start.** ADC readout propagates the event's member pulses
  from vacuum. It does not inherit residual field from earlier events. The
  cavity field carried between evolution spans does not initialize ADC readout.
- **One state per event before ADC jumps.** FTF initializes the whole window
  with the owner block's state, even for samples emitted before that block's
  projection. The projection still occurs at the pulse start. ADC can model
  jumps within an event, but does not switch states at the projection boundary.

The table compares noiseless I/Q sums on the same ADC grids and phases. It uses
ADC 0, the `ftf-demo/initial` R1 readout and the board and FTF settings in
[`tools/data/experiment_wall_time`](../../../../../tools/data/experiment_wall_time/).
The receiver flight time is 400 ns, and R1's photon decay rate is
`κ = 2π × 1.1 MHz`. Each comparison uses four fixed predecessor/owner histories
from the same two DC states, `joint-0` and `joint-2`, corresponding to F1 g and e
with the coupler and F2 in their ground states. These are conditional simulations
without qubit evolution, ADC jumps, sampled noise or rounding.

| Comparison | Worst history | ΔI | ΔQ | \|ΔIQ\| | Reference g/e distance | Single-shot noise RMS | \|ΔIQ\| / distance | \|ΔIQ\| / noise |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chain/t1`, second read, `post_delay = 0 µs` | g→g | +355.3 | −767.6 | 845.8 | 22,572 | 3,961 | 3.75% | 0.214 |
| `chain/t1`, second read, `post_delay = 5/κ ≈ 0.723 µs` | g→g | −23.5 | −65.4 | 69.5 | 21,168 | 3,961 | 0.328% | 0.0175 |
| `rpi_length`, first sweep point with the original fixed 0.11 µs margin, 62 windows crossing a projection | e→g, column 61 (zero-based) | −250.9 | +1,099.6 | 1,127.9 | 117.2 | 1,774 | 9.62× | 0.636 |

The signed differences are approximation minus reference. The table gives the
largest Euclidean I/Q difference in each comparison. Signal, difference, g/e
separation and noise all use ADC sample-sum units. The separation compares
owner g and e in the reference while fixing the predecessor. Noise is the
receiver-only RMS of the integrated I/Q vector, `sqrt(trace(N × PSD / dt_ns))`,
for `N` independent samples. It excludes state mixture, jumps and rounding.

The `chain/t1` comparison (R5) changes only the field at the start of the
second event. The approximation starts that event from vacuum. The reference
carries in the first read's complex field after free decay over 1.91 ns or
725.03 ns. Each 0.99 µs nominal window has 304 samples. Neither window crosses
a projection.

The `rpi_length` comparison (G3) changes the state at the projection in the
reference and uses the owner state throughout the window in the approximation.
Both sides use the same pulses and start from vacuum. The comparison uses the
original fixed 0.11 µs length margin, not the margin now seeded from the readout
module. Each 0.2 µs nominal crossing window has 25 samples before projection
and 36 at or after it. The
other two windows cross no projection and serve as controls. The ratio exceeds
one because the short post-projection segment gives a small g/e separation in
this fixture.

Neither comparison measures the contribution of block 62's pulse to column
63's I/Q when the terminal event shares the pulse union. The numbers do not
estimate hardware error or apply to other timings or calibrations.

### Physical applicability

The model assumes weak dispersive readout, independent weak-coupling Markov
secular baths and exponential echo-characteristic dephasing. Truncations and
numerical tolerances remain user choices. Direct conditional means include
multilevel relaxation. Direct-mean variance includes initial-state mixture,
receiver noise and the integrated rounding term.
Physical raw and traces additionally resolve continuous jump times and cavity
memory. They need only agree with direct means in expectation, not realization.
