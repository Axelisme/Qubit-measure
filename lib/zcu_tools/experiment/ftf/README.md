# FTF environment composition

**Last updated:** 2026-10-31, mux PFB channel selection

**讀者：** lab-author、framework-dev、agent。FTF 環境的組裝、數值接線與 acquisition 設定。

`ftf(configuration, *, acquisition=None, workers=1, warmup=False, time_scale=None)`
builds the same `SimulationEnvironment` for GUI startup and Notebook callers. It creates fixed-unit V/A `FakeDevice` sources in one
`DeviceManager`, uses each source name as its explicit FakeDevice address,
applies initial values at environment construction and binds the derived channel
counts to the fixed mock QICK profile. The returned manager owns cleanup. A
builder failure closes its resources and preserves both startup and close errors.
The GUI example is `zcu_lab/simulation_envs/FTF/startup.py`, which calls `ftf(...)`
with adjacent `ftf.yaml`, a working copy of the tracked `default.yaml`.
The Notebook walkthrough is `notebook_md/single_qubit.md`; it calls the same public
builder with an explicit YAML path. Neither caller uses an environment wrapper.
欄位用途、單位與 entry 接線請讀 [FTF YAML v2 欄位指南](../../../../zcu_lab/simulation_envs/FTF/field-guide.md)。

## Contents

- [Round pacing](#round-pacing)
- [Routing telemetry](#routing-telemetry)
- [Numeric source](#numeric-source)
  - [Acquisition clock and loss drift](#acquisition-clock-and-loss-drift)
  - [Workers and engine budget](#workers-and-engine-budget)
  - [Physical facts and conditional readout](#physical-facts-and-conditional-readout)
- [Startup preparation](#startup-preparation)
- [Acquisition and numeric composition](#acquisition-and-numeric-composition)
  - [Preparation reuse](#preparation-reuse)
  - [Control conversion](#control-conversion)
  - [Flux pulses](#flux-pulses)
  - [Cavity drives](#cavity-drives)
  - [Readout and sampling](#readout-and-sampling)
  - [Physical applicability](#physical-applicability)
  - [Private replay](#private-replay)

## Round pacing

`numerics.time_scale` is a finite nonnegative physical-time multiplier in both
acquisition modes. It defaults to 1.0; 0.1 runs ten times faster and zero skips
waiting. Each round targets `time_scale * reps * sum(timeline duration + relax_delay)`,
with timeline us converted to seconds. Readout and pre/post delays are included.
Snapshot capture and numeric execution consume this budget; only the positive
remainder sleeps. Slow rounds and failures add no wait. Stop interrupts pacing
at checkpoints at most 10 ms apart. Integrated acquisition discards that round;
decimated acquisition keeps the completed trace and stops before the next round.
Startup preparation, program construction and round hooks are outside this
budget. No GUI field is added; final plot-buffer flushing is unchanged.

For profiling or benchmarking, call `ftf(path_or_settings, time_scale=0)`.
This public override disables waiting without changing numeric
settings, RNG streams, density, readout or routes. Alternatively set
`numerics.time_scale: 0` in the caller-owned YAML. With the default
`time_scale=None`, the builder uses the YAML value. Wait for startup
preparation with `env.wait_ready()` before starting a benchmark timer too.

## Routing telemetry

`numerics.telemetry.enabled` defaults to **True**. Each admitted acquire writes
one JSON line, including all software rounds and partial work after a failure.
`numerics.telemetry.directory` defaults to `null`, selecting
`<project root>/.sim_telemetry/routing-YYYYMMDD.jsonl`. Dates and timestamps use
UTC. Relative explicit directories resolve at environment startup. The GUI and
experiment acquisition calls need no changes.

Use `ftf(settings.with_telemetry(enabled=False), time_scale=0)` for
profiles and benchmarks. The public settings method returns a copy, disabling
routing timers and all telemetry I/O without changing the original settings.
No directory is created. `settings.with_telemetry(enabled=True, directory=Path(...))`
selects another directory; omitted directory preserves the current setting.
The YAML form is `numerics: {telemetry: {enabled: false}}`. Routing and
density/readout results are unchanged bitwise.

Each line contains `timestamp`, `points` (request grid, excluding shots),
`rounds` (captured attempts), `wall_s` (admitted acquire time including pacing,
excluding telemetry I/O), `frontend_cpu_s` (geometry and WorkPlan calling-thread
CPU), `estimated_cost_unit` and `families`. Failed calls
before a captured request have zero points and empty families. Family values are
`rows`, `prepare_cpu_s`, `apply_cpu_s`, `actual_cpu_s`, `cache_hits`,
`rejected_rows`, `prepare_builds`, `admission_rows`, `estimated_cost` and nullable
`estimated_cpu_s`; common-frame instruments builds and admission evaluations,
while other families currently leave those two counters zero; see the
[evolution report contract](../../simulate/evolution/README.md#routing-telemetry).
Rows and hits sum across rounds. Noiseless reuse contributes logical rows and
cache hits but zero kernel CPU, rejections and estimated work. Actual CPU excludes
ADC, sampling and routing bookkeeping; wall includes those operations.
Numeric templates contain no program/experiment identifier, so `experiment` is
omitted rather than guessed from controls or caller frames.

`projection_outcomes` counts retained outcome histories per C-order point in the
latest solved round. `projection_pruned_weight` gives each point's cumulative
removed probability before redistribution. Both arrays are empty for
terminal-only templates or when no solve completed. Reused preparations retain
these values. They do not count the semiclassical cavity main state.

The writer caps its `routing-*.jsonl` files at **50 MiB**. Before each append it
deletes oldest files by modification time, breaking ties by filename, until the
line fits. Today's file can be removed and recreated. Unrelated files remain.
A filesystem or serialization failure emits at most one warning per environment
and never aborts acquire, even with warnings configured as errors. Local files
are gitignored. `tools/analyze_routing_telemetry.py <directory or files...>` reports
family usage shares, rejection counts, CPU per relative cost unit, and weighted
`(actual - estimated) / estimated` error where CPU estimates exist.

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

`FTFAcquisitionOptions(clock=None, truth_log=None, private_replay=None)` selects
one environment clock and private persistence paths. Pass it as `acquisition=` to
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
are in the [Drift guide](../../../../zcu_lab/simulation_envs/FTF/field-guide.md#drift跨-acquisition-損耗漂移).
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
directories and appends; callers own retention for both truth and replay paths.
A new source or environment restart creates a new session starting at chip values.
Disabled drift writes nothing. Truth-log serialization and filesystem failures
propagate before request construction.
Routing telemetry remains a separate YAML-controlled policy. Truth and replay
files must not enter ordinary Run, GUI or MCP metadata.

### Workers and engine budget

The segment engine's fast algorithms are the default numeric path.
One source owns one `Engine` with a 4 MiB routing-facts allowance.
Numerical preparations live within one evolve. The source's complete preparation
LRU remains separate from Engine and does not retain noisy outputs.
The keyword `workers` selects the engine's sole parallel width; direct callers
use one worker by default. GUI startup explicitly uses `workers=1`.
The Notebook example selects `max(1, (os.cpu_count() or 1) - 2)` workers,
reserving two host CPUs. Unknown CPU counts use one worker.
An eight-core host uses six Notebook workers. Evolution limits
native BLAS to one thread for the entire call. Overlapping evolves share a
process-level reference count: the first sets the limit and the last restores
it. Do not independently change threadpoolctl limits while an evolve is active.
Empty work, one timeline, and `workers=1` run serially without a joblib pool.
Other calls own a threading pool only during `evolve`; no idle pool survives.
`FTFSource.close()` clears preparations, model-source cumulative gate usage and the engine's routing facts. Dropping the
environment also releases this in-memory state; no disk cache is involved.

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

The Notebook example opts into `warmup=True`. Building that environment
immediately starts one process-wide background numeric worker. All opted-in
environments in the same process share its future and terminal result.
GUI startup explicitly uses `warmup=False`, so it starts no numeric preparation.
Direct `ftf()` callers also default to `warmup=False` and `workers=1` to avoid
unexpected CPU work. Authors choose the policy at each call.
Preparation uses small disposable numeric models, one evolution worker and no
sampling. It never reads or changes the live environment's devices, binding seed,
acquire sequence or resident engine cache. Compilation still consumes CPU and
memory; it overlaps configuration work rather than reducing total compile cost.

`env.warmup` is an `EnvironmentWarmup`, or None when disabled. Its immutable
`status` reports `state` (pending/running/ready/failed), `completed` item names,
`current`, `total` and an optional `error`. Poll it for progress.
`env.warmup.start()` returns the same non-cancellable future on every call;
callbacks receive the terminal status and must marshal GUI updates to the owner
thread. `env.wait_ready(timeout=None)` waits for completion and returns the terminal
status, or None when disabled. A timeout raises `TimeoutError` without cancelling
work. Failure is logged, retained in status and permits ordinary on-demand
compilation. The worker exits after completion. Process exit waits for an active
worker rather than interrupting the compiler. Closing live devices does not cancel
this independent numeric work.

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

### Preparation reuse

`open_acquire(request)` starts a session for a fixed request. Its
`sample_round(snapshot, index)` reuses the preceding round's noiseless event
populations and conditional response when the snapshot is equal. Every round
draws a new stream.
Opening another acquire invalidates the old session and drops its references.
Projection continuation densities exist only during span preparation. A later shot, round or acquire does not inherit a preceding
outcome's terminal state or sampled herald result. Immutable outcome weights,
event populations and conditional sums follow the complete-preparation reuse
rules below. See [ADR-0078](../../../../docs/adr/0078-evolution-module-boundary.md#損耗漂移下的結構與-preparation-lifetime).
A source-owned 4 MiB LRU retains owned, read-only probabilities, conditional sums,
content mappings, path diagnostics and integrated sampling plans across acquires.
Its key reads full current C1 content, layout, mode, reps and readout values,
including absolute phase origins. Physical settings/snapshot/numeric changes,
engine replacement, failed preparation and close clear the preparations. It never
retains request seeds, acquire sequences, noisy IQ or round results. Hits update
numeric-use identity and repeat report-only accuracy warnings through evolution's
`PathDiagnostics.warn_exceeded()`. Hits ask `Engine.reuse` to resolve the current
selection from immutable inventory and pricing facts. FTF neither inspects
cumulative source counters nor infers history sensitivity from count increments. An unchanged
selection returns the saved preparation without template conversion, geometry,
WorkPlan construction, numerical preparation, population conversion or cache put.
A changed selection or evicted routing facts calls `Engine.evolve` and refreshes
populations and paths while retaining ADC channels. Content keys exclude
cumulative counts, so growing usage does not force a preparation-cache miss.
Evolution counts each distinct acquire content's complete row inventory once;
identical repeats do not invent additional numerical playback to amortize a gate.
Other hits return the saved preparation without evolving. Equal consecutive
software rounds remain one preparation. The public
`source.preparation_cache` reports cumulative `builds`, `hits` and `resident_bytes`.
Its `max_bytes` can be adjusted between calls; zero disables residency. The budget
counts Python containers/keys and whole ndarray allocations, including bases of
views. Active/caller references are separate from resident budgets.
Conditional ADC solvers are request-local. A separate source-owned
32 MiB `envelope_encoder` retains a domain-separated SHA256 fingerprint for each
resident immutable `PulseEnvelope`. The full binary content (types, dtypes,
shapes, sample times and complex values) is hashed once on a miss. Each outer key
occurrence hashes only a domain tag and the 32-byte fingerprint. Equal content
produces equal keys with or without an encoder; identities only select resident
reuse. These keys stay in process-local caches, not replay or disk. The budget
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
`2*pi*(f-f_ref)*pulse_start_ns`, with frequencies in GHz. The model reference
is the first carrier; that carrier's phase stays unchanged. The existing
`phase_reference_ticks` contribution, converted through `TimeBase.ns`, stays part of the common-origin phase.
ADC demodulation keeps its absolute-time grid. An isolated addressed carrier's
conditional IQ center is therefore invariant under a common pulse/grid time
shift. Mux ADCs select incident carriers by the fixed profile's PFB channel
before demodulation. The profile determines Nyquist folding, channel spacing and
offset; its PFBv4 channels are 38.4 MHz apart with a ±19.2 MHz passband.
Different-channel carriers are excluded, including when only one carrier is
incident. An addressed single carrier retains its previous output bits. Carriers
in the same channel still superpose, so their finite-window leakage can depend
on readout start. This ideal selection does not model FIR response, filter
transients or anti-alias filtering. Pulse supports use absolute nearest-2-ns rounding, ties even.
The model owner supplies a round-DC `ModelSource`. Charge and small-signal flux
controls evolve every template to readout start. Each drive line selects full charge n or the derivative of the existing
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
`flux_phi0_per_gain` in Phi0. The public `ftf` builder includes these channels in
the mock QICK profile. Line display names may repeat across loops. Pulses on
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
that bound. Evolution stops at terminal readout start even if a tail continues.
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
The [fixed-DC flux benchmark](../../../../docs/discussions/2026-10-28-ftf-fixed-dc-flux-benchmark.md)
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

`tests/experiment/ftf/test_flux_reset.py` checks synthetic templates through
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
without triggers as cavity drives. A block owns triggers from its start until the
next block starts, including ringdown after pulse end. Earlier pulses may end at
the next start, but their last ADC sample must be strictly earlier. A crossing
sample window raises `ValueError` naming the block and trigger.

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
outcome history across later reads. Each outcome has its own density and cavity
field. The projective pulse drives that outcome's cavity during continuation.
A later projection copies the incoming cavity field before continuing each new
outcome. Projection uses all retained energy levels, not IQ-resolvability clusters.
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
sequentially. All state-conditioned fields start each span from the previous
main branch's final field. The first span starts from vacuum. The main branch
has the largest population in the span-start DC density; exact ties select the
lowest energy. Each span supplies ADC's `initial_field` and absolute
`initial_time_ns`; the first supplies zero at 0 ns. Fields retain ringdown after
pulse switch-off. Selecting one
branch for a superposition is a semiclassical approximation, not a description
of quantum cavity-state entanglement.

FTF averages Stark energy shifts and driven dephasing over each span and passes
neutral `SegmentShift` values to evolution. Evolution propagates only that span
from the preceding output density, without normalizing or clipping it. Absolute
control coordinates, carrier phases and pulse-local envelopes keep their common
origin. Within a span, evolution uses the averages rather than instantaneous
fields. Templates with neither cavity drives nor projective reads retain their
original single-evolve path.

Each cavity-driven preparation retains all span reuse keys. Cache hits check
every key through `Engine.reuse`. Projection history, initial density and span
shifts enter continuation digests. Runtime evolves each point/outcome separately,
so one Engine call never contains duplicate point IDs. Evolution's
`merge_path_diagnostics` combines
span work and unions class membership by caller point index, so each point counts
once per class across the acquire.

One ADC `CavityAverageCache` serves the cavity span composition of an acquire.
All g/e points share its 64 MiB budget through serialized calls. ADC owns keys,
arrays and admission; `CavityDrive` only forwards the owner. Composition clears
all resident content on return or exception. Neither Engine nor the long-lived
source retains this owner. Shaped inputs and final branch fields stay uncached.

A Stark shift that reverses state-energy order raises a point/segment-specific
error because it exceeds the dispersive model's validity. States are not reordered.
The linear dispersive cavity model has no measurement-induced state transitions
and does not model MIST onset.

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
It ignores drive-line controls at or after the terminal readout start. Earlier
control portions, including controls after a herald, still evolve normally. Pulse readout requires exactly one active terminal incident pulse per
point, including mutually exclusive slots. Mux readout permits 1..8 coherent incident tones with
a common pulse start. Each mux ADC observes only the shared-line carriers in its
PFB channel through its own demodulation, calibration, delay and sample grid.
The rotating reference is chosen from the first terminal carrier before selection
and stays common to every ADC. Logical events select independently when an ADC
retunes between reads. Multiple grids within one event must address the same PFB
channel; planning raises `ValueError` if they span different channels.
The physical resonator pair
is unchanged; tones are not assigned to individual resonators. Controls evolve
to the common terminal readout start only. Multiple terminal readout blocks are
not supported.
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
Active event generators select the pulse or mux path. Each readout window must
use a receiver and generator from that path. `FTFSource.open_acquire` rejects
pulse and mux generators anywhere in one request's templates and names both
indices. Standalone `plan_template` rejects mixing within its template too.
Separate acquires can select different paths in the same environment.
The runtime sizes generator and ADC arrays from their maximum configured indices
plus one, including both readout paths and all drive lines. It uses the mux mock
profile only when mux is configured and includes both paths in the routes.
`plan_template` and `FTFSource` accept an explicit `readout=` override; private replay rejects an
override that cannot be reconstructed from captured settings. Plans expose
`readouts`, an ordered tuple of `FTFReadoutPlan(channel, batch)`, not a singleton.
Each batch retains the same chronological logical-event axis. An event contains
only its own incident pulse block and that ADC's observed grids. Empty grids mean
that ADC does not observe the event and emits no read column. All ADCs may omit
terminal observations; the terminal event and evolution endpoint remain in the
plan. Each ADC must still observe at least one event. Projective plans retain
point-local event contents. Templates without projective reads keep their existing
terminal-content deduplication and rotating reference. Integrated flat,
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

### Physical applicability

The model assumes weak dispersive readout, independent weak-coupling Markov
secular baths and exponential echo-characteristic dephasing. Truncations and
numerical tolerances remain user choices. Direct conditional means include
multilevel relaxation. Direct-mean variance includes initial-state mixture,
receiver noise and the integrated rounding term.
Physical raw and traces additionally resolve continuous jump times and cavity
memory. They need only agree with direct means in expectation, not realization.

### Private replay

`acquisition=FTFAcquisitionOptions(private_replay=absolute_directory)` opts into
checked private input archives. Acquire and round JSON use schema version 2.
Version 1 archives fail rather than converting float times. Import `FTFAcquisitionOptions` from
`zcu_tools.experiment.ftf`. Relative replay roots raise `ValueError`.
`replay_round(round_path)` recomputes one round without drivers, earlier rounds,
current YAML or numeric records. Import it from `zcu_tools.experiment.ftf`.
`ReplayAcquire.settings` stores that acquisition's effective settings with
`drift=None`, so replay neither reads the simulation clock nor samples drift.
`ReplayRound.snapshot.model_identity` stores the effective model identity.
Backend, engine numeric version, RNG version and dependency versions must match. The stream policy is `PCG64/SeedSequence-v2`.
The backend is `ftf-segment-v13`. Integrated raw and decimated traces resolve
continuous CTMC jump times and cavity memory, with one physical path per shot
shared across mux carriers, ADCs and windows. Archives from `ftf-segment-v12`
and earlier backends fail explicitly because they predate segment-boundary cavity
field inheritance.
Direct means retain conditional moments, their numeric values and their existing
RNG stream. Raw getters do not advance that stream or shift shots to match a
finite-shot direct mean. Each acquisition route reproduces itself for equal
seed, inputs and recorded versions. Evolution uses `segment-engine-v7`.
Old exact-record manifests also fail explicitly. Archives contain model truth and
must not appear in ordinary Run/native/RPC metadata. Normal provenance remains
the opaque simulation identifier.

There is no offline database builder, U-table library or persistent density
record store. Replay retains inputs and recomputes evolution and readout.
