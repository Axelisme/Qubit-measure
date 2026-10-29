# Single-fluxonium acquisition

**Last updated:** 2026-10-30, inactive fast-flux DC shortcut

**Readers:** framework-dev, lab-author, agent. Compose complete fluxonium timelines with one physical readout cavity.

`single_fluxonium(path_or_settings, *, private_replay=None)` returns the normal
`SimulationEnvironment`. It creates one fixed-unit DC FakeDevice, a mock SoC,
one route per named microwave line, an optional fast-flux route and one readout generator/ADC route.
All generators bind to that ADC for transport admission. Only the configured
readout generator supplies physical readout pulses. No real hardware or numeric
warmup runs at construction.

The caller owns resources after return. After acquisition drains, call
`environment.close_backend()` when present, then clean up caller-owned devices.
GUI startup registers the backend cleanup through its normal on-close hook.
Transferring a driver to the GUI transfers only that driver's cleanup responsibility.

## Complete timelines

`plan_template(settings, snapshot, template, *, readout_kind="pulse")` returns a `SingleFluxoniumPlan`.
`readout_kind="mux"` accepts coherent terminal tones on the same physical cavity.
A null or omitted `settings.flux_line` keeps flux fixed at the live DC value.
It binds no fast-flux generator and rejects foreign controls.
Each `drive_lines` generator maps to its own named track, including overlapping events on different generators. Each source multiplies the full charge operator by its own complex calibration. Readout, fast-flux and microwave generator bindings must be distinct. Generator count is the largest bound index plus one; holes remain unbound. Unknown event generators fail in preflight with their event path and channel.
Acquisition may carry named `ChargeDriveRequirement` records in its request.
Preflight requires every requested generator to belong to `drive_lines`, rejecting
flux, readout and unbound generators before waveform construction or evolution.
Other legal fast-flux events keep their ordinary routing. Names are diagnostics,
not role keys. An empty requirement tuple preserves ordinary planning.
It converts QICK us/MHz/degrees into ns/GHz/radians. Shared
`experiment.readout_plan.drive_track` preserves signed gains, complex envelopes
and the common carrier origin. Each template remains one `TimelineBatch`.
`Engine.evolve` owns segmentation, kernel selection and propagation.

With a configured flux line but no active point on any fast-flux track, the
planner keeps flux at the live DC value and skips flux-history sampling.
Active tracks retain the causal response and passive tails described below.

Fast flux uses `simulate.lines.FilteredFlux` with one loop. The planner samples
its complete causal response until terminal readout, including passive tails.
A 2 ns grid, delayed event sides and the endpoint define the continuous waveform.
Bath levels use interval-midpoint flux on the configured bath grid, quantized
relative to live DC in 1e-3 Phi0 steps. Equal adjacent levels merge.
Ideal constant pulses use abrupt schedule levels without a continuous waveform.
The planner never forces the endpoint back to DC.

A nonterminal readout block with an ADC trigger causes one projective read at
its pulse start. Repeated triggers on that block share one projection and CTMC
path. The shared [readout planner](../README.md#explicit-simulation-environments)
binds triggers from this block's start until the next block starts, including
ringdown after pulse end. The last nonterminal sample must precede the next
block start; violations name the block and trigger.
`SingleFluxoniumPlan.read_events` retains projective blocks followed by
terminal, with common-origin phases and channel-local grids. An unobserved
terminal block has empty grids but still ends evolution and contributes its
state to integrated counts. Untriggered earlier blocks remain `cavity_drives`;
execution rejects them before evolution. The source does not approximate their
Stark shifts, cavity memory or backaction. All readout uses the DC model even
when flux has not returned to DC.

`SingleFluxoniumSource.prepare` projects each density into rank-1 outcomes in the
retained DC energy basis. It keeps each outcome's full history across later
reads and continues from that read's start through `Engine.evolve`. ADC CTMC
jumps during an earlier read affect its IQ only, not later density evolution.
This approximation does not group outcomes by IQ distinguishability and does
not add MIST or active reset. Within one preparation, templates must have equal
projective counts and per-event grid counts. Conflicts raise `ValueError` naming
the templates and layouts before evolution or sampling.

`numerics.projection_prune_floor` discards global history weights below its
threshold after each projection, then normalizes survivors per point. The
finite range is `0 <= floor < 1`, with default `1e-9`; zero disables pruning.
If a point loses every outcome, preparation raises `ValueError` naming its ID
and floor. `SingleFluxoniumPreparation.projection_diagnostics` records retained
history counts per point/event, discarded probability mass relative to the
original distribution, and evolution call count. Summing a point's discarded
weights gives its cumulative loss without recounting earlier pruning.
Span reuse digests include continuation densities and absolute time bounds.
Every span must pass the engine's routing validation before preparation reuse.
Without projective reads the source keeps its terminal-only evolution and ADC
sampling path, including RNG consumption.

## Physical and ADC composition

`SingleFluxoniumSource(settings, devices, *, private_replay=None, readout_kind="pulse", runtime=SingleFluxoniumRuntimeOptions())` captures the
startup driver object. `snapshot()` reads its current A/V value and checks the
fixed unit and inclusive limits. Registry removal or renaming does not restore
startup values. `model_facts(snapshot)` builds the retained projection, Gibbs
initial density, full-charge drives, directional baths and one physical ADC model.
The charge calibration uses `Re[c*z] = Re(c)*Re(z) - Im(c)*Im(z)`.

Evolution returns densities in DC energy coordinates. The source supplies
explicit `Basis` records to `batch_populations`, then calls
`normalize_populations` as the sole probability repair. Readout state order is
the retained DC energy order. Cavity frequencies include the model owner's
dressed shifts at that DC point. Receiver PSD converts to sample covariance
using the ADC sample interval. One scattering phase is always supplied explicitly.

`preflight(request, snapshot, receivers=...)` plans the complete observation set
of one physical cavity. Pass every ADC's `resources.simulation_schema.Receiver`
calibration in request order. Omitting `receivers` selects the configured
standalone ADC. Classification uses all triggers, even when an ADC observes
only terminal or only herald. Each ADC keeps empty grids for unobserved events.
Preflight checks every template's projective count and each ADC's event/grid
layout without building a model, evolving a density or solving an ADC response.
Conflicts name the ADC, templates and layouts.

`prepare_readouts(preflight)` evolves one common `ProjectionReadout` and returns
all `channels` and `raw.batches` in receiver order. ADCs observing the same event
share its CTMC path in raw and decimated sampling. Their receiver noise remains
independent. `ConditionalSums.event_indices` retains original logical event IDs,
so a terminal-only ADC does not relabel terminal as event zero. Preflight binds
the current settings and snapshot. Changing source settings before preparation
raises `ValueError`; request arrays must remain unchanged between those calls.

Standalone `prepare(request, snapshot)` uses the same owner with one receiver.
Its `channel` property returns that receiver. Multi-ADC callers use `channels`;
asking for `channel` when several receivers exist raises `ValueError`.
No terminal marginal can replace `raw.readout` when sampling projection histories.

Integrated rounds use ADC `round_statistics` and `PreparedRawShots`.
Decimated rounds use ADC `sample_traces`. No environment-owned random sampler,
synthetic second cavity or alternate IQ generator exists. QICK owns averaging,
round hooks and Stop checkpoints. The binding paces each round to
`time_scale * reps * sum(point timeline duration + relax_delay)`.
`numerics.time_scale` defaults to 1.0 (physical time); zero disables waiting.
Integrated Stop is checked between evolution chunks and during pacing, discarding
the interrupted round while retaining completed rounds. ADC sampling remains synchronous.

## Coherence drift and simulated time

`zcu_tools.simulate.drift.LogOU` owns the numeric log-OU transition and accepts
only numeric parameters and an RNG. This environment's `drift.CoherenceDrift`
composes loss streams, freezes acquisition settings and writes private truth.

The optional top-level YAML `drift` section selects independent log-OU processes:

```yaml
drift:
  q_cap: {sigma: 0.15, tau_s: 3600.0}
  xqp: {sigma: 0.3, tau_s: 7200.0}
  aphi: {sigma: 0.1, tau_s: 3600.0}
```

Omit `drift`, or set it to null, to keep constant losses and disable truth I/O.
Each selected loss must be positive. Omit an individual parameter to keep it
constant. `sigma` is the stationary standard deviation of ln(loss), and `tau_s`
is correlation time in seconds. The stationary log mean is ln(the YAML loss),
not a promise about its linear arithmetic mean. Every new environment starts
at the YAML losses. The first acquisition fixes the OU time origin and uses
those values. Later acquisitions use the exact transition over elapsed time.

One environment owns one `SimulationClock`. Its time is monotonic wall seconds
plus a nondecreasing offset. After each completed round, the clock adds
`max(0, unscaled physical duration - actual wall duration)`. Actual duration
includes snapshot, computation and pacing sleep. Cancelled or failed rounds add
no offset. Idle wall time advances the clock. `time_scale=0` skips sleep without
losing physical time. A new environment resets the offset to zero.

The binding calls `begin_acquire` at admission, before request construction.
`open_acquire` consumes that preparation without resampling. Standalone source
callers can call `open_acquire` directly to sample once. All sweep points and
rounds share the acquisition's effective losses, while live DC snapshots still
refresh each round. Independent streams derived from the acquisition root seed
do not consume ADC sampling randomness. Repeatability requires the same time
sequence as well as the same root seed.

Enabled drift writes one JSONL row per acquisition to
`~/.local/state/zcu-tools/simulation/single-fluxonium/<session>/truth.jsonl`.
Builder `truth_log=Path(...)` selects another file. Standalone sources use
`runtime=SingleFluxoniumRuntimeOptions(clock=..., truth_log=...)`. Each row contains
`session`, `sim_now` in seconds, UTC ISO `wall_time`, `q_cap`, `q_ind`, `xqp` and
`aphi`. Session UUIDs distinguish restarts even when a caller reuses a file.
I/O errors propagate. The log stays on the simulation side, outside Run, GUI,
MCP and Labber metadata. `clock=SimulationClock(...)` provides a wall-clock
injection point for offline callers.

`drift.LossRow` and `drift.write_truth` own the shared single/six/FTF JSONL format
and append operation. The optional `qubit` key identifies a named circuit or FTF
loop. Loss rows allow null q_ind/xqp for circuits without those channels; a
standalone single source omits the qubit key. Each environment owns its session
and path. The writer creates parent directories and propagates serialization
and filesystem failures.

## Reuse and lifetime

`open_acquire(request)` invalidates the preceding session. `sample_round` and
`execute` propagate numeric, ADC and replay errors without partial results.
Invalidated sessions and closed sources raise `RuntimeError`. Direct callers
serialize source methods. `close()` clears structure, models, plans and routing facts,
invalidates the session and leaves caller-owned drivers untouched.

A standalone source uses an Engine with 4 MiB of routing facts.
Composing environments pass an exclusive Engine through immutable
`SingleFluxoniumRuntimeOptions(engine=...)`. The source closes that Engine.
The six-cavity environment divides its environment-wide routing allowance
through this entry point. ADC solvers live only for the current preparation.

The source retains one loss-independent structural generation and one complete
noiseless preparation. The structural key excludes q_cap, q_ind, xqp, aphi and
drift configuration, but includes all other settings, DC values and numeric
versions. The physics owner retains the Hamiltonian projection, eigensystems
and dressed readout frequencies. Loss changes close the old ModelSource and
discard full models, plans and preparation. Rates are rebuilt from the current
acquisition's frozen losses. Temperature, DC, truncation or other structural
settings changes discard structure too. No dissipative propagator or density
survives a loss change through this reuse. See [ADR-0078](../../../../docs/adr/0078-evolution-module-boundary.md).
Keys include settings, the full snapshot, numeric/dependency versions, layout,
all borrowed control/readout arrays and every waveform sample/time. Buffers are
reread on each lookup. Evolution validates reuse against its own routing policy.
Noise, counts, round identities and measured IQ are never cached.
Projection continuation densities exist only while preparing the spans of one
timeline. No later shot, round or acquisition starts from a previous outcome's
terminal density. Reuse of immutable outcome weights, event populations and
conditional sums follows the complete-preparation rules above. It does not
carry forward a sampled herald result.
Reassigning `source.settings` to validated settings changes the snapshot's
settings identity and clears preparations on the next lookup. It does not
replace the startup driver or reset its value. Replay captures the current
settings and validates that same identity when reconstructing the source.
Delivered raw sources retain their own immutable inputs after another acquire,
cache eviction or source close. Repeated materialization returns fresh arrays.

## Private replay

Pass an explicit absolute `private_replay` directory to the builder or source.
The resource owner archives detached settings, controls, envelopes, RNG identity
and the selected snapshot. With drift enabled, archived settings contain the
acquisition's effective losses with drift disabled and a matching model identity.
Replay never samples a new drift state or reads the live clock. It stores no measured IQ. Call
`replay_round(round_json_path)` to recompute only that round without live devices.
The single-fluxonium backend has its own version string. Private single-model replay supports pulse mode only; the constructor rejects mux mode with a replay root. Wrong settings backend,
model identity, dependency versions or numeric references raise `ValueError`.
Integrity and I/O errors propagate. These author-only archives contain model
truth and do not belong in normal GUI/RPC metadata.
