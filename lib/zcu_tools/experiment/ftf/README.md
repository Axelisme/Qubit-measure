# FTF environment composition

**Last updated:** 2026-10-19, source-scoped immutable envelope fingerprints, noiseless preparation and ADC solver reuse, unified resident/pending budget, startup numeric preparation, closed-form flat mux decimated readout, sampled readout, integrated carrier tiles and shared model facts

`ftf(path_or_settings, *, workers=1, warmup=False)` builds the same `SimulationEnvironment` for GUI startup
and Notebook callers. It creates fixed-unit V/A `FakeDevice` sources in one
`DeviceManager`, applies their initial values once and binds the derived channel
counts to the fixed mock QICK profile. The returned manager owns cleanup. A
builder failure closes its resources and preserves both startup and close errors.
The executable example is `zcu_lab/simulation_envs/FTF/environment.py` with adjacent
`ftf.yaml`, a working copy of the tracked `default.yaml`. The Notebook walkthrough is `notebook_md/single_qubit.md`.
欄位用途、單位與 entry 接線請讀 [FTF YAML v2 欄位指南](../../../../zcu_lab/simulation_envs/FTF/field-guide.md)。

`FTFSource.snapshot` reads live DC values before each round. It checks units and
limits without restoring startup values. `execute` consumes detached templates.
The segment engine's fast algorithms are the default numeric path.
One source owns one `Engine`, whose 2 GiB resident/pending prepare budget survives across
acquires. This shared byte budget counts retained model-only payload arrays, not total
process memory; workspaces, active references and ADC allocations remain separate.
The keyword `workers` selects the engine's sole parallel width; direct callers
use one worker by default. GUI and Notebook use the shared FTF environment entry,
which selects `max(1, (os.cpu_count() or 1) - 2)` workers. Unknown CPU counts
therefore use one worker. An eight-core host uses six workers. Evolution limits
native BLAS to one thread for the entire call. Overlapping evolves share a
process-level reference count: the first sets the limit and the last restores
it. Do not independently change threadpoolctl limits while an evolve is active.
Empty work, one timeline, and `workers=1` run serially without a joblib pool.
Other calls own a threading pool only during `evolve`; no idle pool survives.
`FTFSource.close()` clears preparations and the engine cache. Dropping the
environment also releases this in-memory state; no disk cache is involved.
The source retains at most one snapshot's `FTFModelFacts` across acquires.
The exact key includes full settings, the entire snapshot and backend/numeric
revisions. Facts own the joint projection, Gibbs/source, DC eigen/bath memo and
neutral readout model. All templates share these physical inputs. Evolved densities remain acquire/template data. A source also owns an 8 MiB
`ReadoutCache` for ADC solvers, ConstantLine facts and exact conditional responses.
Full numeric model, receiver and solver tolerances select a canonical immutable
model and solver. Existing pulse/grid keys preserve absolute starts, carrier
phases, demodulation and opaque callback identity. Different T1 readout starts
reuse physical facts but compute their own sums. Every round still samples IQ.
The byte limit counts keys, traces, nested line caches and whole allocations held
by array views. Oversized solvers are evicted whole after the response returns.
A changed physical-facts key clears readout residents, including on settings,
snapshot or numeric revision changes. Close clears them too. No projected
operators, rates or readout response cross flux points.

## Startup preparation

The shared production GUI/Notebook environment entry opts into `warmup=True`.
Building that environment immediately starts one process-wide background numeric
worker. All opted-in environments share its future and terminal result. Direct
`ftf()` callers keep `warmup=False`, just as they keep `workers=1`, to avoid
unexpected CPU work. Authors can disable startup work with `ftf(..., warmup=False)`.
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
shaped direct integration, covariant single-tone and simultaneous gate playback,
Dyson gain batches, constant-tone block RWA, dense Floquet periods, t4 and ADC
integrated/decimated/mux responses. It does not promise every possible signature
or kernel. New routes require updating the representative inputs and checking
real acquire compile events with an empty scratch `NUMBA_CACHE_DIR`.

## Acquisition and numeric composition

`open_acquire(request)` starts a session for a fixed request. Its
`sample_round(snapshot, index)` reuses the preceding round's density and
conditional response when the snapshot is equal. Every round draws a new stream.
Opening another acquire invalidates the old session and drops its references.
A source-owned 4 MiB LRU retains owned, read-only probabilities, conditional sums,
content mappings, path diagnostics and integrated sampling plans across acquires.
Its key reads full current C1 content, layout, mode, reps and readout values,
including absolute phase origins. Physical settings/snapshot/numeric changes,
engine replacement, failed preparation and close clear the preparations. It never
retains request seeds, acquire sequences, noisy IQ or round results. Hits update
numeric-use identity and repeat report-only accuracy warnings. The public
`source.preparation_cache` reports cumulative `builds`, `hits` and `resident_bytes`.
Its `max_bytes` can be adjusted between calls; zero disables residency. The budget
counts Python containers/keys and whole ndarray allocations, including bases of
views. Active/caller references are separate from resident budgets. Bounded ADC
solvers and exact contents also remain source-owned. A separate source-owned
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

`plan_template` accepts optional matching `FTFModelFacts` for shared physical
inputs; standalone calls build their own facts. `FTFSource.model_facts` supplies
the source-owned latest facts and rejects calls after close.
`plan_template` converts QICK us/MHz/degrees into ns/GHz/radians with point
arrays. It preserves signed gains, complex sampled envelopes and the common
carrier origin for evolution. Readout converts the program's common-origin
phase to ADC's pulse-local phase by subtracting
`2*pi*(f-f_ref)*pulse_start_ns`, with frequencies in GHz. The model reference
is the first carrier; that carrier's phase stays unchanged. The existing
`phase_reference_us` contribution stays part of the common-origin phase.
ADC demodulation keeps its absolute-time grid. An isolated addressed carrier's
conditional IQ center is therefore invariant under a common pulse/grid time
shift. With multiple nonzero carriers, finite-window cross-tone leakage can
still depend on the readout start. Pulse supports use absolute nearest-2-ns rounding, ties even.
The model owner supplies a fixed-DC `ModelSource` and static schedule for each
round. Charge and small-signal flux controls evolve every template to readout
start. Each drive line selects full charge n or the derivative of the existing
fixed-coordinate Hamiltonian at that round's DC point. Omitted operator means
charge. Charge calibration uses GHz/gain; flux calibration uses Phi0/gain and
multiplies the derivative in GHz/Phi0. Signed complex crosstalk uses
Re[c*z] = Re(c)*Re(z) - Im(c)*Im(z) for either operator.
Flux drives keep that linearized DC operator throughout the round, including
secular energy-basis rotations. They do not activate the shared engine's
large-amplitude flux controls, filtering or flux-tail paths. Version 2 settings
group physical values, named DC/drive lines, readout and numerics. Version 1 is
not accepted.

Readout uses the DC model and populations in its energy basis. FTF explicitly
calls ADC's `normalize_populations` once after `batch_populations`, before
conditional responses or sampling. ADC owns the 1e-6 probability tolerance,
tolerated-negative cleanup and row normalization. Invalid rows fail without
sampling. ADC raises `PopulationValidationError` with `rejected_rows` for the
reported validation rule. FTF catches only that type, retains its count in
`FTFNumericUse.rejected_rows`, then re-raises without sampling. Successful
preparations retain zero rejected rows. The source retains the latest
evolve diagnostics even when this preparation fails, including the engine's
report-only maximum trace error. Raw evolution densities remain unchanged.
It ignores drive-line controls during or after readout. Earlier control portions still evolve
normally. Pulse readout requires exactly one active incident pulse per point, including
mutually exclusive slots. Mux readout permits 1..8 coherent incident tones with
a common pulse start. Every ADC observes the total shared-line field through its
own demodulation, calibration, delay and sample grid. The physical resonator pair
is unchanged; tones are not assigned to individual resonators. Controls evolve
to the common readout start only. Separate readout blocks are not supported.
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

`FTFReadoutConfig(kind, receivers)` is the detached author configuration below
settings. `readout_configuration` is its settings adapter. The default
`readout.drive.kind: pulse` retains the single ADC object; `kind: mux` requires
an ADC receiver list with 1..8 unique `ro_ch` values. The builder selects a fixed
mixmux8/PFBv4 compile profile only for mux. `plan_template` and
`FTFSource` accept an explicit `readout=` override; private replay rejects an
override that cannot be reconstructed from captured settings. Plans expose
`readouts`, an ordered tuple of `FTFReadoutPlan(channel, batch)`, not a singleton.
Each batch contains every incident tone and that ADC's grids. Integrated flat,
common-start mux batches use ADC's carrier-wise linear sums. Each carrier keeps
the guarded direct-sum or constant-field/full-exponential route. Delay, mismatch,
ringdown and per-receiver offset remain part of ADC's response contract. Shaped
mux batches use ADC affine-input convolution propagation for known numeric
envelopes. Opaque callbacks and unsupported timing retain the field solver. Flat common-start
decimated batches superpose vacuum-start carrier fields through the same line,
then calibrate once per receiver. Single-carrier fast paths and evolution are unchanged.

The receiver noise PSD is two-sided, in ADC sample units squared times ns.
Discrete sample covariance is PSD/dt_ns using the ADC output rate. Receiver gain
does not scale noise. Integrated acquisition returns `RoundStatistics` with joint
state counts shared across all ADCs/reads, and separate channel totals. Decimated acquisition returns `RoundSignals`,
reordering point-major samples into QICK loop order. The adapter owns packing,
rounding, deferred raw generation and averaging. All channels at the same shot
index share one sampled joint state; receiver noise is independent. There is no
extra tone axis: each declared ADC has its own raw-shot array. Output follows
program ADC declaration order, which is mux tone order, not sorted channel index.

The model assumes weak dispersive readout, independent weak-coupling Markov
secular baths and exponential echo-characteristic dephasing. Truncations and
numerical tolerances remain user choices. Conditional means include multilevel
relaxation, not random jump times. Variance includes initial-state mixture,
receiver noise and the integrated rounding term.

`private_replay=absolute_directory` opts into checked private input archives.
`replay_round(round_path)` recomputes one round without drivers, earlier rounds,
current YAML or numeric records. Backend, engine numeric version, RNG version and
dependency versions must match. The stream policy is `PCG64/SeedSequence-v2`.
The backend is `ftf-segment-v10`, which identifies closed-form sampled-envelope
readout and tiled flat integrated sums, retaining common-origin phase conversion.
Archives from `ftf-segment-v9` and earlier backends fail explicitly because
readout floating-point results change. Evolution's numeric version and RNG
sampling contract are unchanged.
Old exact-record manifests also fail explicitly. Archives contain model truth and
must not appear in ordinary Run/native/RPC metadata. Normal provenance remains
the opaque simulation identifier.

There is no offline database builder, U-table library or persistent density
record store. Replay retains inputs and recomputes evolution and readout.
