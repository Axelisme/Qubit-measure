# Segment evolution

**Last updated:** 2026-11-23, fixed-H collapse operators and frozen physical v2

**讀者：** framework-dev、agent。Segment evolution 的公開輸入、路由與數值邊界。

`simulate.evolution` evolves numeric models to readout start. Callers use the
package root; dispatch, kernels, scheduling and caching are internal.
[ADR-0078](../../../../docs/adr/0078-evolution-module-boundary.md) records this
boundary and the decision to keep algorithm knowledge in dispatch.

## Contents

- [Public entry points](#public-entry-points)
- [GPU backend](#gpu-backend)
- [Frozen numeric inputs and references](#frozen-numeric-inputs-and-references)
- [Dispatch inspection](#dispatch-inspection)
- [Engine capture](#engine-capture)
- [Joint-Fock prototype](#joint-fock-prototype)
- [Routing telemetry](#routing-telemetry)
- [Decision graph](#decision-graph)
- [Kernel summary](#kernel-summary)
- [Adding a family](#adding-a-family)
- [Row plans and the kernel seam](#row-plans-and-the-kernel-seam)
- [WorkPlan and cache ownership](#workplan-and-cache-ownership)
- [Algorithm ownership and reuse](#algorithm-ownership-and-reuse)
- [Accuracy, failures and lifecycle](#accuracy-failures-and-lifecycle)
- [Research records](#research-records)

## Public entry points

Importing this package initializes the shared simulator Numba cache policy
before creating dispatchers. ADC and evolution share one successful bootstrap
per process. Basic `simulate.value2flux` and `simulate.flux2value` imports do not
initialize it. See the [cache contract](../README.md#numba-machine-code-cache)
for source/version isolation, fail-fast storage and process-global effects.

The exports in [__init__.py](__init__.py) fall into these groups:

| Purpose | Exports |
| --- | --- |
| Execution and results | `Engine`, `EvolutionResult`, `PathDiagnostics`, `FamilyRouting`, `StarkOrderInversion`, `Cancelled`, `SEGMENT_NUMERIC_VERSION` |
| Execution capture | `CaptureBuffer`, `CaptureMetadata`, `CaptureRoute`, `CaptureCandidate`, `CapturePrefixRun`, `CapturePosition`, `CaptureAdopter`, `CaptureSampledAdmission`, `CaptureSampledSpot`, `CaptureRowKey` |
| Dispatch inspection | `DispatchInspection`, `DispatchRow`, `DispatchCandidate`, `RowKey` |
| Engine model source and routing history | `ModelSource`, `SecularModel`, `SecularDrive`, `RoutingHistory` |
| Flux description | `FluxExpansion`, `FluxTerm`, `FluxLevel`, `FluxSchedule`, `FluxWaveform` |
| Batched timelines | `TimelineBatch`, `DriveTrack`, `EnvelopeSpec`, `SegmentShift`, `segment_spans` |
| Control description | `ControlPulse`, `ControlSequence`, `Envelope`, `PhaseEvent`, `quantize_ns` |
| Fixed-coordinate model records | `NumericModel`, `DriveSource` |
| States and population conversion | `Basis`, `DensityState`, `Populations`, `populations`, `batch_populations` |
| Explicit joint-Fock prototype | `JointFockRequest`, `JointFockDiagnostic`, `JointFockResult`, `joint_fock_spectrum` |
| Shared RWA Hamiltonian coefficients | `RWACoefficients`, `rwa_coefficients`, `interaction_hamiltonian` |
| Read-only constant RWA guard query | `single_tone_rwa_diagnostic`, `SingleToneRWADiagnostic`, `RWAGuardMetric`, `RWAUncomputed` |
| Read-only constant-pair RWA guard query | `multitone_rwa_diagnostic`, `MultitoneRWADiagnostic` |
| Read-only common-frame guard query | `common_frame_diagnostic`, `CommonFrameDiagnostic` |
| Indexed scene cache | `SCENE_CACHE_VERSION`, `FrozenIndexEntry`, `read_frozen_index`, `write_frozen_index`, `read_scene_records`, `write_scene_record` |
| Frozen numeric inputs | `FROZEN_PHYSICAL_VERSION`, `FrozenInput`, `FrozenRecord`, `physical_input`, `canonical_json`, `digest`, `read_physical_input`, `read_frozen_input`, `write_frozen_input`, `read_frozen_manifest`, `write_frozen_manifest` |
| Solver-free reference storage | `SolverTolerances`, `ReferenceResult`, `reference_key`, `load_reference`, `write_reference` |

`Engine(workers, *, families=None, kernels=None, chunks_per_worker=None,
routing_cache_bytes=4*1024**2, backend="auto")` owns immutable routing facts.
Numerical preparations live only within one evolve. `evolve(source, batches, *, budget=1e-3, cancel=event)` consumes a
`ModelSource` and all `TimelineBatch` inputs for one acquire. `NumericModel` is a
separate fixed-coordinate record, not an Engine input. Constructor and field
contracts live with their types in [engine.py](engine.py), [secular.py](secular.py),
[inspection records](_inspection.py), [controls.py](controls.py), [model.py](model.py)
and [state.py](state.py).

Units are ns, GHz and rad. A [`DriveTrack`](engine.py) carries a signed numeric
carrier in `frequency_ghz`, including zero. This field does not record emitted
RF. The track's `port` matches a [`SecularDrive`](secular.py) by name. That record
supplies Hermitian I/Q operators `X` and `Y` in the model's energy basis. For
complex envelope value `z`, the drive Hamiltonian is `H_drive/h = Re(z)*X + Im(z)*Y`.

`SecularModel` supplies energies, drive quadratures, directional rates and pure
dephasing in each quantized flux level's energy basis. The optional
`collapse_operators` tuple supplies full complex `L` matrices in that basis,
in ns⁻½. CPU `dop853` adds each `LρL† − {L†L,ρ}/2` to the secular bath and
transforms it with the Hamiltonian into the integration frame.
Keep the coherent components of one physical jump in one matrix. Use `()` when
there are no explicit operators; the existing secular routes are unchanged.
The model builder owns physical applicability. Records own read-only copies
and reject invalid shapes or nonfinite operators. `SecularModel.identity` is the
caller's model namespace, not a complete physics key. A shifted model appends
the correction fingerprint to that namespace. `SecularModel.content_identity`
adds the ordered complex jump matrices to the namespace and equals `identity`
when there are no jumps. The Engine uses it for preparation and routing reuse.
The frozen codec stores the namespace and matrices separately.

For a row with explicit operators, `dop853` is the only candidate. The timeline
must stay at a fixed flux level without a supplied flux waveform. `families=` can
exclude `dop853`, in which case both inspection and evolution raise
`NoValidKernel`; neither falls back to a secular approximation. Explicit
operators never use OpenCL, including on an Engine configured for that device.

[`EnvelopeSpec`](engine.py) defines envelope sampling independently of
`DriveTrack` pulse timing. `EnvelopeSpec.values(local_ns)` returns unity for a
constant envelope. A sampled envelope linearly interpolates the real and
imaginary parts at pulse-local ns, using its increasing `sample_ns` grid starting
at zero. The grid need not be uniform. The final sample is held through pulse end.
Tracks and segments enforce pulse support, not the evaluator.
`DriveTrack.phase_rad` defines the carrier phase at the shot's absolute time
zero, not at the start of the pulse or a continuation span.

`TimelineBatch.segment_boundaries` adds point-aligned absolute ns boundaries.
`segment_spans(batch)` is a pure query of the same final partition used by
`evolve`, including active microwave plateaus and flux events. Callers can
average external energy shifts and dephasing over these spans, then attach
point-aligned `SegmentShift` values through `segment_shifts`. Each supplied span
must match one queried span exactly; missing spans have zero shift. Shifts do
not add boundaries, rotate bases or change physical `FluxLevel` values. Model,
prepare and gate-usage identities include shift content. Shifts keep explicit
collapse matrices in the unchanged eigenbasis. Without shifts,
the original model source, initial state and reuse semantics remain unchanged.

A shift may reverse the order of two adjacent physical state energies. Evolution
keeps their original indices and uses the corrected energies. A standalone
`Engine.evolve` issues one `UserWarning` per reversed pair, with the first point
and segment, counts, and the tightest upper limit on the lower-minus-upper
Stark shift difference in GHz. Equal energies do not warn. Callers that compose
several evolves can use `Engine.defer_stark_order_warnings()` and report the
observations in `PathDiagnostics.stark_order_inversions` together. Other
validation and numerical failures still propagate.

`TimelineBatch.start_ns` and `initial_densities` optionally continue each point
from an existing DC-energy-basis density. Supply both or neither. Starts must
be zero or final-span boundaries no later than `end_ns`. The engine plans with
the common origin and propagates only `[start_ns,end_ns]`, without replaying the
prefix or restarting carrier phases and envelopes. The full boundary query
still covers `[0,end_ns]`. Batch construction retains the existing density
invariants without repair. Continuation inputs participate in acquire digests, not model-only prepare identities.

`PathDiagnostics.point_members` records each class's increasing unique caller
point IDs. `merge_path_diagnostics(parts)` unions these memberships, sums work
and CPU, and takes the maximum trace error. Reuse diagnostics participate as
returned. These diagnostics remain author-only, separate from acquisition data.

`Engine.evolve` returns `EvolutionResult` densities at readout start in the DC
energy eigenbasis. Every returned density can serve as a continuation input.
The environment explicitly converts them to populations, then calls ADC's
normalization and sampling tools. Evolution neither imports ADC nor produces
I/Q. ADC owns its probability tolerance and readout sums; see the
[ADC README](../adc/README.md).

### Constant RWA guard diagnostics

Call `single_tone_rwa_diagnostic(model, port, frequency_ghz, amplitude,
duration_ns, budget=1e-3)` from the package root. Supply the row's `SecularModel`
at fixed flux, including any segment shifts, and a drive port name. Frequency
is any finite signed GHz, including zero. Amplitude is signed dimensionless
gain, and duration is nonnegative ns. All numeric inputs must be finite and
budget must be positive. Explicit collapse operators, unknown ports, negative
duration and nonfinite inputs raise `ValueError`.
Numerical construction failures propagate. The query does not evolve a density,
select a family or modify routing history. It releases its temporary facts
before returning.

The frozen `SingleToneRWADiagnostic` returns the following comparisons. A
computed field is a frozen `RWAGuardMetric` with `value`, `threshold`, inclusive
`comparison` and `passed`. A skipped field is `RWAUncomputed`, a frozen marker
with neither a value nor a passed status. `accepted` requires every field to be
computed and pass. Low and zero carriers return `accepted=False`, not an exception.

| Field | Value and units | Admission |
| --- | --- | --- |
| `budget` | Per-segment allowance, dimensionless | ≥ 1e-3 |
| `carrier_ghz` | Absolute carrier, GHz | ≥ 0.05 |
| `kappa2` | Spectral norm of the entrywise absolute omitted c/d coefficients summed over both Fourier signs after adaptive edge restoration, dimensionless | ≤ 0.08 |
| `strength_ratio` | abs(gain) × sum of full quadrature spectral norms / abs(carrier), dimensionless | ≤ 0.25 |
| `phase_rad` | 2π × duration × squared gain-weighted retained positive Fourier norm / abs(carrier), rad | ≤ 0.06 |
| `pruning` | Phase-graph population omission bound, dimensionless | ≤ min(budget, 1e-3) |
| `real_dimension` | Populations plus real within-component coherence coordinates, count | ≤ 64 |

The [single constant RWA owner](kernels/single_tone_rwa.py) defines the query and
graph comparisons. It shares retention, kappa2, strength and phase calculations with the guard
through [RWA numerics](kernels/_rwa_numeric.py). `kappa2` replaces the public
`omission_transfer` field without an alias. Its value is
`||sum_sign abs(c/d)||_2`, where c includes gain and d is signed detuning in GHz.
It uses the endpoint kick's omitted edges, including diagonal ±carrier terms,
from the complement of the resolved retention mask. Phase-graph pruning does
not change this edge set.

Retention still starts with the 50 MHz window and restores forward edges and
their adjoints using first/second-order population transfer and the existing
5.8e-3 threshold. Kappa2 replaces only the final single-constant transfer
comparison. The shared transfer constant and `multitone_rwa` are unchanged.

The query preserves guard early returns. Budget always has a value. A failed
budget leaves every later metric uncomputed. A failed carrier leaves kappa2,
strength, phase, pruning and dimension uncomputed. Kappa2 and strength have
values after these domain checks. A failed strength ratio leaves the phase in
radians uncomputed. Any base rejection leaves pruning and dimension uncomputed.
A pruning rejection still returns dimension because the guard computes both.

Kappa2 is computed after retention, before strength and phase comparisons.
Its value and passed status remain visible even when either later comparison
rejects the row. The diagnostic is a numeric table, not a decision trace.
A value and passed status do not mean that comparison participated in the
particular rejection. `accepted` equals the guard result for the same row.

The query does not diagnose sampled envelopes, multiple tones or moving flux.
Carrier phase and initial density do not enter the comparisons. These metrics
are admission proxies, not a certificate of final full-density accuracy.
The 0.08 threshold has margin on the fixed acceptance set described below.
It is not a maximum usable threshold or an accuracy guarantee for arbitrary
inputs. Dispatch order and other families are unchanged.

## GPU backend

OpenCL is an execution backend for `dop853`, not an extra family. On Windows
or Linux, install a GPU driver that exposes `cl_khr_fp64` and add the optional
`gpu` extra. For a GUI checkout, run:

```bash
uv sync --directory <worktree> --locked --no-dev --extra gui --extra gpu
```

The extra installs `pyopencl` from `uv.lock`, not the driver. The `all` extra
does not include `gpu`. The default development group includes `gpu` so
managed lanes can run GPU tests on machines with a compatible driver.

| `Engine(backend=...)` | Behavior |
| --- | --- |
| `"auto"` (default) | Detects OpenCL on the first eligible row. For eligible DOP853 work, compares CPU and GPU cost, including warm program setup, transfers and batch size. |
| `"cpu"` | Never probes or uses OpenCL. |
| `"opencl"` | Requires a device that passes detection at construction. Sends eligible DOP853 rows to that device without price comparison or fallback. A build failure for the selected dimension and tableau raises during evolve. Ineligible rows can still run on CPU. |

Eligibility depends on structure, not envelope shape. The row must be an
independent, unchained interaction-frame `dop853` endpoint at fixed DC. OpenCL
rejects explicit collapse operators, flux modulation and segment shifts. The
row must last 100 ns to 10 µs and have at most 32 retained levels. Two sampled
tones must cover the whole row. The union of
their sample grids can contain at most 960 knots, because the runner restarts
integration at knots from either tone. Other rows in the same timeline may
run on CPU.

Detection imports `pyopencl`, finds an FP64 GPU and compiles both tableaux
for a fixed three-level endpoint. Each result must differ from its CPU
counterpart by at most `1e-5` per complex entry. The Engine can reuse the
stored probe result for the same device, driver and kernel. Failure to write
that result does not change the probe outcome. If detection fails, `opencl`
raises during construction and `auto` uses CPU with the same numerical result
as `"cpu"`. If a program fails to build before execution, `auto` falls back
only for that dimension and tableau. Enable routing telemetry to read the
reason in `opencl_detection_error`. GPU execution errors raise without a CPU retry.

One work-group integrates one row on device. At budgets ≥ `1e-3`, eligible
rows use DP5(4) with one third of the DOP853 local tolerances. Every eligible
row is at most 10 µs, so it meets the same duration limit as CPU DP5(4).
Tighter budgets retain DOP853. Dispatch selects the OpenCL tableau once per
`evolve` call. Pricing, program lookup and build, execution and routing
telemetry use that same choice. The GPU computes
the starting interaction-frame phasor of each step in FP64 and uses FP32
within the step. Before handing off the density, the runner returns it to the
model eigenbasis, symmetrizes it and normalizes its trace. The scheduler
batches ready rows across the acquire by dimension and model key. It uses
measured dispatch times to target about 0.1 s per batch without reading OS or
driver timeout settings.

The CPU/GPU cost choice uses the ready rows, worker count and evolve budget.
For DP5(4), the GPU estimate multiplies matrix work by 0.68 and keeps the
launch, transfer and per-trial floors unchanged. The factor uses the slower
of two warm 24-point `simultaneous_drive` measurements on Intel Arc.
At d=24 the median fell from 0.303 s to 0.206 s (1.47×); at d=32 it fell from
0.502 s to 0.308 s (1.63×). Maximum absolute complex-density errors against
the independent reference were `3.04e-5` and `2.76e-5`, respectively.
Disk cache warmth and measured batch times do not enter routing.
Program builds and detection records use `<user-cache-root>/opencl/dop853/<digest>/`;
the digest covers the kernel source and options, including dimension and tableau,
plus the device and driver. The driver can also cache
builds itself. The simulator's [cache root](../README.md#numba-machine-code-cache)
can be set with an absolute `ZCU_TOOLS_CACHE_DIR`. A cold build populates the
cache for later Engines and is not charged in the routing cost. The Engine
owns the OpenCL context and releases it on `close()` after active work ends.

## Frozen numeric inputs and references

[`FROZEN_PHYSICAL_VERSION`](frozen.py) owns the physical decoding contract for
manifests and standalone inputs. It changes when fields, units, dtypes, defaults
or Engine-input reconstruction change. [`SCENE_CACHE_VERSION`](scene_cache.py)
owns the index JSON and cache-directory revision. The physical version is 2;
the scene-cache version is 1. Consumers compare both without parsing storage
layout. `SEGMENT_NUMERIC_VERSION` records numerical algorithm semantics, not
input-format compatibility.

The public codec snapshots a `ModelSource` and one `TimelineBatch` without
running an Engine or a solver. `write_frozen_input(path, source, batch)` writes
physical version 2, a complete `physical` record and its `physical_sha256`.
`read_frozen_input(path)` checks the version and hash, then returns a
`FrozenInput(source, batch)`. The caller closes the returned source.

`physical_input(source, batch)` returns the same numeric record in memory.
`read_physical_input(record)` reconstructs it, including records from JSON.
The record contains the initial density, scheduled models plus DC zero,
continuous-flux expansion when used, slow-envelope subdivisions and complete
batch data. Point order, complex128/float64/int64/bool record dtypes, segment
corrections and continuation fields survive the round trip. The reconstructed
source has a fresh owned history and a finite level-to-model mapping. Each flux
level may appear only once in the model table. Unlisted levels raise `KeyError`.
Omit `collapse_operators` for models without explicit jumps. When present, it
must be a nonempty ordered list of finite complex matrices of the model's size.
The v2 standalone reader rejects v1 containers even with matching hashes.
No-jump snapshots keep their v1 physical and oracle hashes. The codec does not
store Python builders, devices or registered source shifts.

`canonical_json` uses sorted keys, compact separators, nested array lists and
real/imag objects for complex values. It rejects nonfinite values. `digest`
returns its SHA-256. `physical_input(..., include_identity=False)` removes only
model namespaces. Complete physical hashes include those namespaces;
identity-free hashes allow oracle sharing. Neither hash is a reference key.

`write_frozen_manifest(path, records)` writes ordered `FrozenRecord` values as
physical-version-2 canonical JSON inside stdlib gzip. `FrozenRecord(name, physical,
metadata=None)` keeps caller metadata opaque. Its `physical_sha256` and
`oracle_sha256` properties calculate the complete and identity-free hashes.
The writer calculates both hashes. `read_frozen_manifest(path)` checks the
version and both hashes, then returns a tuple in original order. Both reject
duplicate names, duplicate flux levels and invalid collapse operators. The
gzip header has mtime=0 and no filename, so identical records produce identical bytes
at different paths. Reconstruct a source separately with
`read_physical_input(record.physical)` and close it after use. The container
neither interprets scene declarations nor selects a reference solver.

`FrozenIndexEntry.from_record(record)` pins the name and physical, oracle and
metadata digests. Metadata uses the public `digest` without caller interpretation.
`write_frozen_index(path, entries)` atomically publishes version 1 and ordered
identity rows; `read_frozen_index(path)` validates them. Unknown versions fail.
`write_scene_record(directory, record)` returns its entry after publishing a
single-record manifest. Filenames contain the name and physical/metadata hash
prefixes of 16 characters. `entry.cache_path(directory)` exposes that path when
needed for explicit cache maintenance. Valid existing files are reused; conflicting
files fail validation. Both writers stage files in the destination directory and
publish with replace, preserving existing pins on a pre-publication failure.
`read_scene_records(directory, entries)` reads only the indexed files in entry
order and verifies their full identities, including opaque metadata. A missing
file raises FileNotFoundError with the record name and hashes. None of these APIs
constructs a model, runs an Engine/solver, resolves Git or deletes old cache files.
For the scene-cache location and producer commands, see
[Freeze numeric scenes](../../../../tools/README.md#freeze-numeric-scenes).

[`reference_key(source, batch)`](reference_cache.py) identifies a cached NPZ
result. It hashes the key's own version 1, identity-free physics, coarse
tolerances 2e-10/2e-12, fine tolerances 2e-12/2e-14 and convergence limit 2e-6.
Callers can override both tolerance records. Solver identity and routing history
do not enter the key.
`load_reference(cache_dir, source, batch)` only reads the matching NPZ, without
pickle or directory creation. A miss raises `FileNotFoundError`; it never solves.
`write_reference` atomically publishes `density` and `convergence_error` under
that same key. The producer certifies convergence, and measurement preflight
checks the loaded result before evolution.

The independent preparation entry point is
[`family_pareto_reference_producer.py`](../../../../tests/simulate/evolution/contract/family_pareto_reference_producer.py).
It reads only frozen record names and physics. The caller explicitly maps each
index to the independent lab-frame or interaction-picture solver and supplies
an explicit scene cache. The producer checks the reference cache first. A miss
calls only the selected solver. If the selected bath has explicit collapse
operators, the built-in solvers raise `ValueError` and the producer writes no
result.
`--scenario NAME` requires one matching row across the
supplied indexes before reading payloads. Zero or multiple matches fail without
solving; other rows need no payload. Without a scenario, all indexed rows remain
selected. All selected numeric files load before any solver runs. Metadata
and scene names do not select the solver. See
[Reference solver limits](#reference-solver-limits) for the bath inputs both
solvers use. Both solvers stay in tests. The codec imports neither tests nor
tools.
See [ADR-0078](../../../../docs/adr/0078-evolution-module-boundary.md).

## Dispatch inspection

`Engine.inspect_dispatch(source, batches, *, budget=1e-3, cancel=None)` returns a
`DispatchInspection` without evolving the user's timelines. It uses evolve's
classifier, candidate ordering, guards and complete-inventory pricing.
Guards can evolve their own probes: Floquet computes a one-period propagator,
and a sampled multitone RWA group that wins pricing compares fixed probe
endpoints. When OpenCL availability could change a sampled group's selection,
inspection uses the same lazy DOP853 device check as evolve. It reuses an
existing result, including the check made by a forced OpenCL engine. The check
can run a device self-test and write machine-level OpenCL detection and program
caches. Inspection does not compile a workload point variant. CPU-only engines
do not check a device.
Source, batch, budget, cancellation and failure
contracts match `evolve`; an exhausted candidate round raises `NoValidKernel`.

`rows` follows input batch order, point order, then split-span order.
Each row's `RowKey` contains the zero-based batch index, caller point ID and
zero-based chronological segment index. Segment indices include drive, flux,
plateau and explicit boundary splits. Continuations number only retained spans
at or after `start_ns`. Each row also reports absolute `start_ns` and `duration_ns`.

A `selected` row's `family` matches evolve from the same initial history.
A `deferred` row still needs its timeline worker to finish routing. A local
prepare key can already have a decision if evolve would resolve its guard
before scheduling. A `needs_prefix_input` row needs a driven prefix density
that evolve would compute first. Unresolved rows return `family=None`.
Inspection neither runs timeline workers nor computes prefix densities.

`candidates` records offered families and guard decisions in first-offer order.
An evaluated guard reports `accepted` or `rejected`. A guard left to a timeline
worker reports `deferred`. Candidates after a stopped round report
`untested_after_short_circuit`, unless an earlier round already evaluated them.
Structurally inapplicable and cost-skipped families are absent unless offered.
Inspection can select `multitone_gate` after structural checks and pricing.
It does not build the gate. Without `families=`, a gate skipped on price is
absent from `candidates`. With an allowlist, a later fallback round can name
it after another family has
stopped the row. The gate then reports `untested_after_short_circuit`, not a
guard rejection.

Candidate `scheduling_cost` is `Kernel.cost`'s nonnegative relative row work
for chunk interleaving. It is neither a time estimate nor a numerical
certificate. For `dop853` it is the CPU fallback cost, reduced for rows
eligible for CPU DP5(4). The scheduler selects a backend when a wave is ready.
Inspection does not expose the cold-amortized estimate used to order candidates.
Short-circuited candidates have `scheduling_cost=None`; evaluated and deferred
candidates include their value.

Inspection does not modify routing history. It projects usage without writing
counters, content membership or generation, and does not create routing-reuse
snapshots. Querying A before evolving B leaves B's route and density unchanged.
Guard preparations expire on return and do not enter Engine caches. Source
model caches and shift registrations may be populated. Calls sharing a history
must serialize, just as for evolve.

### Constant-pair RWA guard diagnostics

Call `multitone_rwa_diagnostic(model, ports, frequencies_ghz, amplitudes,
duration_ns, budget=1e-3)` from the package root. Supply the fixed-flux row's
`SecularModel`, including segment shifts, and three aligned length-two tuples.
`ports` names known model drives, `frequencies_ghz` contains finite signed GHz,
and `amplitudes` contains finite signed dimensionless gains. Duration is finite
nonnegative ns, and budget is finite and positive. Explicit collapse operators,
unknown ports, wrong tuple lengths, nonfinite inputs, negative duration and
invalid budget raise `ValueError`. Numerical construction failures propagate.

The frozen `MultitoneRWADiagnostic` uses the same `RWAGuardMetric` and
`RWAUncomputed` leaves as the single-tone query. It has no `reason` field.
`accepted` reports numerical admission, not dispatch selection. Same-port,
low/zero carrier and positive budget below `1e-3` return `accepted=False`.

| Field | Value and units | Inclusive admission |
| --- | --- | --- |
| `budget` | Per-segment allowance, dimensionless | ≥ 1e-3 |
| `carrier_ghz` | Smaller absolute carrier, GHz | ≥ 0.05 |
| `omission_transfer` | Final mask's remaining first/second-order transfer proxy, dimensionless | ≤ 0.0058 |
| `omission_phase_rad` | Final mask's remaining omission phase, rad | ≤ 0.0045 |
| `real_dimension` | Sum of squared retained component sizes, including isolated populations | ≤ 256 |
| `secular_accumulation` | ε_dom² × Ω_ret × row duration, cycles | ≤ 1.476e-4 |

For the final resolved Fourier mask, ε_dom is the largest omitted nonzero
coupling magnitude divided by absolute detuning, including diagonal terms.
Ω_ret is twice the largest retained coupling magnitude over both tones and
signs, in cycles/ns. Couplings include signed gains. Empty maxima are zero.
The accumulation proxy uses row-local duration and no `2π` conversion.
It supplements endpoint kicks; it does not bound final density error.

Budget is always computed. A budget or same-port rejection leaves later
metrics uncomputed. A carrier rejection skips retention and accumulation.
Computed final-mask metrics remain visible even when a later comparison rejects
the row. If restoration crosses the dimension limit before final-mask omission
analysis, dimension is visible, but transfer, phase and accumulation are
uncomputed. The query never reports earlier-mask values or infinity sentinels
as final metrics.

The [family query](kernels/multitone_rwa.py) and kernel guard project the same
[admission fact](kernels/_rwa_numeric.py). One WorkPlan shares that fact without
recomputing it. Each independent query closes its temporary plan before return;
it does not evolve density, select a family, mutate history or cache facts across
calls. This query covers constant pairs only, not sampled controls or moving
flux. To inspect sampled admission, use `Engine.inspect_dispatch`. The
constant-pair query does not change dispatch prices.

### Common-frame guard diagnostics

Call `common_frame_diagnostic(model, ports, frequencies_ghz, amplitudes,
duration_ns, envelopes=None)` from the package root. Supply the row's fixed-flux
`SecularModel`, including segment shifts. For the producer of `segment_shifts`,
see [FTF cavity drives](../../simulators/ftf/environment/README.md#cavity-drives).
Each tuple describes two tones in the same order. Port names may repeat.
Frequencies are finite signed GHz, gains are finite signed dimensionless
values, and duration is finite nonnegative ns.
Optional `EnvelopeSpec` values describe the complete envelopes. The default is
unity. Full-envelope peak and slope bounds apply even when the row slices a
pulse. Phase, offsets and initial density do not affect these indicators.
Explicit collapse operators, unknown ports, invalid tuple lengths, nonfinite
values and negative duration raise `ValueError`.

The frozen `CommonFrameDiagnostic` reports `accepted` and `reason` from the
numerical guard. It does not predict which family wins dispatch pricing.
The comparisons do not depend on the Engine's per-segment budget.

| Field | Value and units |
| --- | --- |
| `denominator_min_ghz` | Smallest absolute SW denominator, GHz. Must exceed 1e-10. |
| `coherent` | `RWAGuardMetric` for the dimensionless sum of `second`, `third`, `fourth`, `tail` and `derivative`, required ≤ 0.0018. |
| `bath` | `RWAGuardMetric` for dimensionless oscillatory bath dressing plus secular T·gamma·eta², required ≤ 0.01. |
| `eta` | Maximum absolute peak-weighted SW generator matrix row sum, dimensionless. |
| `second`, `third`, `fourth` | Omitted-channel integral indicators at each order, dimensionless. |
| `tail` | Fifth-and-higher BCH norm tail without oscillatory cancellation, dimensionless. |
| `derivative` | Envelope-slope omission indicator, dimensionless. |

To investigate a rejection (search term `common_frame rejected`), read the
diagnostic's `reason`: `accepted`, `carrier`, `denominator`, `coherent` or `bath`.
A zero mean carrier returns `accepted=False` and leaves all numeric fields
`RWAUncomputed`. A denominator rejection exposes its minimum but leaves the
later indicators uncomputed. A coherent rejection still exposes bath and all
components because admission computes them together. Computed values do not
imply that every comparison participated in the rejection.

The [common-frame owner](kernels/common_frame.py) builds one temporary coefficient
product and calls the same row-local admission used by the guard once.
The query does not evolve a density, change inputs or routing history, retain
preparations, or compute extra indicators. These calibrated proxies do not
certify full-density accuracy. The coherent limit is 0.0018; the bath limit
remains 0.01. A rejected pinned row raises `NoValidKernel`. Default dispatch
compares the remaining admissible families by price.

For matrix coverage of shifted spans, see [Reference solver limits](#reference-solver-limits).

To read recorded routes and guard decisions (search term `inspect-dp54-summary`),
use the [simulator capture inspector](../../../../tools/README.md#simulator-capture-inspection).
A recorded `rejected` decision does not include the reason. The inspector does
not reevaluate the guard.

## Engine capture

`CaptureBuffer()` is a caller-owned in-memory recording. Pass it to
`Engine.evolve(..., capture=buffer)` and read `buffer.records` after return or
failure. The default `capture=None` does not freeze inputs or collect routes.
Capture buffering does not write files. After all evolves return or raise, call
`write_capture(buffer)` to publish the accumulated records. Explicit-path
consumers can still use `write_frozen_manifest(path, buffer.records)`.

Each evolve gets a fresh random `capture_id` and appends one `FrozenRecord`
per input batch, including empty batches, before prefix execution or dispatch.
Names are `<capture_id>-<batch_index>`. A buffer can accumulate multiple calls.
`CaptureBuffer.recording(source, batches, budget, families)` owns the Engine's
recording scope and yields the call's metadata list. Other callers should pass
the buffer to evolve instead. Direct scope entry only freezes inputs, produces
no route events, and still marks complete on normal exit. The method sorts
events and releases buffer ownership on every exit.
A concurrent call using the same buffer raises `ValueError` at entry, before
waiting for an Engine lock. `records` is a tuple snapshot. Metadata dictionaries
are not deeply frozen; callers must not modify them.

Read a saved capture with `records = read_frozen_manifest(path)`, then call
`read_capture_metadata(record)` for each record. The latter returns
`CaptureMetadata`. Non-dict metadata, missing fields, wrong types or Literal
values, and extra fields at any depth raise the original
`pydantic.ValidationError` with field paths. It neither supplies defaults nor
checks cross-field relationships, solver conditions or guard numerics. The
frozen codec still treats metadata as opaque and does not construct models.
The [capture inspector](../../../../tools/README.md#simulator-capture-inspection)
reports these observations and joins same-acquire telemetry without solving.

`CaptureMetadata` defines the opaque JSON metadata. It contains `capture_id`,
zero-based `batch_index`, `batch_count`, `budget`, sorted `families` or null,
the frozen DC model's `source_identity`, `segment_numeric_version` as a string,
`frozen_physical_version` as an integer, `complete`, `routes`, `prefix_runs` and
`sampled_rwa_admissions`.
Normal return marks all records for that call complete. Freezing, numerical and
cancellation errors propagate unchanged. Already frozen batches retain
`complete=false` and observed events; a batch that failed to freeze has no record.

Each `CaptureRoute` describes one actual classification round. It has caller
`point` and split `segment`, absolute `start_ns`, `duration_ns`, `round` as caller
or worker, `prefix_run` as an index or null, the selected `family` or null, and
ordered `{family, guard}` candidates. Guards use the same four decisions as
inspection, including `untested_after_short_circuit`. Caller deferral and worker
resolution remain separate events. Capture does not rerun guards, call inspection
or add `Kernel.cost` calls. Route events contain neither costs nor guard certificates.

Each `CapturePrefixRun` names the representative group row after a shared prefix
and all adopting `{batch, point, segment}` rows, including itself. It lives in
the representative batch's record. Only the representative timeline has prefix
events. Adopters reuse that input; they do not claim a second execution.
Events list prefix runs in Engine order, then the main run. Within each run,
caller events precede worker events. Each round follows input point order and
segment order, independent of worker scheduling.

`sampled_rwa_admissions` lists the sampled amplitude groups considered for
RWA. Each entry lives in the first member's batch. Its `rows` and `spots` use
`{batch, point, segment}` coordinates, even when a group spans batches.
Each completed spot records `max_density_error`, the maximum elementwise
absolute difference between the RWA and CPU DOP853 probe densities.
`accepted` and `rejected` report the numerical guard, not the final route.
If no comparison completed, `cost_skipped` reports a losing price and
`not_probed` means the group could try RWA. Both have null spot errors.
Capture records existing results, not probe densities.

The physical codec preserves `batch.segment_shifts`, so reconstruction with
`read_physical_input(record.physical)` restores those corrections. Engine
currently registers shifts only through these batch inputs. The source registry
itself is not serialized. A future registration path outside batch corrections
must add an explicit non-reconstructibility marker. Capture supplies no reference
error. See the [capture discussion](../../../../docs/discussions/2026-11-17-engine-capture-events.md)
for the lifecycle and event-log choices.

## Capture persistence

`write_capture(buffer, *, metadata=None)` snapshots records once
and publishes a new `capture-<uuid4 hex>.json.gz` file. Rewriting the same
buffer keeps earlier records in a different file. An empty buffer produces
a zero-record manifest, not None. Read it with `read_frozen_manifest(path)`.
Optional JSON `metadata` describes the whole capture and is readable in the
manifest. `read_frozen_manifest` returns only the Engine records and requires
the current physical version.

`capture_directory() -> Path` returns the absolute
`<user-cache-root>/simulator-debug/v2` directory without creating it.
The writer and Numba use the same resolver. Set `ZCU_TOOLS_CACHE_DIR` before
use to select another non-empty absolute cache root. Without an override,
platformdirs selects the root, with the system temporary directory as fallback.
The writer leaves old files untouched. To inspect a v1 capture, run the v1
checkout's inspector and pass the path to that capture.

The writer serializes into a same-directory temporary file and publishes it
with `os.replace`. It attempts to remove temporary files on failure.
Invalid cache configuration and errors reading records propagate before
publication. Publication OSError and codec ValueError/TypeError emit at most
one RuntimeWarning per call, then return None, even under warnings-as-errors.
Nonfinite and unsupported JSON values cannot be recorded. Other errors propagate.

The writer has no automatic size limit, truncation or eviction. Files persist across
tasks until the user deletes them, for example by unlinking selected paths from
`capture_directory().glob("capture-*.json.gz")`.
During acquire, memory grows with the recorded evolves and their frozen inputs
and events. Publication also allocates the full canonical JSON and gzip bytes.
The [environment switch](../../simulators/README.md#debug-capture) publishes once
per acquire and links the basename to telemetry.
The [writer discussion](../../../../docs/discussions/2026-11-17-capture-publication.md)
records the API and failure-policy choices.

### Capture measurements

A single offline batch on 2026-11-17 measured revision `eb095cfcc77d213b728bf1f7662631e3b229960d`,
whose tree was `d38ca3f84ae7594d74822c3727f790eb96d4ddaa`.
The runner held both CPU slots, CPU union 0–7. The payload and all fresh children
used CPU affinity 0–3, one native thread, 8 GiB memory and zero swap.
MemAvailable exceeded 12 GiB before and after admission. Payload elapsed was
30.56 s under a 600 s bound. Queue time was 383.85 s and was not sample time.
No hardware, GUI, reference solver or shared scene cache participated.

The tiny Engine scene used three levels, one four-point DC batch and one empty
batch, with two evolves per unit. All modes used one worker, t4 and telemetry.
A disabled capture, B recorded in memory, C also published one manifest.
Each cold sample used a fresh process and empty private Numba cache. Imports
and scene construction stayed outside the timer. Cold measured the first Engine
calls, not application startup. Each mode has only one cold observation.

| Mode | Cold wall ms | Cold process CPU ms | Warm wall median ms, range | Warm process CPU median ms, range |
| --- | ---: | ---: | ---: | ---: |
| A, off | 327.396 | 288.432 | 5.726, 5.539–5.777 | 4.786, 4.720–4.850 |
| B, memory | 327.072 | 287.516 | 5.708, 5.622–5.826 | 4.984, 4.916–5.147 |
| C, published | 348.389 | 307.809 | 7.149, 6.765–7.395 | 5.885, 5.769–6.021 |

The resident process warmed each mode once, then ran ABC/CBA twice, four samples
per mode. History clearing and density/manifest checks stayed outside the timer.
B minus A was −0.017 ms wall, −0.31%, and +0.197 ms CPU, +4.12%.
The overlapping wall ranges do not establish a memory-recording speedup.
C minus A was +1.423 ms wall, +24.84%, and +1.099 ms CPU, +22.95%.
Each C file contained four records, two evolve IDs and two empty batches.
Warm gzip sizes were 1,018–1,021 bytes, with 9,327 uncompressed JSON bytes.
The cold C file was 1,019 bytes. All compared densities were identical.

Capacity samples used synthetic source/binding acquire inputs, not program
compilation or user YAML. FTF had 32 points, single 64, six 128. Six was the
largest grid in this batch, with every source seeing all 128 mux points.
All had two read windows, two rounds and two shots. FTF retained two levels;
single and each six source used two levels from a six-state native basis.
Each environment used ACCA order in four fresh processes, two samples per mode.
A complete capture-off warmup preceded each sample. A fresh environment prevented
warmup preparation reuse. Before timing, each process wrote 5 to its own
`/proc/self/clear_refs`, verified VmHWM reset to current RSS, then read VmHWM
immediately after acquire and before codec validation. This excludes historical
import/JIT peaks and captures short-lived JSON/gzip allocations without polling.

| Environment | A/C mean wall ms | A/C mean process CPU ms | A peak RSS MiB, range | C peak RSS MiB, range | Mean C−A peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| FTF, 32 points | 159.285 / 170.540 | 158.475 / 169.687 | 273.121–276.965 | 277.199–277.289 | +2.201 |
| Single, 64 points | 31.872 / 33.942 | 31.725 / 33.801 | 445.422–445.523 | 445.648–445.699 | +0.201 |
| Six, 128 points | 337.469 / 359.277 | 336.054 / 357.806 | 447.887–447.961 | 449.062–449.137 | +1.176 |

Reset baseline RSS ranges in MiB were A/C 273.121–276.965 / 277.043–277.098
for FTF, 445.418–445.520 / 445.645–445.695 for single, and
447.879–447.934 / 447.984–448.047 for six. Per-process peak above baseline was
A/C 0 / 0.156–0.191 MiB for FTF, 0.004 / 0.004 MiB for single, and
0.008–0.027 / 1.078–1.090 MiB for six. The absolute C−A comparison includes
process-baseline differences. It is not an isolated allocation estimate.

| Environment | Records / evolve IDs per file | Gzip bytes, range | Uncompressed JSON bytes |
| --- | ---: | ---: | ---: |
| FTF | 64 / 64 | 9,281–9,285 | 148,561 |
| Single | 2 / 2 | 1,697 | 33,935 |
| Six | 12 / 12 | 12,430–12,437 | 387,705 |

Every file decoded, all recorded evolves were complete, capture_file matched
telemetry, and A/C output fingerprints agreed. Reused round preparations did not
invent extra evolve records. The largest gzip was 12,437 bytes and the largest
mean peak increment was 2,308,096 bytes, below the 256 MiB and 2 GiB review
thresholds. These small synthetic samples do not bound user workloads or certify
numerical accuracy. Different process baselines and two memory samples per mode
limit the peak comparison. V1 retention still requires manual deletion.

## Joint-Fock prototype

`joint_fock_spectrum(JointFockRequest(...))` is an independent numeric entry.
It does not connect to acquisition or ADC, and it does not change Engine's
flat or shaped routes. Request fields and refusal conditions are documented in
[joint_fock.py](joint_fock.py). All units are GHz, ns and radians.

The main path retains physical states [0,2] and two quantum cavity modes.
It constructs the finite-pi and ground preparations with analytic one-jump
prefill. It rejects upward rate times prefill duration above 1e-12.
Each rf shares that preparation across both branches and all probe carriers.
The common displaced cavity basis retains n1+n2<=N. Sampled coherent pointers
select the smallest N with Poisson tail <=1e-3, then add one excitation.
The main scan starts at prefill end because it does not propagate the earlier
vacuum. Constant CSR generators use fixed RK4 steps no larger than 1 ns.

An explicit `JointFockDiagnostic` changes preparation to numeric physical
vacuum, finite-pi, prefill and probe. It retains [0,2] or [0,1,2,3].
Its `step_ns` sets the fixed RK4 step upper bound for every numeric segment.
It defaults to 1 ns and accepts only finite values with 0 < step_ns <= 1.
Each segment takes ceil(duration / step_ns) equal steps, so its actual step
never exceeds the request. Zero-duration segments leave the state unchanged.
The analytic-preparation path keeps its 1 ns upper bound.

The diagnostic builds RWA coefficients on the full supplied `SecularModel`, including
virtual sums, before selecting dynamic indices. Its cutoff scan starts at t=0
to include displaced physical vacuum. Both paths report the cutoff inventory,
probe-start and final density diagnostics, and the finite vacuum representation's
trace and edge population. These scans cover sampled coherent [0,2] no-flip
and one-jump paths, not every multilevel jump history. They do not bound the
complete density's truncation error. Differences between the paths include
preparation and cutoff selection, so they are not pure truncation errors.

The prototype does not clip populations, repair densities or retry another
solver. It does not reduce the requested step automatically. Nonfinite evolution
raises `FloatingPointError`; callers must choose any smaller step explicitly.
Fixed-step RK4 does not certify positivity or accuracy at arbitrary drive and
frequency scales. The small-model tests
compare analytic Rabi and relaxation, an independent physical-frame Lindblad
ODE and a virtual shift from a state outside the dynamic projection.
Full-map accuracy and wall measurements are separate validation work.

## Routing telemetry

`Engine.evolve(..., telemetry=True)` adds immutable `FamilyRouting` values to
`result.paths.routing`. The default is False, with no thread CPU clock reads and
an empty mapping. Telemetry does not write files. OpenCL device detection and
program builds can access the simulator user cache. `paths.frontend_cpu_s` separately
reports calling-thread geometry and WorkPlan CPU.
`prepare_builds` counts common-frame coefficient products or Floquet exact-key
frame construction attempts, including invalid frames and frames whose rows
are all refused. Other families leave `prepare_builds` at zero.
`admission_rows` counts common-frame row indicator evaluations or completed
sampled spot comparisons for `multitone_rwa`. Both count refused rows, but not
apply or reuse hits. Sampled row proxies and cost skips do not count.
Other families leave `admission_rows` at zero.

`prepare_key_builds` counts key slots passed to scheduled preparation or returned
as guard products, including discarded guard products. Building the same key
again counts again; a guard without a returned product does not count. Several
key slots can share spectral work, so this counter does not count spectral
constructions. For `multitone_gate`, it counts completed numerical builds;
later consumers of the same evolve-local key do not increment it.
Merge adds key builds, and result reuse reports zero.
These key counts are available in the Engine report; acquire JSONL keeps its
[explicit field set](../../simulators/README.md#acquire-telemetry).
Telemetry does not change guards, routes, densities or cancellation.

`PathDiagnostics.duration_ns_by_kernel` always reports logical physical ns per
family, even with telemetry off, without CPU clocks. Its immutable mapping
has the same keys and consumers as segment counts. Merge adds lengths and
result reuse preserves them. Enabled `FamilyRouting.duration_ns` reports the
same logical length, including every adopting timeline for a shared prefix.

Each family reports logical segment `rows`, `duration_ns`, `prepare_cpu_s`, `apply_cpu_s`,
`actual_cpu_s` (their sum), `cache_hits`, `rejected_rows`, `estimated_cost` and
nullable `estimated_cpu_s`. Calling-thread CPU measures guards/certificates,
prepare, input binding and apply, in the thread
performing each call. It excludes geometry, dispatch bookkeeping, ADC and I/O.
Rejections count actual numerical guard refusals, not cost-based skips or planner
feasibility probes. A shared prefix executes once although logical rows include
all adopting timelines. Caller-retained complete preparation hits report logical
rows as `cache_hits`, with zero CPU, rejected rows and estimated work.
Engine evolves always execute numerical work. For `dop853`, `opencl_rows`
counts endpoints computed on GPU and `opencl_detection_error` reports why
automatic detection or compilation for the active dimension and tableau failed.
It is `None` when neither failed. `dp54_rows` counts physical DP5(4) endpoints;
`dop853_rows` counts physical DOP853 endpoints. Both counts include CPU and GPU.
Both tableau counts add across merged reports and are zero on exact-result reuse.
A shared prefix contributes one execution, not one execution per consumer.
The paired benchmark reads these counts from Engine results. The
[acquire routing JSONL](../../simulators/README.md#acquire-telemetry) does not store them.

`PathDiagnostics.execution_device` is read-only and available with telemetry
off. It is `cpu` if no row ran on GPU; otherwise it contains the OpenCL device
name, driver version, kernel version and pyopencl version. Merging CPU and GPU
parts keeps the GPU identity; merging different GPU identities raises
`ValueError`. Exact-result reuse keeps the device that produced the result.

`estimated_cost` uses **relative scheduling units**, not seconds. The existing
constant two-tone plan supplies seconds estimates for selected Floquet/DOP853
rows. Other rows have `estimated_cpu_s=None`; mixed-plan totals are not assigned
to a single family. Costs and CPU describe physical execution, so prefix costs
are counted once. A family with only rejected work can have zero selected cost.

## Decision graph

[Geometry construction](_planning.py) partitions pulse, flux and readout events
into rows. [Dispatch](_dispatch.py) selects candidates from flux motion, active
tones, envelope shape, known inputs and acquire-wide prepare-key reuse, not
experiment names. Numerical guards decide validity separately from cost.
A rejected guard leaves the row for the next candidate.
Floquet admission is per row. With the default family set, a refused row
continues to another candidate while admitted rows keep their Floquet plans.
A fixed `families` set can remove that fallback. If any row then has no admitted
candidate, the whole evolve call raises `NoValidKernel` without partial results.
This does not move Floquet ahead of `multitone_rwa` in the candidate order.

For plateau rows, default dispatch tries `single_tone_rwa` before
`floquet_period` without comparing their costs. On the L and I workloads,
kick-enabled `single_tone_rwa` is more than 10% slower than `floquet_period`.

`Engine(..., families=("t4", "dop853"))` restricts every segment in the acquire,
including shared prefixes and deferred timeline workers, to those families.
Pass a collection of family names, not a per-segment mapping or preference list.
The constructor copies the names and raises `ValueError` for unknown names.
`None` preserves default routing. An empty collection rejects any nonempty
domain. Empty work still returns an empty result.

Dispatch keeps its cost ordering among allowed candidates. If cost shortcuts
skip an expensive allowed candidate, dispatch tries it before rejecting the
domain. Dispatch excludes rows that violate a family input precondition, such
as common-frame carriers with an absolute mean below 1e-10 GHz. The kernel
owner defines each precondition once. Legal preparation failures still propagate;
dispatch does not catch them to retry another family.
The allowlist never changes numerical guards, budget or complete-acquire
inventory counts. A rejected pinned family cannot select an excluded family.
If no allowed candidate passes, `NoValidKernel` names the domain, caller point
IDs and affected segment rows. Its `attempted` tuple lists the allowed candidates
in the final dispatch round's offer order, including cost-skipped families.
An empty tuple means none of the allowed families is a candidate for those rows.
The allowlist does not add fallback routes to other domains. `kernels=` remains
an internal numerical-collaborator seam.

The family names are `t4`, `adiabatic_map`, `flux_dop853`, `slow_envelope`,
`dyson`, `single_tone_rwa`, `floquet_period`, `t6`, `single_tone_gate`,
`shaped_sambe`, `gate_cache`, `period_table`, `common_frame`, `multitone_gate`,
`multitone_rwa`, `floquet` and `dop853`.

No-drive rows use secular propagation or adiabatic transport. Driven rows use
constant-tone, shaped-tone or continuous-flux families. Candidate order,
thresholds and algorithm details live with their owners:

| Concern | Owner |
| --- | --- |
| Row domains, named candidate order and terminal routes | [dispatch](_dispatch.py); [kernel collection](kernels/__init__.py) instantiates implementations only |
| Shaped and two-tone cold-cost choices | [shaped dispatch](_shaped_dispatch.py), [two-tone dispatch](_two_tone_dispatch.py) return family names, not positional indices or unnamed flags |
| Multitone gate amortization | [multitone dispatch](_multitone_dispatch.py) compares cold per-key gate prices with eligible direct/RWA alternatives, using complete-acquire canonical-key counts |
| Sampled multitone RWA pricing | [sampled RWA dispatch](_sampled_rwa_dispatch.py) prices fixed probes and playback; it retains the prior CPU and GPU wave comparison outside the overlap |
| Overlapping sampled gate and RWA offers | [shared multitone dispatch](_shared_multitone_dispatch.py) compares complete amplitude groups in CPU seconds and rechecks build amortization after RWA probes |
| Dissipative DC channel pricing | [slow dispatch](_slow_dispatch.py) compares M⁶ channel construction amortized over complete-acquire exact prepare keys with DOP853 |
| Slow/plateau carrier boundary | [slow-envelope comparison](kernels/slow_envelope.py) owns the strict absolute-carrier comparison shared by dispatch and its guard |
| Common-frame pricing and numerical construction | [common-frame dispatch](_common_frame_dispatch.py), [kernel](kernels/common_frame_kernel.py) |
| Constant and shaped period builders | [Floquet period](kernels/floquet_period.py), [period table](kernels/period_table.py), [shaped Sambe](kernels/shaped_sambe.py), [gate cache](kernels/gate_cache.py) |
| Two-tone Floquet admissions and variants | [Floquet](kernels/floquet.py) |
| Shared RWA admission, retention and Hamiltonian coefficients | [RWA numerics](kernels/_rwa_numeric.py) owns policy, model/WorkPlan adapters and facts; [Fourier numerics](kernels/_rwa_fourier.py) owns the coefficient record and compiled Hamiltonian assembly. Family playback remains in [multitone RWA](kernels/multitone_rwa.py) and [single constant RWA](kernels/single_tone_rwa.py). |
| Single-tone covariant gate admission, splitting and reuse | [single-tone gate](kernels/single_tone_gate.py), [RWA builder](kernels/_rwa_gate.py), [phase graph](kernels/_rwa_graph.py) |
| Full-drive dual-phase gates | [multitone gate](kernels/multitone_gate.py), [phase grid](kernels/_multitone_gate_phase.py), [Hilbert endpoints](kernels/_gate_endpoints.py) |
| Direct integration and chronological chains | [DOP853 planning](_dop853_dispatch.py), [DOP853](kernels/dop853.py), [flux DOP853](kernels/flux_dop853.py) |

Cold-cost choices use the complete acquire inventory rather than resident cache
warmth. Prefix and worker subsets borrow complete-acquire counts. A shared
prefix runs on the CPU. `multitone_gate` charges one cold construction per key:
30 full-drive endpoint nodes and Fourier fitting. Dispatch divides that price
by current-acquire key uses, then adds per-row key work and playback. The price
must beat an eligible alternative before preparation. Constant-pair RWA keeps
its history-based cold price. Neither gate amortization nor sampled RWA probe
pricing reads history.

The estimator uses the raw full-drive work scale for gate endpoints and the
grid size for Fourier and playback. The calibrated CPU
`dop853` competitor multiplies the raw scale by 0.0891 through 25 ns, then
linearly increases the factor to 0.412 at 100 ns and holds it constant.
The calibration uses seven CPU DP5(4) measurements at budget 1e-3, including two
DRAG windows. The factor does not change gate work, the independent RWA probe
estimate, or the direct solver. See the
[pricing discussion](../../../../docs/discussions/2026-11-18-gate-direct-duration-price.md).

Insufficient current reuse can leave a structurally eligible gate unbuilt.
These are CPU estimates, not wall-time or accuracy guarantees.

For a proxy-eligible amplitude group with an allowed gate candidate and no
eligible original GPU route, dispatch compares total CPU seconds, without
dividing by worker count. The original route excludes sampled RWA and gate.
If the existing common-frame offer and guard accept a row, the original price
is the common-frame price. Otherwise it is the calibrated CPU `dop853` price
when allowed. The RWA offer includes one set of spot checks and all group
playback. The gate offer takes the cheaper of the gate and original costs for
each key. Each gate key charges its group a share of the build proportional to
that group's uses. The denominator includes eligible uses in other phase
groups and outside sampled RWA groups. While pricing a CPU prefix, dispatch
excludes a suffix-only group if its independent DOP853 waves win the legacy
GPU comparison even with unresolved driven neighbors treated as direct. It
uses the normal lazy device check. Groups spanning prefix and suffix, or
requiring neighbor routes to be resolved first, retain their existing counts.

Dispatch runs probes for the groups that select RWA first. Before building a
gate, it rechecks the full build against the remaining uses. An RWA refusal
releases its group to the next offer. If any key still pays for its build,
its rows use the gate and other keys use the original route. If no key pays
for its build, dispatch can try RWA if it still beats the original route.
Once selected, the gate prepares and plays back. Numerical failures propagate
rather than switching families.

Groups without an allowed gate candidate retain the earlier RWA comparison.
When the original GPU wave is eligible and wins, the entire group uses the
earlier RWA, common-frame and GPU comparison. That path does not compare GPU
wall seconds with gate CPU seconds. A single-family pin skips price, not the
guard. See the [routing discussion](../../../../docs/discussions/2026-11-19-shared-multitone-pricing.md).

Historical calibration note for [slow-envelope propagation](kernels/slow_envelope.py):
the calibrated population tolerance is 1e-2, with about 1e-4 or less measured on
21-level Q1 and a 1e-3 synthetic regression limit. It does not adapt to budget
or certify arbitrary pulse strengths and coherences.

## Kernel summary

The domains below use the names in [dispatch](_dispatch.py).
`idle` has no tones or continuous flux motion. `adiabatic` has flux motion
without tones; `flux` has flux motion with tones. At fixed flux, `slow` has one
constant tone with carrier magnitude below 30 MHz. `plateau` has one constant
tone with carrier magnitude at or above 30 MHz. `shaped` has one sampled tone. `two` has two constant tones. `multiple`
has two tones with a sampled envelope, or at least three tones.
A family in a domain is a candidate, not a promise of selection. Cost ordering,
input preconditions and numerical guards still apply with `families=`.

For shaped rows with `period_table` as the first allowed fallback and an exact
table key shared across timelines, dispatch resolves the `shaped_sambe` guard
before scheduling. After a refusal the scheduler can build one fallback table
for those timelines. Dispatch batches local Sambe keys that share the same
spectral work in that timeline. If a key joins the batch only for spectral reuse
and Sambe refuses it, its worker still handles the fallback. Other local Sambe
keys stay deferred. This does not change guard thresholds or candidate order.

In the table, `B` is the per-segment `budget`, `d` is duration in ns and `T` is
one carrier period in ns. Split, motion, omission and phase indicators use the
bounds defined by each linked numerical owner. They are admission criteria,
not measurements of final full-density error. RWA slope and curvature limits
are normalized by envelope peak. Retained real dimension counts populations
and both real coordinates of within-component coherences.

The matrix exercises 17 families. Small-model `multitone_gate` contracts
cover one or seven gates, fixed or independently varying dual phases, and
Gaussian, flat-top, DRAG and sampled envelopes. They also cover mixed waveform
keys, including constant+sampled pairs in one evolve. Routing contracts check
build counts, price losses and numerical failure propagation. These cases do
not qualify the full FTF sequences.
Two existing guard-edge cases remain strict xfails: `gate_cache` at
`9.296548e-3` and `shaped_sambe` at `3.091554e-3`.
The `single_tone_rwa` kappa2 inner-edge case has full-density error
`3.061429e-13` and reference convergence difference `8.564902e-13`.
Its two-level diagonal drive makes kappa2 equal to gain, set just below 0.08.
The public diagnostic agrees with guard admission on both sides of that edge.
This focused case replaces the older retained-phase edge and is separate from
the real-workload calibration below. Representative inputs are pinned by
[`test_pinned_family_density`](../../../../tests/simulate/evolution/contract/test_accuracy_matrix.py);
[`test_guard_edge_density`](../../../../tests/simulate/evolution/contract/test_accuracy_matrix.py)
pins the [guard-edge inputs](../../../../tests/simulate/evolution/contract/guard_edge_cases.py).
The same named inputs also run with
[default empty and warm routing history](../../../../tests/simulate/evolution/contract/test_accuracy_matrix.py);
those runs measure the families actually selected, not every candidate.
The [matrix instructions](../../../../tests/README.md#evolution-精度矩陣)
own the command and report format.

Sampled dual tones that pass the row proxies form `multitone_rwa` groups whose
members differ only in each tone's amplitude. A group matches model, duration,
ports, carriers, effective start phases and pulse-local offsets exactly.
Sampled envelopes must share a native grid and have exactly proportional I/Q
samples, with no phase rotation or tolerance. The guard probes the largest
amplitude for each tone, checking a row only once if it has both maxima.

Dispatch pricing includes RWA playback and caller-thread spot checks, each
charging a fresh RWA build and apply plus full-drive CPU DOP853 at the fixed
probe tableau. The direct playback alternative is priced with its production
tableau. Dispatch skips the checks when an eligible alternative is at least
as cheap. An RWA-only allowlist skips pricing, not the guard. Price
amortization and guard decisions do not carry over to another evolve.

The guard evolves the same fixed coherent probe with RWA and full-drive CPU
`dop853` at the requested budget. Every probed row must have maximum absolute
full-density difference ≤ 0.1 × B. If any fails, only its group falls back
to other candidates such as `common_frame` or `dop853`. This empirical check replaces the
[2026-11-14 interim rejection](../../../../docs/discussions/2026-11-14-sampled-multitone-interim.md)
for sampled RWA, but does not certify arbitrary input states or accumulated
sequence error. The matrix covers rejected phase-varied sequences and short
32-level frequency scans.

`multitone_gate` uses no fixed reuse threshold. See the [decision graph](#decision-graph)
for how dispatch prices eligible uses and compares the gate with sampled RWA
and the original route.

| Family and use | Dispatch domain | Approximation and numerical admission | Pinned accuracy cases | Full density ≤ 2e-3 |
| --- | --- | --- | --- | --- |
| [`t4`](kernels/t4.py), static idle evolution | `idle` | Exact secular population semigroup and coherence phases/decay. Guard requires zero tones; no approximation threshold. | [`idle_coherence`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |
| [`adiabatic_map`](kernels/adiabatic_map.py), undriven flux transport | `adiabatic` | Preserve ordered populations, integrate dynamical/gauge phases and decay coherences. Requires waveform samples. Quadrature doubles from 4 to 4096 nodes per knot interval until phase-factor change ≤ B/10; overlap < 0.25 forces refinement. This does not check nonadiabatic transitions, degeneracy or crossings. | [`adiabatic_ramp`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |
| <a id="flux-dop853"></a>[`flux_dop853`](kernels/flux_dop853.py), driven continuous flux | `flux` | Integrate full H(flux(t)) with a fixed quantized secular bath per row, stopping at active flux/envelope knots. Requires tones and waveform samples. Local rtol = min(1e-4, 0.03B), atol = min(1e-6, 0.0003B), from the [shared solver](kernels/_dop853.py). | [`continuous_flux`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |
| [`slow_envelope`](kernels/slow_envelope.py), low-carrier constant drive | `slow`, carrier magnitude strictly below 30 MHz | Nonzero carriers use midpoint-held full Hamiltonians with Strang bath half-steps, default 20 holds per period. Dissipative DC uses the exact constant Lindblad channel without splitting, storing O(M⁴) complex values for M levels. Bath-free DC uses one unitary. Dispatch and kernel guard share the strict absolute-carrier < 30 MHz comparison; exactly ±30 MHz belongs to plateau. No budget adaptation; historical population calibration is separate from this density measurement. | [`low_carrier`, `dephasing_pump`, `raman_decay`, `raman_weak_legs`](../../../../tests/simulate/evolution/contract/accuracy_cases.py); [boundary contract](../../../../tests/simulate/evolution/contract/test_routing_boundaries.py) | PASS |
| [`dyson`](kernels/dyson.py), weak constant drive | `plateau` | Second-order full-carrier modal Dyson. Calibrated guards require r ≤ min(0.1, 0.25B^(1/3)), r2 ≤ 0.05, and second-order population indicator B2 + B/8 ≤ min(0.02, (1.5B)^(2/3)). Certified shared no-drive prefixes omit the B2 check. B/8 reserves path pruning; this is not a multilevel remainder proof. | [`weak_plateau`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |
| [`single_tone_rwa`](kernels/single_tone_rwa.py), constant rotating-frame block evolution | `plateau` | Fixed flux, B ≥ 1e-3, carrier magnitude ≥ 50 MHz. Initial 50 MHz edge window with first/second-order population-transfer restoration at 5.8e-3, including forward edges and adjoints. Final admission uses kappa2 = spectral norm of sum_sign abs(c/d) over the resolved omitted edges, ≤ 0.08, in place of the remaining transfer comparison. Full-operator/carrier ratio ≤ 0.25 and retained-norm phase proxy ≤ 0.06 rad. Phase-graph pruning ≤ min(B, 1e-3); real generator dimension ≤ 64. First-order endpoint kicks include every edge outside the resolved retention mask, including diagonal ±carrier terms; phase-graph pruning is separate. Cross-component density entries below 1e-5 after the initial kick are dropped. Uses the full secular bath without Strang splitting. | [`linear_plateau`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS at kappa2 inner edge: 3.061429e-13 |
| [`floquet_period`](kernels/floquet_period.py), full-carrier constant-tone period reuse | `plateau` | Hilbert period trajectories with secular split playback. Requires carrier magnitude ≥ 0.03 GHz and some subdivision count m in 1..64 with dT²ΓV²/m² ≤ 1.0, using this owner's rad/ns drive bound. Below 1 GHz, CF4 or adjacent-cutoff Sambe uses endpoint defect ≤ min(1e-4, 0.1B). Higher carriers use DOP853 rtol min(1e-8, 1e-4B), max step T/4. These are calibrated local controls. | [`weak_plateau`](../../../../tests/simulate/evolution/contract/accuracy_cases.py), [`strong_plateau`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS; largest family error 1.375101e-3 |
| [`t6`](kernels/t6.py), full dissipative constant-tone evolution | `plateau`, terminal candidate | One constant tone at any strength. Direct integrates each density in the H0 frame with the full secular bath and the shared dop853 policy: local rtol = min(1e-4, 0.03B), atol = min(1e-6, 0.0003B), without a fixed step ceiling. These are not whole-sequence error bounds. All rows use direct integration, irrespective of key reuse. It does not do RWA or density repair; numerical failure raises rather than reroutes. | [`weak_plateau`](../../../../tests/simulate/evolution/contract/accuracy_cases.py), [`strong_plateau`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |
| [`single_tone_gate`](kernels/single_tone_gate.py), phase-covariant shaped RWA gates | `shaped` | Fixed flux, B ≥ 1e-3, real sampled envelope with at least 3 samples, slope ≤ 0.025/ns, curvature ≤ 0.003/ns², carrier magnitude ≥ 50 MHz. Omission restoration requires transfer ≤ 0.0058, residual phase ≤ 0.0045 rad and retained real dimension ≤ 256. Pruning ≤ min(B, 1e-3); adaptive secular split uses n ≥ 4 with d³ΓV²/n² ≤ 0.01. [Shared gate owner](kernels/_rwa_gate.py). | [`linear_shaped`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |
| [`shaped_sambe`](kernels/shaped_sambe.py), frozen-envelope finite-Fourier periods | `shaped` | Nonzero peak, carrier magnitude ≥ 0.03 GHz, subdivisions m ≤ 32, envelope slope ≤ 2·peak·abs(f)·m and peak drive bound ≤ 0.5·abs(f)·m. Subdivision policy limits real/complex motion to 0.01/0.0025 and secular split to 0.01. Compare adjacent harmonic cutoffs 2/3 or 3/4 and use 4 or 8 amplitude nodes; accumulated spectral plus interpolation estimate must be ≤ B/2. | [`linear_shaped`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | FAIL at guard edge: 3.091554e-3; strict xfail |
| [`gate_cache`](kernels/gate_cache.py), phase-interpolated full-carrier shaped gates | `shaped` | Four Hilbert subsegments with secular half-steps. Model-only secular split d³ΓV²/16 ≤ 0.01. For 0 < d ≤ 30 ns, dispatch tries 6 phase nodes with 2 probes and full-unitary defect ≤ min(1e-4, 0.1B); other keys or failed convergence use 16 nodes. Fixed-order playback does not certify arbitrary-drive error against B. | [`shaped_complex`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | FAIL at guard edge: 9.296548e-3; strict xfail |
| [`period_table`](kernels/period_table.py), shaped amplitude-band period playback | `shaped` | 8 Chebyshev amplitude nodes and 16 phase samples. Guard requires envelope-motion indicator ≤ 0.05, amplitude-band bandwidth ≤ 0.5 and secular split ≤ 0.01. These calibrated limits do not scale with B. DC is guard-admissible with local direct integration, but default costs prefer terminal DOP853. | [`linear_shaped`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |
| [`common_frame`](kernels/common_frame.py), common-frame SW2 beat dynamics | `two` or two-tone `multiple` only | Fixed-flux common frame with second-order beat/two-photon terms, first-order endpoint dressing and CF4/secular playback. Absolute mean carrier must be ≥ 1e-10 GHz. CF4 steps are capped at 8 ns and 0.08 cycles of the fastest retained channel, with no lower cutoff. Dispatch prices first-order playback steps, so cost rises with active carrier spacing; numerical guards determine applicability. Fast-channel denominator must exceed 1e-10. Row-local coherent/bath indicators must be ≤ 0.0018/0.01, independent of B. | [`nearby_carriers`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS at the new coherent guard edge |
| [`multitone_gate`](kernels/multitone_gate.py), full-drive dual-phase gates | `multiple` | Two tones, fixed flux, positive duration, at least one sampled envelope. Four D(h/2)·U·D(h/2) quarters with complete drive, no RWA or pruning. Each key fits one 5×6 Fourier grid without refinement, at endpoint rtol/atol 2.75e-7/2.75e-9. Keys omit start and both phases. Dispatch compares cold build and playback costs before construction. Dispatch builds each selected key at most once per evolve and uses that product without a numerical admission probe. Numerical failures propagate without fallback. A pin skips pricing, not structural checks. | [Four envelopes and mixed waveform keys](../../../../tests/simulate/evolution/contract/test_accuracy_matrix.py), [per-key routing](../../../../tests/simulate/evolution/contract/test_shared_multitone_routing.py) | Small-model contracts; long-sequence qualification separate |
| [`multitone_rwa`](kernels/multitone_rwa.py), simultaneous omission-restored RWA | `two` for distinct-port constant pairs; two-tone `multiple` | Fixed flux, B ≥ 1e-3 and each carrier magnitude ≥ 50 MHz. Restore omitted paths until transfer proxy ≤ 0.0058 and residual phase ≤ 0.0045 rad, with retained real dimension ≤ 256. For sampled pairs, slope and curvature do not reject rows. For a group selected for comparison, choose the row with maximum `abs(gain) × envelope peak` for each tone, deduplicate rows and require agreement with full-drive `dop853` at ≤ 0.1 × B. Sampled rows have a 4 ns step ceiling and no kick. Constant pairs additionally require final-mask ε_dom²·Ω_ret·T ≤ 1.476e-4 cycle and use summed first-order endpoint kicks with a 16 ns step ceiling. The [constant-pair query](#constant-pair-rwa-guard-diagnostics) reports the constant metrics. | [`joint_constant`, `joint_gate_inventory`, `cheap_sampled_nearby`, `sampled_multitone_short_sweep_32`](../../../../tests/simulate/evolution/contract/accuracy_cases.py); [sampled spot edge and phase sequence fallback](../../../../tests/simulate/evolution/contract/test_accuracy_matrix.py) | PASS for admitted rows; rejected groups use fallback |
| [`floquet`](kernels/floquet.py), constant Stark tone plus probe | `two` | Larger absolute gain selects the Stark slot, with ties choosing the first; that slot requires positive carrier. Frame overlap ≥ 0.5, modal condition ≤ 1e8, maximum real eigenvalue ≤ 1e-10/ns and residual ≤ 1e-9; sampling extent ≤ 128 with at most 4096 phase samples. k=0 omission and convergence each fit B/3. Modal Dyson additionally uses r ≤ 0.1, r2 ≤ 0.05 when probe gain magnitude ≥ 0.1, Dyson norm-duration v ≤ 0.5, transfer indicator v + v²/2 ≤ 0.02 and higher-order remainder exp(v)·v³/6 ≤ B/3; otherwise use periodic channel playback. Retained-sideband admission instead requires convergence ≤ B/40 and bound ≤ 0.95B, retaining edges within 0.1/ns and averaging other harmonics through order 4. Admission is per row; refused rows continue to other allowed candidates. This finite-Fourier admission is not a density-error guarantee. | [`two_carriers`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |
| [`dop853`](kernels/dop853.py), full-drive direct density integration | `slow`, `shaped`, `two`, `multiple`, terminal candidate; `full_loss`, sole candidate | No RWA. Stops at active control knots. DOP853 local rtol = min(1e-4, 0.03B), atol = min(1e-6, 0.0003B). Secular-only CPU and [eligible OpenCL](#gpu-backend) rows with d ≤ 10,000 ns and B ≥ 1e-3 use DP5(4) at one third of both tolerances. Explicit collapse operators require fixed flux with no waveform; run CPU DOP853 even at the default budget. | [`shaped_complex`, `two_carriers`, `joint_gate_inventory`](../../../../tests/simulate/evolution/contract/accuracy_cases.py); [CPU and OpenCL DP5(4) cases](#cpu-dp54-accuracy); [complete jump density](../../../../tests/simulate/evolution/test_collapse_operators.py) | PASS |

For dissipative DC rows, [slow dispatch](_slow_dispatch.py) charges channel
construction as `3.8 * (M/32)^6` seconds, divided by the exact prepare key's
use count in this acquire. It adds `0.000422 * (M/32)^4` seconds per application.
The key includes level, shift, port, carrier, gain, duration, phase and
subdivision policy. Prior acquires and resident cache warmth do not subsidize
construction. Dispatch compares this cold price with a slow-domain calibration
of the DOP853 bandwidth/action estimate. Pinning either family preserves its
numerical guard. Bath-free DC and nonzero carriers keep `slow_envelope` first.
`Kernel.cost` remains a relative chunk-scheduling weight, not this cold price.

The calibration covers 11 synthetic plateaus at M=16, 32 and 48 and durations
of 40–350 ns. Estimates were 0.53–1.36 times measured single-thread wall time.
The default route for a 32-level 100 ns DC segment and the three small
dissipative DC matrix cases stays within 10% of the fastest admitted family,
with full-density error ≤2e-3. The 100-point shared-key case now selects DOP853.
Its paired wall-time ratio against the previous `slow_envelope` default is 0.44.
These prices are not runtime bounds.

The `common_frame` coherent limit is 0.0018. In the retained two-tone grid,
14 admitted cells had coherent indicators at most 0.001469411 and historical
full-density error at most 8.234e-4. The nearest known failure had coherent
0.002623604 and density error 2.1542e-3. The limit leaves 22.5% headroom above
the retained maximum and 31.4% clearance below that failure.
The limit applies to every `common_frame` row, including two-tone sampled rows.
The calibration grid holds constant pairs only; sampled rows have no separate
calibration, and the new limit is stricter than their former 0.03.
These are calibrated proxies, not a proof for arbitrary models or durations.
The bath limit stays 0.01; no additional comparison is used.

This also rejects five historically qualified cells: `A_x3.5_T400`,
`B_f0.25_g0.038_T1000`, `B_f0.25_g0.038_T250`, `B_f0.45_eps0.029_T400`
and `C_g0.1_T512`. Rejected rows use whichever remaining admissible family
wins the default price comparison, not a forced `dop853` route.

The common-frame matrix edge now uses the two-level linear-drive model at
0.6/0.63 GHz, 6.5 ns, and gains 14.175938164450116/8.859961352781323.
Its coherent indicator is 0.0017999982, one part per million inside the new
limit. The matrix checks the complete complex density against a two-tolerance
full-Hamiltonian Lindblad reference with the unchanged 2e-3 acceptance bound.
The earlier 8.066669e-3 edge is replaced, not retained as an expected failure.

### Reference solver limits

Both independent timeline solvers,
[lab-frame `reference_timeline`](../../../../tests/simulate/evolution/contract/reference.py)
and [interaction-picture `interaction_density`](../../../../tests/simulate/evolution/contract/interaction_reference.py),
build their dissipative terms from the secular bath's `rates_per_ns` and
`dephasing_per_ns`. On inputs both support, they integrate the same Lindblad
equation with this secular bath and retain all Hamiltonian drive terms, without
a rotating-wave approximation. Both raise `ValueError` for explicit collapse
operators in a selected bath. Neither accepts a full Lindblad superoperator.
Both also reject batches with `segment_shifts`. The matrix does not certify
shifted spans.
For the FTF producer and guard query, see [Common-frame guard diagnostics](#common-frame-guard-diagnostics).

### Constant-pair guard evidence

The [constant-pair edge test](../../../../tests/simulate/evolution/contract/test_accuracy_matrix.py)
uses two circular-drive ports with a diagonal pump component, relaxation and
dephasing. Its two phase points have an inside duration of `184.5*(1-1e-6)` ns.
The public query reports `1.475998524e-4` cycles, one part per million inside
the `1.476e-4` limit. Pinned `multitone_rwa` accepts it. The maximum complete
complex-density error is `1.014148242e-3`, against independent full-Hamiltonian
Lindblad references that agree within `4.751899e-11`. The fixed outside duration
`184.5*1.01` ns reports `1.49076e-4` cycles and is rejected by both query and
pinned Engine. The carrier guard edge uses a separate constant-pair case.

The [ground-state A16 tracer](../../../../tests/simulate/evolution/contract/multitone_cases.py)
builds a fixed 1600 ns constant pair through public physical/model interfaces.
Before the secular guard, endpoint-kicked playback admitted this input with
complete density error `7.352059944e-3`; its two reference tolerances agreed
within `3.335506316e-10`. The permanent [rejection test](../../../../tests/simulate/evolution/kernels/test_multitone_rwa.py)
checks public `NoValidKernel`, without solving a reference or loading saved
matrices. These inputs do not establish a new real-workload timing or price.

Constant-pair cold prices include an endpoint-prepare reserve and restored-edge
bandwidth work. The three restored-zero C2b controls retain the small-graph
shortcut because kicked RWA is faster than their qualified competitors.
Complete inventories and prefix/worker subsets use one small-graph predicate.
The endpoint-prepare reserve amortizes over exact RWA prepare-key consumers,
not over the broader common-frame coefficient group.
The [fixed scene declarations](../../../../tools/family_pareto_multitone_scenes.py)
add three C2b controls, a nine-point synthetic feature star at fixed M=16,
and consecutive base supports at N=20/100. Numeric payloads stay outside Git;
[the indexes](../../../../tools/data/family_pareto/README.md) pin their identities.
The star records price residuals; it does not extend the accuracy guarantee to
all constant-pair inputs. O1 comparisons retain its complete 24-point acquire
and the same `t4` idle companion for every pinned driven family.

這份證據只涵蓋列出的凍結輸入；日常 FTF 常數雙 tone 的 32、288、512 ns、1 µs 未驗證。未驗證不是不支援，也不代表算錯。

### Recorded real-workload evidence

A separate fixed-candidate acceptance run calibrated the single-constant
kappa2 threshold of 0.08 with endpoint kicks. It covered 395 inputs and
561 points with 136 references, including both eight-point RB workloads.
Of the 561 pinned attempts, 512 admitted points met the full complex-density
`2e-3` criterion. The 49 `NoValidKernel` refusals comprise 42 period points and
seven nonzero shaped-RB points. Refusals have no density comparison and do not
count as PASS or zero error. The 273 permitted `floquet_period` surrogate
attempts admitted. There were no new A3 refusals and no default attempts, so
this run supplies no new default-route observations.

The worst admitted pinned point was `period/0.4/0.1/385.0`, point 0.
Its full-density error was `0.00016965243421764126`, leaving
`0.0018303475657823589` below `2e-3`. This is a real-workload point, not the
synthetic guard-edge case. Both zero-segment RB points retained complete
density/reference comparisons without invented route or kappa2 values.

These results support 0.08 on this candidate and acceptance set, not a larger
threshold or arbitrary-input accuracy. The parent reviewed source-generated
reference identity, convergence and error metadata, and checked the saved
densities' shape, complex dtype and finite values. That review did not solve
references independently or repeat evolution. This accuracy run does not
remeasure the timings or dominance counts in the historical tables below.
Performance remeasurement remains pending.

The [thirteen recorded summaries](../../../../tools/README.md#recorded-scenario-summaries)
retain the final errors, three history modes, wall/process CPU medians, companions
and measurement provenance. Dominance compares only policies whose maximum final
whole-sequence density error is at most `2e-3`. Qualified policies are equally
accurate; another qualified candidate must be more than 10% faster to dominate.
Default/fixed groups and the six directly retained families do not compete.
Counts below use equal-companion groups and apply separately to all three modes
and both clocks. Refusals, blocked companions and FAIL cells are not dominance
evidence. These observations do not authorize deletion or guard changes.

| Family | Not dominated on recorded workloads | Evidence and limits |
| --- | --- | --- |
| `t4` | Directly retained, exclusive idle domain | Pinned policies refuse driven spans; idle companion work is not an isolated target measurement. |
| `adiabatic_map` | Directly retained, exclusive undriven-flux domain | No continuous-flux workload in these thirteen tables. |
| `flux_dop853` | Directly retained, exclusive driven-flux domain | No continuous-flux workload in these thirteen tables. |
| `slow_envelope` | Directly retained, exclusive low-carrier domain | These inputs do not exercise its domain. |
| `t6` | Directly retained, terminal plateau candidate | [N-long](../../../../tools/data/family_pareto/summaries/long_gate_sequence.md) qualifies; local budget controls are not a final-error guarantee. |
| `dop853` | Directly retained, terminal shaped/two/multiple candidate | [A](../../../../tools/data/family_pareto/summaries/gain_sweep.md) qualifies at 4.14019e-6 on remeasurement; retained does not mean accurate on every input. |
| `dyson` | No measured target | Offered plateau groups refuse; no dominance conclusion. |
| `single_tone_rwa` | 4: L, I, N, N-long | [N-long](../../../../tools/data/family_pareto/summaries/long_gate_sequence.md) qualifies at 1.24701e-3 over 50/100/200 gates. F and both bias045 tables fail in these historical measurements, not in the fixed-candidate calibration above. Timing and dominance remeasurement remains pending; no deletion is authorized. |
| `floquet_period` | 4: F, I, F-bias045, N-bias045 | L, N and N-long are dominated by qualified `single_tone_rwa`; [F-bias045](../../../../tools/data/family_pareto/summaries/carrier_sweep_bias045.md) supplies a qualified alternative to RWA. |
| `single_tone_gate` | No measured target | Shaped target groups refuse; no dominance conclusion. |
| `shaped_sambe` | 3: A, S, S-long | [S-long](../../../../tools/data/family_pareto/summaries/long_shaped_gate_sequence.md) qualifies at 8.42157e-5; O2 is companion-blocked, not target-refused. |
| `gate_cache` | 0 of 3 qualified workloads | A, S and S-long are dominated by `shaped_sambe` and `period_table`. A clear-history is unmeasured. No deletion is authorized. |
| `period_table` | 1: O2 | [O2](../../../../tools/data/family_pareto/summaries/overlapping_shaped_tones.md) qualifies with `dop853` plus `t4` companions; A, S and S-long are dominated by `shaped_sambe`. |
| `common_frame` | No measured target | M, O1 and O2 target guards/preconditions refuse; no dominance conclusion. |
| `multitone_gate` | No measured target | M and O2 did not run this family in these historical measurements; no dominance conclusion. |
| `multitone_rwa` | No qualified target in these historical tables | [O1](../../../../tools/data/family_pareto/summaries/overlapping_constant_tones.md) records 2.59000e-3 before the constant-pair changes; M and O2 refuse there. The fixed-input price calibration above qualifies the current O1 acquire with its `t4` companion. It does not replace the historical all-mode Pareto table or remeasure M and O2. |
| `floquet` | No measured target | O1 target guard/precondition refuses; other inputs are not its candidates. |

The long shaped sequence supplies final-error evidence for `shaped_sambe` and
`period_table` as faster alternatives to `gate_cache`. It does not establish
coverage of every shaped workload. Cavity-drive, multiple-readout and continuous
flux workloads remain outside these tables. Family removal still requires the
relevant workload evidence and user approval.

[t6](kernels/t6.py) always integrates each density directly in the H0 frame.
Preparation packs model-only operators without scanning the complete acquire
inventory. Prepare keys still contain flux level, shift content, port, signed
carrier, signed gain and duration; phase and start time are apply-only.
`cost(table, rows)` supplies table-only chunk-interleaving weights.
It does not determine candidate order.
[ADR-0078](../../../../docs/adr/0078-evolution-module-boundary.md#t6-只保留-direct)
records why the reusable-channel strategy was removed.

The matrix measures the maximum absolute difference over every complex density
entry at readout, against an independent full-Hamiltonian Lindblad DOP853
reference. It includes coherences, relaxation and dephasing. The reference must
agree within `2e-6` between rtol/atol pairs `2e-10/2e-12` and `2e-12/2e-14`.
The matrix emits density and reference-agreement measurements for each case;
the constant-pair edge measurements are recorded above.
[Multisegment](../../../../tests/simulate/evolution/contract/test_accuracy_matrix.py)
observations measure the final density of the whole timeline, not each kernel's
local contribution. The separate
[history case](../../../../tests/simulate/evolution/contract/test_accuracy_matrix.py)
checks both densities when accumulated inventory changes the default route from
`shaped_sambe` to `single_tone_gate`.

### CPU DP5(4) accuracy

The `dop853` family uses DP5(4) on CPU for secular-only rows up to and including 10 µs
when the evolve budget is at least `1e-3`. The spike covered durations up to
10 µs at the default budget; it did not cover tighter budgets. Dividing both
local tolerances by three kept all six spike cells below `1e-3` full-density
error. At the default budget the uncapped DP5(4) tolerances are
`rtol=1e-5`, `atol=1e-7`. Lab-frame chains retain their bare-phase cap
before the division. DP5(4) uses a fifth-order controller exponent (`-1/5`).
Both CPU tableaux share the adaptive loop, frames, workspace and knot stops.
DP5(4) can put a nearly pure state outside the positivity tolerance. At
readout, CPU and OpenCL use the same [budgeted density projection](#accuracy-failures-and-lifecycle)
as every other kernel. Tableau selection and the continuation input tolerance
are unchanged.

The [CPU matrix](../../../../tests/simulate/evolution/contract/test_accuracy_matrix.py)
checks a 100 ns sampled row, a synthetic d=24 two-tone row with 960 knots per
tone, and a sampled row exactly at 10 µs. It confirms DP5(4) through routing
telemetry and compares every complex density entry at readout to the independent
two-tolerance interaction-frame Lindblad reference. The OpenCL matrix shares
the synthetic 960-knot case and duration edge, plus other sampled cases and a
tighter-budget DOP853 control. It checks GPU row and tableau counts and compares
full density to the same reference solver. The [FTF discussion](../../../../docs/discussions/2026-11-18-short-dop853-dp54.md)
records the separate evidence from real FTF workloads.

These measurements cover the linked inputs, not every input admitted by a
guard. Historical population calibrations, omission proxies and local solver
controls do not establish a whole-sequence full-density bound. In particular,
the slow-envelope population tolerance of `1e-2` above is not the matrix's
`2e-3` full-density acceptance threshold. The matrix does not measure environment
I/Q accuracy or model/basis convergence.

## Adding a family

Implement the kernel's numerical guard, prepare keys, preparation and apply in
its own module. Instantiate it in [kernels/__init__.py](kernels/__init__.py).
Add its named choice to the owning domain classifier in
[_dispatch.py](_dispatch.py). Domain masks contain no duplicate family lists.

A family without a new cost comparison needs three production locations in
three files: its implementation, its collection entry and its domain classifier.
For a cost-ranked shaped fallback, add its score formula and named score-map
entry in [_shaped_dispatch.py](_shaped_dispatch.py). With its implementation and
collection entry, that is four locations in three files. The classifier consumes
the sorted family names without changes. No parallel tuple or integer permutation
needs updating. Family-specific inventory or frame planning belongs
to its existing dispatch helper when needed, not a capability registry.

Add Engine accuracy inputs and the family's numerical-guard edge under
[contract tests](../../../../tests/simulate/evolution/contract/), then update the
kernel table above. Those tests and this documentation are additional to the
production-location counts.

## Row plans and the kernel seam

[kernel.py](kernel.py) defines `SegmentTable`, `Kernel`, `Prepared`, `RowPlan`,
`ApplyContext` and guard handoff records. These are internal contracts, not
additional package integration entry points.

[Adiabatic transport](kernels/adiabatic_map.py) takes density in the continuous
row-start eigenbasis and returns it in the continuous row-end eigenbasis.
Populations stay unchanged; coherences acquire integrated dynamical phase,
row-bath decay and the parallel-transport gauge correction. Quadrature splits
at waveform knots and doubles its order until complex phase factors converge
within one tenth of the segment budget. The kernel docstring specifies the
node limits and failure semantics. The scheduler carries the actual endpoint
basis between these rows, without intermediate T5 bath-basis changes. It
converts to a quantized basis only when entering another family, and returns
the final density to the DC basis. Slow motion and a nondegenerate, noncrossing
spectrum are the model owner's applicability conditions; quadrature convergence
does not certify the adiabatic approximation.

- `Prepared` is an opaque read-only numerical payload. `prepare` and guard
  handoffs build model-only data without reading density; `bind_input` may wrap
  that data in an evolve-local input-bound payload. `nbytes` charges owned arrays
  for the entire payload batch.
- `RowPlan` holds read-only decisions indexed by global `SegmentTable` row for
  one family in one evolve. Floquet stores admissions and retained equations;
  the direct solvers store frame or chain decisions. Plans never enter
  persistent kernel state.
- `KernelContext` supplies preparation, key and guard calls with one evolve
  owner. It carries the CPU worker width for sampled pricing, the offset for
  prefix row IDs, and optional capture. For ordinary timeline work it also
  carries any OpenCL backend and dispatch's tableau choice. Shared prefixes
  are priced and run on CPU.
  `ApplyContext` supplies the finite positive per-segment allowance and the
  family's CPU plan or, for admitted OpenCL rows, the runner and selected
  tableau. Floquet and both CPU direct solvers require their own plan; a missing
  plan raises `ValueError`, and a wrong family type raises `TypeError`.
  Families without plans, including `single_tone_rwa`, `multitone_rwa`,
  `single_tone_gate` and `multitone_gate`, reject a non-`None` plan with `ValueError`.

Dispatch returns at most one plan per family and a separate global-row chain
array. [Scheduling](_schedule.py) keeps the originating plan through preparation
bindings and input-bound bindings, then supplies it unchanged in `ApplyContext`.
It neither slices nor interprets plans. A timeline worker resolves deferred
neighbors before building its local plan and chains. It does not mutate the
caller dispatch's shared decisions.

Splitting or reordering a guard call preserves its acceptance mask. A guard
may read the source initial density or a certified prefix input, but it does not
evolve them or inspect worker states. A measured guard can evolve its own
fixed probes. Complete-group decisions are shared through the WorkPlan.
`guard_builds` identifies table-building guards;
`guard_prepared` hands accepted model-only products to apply without rebuilding.
When `guard_builds` is true, the hook must return `GuardResult`, even for empty
or rejected rows. Returning `None` raises rather than retrying `guard`.
When false, classification calls `guard` directly. Multitone dispatch owns the
per-key gate price comparison; prefix and worker calls borrow the complete-acquire
inventory, never subset counts, routing history or resident warmth.
The `multitone_gate` guard checks structural eligibility. The scheduler calls
prepare after selection. Prefixes and workers share the immutable product for
each key through the WorkPlan until evolve ends. With telemetry on,
`prepare_key_builds` counts the numerical construction, not its consumers.

Cross-timeline keys are guarded before shared preparation. Local builders usually
run inside their timeline worker. A guard resolved before scheduling, such as
Sambe with a shared table fallback, uses the same handoff.

`bind_input` can contract a common known input into an evolve-local read-only
payload. Dyson uses it for source or shared-prefix inputs. Prefix resolution
evolves one representative per identical prefix group in time order, including
nested and driven prefixes. Bound payloads expire with the evolve.
Returning `None` retains ordinary per-batch apply. Apply advances worker
densities in place and returns a row-aligned boolean exceedance mask.

## WorkPlan and cache ownership

Engine owns one [WorkPlan](workplan.py) per call.
[Dispatch](_dispatch.py) populates it from the complete evolve table before
workers start. Evolve records history usage; inspection projects usage with the
same inventory and cost formulas. Prefix representatives, caller classification
and timeline workers share the plan through `KernelContext` and `ApplyContext`.
Engine closes the plan on success, cancellation or failure, even if population
stops before workers start. It drains any workers before closing the plan.
The single `complete_table` field supplies full immutable geometry to common-frame,
multitone dispatch and Floquet frame construction. Closing drops it along with
the evolve-local facts.
The registry kernels retain no evolve-local data.

### Facts and algorithm groups

WorkPlan publishes each lazy fact once, including concurrent requests for the
same key. WorkPlan copies builder results and publishes read-only arrays.
RWA graphs share only within a complete model identity, level, ordered ports,
policy and exact final Fourier-edge mask, including adaptively restored edges.
The same mask can serve different carriers, but Stark diagonals remain carrier-local.
Constant RWA guards and preparation reuse graph topology and labels. Port quadrature
norms, packed models and full secular baths are also evolve-local facts. RWA uses
the sum of quadrature norms; shaped admission uses their maximum. Fact hits never
replace row-local fixed-flux, budget, strength, phase, omission, pruning or dimension
checks. Constant-RWA payloads own their array copies for this evolve.
The shared facts include envelope peak/slope/real flags, period split rates, packed
DOP853 samples and covariant gate admissions. Gate admission facts use full canonical physics, budget and numerical
policies. Each row still checks its own continuous flux support before borrowing an
admission; only absolute start and covariant phase are omitted from the fact.
Floquet consumes full Fourier averages during frame construction. WorkPlan
publishes each compact frame and optional sideband candidates once per source
instance, budget and complete prepare key. Prefix and deferred classifications
share these facts, including invalid frames and frames whose rows are all
refused. The complete evolve table determines sideband demand before the first
build; row-local certificates and admissions are not facts. Model facts include the full
physical coordinates in their identity. Table-indexed facts
also include table identity, so renumbered prefix representatives cannot alias
the complete table. DOP853 packing also distinguishes flux-inclusive samples.
Densities and adaptive solver history are not facts. Lab-frame DOP853 gives each
row its own Hamiltonian, matrix-product and derivative buffers. These mutable
buffers never enter shared preparations or WorkPlan facts.

The planner publishes immutable algorithm-named records from the complete
candidate inventory, before numerical admission:

- `GainGroupPlan` and `GainBuildPlan` fix period builders, work limits and gain
  anchors. `CF4CarrierGroupPlan` fixes narrow constant-CF4 carrier inventories and
  four carrier anchors. Cache warmth and rejected rows do not change either
  axis's anchor coordinates.
- `CarrierFamilyPlan` fixes physical carrier groups. More than nine unique,
  nonzero, same-sign carriers qualify when the largest magnitude is at most
  1.1 times the smallest.
- `RampFamilyPlan` groups rise/fall candidates by physical spectral inputs.
  Duration, slope and prepare key can differ within a group. Kernels still own
  harmonic cutoffs, amplitude nodes, spectral products and convergence tests.

Resource handles distinguish planned inventory from incrementally admitted
actual key builds. Gain anchors use single-flight immutable resources.
Rejected rows are not actual consumers. Caller-only groups seal before
building; the last actual handoff then releases the product. Groups with deferred
consumers stay open until WorkPlan closes. Prefix preparation cannot seal a
group needed by later timeline work. Gain and constant-CF4 carrier anchors share
this admission and handoff mechanism. Carrier interaction-picture prefixes belong
only to the evolve-local anchor resource, not to resident period payloads.
Non-anchor carrier guards also share single-flight CF4 node eigensystems and
adjacent eigenbasis overlaps, partitioned by exact model, port, gain, carrier
sign and coarse phase grid. Each carrier still advances every coarse interval
and checks the original nine probes at the same tolerance. A comparison within
a grid/dimension-scaled roundoff margin of the threshold uses the original
Padé coarse grid; rejected predictions also refine from that original grid.
These spectral products follow the same admission, sealing and handoff lifetime
as anchors. They do not enlarge resident period payloads. Standalone builds and
gain interpolation retain their original Padé coarse guards.

Carrier interpolation uses a mutable `CarrierWorkspace`, separate from immutable
facts and resource products. WorkPlan creates it lazily from published carrier
records under its inventory lock. Prefix and timeline consumers share that one
workspace; its own lock serializes tile construction. WorkPlan drops ownership
at close, after workers drain. A closed plan rejects further workspace requests.
The kernel registry does not retain this workspace.

### Evolve-local preparations and caller reuse

The engine deduplicates prepare keys across one evolve. Keys used by distinct
timelines enter shared preparation. Local builders and repeated local keys stream
one timeline at a time, with one static-map exception. A complete classified
inventory containing only `t4` and `shaped_sambe`, both present and no deferred
rows, can batch static keys within each timeline chunk. The default scheduler
requires at least two timelines per worker and an average of at least 64 rows
per timeline, then uses one timeline chunk per worker. Prepare partitioning stays unchanged. Explicit
`chunks_per_worker` keeps the previous streaming policy.

In that narrow case, ordinary `t4` maps prepare once per complete local key in
the chunk instead of forcing all shared shaped playback into singleton timelines.
The existing cursor and scatter advance each timeline's own density and carrier
phase in order. Keys, numerical routes, budgets and density operations do not
change. Static maps expire with the chunk. Local static-map array storage is
`24 * d**2 * K` bytes for dimension `d` and `K` distinct local keys in that chunk.
`K` can grow with timeline and row count. Other local builders usually keep a
one-timeline lifetime. Shared preparations and `multitone_gate` key products
live until the evolve ends.

Single-use ordinary keys prepare at their apply wave.
Accepted guard products go directly to apply, even when only a subset of rows
accepts. Fully rejected guard products have no apply consumers and are released.
Floquet's compact frame fact remains available to later classifications until
WorkPlan closes; no frame or refusal list survives the evolve.
Input-bound payloads, densities and dissipative preparations never cross evolves
through Engine storage.

Callers retaining complete exact-content preparations pass
`reuse_key=content_digest` to `evolve`, then call
`Engine.reuse(source, content_digest, cancel=event)` before reuse.
The caller invalidates on input, source, Engine or budget changes.
Evolution retains immutable inventory and usage-independent pricing facts.
It reprices current source usage without constructing geometry or a WorkPlan.
Unchanged relevant key totals reuse the policy answer. Changed totals reselect
actual keys and rows. True authorizes reuse. False records no usage and requires
fresh evolve. A different source, eviction or disabled residency also returns
False. FTF does not build geometry or duplicate algorithm cost rules.

The routing-facts LRU charges immutable Python records and scalars up to
`routing_cache_bytes`, default 4 MiB. The cap is a nonnegative integer without
a fixed upper bound. Zero disables retention. Weak references do not retain
sources. Oversized facts compute without evicting existing entries.
`close()` clears routing facts. Models, output densities, numerical scratch,
active payloads, ADC preparation and Numba's compilation cache are separate.
The byte cap is not a process RAM limit.

Engine prepare residency, exact-result memoization and source ADC solver
residency are removed. Representative single CPMG, six-cavity and FTF
acquisitions had zero hits in those layers within an acquisition.
Recomputing across acquisitions avoids retaining loss-dependent products as
losses drift. Single-source complete preparation and model reuse, and FTF's
acquire-local `CavityAverageCache`, retain their separate lifetimes.
See [ADR-0078](../../../../docs/adr/0078-evolution-module-boundary.md).

## Algorithm ownership and reuse

Each family owns its build, apply and numerical certificates. Shared numerical
operations have one owner, rather than one kernel calling another kernel
class's methods.

| Shared operation | Owner and users |
| --- | --- |
| Flux-level and shift-content model keys | `model_keys` in [kernels/_workload_facts.py](kernels/_workload_facts.py), shared by Dyson preparation and flux DOP853 grouping. Dyson modal packing and model iteration remain in `_dyson_model.py`. |
| Continuous flux waveform interpolation, term-kind interpretation and Hamiltonian accumulation | [kernels/_flux_expansion.py](kernels/_flux_expansion.py), shared by [adiabatic_map](kernels/adiabatic_map.py) and [flux_dop853](kernels/flux_dop853.py). Each kernel owns packing and integration. |
| Rates → population generator Q | [../baths.py](../baths.py), shared by evolution propagators, ADC conditional responses and fluxonium manifold modes. |
| Secular coherence decay and split-step maps | [kernels/_secular.py](kernels/_secular.py) provides decay to static, split, modal and direct propagators. `multitone_gate`, `gate_cache` and `single_tone_gate` use its population and coherence half-step maps with their own durations. |
| Density basis conversion | `transform_density` in [state.py](state.py), shared by population conversion, scheduler boundaries, Dyson input bounds and compiled DOP853 chains. Callers retain basis selection and traversal. |
| Adaptive DOP853 and DP5(4) tableaux, tolerances and integration | [kernels/_dop853.py](kernels/_dop853.py), used by density solvers and Hilbert endpoint builders. Packing and chain traversal live in [kernels/_dop853_data.py](kernels/_dop853_data.py) and [kernels/_dop853_chain.py](kernels/_dop853_chain.py). [kernels/_dop853_intervals.py](kernels/_dop853_intervals.py) owns density knot partitioning and source-coordinate playback clocks. |
| Ordinary density endpoints, including H0-rotating-frame integration | `advance_endpoints` in [kernels/dop853.py](kernels/dop853.py) owns the full-drive density equation and exact return to the model eigenbasis. CPU `dop853` rows, `t6` direct rows, period fractional tails and sampled multitone RWA probes share this entry point, supplying explicit frames and budgets. The CPU `dop853` plan selects DP5(4) and its tighter local tolerances for eligible rows. [OpenCL `dop853`](#gpu-backend) uses the same tableaux and tolerance rule in its device runner. |
| Period interpolation, secular playback and fractional tails | [kernels/_period_table.py](kernels/_period_table.py), shared by [period_table](kernels/period_table.py) and [shaped_sambe](kernels/shaped_sambe.py). |
| Gate Hilbert endpoint integration | [kernels/_gate_endpoints.py](kernels/_gate_endpoints.py), shared by sixteen-node, [adaptive single-phase](kernels/gate_phase.py) and [dual-phase](kernels/_multitone_gate_phase.py) builders. Single-tone integration is unchanged. |
| Constant-tone Hilbert period build and playback | [kernels/floquet_period.py](kernels/floquet_period.py) owns period builds and trajectories, with [CF4 grid](kernels/_cf4_grid.py) and [Sambe](kernels/_period_sambe.py) helpers. [kernels/_period_channel.py](kernels/_period_channel.py) owns complete secular-channel playback. Separate from two-tone Floquet frames. |
| RWA row admission, retention orchestration, policy, model/WorkPlan coefficient adapters and facts | [kernels/_rwa_numeric.py](kernels/_rwa_numeric.py), shared by ordinary multitone, single-constant RWA, covariant gates and dispatch. The package root imports `rwa_coefficients` from this owner; it returns the Fourier owner's record. Family preparation and playback remain with their kernels. |
| RWA Fourier coefficient record and numerics, compiled Hamiltonian assembly, omitted-edge endpoint kicks, omission estimates and retained-graph dimension | [kernels/_rwa_fourier.py](kernels/_rwa_fourier.py) owns `RWACoefficients` and `interaction_hamiltonian`, imported directly by the package root, RWA families, gate builder and joint-Fock diagnostic. Single-constant RWA and constant dual-tone `multitone_rwa` apply endpoint kicks using the resolved retention mask. Phase graphs and covariant gate construction remain in `_rwa_graph.py` and `_rwa_gate.py`. |
| Two-tone k=0 admission and propagation, shared Fourier frame evaluation | [kernels/_floquet_k0.py](kernels/_floquet_k0.py) owns omission/Dyson admission, modal Dyson and periodic channels. Both k=0 and retained-sideband apply use its Fourier frame evaluation. Stark frame construction and row plans remain in [floquet](kernels/floquet.py); retained-sideband equations remain in `_floquet_sidebands`. Separate from single-constant-tone `_period_channel`. |
| Two-tone retained-sideband certificates and equations | [kernels/_floquet_sidebands.py](kernels/_floquet_sidebands.py), used by [floquet](kernels/floquet.py) guard admission and apply. |
| Common-carrier frame, SW2 beat channels, omitted-channel indicators and dressed CF4/secular propagation | [kernels/common_frame.py](kernels/common_frame.py), the model-only numerical owner with row-local admission. [kernels/common_frame_kernel.py](kernels/common_frame_kernel.py) hands charged products to the engine; [_common_frame_dispatch.py](_common_frame_dispatch.py) owns cold variant pricing. |

The package-root coefficient names, record fields and numerical behavior are unchanged.
`RWACoefficients.__module__` is now
`zcu_tools.simulate.evolution.kernels._rwa_fourier`, rather than `_rwa_numeric`.
The compiled `interaction_hamiltonian` source location also moves to
`kernels/_rwa_fourier.py`. The old owner provides no compatibility shim.

## Accuracy, failures and lifecycle

Ordinary and continuous-flux density DOP853 stop at every active sampled-control
knot and restart the initial step per interval. A nextafter event bridge limits
only its own interval, never the full row. Playback retains both source-coordinate
endpoints and the original row-local carrier and interaction-picture phase.
The full row still determines local tolerances and bath basis changes. Flux
samples retain linear interpolation, including positive-duration event bridges.

`budget` is an absolute per-segment population and low-four allowance, not a
certified global error bound. Guards combine analytic bounds and calibrated
approximations. In particular, retained-sideband Floquet admission does not
certify aliasing beyond the Fourier grid, full finite-Fourier contractivity,
global ODE error or directed rounding of norms and eigensystems.

After converting to the DC basis, evolution passes through densities that are
Hermitian within `1e-10`, have trace one within `1e-8` and have no eigenvalue
below `-1e-8`. For an invalid finite density, it takes the Hermitian part, clips
negative eigenvalues to zero and normalizes the remaining weights to sum to one.
The largest absolute change to any complex entry must be at most the supplied
`budget` for that evolve call (default `1e-3`). This uses the same max-entry
metric as full-density reference comparisons (ADR-0078), but bounds the repair,
not the error against a reference solution. If the budget is exceeded or no
positive weight can be normalized, `evolve` raises `RuntimeError` naming the point.
It does not retry or reclassify the row. Projection happens only on returned
densities, not at internal segment or prefix boundaries.

A kernel's post-apply accuracy exceedance does not trigger a retry or
reclassification. `PathDiagnostics.exceeded_by_kernel` counts those segments
by family, and one `RuntimeWarning` reports the evolve call's total.
`PathDiagnostics.warn_exceeded()` owns the aggregation and warning text, so callers
reusing a result can report the same exceedances without changing it.
All routing diagnostics use family names, not builder or frame variants. Shared
prefix counts and exceedances scale by the adopting timelines.
`max_trace_error` reports the maximum absolute complex trace-minus-one over
readout densities before projection, or zero for empty work. The diagnostic
itself never rejects output or applies ADC's tolerance.

Invalid inputs fail at construction or entry. Missing required registry entries
raise with segment and family context. An exhausted candidate round raises
`NoValidKernel`, a `RuntimeError` subclass with the unchanged `no valid kernel`
message. Import it from `zcu_tools.simulate.evolution`. Its read-only `domain`,
`points` and `segments` fields match the message; point IDs and segment rows are
tuples. `attempted` is the tuple of family names offered by that final round,
none of which accepted those rows. With `families=`, it includes all allowed
candidates, including families tried earlier or skipped for cost. The order is
the final round's offer order, not necessarily first-attempt order. With
`families=None`, it excludes families tried or skipped by earlier cost shortcuts.

A listed family was rejected by a numerical guard or an input precondition,
possibly through a retained guard rejection. Under an allowlist, an unlisted
family is not a candidate for those rows. The exception describes only the first
failing row group; later groups in the acquire have not been checked.
Preparation and apply failures propagate; they do not restart candidate
selection. Cancellation raises `Cancelled` without partial results. The
`simulate.evolution` export is the same exception defined in `simulate.boundary`.

Calls on one Engine serialize. Joblib provides host parallelism.
CPU kernels are single-threaded and do not open pools. Multi-timeline calls with
multiple workers create and close a pool per evolve; empty work, one timeline
and one worker run inline. By default, complete-acquire row density and the
fraction of cheap gate/no-drive rows select one or six timeline chunks per
worker. Prepare retains six chunks per worker to spread cold key builds.
An explicit `chunks_per_worker` fixes CPU prepare and timeline chunks. It does
not split GPU batches. This choice does not change admission, WorkPlan facts or RNG.
Engine checks Stop after each geometry batch and again after merging geometry.
During WorkPlan population, it checks after gate inventory and before single-gate
pricing or usage recording. It also checks at entry, at chunk and preparation
boundaries, and before return. A check does not preempt work inside a batch or
planner. See the [FTF Stop design target](../../../../docs/adr/0078-evolution-module-boundary.md#ftf-stop-用合作式檢查點設計上-5-秒內).
Native BLAS stays at one thread throughout evolve. Overlapping engines share a
process-level reference count and the last call restores the original limits;
external threadpoolctl users must not change them during active evolves.
`close()` is idempotent. It waits for active work, then clears routing facts and releases OpenCL resources.

`RoutingHistory` owns distinct acquire content's complete row inventories,
separate from resident preparations and numerical guards. Each numerical input digest
contributes its canonical gate/multitone key row multiplicities once. Identical
repeats need no numerical playback, so counting them again would overstate gate
amortization and select unnecessary builds. Distinct contents still accumulate
all their rows, including rejected or nonresident keys, before prefix work.
The digest covers model, initial density, budget and full batch content, not
sampling sequence or cache state. Preparation-cache hits, clearing, eviction,
disabled residency and Engine replacement do not reset this history. Thus clearing a
preparation cache and rerunning selects the same current route. Callers serialize
acquires sharing a history, even across sources and Engines. Content membership
and cumulative counters are not an LRU: storage grows with distinct content/key
pairs until the history is cleared or released. Only the current inventory is
visited on a hit, not the historical set. Model identities namespace usage, so
different physics do not share amortization.

To retain routing history across model sources or clear it between acquires:

```python
from zcu_tools.simulate.evolution import ModelSource, RoutingHistory

history = RoutingHistory()
source = ModelSource(initial_density, build_model, history=history)
# Reuse history across acquires or lend it to another matching model source.
history.clear()  # Call after all acquires sharing history have drained.
```

`ModelSource(..., history=None)` creates its own history by default. Its routing
and distinct-content counting remain unchanged. `ModelSource.close()` clears
model caches and this owned history after active work drains. A supplied history
is borrowed: closing the source does not clear it. Immutable builder inputs
remain reusable after close.

`history.clear()` discards all usage and content membership. The next evolve
selects from empty history. It also invalidates every pre-clear preparation:
`Engine.reuse` returns False until a new evolve registers current-generation
facts, even if the route would stay the same. Repeated reuse checks cannot
revive an old preparation. Clearing history does not clear model caches, alter
initial density, reset sampling RNG or reset a simulation clock. It owns none
of those states. Callers with history-sensitive preparations use `Engine.reuse`
or `Engine.evolve` to resolve the current route.

`SEGMENT_NUMERIC_VERSION` is `segment-engine-v15`. Engine captures and paired
benchmark reports record the numeric version. Evolution does not persist
densities or offline tables across processes or Engine lifetimes.

## Research records

Experiments, rejected alternatives and historical measurements live in the
tracked discussion snapshots, not in the runtime contract above:

- [Constant-pair guard and diagnostic](../../../../docs/discussions/2026-11-16-constant-multitone-guard-diagnostic.md).
- [Dyson input reuse](../../../../docs/discussions/2026-10-18-dyson-input-reuse.md).
- [Constant-period builders](../../../../docs/discussions/2026-10-18-constant-period-builders.md).
- [Shaped periods and gates](../../../../docs/discussions/2026-10-18-shaped-period-and-gates.md).
- [Two-tone Floquet](../../../../docs/discussions/2026-10-18-two-tone-floquet.md).
- [Direct integration](../../../../docs/discussions/2026-10-18-direct-integration.md).
- [Planner evidence and overhead](../../../../docs/discussions/2026-10-18-planner-evidence-and-overhead.md).

The shared numerical contract suite is
[tests/simulate/evolution/contract/kernel_suite.py](../../../../tests/simulate/evolution/contract/kernel_suite.py).
