# Segment evolution

**Last updated:** 2026-11-17, kappa2 admission and current plateau dispatch

**讀者：** framework-dev、agent。Segment evolution 的公開輸入、路由與數值邊界。

`simulate.evolution` evolves numeric models to readout start. Callers use the
package root; dispatch, kernels, scheduling and caching are internal.
[ADR-0078](../../../../docs/adr/0078-evolution-module-boundary.md) records this
boundary and the decision to keep algorithm knowledge in dispatch.

## Contents

- [Public entry points](#public-entry-points)
- [Frozen numeric inputs and references](#frozen-numeric-inputs-and-references)
- [Dispatch inspection](#dispatch-inspection)
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
| Execution and results | `Engine`, `EvolutionResult`, `PathDiagnostics`, `FamilyRouting`, `Cancelled`, `SEGMENT_NUMERIC_VERSION` |
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
| Indexed scene cache | `SCENE_CACHE_VERSION`, `FrozenIndexEntry`, `read_frozen_index`, `write_frozen_index`, `read_scene_records`, `write_scene_record` |
| Frozen numeric inputs | `FROZEN_PHYSICAL_VERSION`, `FrozenInput`, `FrozenRecord`, `physical_input`, `canonical_json`, `digest`, `read_physical_input`, `read_frozen_input`, `write_frozen_input`, `read_frozen_manifest`, `write_frozen_manifest` |
| Solver-free reference storage | `SolverTolerances`, `ReferenceResult`, `reference_key`, `load_reference`, `write_reference` |

`Engine(workers, *, families=None, kernels=None, chunks_per_worker=None,
routing_cache_bytes=4*1024**2)` owns immutable routing facts.
Numerical preparations live only within one evolve. `evolve(source, batches, *, budget=1e-3, cancel=event)` consumes a
`ModelSource` and all `TimelineBatch` inputs for one acquire. `NumericModel` is a
separate fixed-coordinate record, not an Engine input. Constructor and field
contracts live with their types in [engine.py](engine.py), [secular.py](secular.py),
[inspection records](_inspection.py), [controls.py](controls.py), [model.py](model.py)
and [state.py](state.py).

Units are ns, GHz and rad. `SecularModel` supplies energies, drive quadratures,
directional rates and pure dephasing in each quantized flux level's energy
basis. The model builder owns physical applicability. Records own read-only
array copies and reject invalid shapes or numeric invariants.

`TimelineBatch.segment_boundaries` adds point-aligned absolute ns boundaries.
`segment_spans(batch)` is a pure query of the same final partition used by
`evolve`, including active microwave plateaus and flux events. Callers can
average external energy shifts and dephasing over these spans, then attach
point-aligned `SegmentShift` values through `segment_shifts`. Each supplied span
must match one queried span exactly; missing spans have zero shift. Shifts do
not add boundaries, rotate bases or change physical `FluxLevel` values. Model,
prepare and gate-usage identities include shift content. Without shifts,
the original model source, initial state and reuse semantics remain unchanged.

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

`EvolutionResult` contains densities at readout start in the DC eigenbasis.
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
budget must be positive. Unknown ports, negative duration and nonfinite inputs
raise `ValueError`.
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

## Frozen numeric inputs and references

`FROZEN_PHYSICAL_VERSION` owns the physical decoding contract for manifests
and standalone inputs. It changes when fields, units, dtypes, defaults or
Engine-input reconstruction change. `SCENE_CACHE_VERSION` owns the index JSON
and cache-directory revision. Both are currently 1. Consumers compare both
without parsing storage layout. `SEGMENT_NUMERIC_VERSION` records numerical
algorithm semantics, not input-format compatibility.

The public codec snapshots a `ModelSource` and one `TimelineBatch` without
running an Engine or a solver. `write_frozen_input(path, source, batch)` writes
version 1, a complete `physical` record and its `physical_sha256`.
`read_frozen_input(path)` checks the version and hash, then returns a
`FrozenInput(source, batch)`. The caller closes the returned source.

`physical_input(source, batch)` returns the same numeric record in memory.
`read_physical_input(record)` reconstructs it, including records from JSON.
The record contains the initial density, scheduled models plus DC zero,
continuous-flux expansion when used, slow-envelope subdivisions and complete
batch data. Point order, complex128/float64/int64/bool record dtypes, segment
corrections and continuation fields survive the round trip. The reconstructed
source has a fresh owned history and a finite level-to-model mapping. Unlisted
levels raise `KeyError`. Python builders, devices and registered source shifts
are not serialized.

`canonical_json` uses sorted keys, compact separators, nested array lists and
real/imag objects for complex values. It rejects nonfinite values. `digest`
returns its SHA-256. `physical_input(..., include_identity=False)` removes only
model namespaces. Complete physical hashes include those namespaces;
identity-free hashes allow oracle sharing. Neither hash is a reference key.

`write_frozen_manifest(path, records)` writes ordered `FrozenRecord` values as
version-1 canonical JSON inside stdlib gzip. `FrozenRecord(name, physical,
metadata=None)` keeps caller metadata opaque. Its `physical_sha256` and
`oracle_sha256` properties calculate the complete and identity-free hashes.
The writer calculates both hashes; `read_frozen_manifest(path)` verifies them
and returns a tuple in original order. Both reject duplicate names. The gzip
header has mtime=0 and no filename, so identical records produce identical bytes
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
Callers own cache locations and producer instructions. `SCENE_CACHE_VERSION` is
1; callers negotiating a shared numeric format can compare it without parsing
cache filenames or containers.

`reference_key(source, batch)` preserves the historical key formula. It hashes
version 1, identity-free physics, coarse tolerances 2e-10/2e-12, fine tolerances
2e-12/2e-14 and convergence limit 2e-6. Callers can override both tolerance
records. Solver identity and routing history do not enter the key.
`load_reference(cache_dir, source, batch)` only reads the matching NPZ, without
pickle or directory creation. A miss raises `FileNotFoundError`; it never solves.
`write_reference` atomically publishes `density` and `convergence_error` under
that same key. The producer certifies convergence, and measurement preflight
checks the loaded result before evolution.

The independent preparation entry point is
[`family_pareto_reference_producer.py`](../../../../tests/simulate/evolution/contract/family_pareto_reference_producer.py).
It reads only frozen record names and physics. The caller explicitly maps each
index to the independent lab-frame or interaction-picture solver and supplies
an explicit scene cache. `--scenario NAME` requires one matching row across the
supplied indexes before reading payloads. Zero or multiple matches fail without
solving; other rows need no payload. Without a scenario, all indexed rows remain
selected. All selected numeric files load before any solver runs. Metadata
and scene names do not select the solver. Both solvers stay in tests. The codec imports neither tests nor tools.
See [ADR-0078](../../../../docs/adr/0078-evolution-module-boundary.md).

## Dispatch inspection

`Engine.inspect_dispatch(source, batches, *, budget=1e-3, cancel=None)` returns a
`DispatchInspection` without evolving density. It uses evolve's classifier,
candidate ordering, guards and complete-inventory pricing. Floquet admission
still computes its one-period propagator. This method is not a no-computation
shortcut or a second dispatcher. Source, batch, budget, cancellation and failure
contracts match `evolve`; an exhausted candidate round raises `NoValidKernel`.

`rows` follows input batch order, point order, then split-span order.
Each row's `RowKey` contains the zero-based batch index, caller point ID and
zero-based chronological segment index. Segment indices include drive, flux,
plateau and explicit boundary splits. Continuations number only retained spans
at or after `start_ns`. Each row also reports absolute `start_ns` and `duration_ns`.

A `selected` row's `family` matches evolve from the same initial history.
A `deferred` row has a local prepare key whose guard belongs to its timeline
worker. A `needs_prefix_input` row needs a driven prefix density that evolve
would compute first. These unresolved rows return `family=None`; inspection
neither runs timeline workers nor computes prefix densities.

`candidates` records offered families and guard decisions in first-offer order.
An evaluated guard reports `accepted` or `rejected`. A local key reports
`deferred`. Candidates after a stopped round report
`untested_after_short_circuit`, unless an earlier round already evaluated them.
Structurally inapplicable and cost-skipped families are not listed unless
actually offered. Candidate `scheduling_cost` is `Kernel.cost`'s nonnegative
relative row work for chunk interleaving, not seconds or a numerical certificate;
it cannot explain why a family was chosen. The cold-amortized estimate used to
order candidates is not exposed. Short-circuited candidates have
`scheduling_cost=None`; evaluated and deferred candidates include their value.

Inspection does not modify routing history. It projects usage without writing
counters, content membership or generation, and does not create routing-reuse
snapshots. Querying A before evolving B leaves B's route and density unchanged.
Guard preparations expire on return and do not enter Engine caches. Source
model caches and shift registrations may be populated. Calls sharing a history
must serialize, just as for evolve.

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
an empty mapping. Evolution never writes files. `paths.frontend_cpu_s` separately
reports calling-thread geometry and WorkPlan CPU.
Common-frame reports also count `prepare_builds` and `admission_rows`.
Floquet's `prepare_builds` counts exact-key frame construction attempts,
including invalid frames and frames whose rows are all refused. Other families
leave these counters at zero. Telemetry does not change guards, routes,
densities or cancellation.

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
Engine evolves always execute numerical work.

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
| Multitone gate amortization | [multitone dispatch](_multitone_dispatch.py) owns the 20-use cost preference and complete-acquire canonical-key inventory |
| Slow/plateau carrier boundary | [slow-envelope comparison](kernels/slow_envelope.py) owns the strict absolute-carrier comparison shared by dispatch and its guard |
| Common-frame pricing and numerical construction | [common-frame dispatch](_common_frame_dispatch.py), [kernel](kernels/common_frame_kernel.py) |
| Constant and shaped period builders | [Floquet period](kernels/floquet_period.py), [period table](kernels/period_table.py), [shaped Sambe](kernels/shaped_sambe.py), [gate cache](kernels/gate_cache.py) |
| Two-tone Floquet admissions and variants | [Floquet](kernels/floquet.py) |
| Shared RWA admission, retention and Hamiltonian coefficients | [RWA numerics](kernels/_rwa_numeric.py), using [Fourier numerics](kernels/_rwa_fourier.py); family playback remains in [multitone RWA](kernels/multitone_rwa.py) and [single constant RWA](kernels/single_tone_rwa.py) |
| Covariant gate admission, splitting and reuse | [single-tone gate](kernels/single_tone_gate.py), [multitone gate](kernels/multitone_gate.py), [shared builder](kernels/_rwa_gate.py), [phase graph](kernels/_rwa_graph.py) |
| Direct integration and chronological chains | [DOP853 planning](_dop853_dispatch.py), [DOP853](kernels/dop853.py), [flux DOP853](kernels/flux_dop853.py) |

Cold-cost choices use the complete acquire inventory rather than resident cache
warmth. Prefix and worker subsets share those choices. Numerical approximations
retain their own admission limits; they do not certify arbitrary-input or
whole-sequence accuracy.

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

In the table, `B` is the per-segment `budget`, `d` is duration in ns and `T` is
one carrier period in ns. Split, motion, omission and phase indicators use the
bounds defined by each linked numerical owner. They are admission criteria,
not measurements of final full-density error. RWA slope and curvature limits
are normalized by envelope peak. Retained real dimension counts populations
and both real coordinates of within-component coherences.

The matrix exercises 16 active families, including a numerical-guard edge for
each active guarded family. `multitone_gate` rejects every row and reports
`NOT EXERCISED`, with no density error. Three existing edge cases exceed
`2e-3` and remain strict xfails: `gate_cache` at `9.296548e-3`,
`common_frame` at `8.066669e-3` and `shaped_sambe` at `3.091554e-3`.
The other 13 active families pass in the recorded observations.
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

Sampled multitone rejection is the current admission contract, not a temporary
route pending a numerical fix. It leaves candidate order and common-frame pricing
unchanged. Rejected sampled rows fall back to the next admitted candidate,
often `dop853`. Previously cheap sampled RWA graphs lose their exemption from
common-frame pricing. The matrix's `cheap_sampled_nearby` and
`cheap_sampled_separated` cases check both price branches, their empty/warm
routes and final densities. This does not force every sampled row to DOP853.

| Family and use | Dispatch domain | Approximation and numerical admission | Pinned accuracy cases | Full density ≤ 2e-3 |
| --- | --- | --- | --- | --- |
| [`t4`](kernels/t4.py), static idle evolution | `idle` | Exact secular population semigroup and coherence phases/decay. Guard requires zero tones; no approximation threshold. | [`idle_coherence`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |
| [`adiabatic_map`](kernels/adiabatic_map.py), undriven flux transport | `adiabatic` | Preserve ordered populations, integrate dynamical/gauge phases and decay coherences. Requires waveform samples. Quadrature doubles from 4 to 4096 nodes per knot interval until phase-factor change ≤ B/10; overlap < 0.25 forces refinement. This does not check nonadiabatic transitions, degeneracy or crossings. | [`adiabatic_ramp`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |
| [`flux_dop853`](kernels/flux_dop853.py), driven continuous flux | `flux` | Integrate full H(flux(t)) with a fixed quantized secular bath per row, stopping at active flux/envelope knots. Requires tones and waveform samples. Local rtol = min(1e-4, 0.03B), atol = min(1e-6, 0.0003B), from the [shared solver](kernels/_dop853.py). | [`continuous_flux`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |
| [`slow_envelope`](kernels/slow_envelope.py), low-carrier constant drive | `slow`, carrier magnitude strictly below 30 MHz | Midpoint-held full Hamiltonians with Strang bath half-steps, default 20 holds per period and one hold for DC. Dispatch and kernel guard share the strict absolute-carrier < 30 MHz comparison; exactly ±30 MHz belongs to plateau. No budget adaptation; historical population calibration is separate from this density measurement. | [`low_carrier`](../../../../tests/simulate/evolution/contract/accuracy_cases.py); [boundary contract](../../../../tests/simulate/evolution/contract/test_routing_boundaries.py) | PASS |
| [`dyson`](kernels/dyson.py), weak constant drive | `plateau` | Second-order full-carrier modal Dyson. Calibrated guards require r ≤ min(0.1, 0.25B^(1/3)), r2 ≤ 0.05, and second-order population indicator B2 + B/8 ≤ min(0.02, (1.5B)^(2/3)). Certified shared no-drive prefixes omit the B2 check. B/8 reserves path pruning; this is not a multilevel remainder proof. | [`weak_plateau`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |
| [`single_tone_rwa`](kernels/single_tone_rwa.py), constant rotating-frame block evolution | `plateau` | Fixed flux, B ≥ 1e-3, carrier magnitude ≥ 50 MHz. Initial 50 MHz edge window with first/second-order population-transfer restoration at 5.8e-3, including forward edges and adjoints. Final admission uses kappa2 = spectral norm of sum_sign abs(c/d) over the resolved omitted edges, ≤ 0.08, in place of the remaining transfer comparison. Full-operator/carrier ratio ≤ 0.25 and retained-norm phase proxy ≤ 0.06 rad. Phase-graph pruning ≤ min(B, 1e-3); real generator dimension ≤ 64. First-order endpoint kicks include every edge outside the resolved retention mask, including diagonal ±carrier terms; phase-graph pruning is separate. Cross-component density entries below 1e-5 after the initial kick are dropped. Uses the full secular bath without Strang splitting. | [`linear_plateau`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS at kappa2 inner edge: 3.061429e-13 |
| [`floquet_period`](kernels/floquet_period.py), full-carrier constant-tone period reuse | `plateau` | Hilbert period trajectories with secular split playback. Requires carrier magnitude ≥ 0.03 GHz and some subdivision count m in 1..64 with dT²ΓV²/m² ≤ 1.0, using this owner's rad/ns drive bound. Below 1 GHz, CF4 or adjacent-cutoff Sambe uses endpoint defect ≤ min(1e-4, 0.1B). Higher carriers use DOP853 rtol min(1e-8, 1e-4B), max step T/4. These are calibrated local controls. | [`weak_plateau`](../../../../tests/simulate/evolution/contract/accuracy_cases.py), [`strong_plateau`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS; largest family error 1.375101e-3 |
| [`t6`](kernels/t6.py), full dissipative constant-tone evolution | `plateau`, terminal candidate | One constant tone at any strength. Direct integrates each density in the H0 frame with the full secular bath and the shared dop853 policy: local rtol = min(1e-4, 0.03B), atol = min(1e-6, 0.0003B), without a fixed step ceiling. These are not whole-sequence error bounds. All rows use direct integration, irrespective of key reuse. It does not do RWA or density repair; numerical failure raises rather than reroutes. | [`weak_plateau`](../../../../tests/simulate/evolution/contract/accuracy_cases.py), [`strong_plateau`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |
| [`single_tone_gate`](kernels/single_tone_gate.py), phase-covariant shaped RWA gates | `shaped` | Fixed flux, B ≥ 1e-3, real sampled envelope with at least 3 samples, slope ≤ 0.025/ns, curvature ≤ 0.003/ns², carrier magnitude ≥ 50 MHz. Omission restoration requires transfer ≤ 0.0058, residual phase ≤ 0.0045 rad and retained real dimension ≤ 256. Pruning ≤ min(B, 1e-3); adaptive secular split uses n ≥ 4 with d³ΓV²/n² ≤ 0.01. [Shared gate owner](kernels/_rwa_gate.py). | [`linear_shaped`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |
| [`shaped_sambe`](kernels/shaped_sambe.py), frozen-envelope finite-Fourier periods | `shaped` | Nonzero peak, carrier magnitude ≥ 0.03 GHz, subdivisions m ≤ 32, envelope slope ≤ 2·peak·abs(f)·m and peak drive bound ≤ 0.5·abs(f)·m. Subdivision policy limits real/complex motion to 0.01/0.0025 and secular split to 0.01. Compare adjacent harmonic cutoffs 2/3 or 3/4 and use 4 or 8 amplitude nodes; accumulated spectral plus interpolation estimate must be ≤ B/2. | [`linear_shaped`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | FAIL at guard edge: 3.091554e-3; strict xfail |
| [`gate_cache`](kernels/gate_cache.py), phase-interpolated full-carrier shaped gates | `shaped` | Four Hilbert subsegments with secular half-steps. Model-only secular split d³ΓV²/16 ≤ 0.01. For 0 < d ≤ 30 ns, dispatch tries 6 phase nodes with 2 probes and full-unitary defect ≤ min(1e-4, 0.1B); other keys or failed convergence use 16 nodes. Fixed-order playback does not certify arbitrary-drive error against B. | [`shaped_complex`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | FAIL at guard edge: 9.296548e-3; strict xfail |
| [`period_table`](kernels/period_table.py), shaped amplitude-band period playback | `shaped` | 8 Chebyshev amplitude nodes and 16 phase samples. Guard requires envelope-motion indicator ≤ 0.05, amplitude-band bandwidth ≤ 0.5 and secular split ≤ 0.01. These calibrated limits do not scale with B. DC is guard-admissible with local direct integration, but default costs prefer terminal DOP853. | [`linear_shaped`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |
| [`common_frame`](kernels/common_frame.py), common-frame SW2 beat dynamics | `two` or two-tone `multiple` only | Fixed-flux common frame with second-order beat/two-photon terms, first-order endpoint dressing and CF4/secular playback. Absolute mean carrier must be ≥ 1e-10 GHz. CF4 steps are capped at 8 ns and 0.08 cycles of the fastest retained channel, with no lower cutoff. Dispatch prices first-order playback steps, so cost rises with active carrier spacing; numerical guards determine applicability. Fast-channel denominator must exceed 1e-10. Row-local coherent/bath indicators must be ≤ 0.03/0.01, independent of B. | [`nearby_carriers`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | FAIL at guard edge: 8.066669e-3; strict xfail |
| [`multitone_gate`](kernels/multitone_gate.py), retained simultaneous RWA gate family | `multiple`, disabled | Rejects every row. The retained 20-use inventory cost preference cannot restore numerical admission at 19 or 20 uses; pinning cannot bypass rejection. Shared single-tone gate numerics are unchanged. | [19/20 and pinned rejection contract](../../../../tests/simulate/evolution/contract/test_routing_boundaries.py) | NOT EXERCISED; no density error |
| [`multitone_rwa`](kernels/multitone_rwa.py), simultaneous omission-restored RWA | `two`, distinct ports only | Exactly two constant envelopes, fixed flux, B ≥ 1e-3 and each carrier magnitude ≥ 50 MHz. Sampled rows are rejected. Start with 50 MHz edges, restore omitted paths until transfer proxy ≤ 0.0058 and residual phase ≤ 0.0045 rad; retained real dimension ≤ 256. No carrier-spacing, full-operator/carrier ratio or retained-norm phase cap. Apply step ceiling is 16 ns. | [`joint_constant`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |
| [`floquet`](kernels/floquet.py), constant Stark tone plus probe | `two` | Larger absolute gain selects the Stark slot, with ties choosing the first; that slot requires positive carrier. Frame overlap ≥ 0.5, modal condition ≤ 1e8, maximum real eigenvalue ≤ 1e-10/ns and residual ≤ 1e-9; sampling extent ≤ 128 with at most 4096 phase samples. k=0 omission and convergence each fit B/3. Modal Dyson additionally uses r ≤ 0.1, r2 ≤ 0.05 when probe gain magnitude ≥ 0.1, Dyson norm-duration v ≤ 0.5, transfer indicator v + v²/2 ≤ 0.02 and higher-order remainder exp(v)·v³/6 ≤ B/3; otherwise use periodic channel playback. Retained-sideband admission instead requires convergence ≤ B/40 and bound ≤ 0.95B, retaining edges within 0.1/ns and averaging other harmonics through order 4. Admission is per row; refused rows continue to other allowed candidates. This finite-Fourier admission is not a density-error guarantee. | [`two_carriers`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |
| [`dop853`](kernels/dop853.py), full-drive direct density integration | `shaped`, `two`, `multiple`, terminal candidate | No RWA or fast-path admission threshold. Accepts valid rows with finite positive B, including complex sampled envelopes. Stops at active control knots. Local rtol = min(1e-4, 0.03B), atol = min(1e-6, 0.0003B), from the [shared solver](kernels/_dop853.py); these are not global error bounds. | [`shaped_complex`](../../../../tests/simulate/evolution/contract/accuracy_cases.py), [`two_carriers`](../../../../tests/simulate/evolution/contract/accuracy_cases.py), [`joint_gate_inventory`](../../../../tests/simulate/evolution/contract/accuracy_cases.py) | PASS |

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
| `multitone_gate` | No measured target | M and O2 target guards/preconditions refuse; no dominance conclusion. |
| `multitone_rwa` | No qualified target | [O1](../../../../tools/data/family_pareto/summaries/overlapping_constant_tones.md) fails at 2.59000e-3; M and O2 refuse. |
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
[Multisegment](../../../../tests/simulate/evolution/contract/test_accuracy_matrix.py)
observations measure the final density of the whole timeline, not each kernel's
local contribution. The separate
[history case](../../../../tests/simulate/evolution/contract/test_accuracy_matrix.py)
checks both densities when accumulated inventory changes the default route from
`shaped_sambe` to `single_tone_gate`.

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
- `KernelContext(source, budget, work_plan)` supplies preparation, key and guard
  calls with one evolve owner. `ApplyContext(budget, plan, work_plan)` supplies the finite positive per-segment allowance
  and the family's plan. Floquet and both direct solvers require their own plan;
  a missing plan raises `ValueError`, and a wrong family type raises `TypeError`.
  Families without plans, including `single_tone_rwa`, `multitone_rwa`,
  `single_tone_gate` and `multitone_gate`, reject a non-`None` plan with `ValueError`.

Dispatch returns at most one plan per family and a separate global-row chain
array. [Scheduling](_schedule.py) keeps the originating plan through preparation
bindings and input-bound bindings, then supplies it unchanged in `ApplyContext`.
It neither slices nor interprets plans. A timeline worker resolves deferred
neighbors before building its local plan and chains. It does not mutate the
caller dispatch's shared decisions.

Guards are row-local: splitting or reordering a guard call must preserve its
acceptance mask. A guard may read the source initial density or an
engine-computed certified prefix input, never a worker's evolving density.
Guards do not evolve density. `guard_builds` identifies table-building guards;
`guard_prepared` hands accepted model-only products to apply without rebuilding.
When `guard_builds` is true, the hook must return `GuardResult`, even for empty
or rejected rows. Returning `None` raises rather than retrying `guard`.
When false, classification calls `guard` directly. Multitone dispatch owns the
20-use gate cost preference; prefix and worker calls borrow the complete-acquire
inventory, never subset counts or resident warmth. The kernel guard checks only
numerical admission. The `multitone_gate` family rejects all rows, including
pinned calls and inventories of 19 or 20 uses.
Cross-timeline keys are guarded before shared preparation. Local builders run
inside their timeline worker and retain the same handoff.

`bind_input` can contract a common known input into an evolve-local read-only
payload. Dyson uses it for source or shared-prefix inputs. Prefix resolution
evolves one representative per identical prefix group in time order, including
nested and driven prefixes. Bound payloads expire with the evolve.
Returning `None` retains ordinary per-batch apply. Apply advances worker
densities in place and returns a row-aligned boolean exceedance mask.

## WorkPlan and cache ownership

[Dispatch](_dispatch.py) creates one [WorkPlan](workplan.py) from the complete
evolve table before workers start. Its history mode records usage for evolve
and projects usage for inspection, with the same inventory and cost formulas. Prefix representatives, caller classification
and timeline workers share it through `KernelContext` and `ApplyContext`. Engine
drains its pool before closing the plan on success, cancellation or failure.
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
change. Static maps expire with the chunk; other local builders retain their
one-timeline lifetime. Local static-map array storage is `24 * d**2 * K` bytes
for dimension `d` and `K` distinct local keys in that chunk. `K` can grow with
timeline and row count. Shared preparations retain their separate evolve lifetime.

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
| Secular coherence decay | [kernels/_secular.py](kernels/_secular.py), used by static, split, modal and direct propagators. |
| Density basis conversion | `transform_density` in [state.py](state.py), shared by population conversion, scheduler boundaries, Dyson input bounds and compiled DOP853 chains. Callers retain basis selection and traversal. |
| Adaptive DOP853 tableau, tolerances and integration | [kernels/_dop853.py](kernels/_dop853.py), used by density solvers and Hilbert endpoint builders. Packing and chain traversal live in [kernels/_dop853_data.py](kernels/_dop853_data.py) and [kernels/_dop853_chain.py](kernels/_dop853_chain.py). [kernels/_dop853_intervals.py](kernels/_dop853_intervals.py) owns density knot partitioning and source-coordinate playback clocks. |
| Ordinary density endpoints, including H0-rotating-frame integration | `advance_endpoints` in [kernels/dop853.py](kernels/dop853.py) owns the full-drive secular density equation and exact return to the model eigenbasis. Ordinary DOP853 apply, `t6` direct rows and period fractional tails share this entry point and its budget-scaled local tolerance policy, supplying explicit frames. Budget validation, knot stops and numerical failures are unchanged. |
| Period interpolation, secular playback and fractional tails | [kernels/_period_table.py](kernels/_period_table.py), shared by [period_table](kernels/period_table.py) and [shaped_sambe](kernels/shaped_sambe.py). |
| Gate Hilbert endpoint integration | [kernels/_gate_endpoints.py](kernels/_gate_endpoints.py), shared by sixteen-node and [adaptive phase](kernels/gate_phase.py) builders. |
| Constant-tone Hilbert period build and playback | [kernels/floquet_period.py](kernels/floquet_period.py) owns period builds and trajectories, with [CF4 grid](kernels/_cf4_grid.py) and [Sambe](kernels/_period_sambe.py) helpers. [kernels/_period_channel.py](kernels/_period_channel.py) owns complete secular-channel playback. Separate from two-tone Floquet frames. |
| RWA row admission, retention orchestration, WorkPlan facts and Hamiltonian coefficients | [kernels/_rwa_numeric.py](kernels/_rwa_numeric.py), shared by ordinary multitone, single-constant RWA, covariant gates and dispatch. Policy constants live at its top; family preparation and playback remain with their kernels. The package-root coefficient API imports this owner directly. |
| RWA Fourier coefficients, omitted-edge endpoint kicks, omission estimates and retained-graph dimension | [kernels/_rwa_fourier.py](kernels/_rwa_fourier.py), used by `_rwa_numeric.py`. Only single-constant RWA currently applies endpoint kicks. Phase graphs and covariant gate construction remain in `_rwa_graph.py` and `_rwa_gate.py`. |
| Two-tone k=0 admission and propagation, shared Fourier frame evaluation | [kernels/_floquet_k0.py](kernels/_floquet_k0.py) owns omission/Dyson admission, modal Dyson and periodic channels. Both k=0 and retained-sideband apply use its Fourier frame evaluation. Stark frame construction and row plans remain in [floquet](kernels/floquet.py); retained-sideband equations remain in `_floquet_sidebands`. Separate from single-constant-tone `_period_channel`. |
| Two-tone retained-sideband certificates and equations | [kernels/_floquet_sidebands.py](kernels/_floquet_sidebands.py), used by [floquet](kernels/floquet.py) guard admission and apply. |
| Common-carrier frame, SW2 beat channels, omitted-channel indicators and dressed CF4/secular propagation | [kernels/common_frame.py](kernels/common_frame.py), the model-only numerical owner with row-local admission. [kernels/common_frame_kernel.py](kernels/common_frame_kernel.py) hands charged products to the engine; [_common_frame_dispatch.py](_common_frame_dispatch.py) owns cold variant pricing. |

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
global ODE error or directed rounding of norms and eigensystems. Evolution does
not clip or renormalize density.

A post-apply exceedance keeps the uncorrected approximation. It does not trigger
a retry or reclassification. `PathDiagnostics.exceeded_by_kernel` counts those
segments by family, and one `RuntimeWarning` reports the evolve call's total.
`PathDiagnostics.warn_exceeded()` owns the aggregation and warning text, so callers
reusing a result can report the same exceedances without changing it.
All routing diagnostics use family names, not builder or frame variants. Shared
prefix counts and exceedances scale by the adopting timelines.
`max_trace_error` reports the maximum absolute complex trace-minus-one over final
readout densities, or zero for empty work. It never rejects output or applies
ADC's tolerance.

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
selection. Cancellation raises `Cancelled` without partial results.

Calls on one Engine serialize. Joblib threading is the only parallel layer;
kernels are single-threaded and do not open pools. Multi-timeline calls with
multiple workers create and close a pool per evolve; empty work, one timeline
and one worker run inline. By default, complete-acquire row density and the
fraction of cheap gate/no-drive rows select one or six timeline chunks per
worker. Prepare retains six chunks per worker to spread cold key builds.
An explicit `chunks_per_worker` fixes both stages. This choice occurs after
classification and does not change admission, WorkPlan facts or RNG.
Cancellation is checked at chunk and preparation boundaries, and before return.
Native BLAS stays at one thread throughout evolve. Overlapping engines share a
process-level reference count and the last call restores the original limits;
external threadpoolctl users must not change them during active evolves.
`close()` waits for active work, clears routing facts and is idempotent.

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

`SEGMENT_NUMERIC_VERSION` is `segment-engine-v7`. `simulators.ftf.environment` owns private
replay and checks backend, model, numerical, RNG and dependency versions before
recomputing a round. Evolution does not persist densities or offline tables
across processes or Engine lifetimes.

## Research records

Experiments, rejected alternatives and historical measurements live in the
tracked discussion snapshots, not in the runtime contract above:

- [Dyson input reuse](../../../../docs/discussions/2026-10-18-dyson-input-reuse.md).
- [Constant-period builders](../../../../docs/discussions/2026-10-18-constant-period-builders.md).
- [Shaped periods and gates](../../../../docs/discussions/2026-10-18-shaped-period-and-gates.md).
- [Two-tone Floquet](../../../../docs/discussions/2026-10-18-two-tone-floquet.md).
- [Direct integration](../../../../docs/discussions/2026-10-18-direct-integration.md).
- [Planner evidence and overhead](../../../../docs/discussions/2026-10-18-planner-evidence-and-overhead.md).

The shared numerical contract suite is
[tests/simulate/evolution/contract/kernel_suite.py](../../../../tests/simulate/evolution/contract/kernel_suite.py).
