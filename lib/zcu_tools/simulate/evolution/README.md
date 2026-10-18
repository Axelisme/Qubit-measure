# Segment evolution

**Last updated:** 2026-10-18, evolve-local work ownership

`simulate.evolution` evolves numeric models to readout start. Callers use the
package root; dispatch, kernels, scheduling and caching are internal.
[ADR-0078](../../../../docs/adr/0078-evolution-module-boundary.md) records this
boundary and the decision to keep algorithm knowledge in dispatch.

## Public entry points

The exports in [__init__.py](__init__.py) fall into these groups:

| Purpose | Exports |
| --- | --- |
| Execution and results | `Engine`, `EvolutionResult`, `PathDiagnostics`, `Cancelled`, `SEGMENT_NUMERIC_VERSION` |
| Engine model source | `ModelSource`, `SecularModel`, `SecularDrive` |
| Flux description | `FluxExpansion`, `FluxTerm`, `FluxLevel`, `FluxSchedule`, `FluxWaveform` |
| Batched timelines | `TimelineBatch`, `DriveTrack`, `EnvelopeSpec` |
| Control description | `ControlPulse`, `ControlSequence`, `Envelope`, `PhaseEvent`, `quantize_ns` |
| Fixed-coordinate model records | `NumericModel`, `DriveSource` |
| States and population conversion | `Basis`, `DensityState`, `Populations`, `populations`, `batch_populations` |

`Engine(workers, cache_bytes, *, kernels=None, chunks_per_worker=6)` owns a
prepare cache. `evolve(source, batches, *, budget=1e-3, cancel=event)` consumes a
`ModelSource` and all `TimelineBatch` inputs for one acquire. `NumericModel` is a
separate fixed-coordinate record, not an Engine input. Constructor and field
contracts live with their types in [engine.py](engine.py), [secular.py](secular.py),
[controls.py](controls.py), [model.py](model.py) and [state.py](state.py).

Units are ns, GHz and rad. `SecularModel` supplies energies, drive quadratures,
directional rates and pure dephasing in each quantized flux level's energy
basis. The model builder owns physical applicability. Records own read-only
array copies and reject invalid shapes or numeric invariants.

`EvolutionResult` contains densities at readout start in the DC eigenbasis.
The environment explicitly converts them to populations, then calls ADC's
normalization and sampling tools. Evolution neither imports ADC nor produces
I/Q. ADC owns its probability tolerance and readout sums; see the
[ADC README](../adc/README.md).

## Decision graph

[Geometry construction](_planning.py) combines pulse, flux and readout events.
Pulse support uses half-open absolute intervals. Flat-top plateaus of at least
10 ns become constant rows; shorter plateaus stay shaped. Evolution stops at
readout start. FTF readout then uses the DC model, ignoring residual flux tails
and overlapping controls.

[Dispatch](_dispatch.py) sees flux motion, active tones, envelope shape, known
inputs and acquire-wide prepare-key reuse. It does not use experiment names.
Candidate order and numerical validity are separate decisions. A rejected guard
tries the next candidate in the following graph.

| Row condition | Candidate order and numerical route |
| --- | --- |
| No drive, fixed or quantized varying flux | `t4`, analytic secular propagation with basis changes between flux levels. The varying-flux path is also called T5, not a separate kernel. |
| Driven continuous flux | `flux_dop853`, continuous Hamiltonian evaluation from `FluxExpansion`, with baths switching at quantized segment boundaries. |
| One constant tone, unshared driven input, duration at most 10 us | `floquet_period` → `dyson` → `t6`. |
| Other single constant tone | `dyson` → `floquet_period` → `t6`. Dyson checks perturbative ratios and an analytic B2 bound, except that certified no-drive prefixes need no B2 check. Source and shared-prefix densities supply known inputs; other later rows use an arbitrary-input bound. |
| One shaped tone | Try `shaped_sambe` on every eligible row, without a reuse threshold. Remaining rows try `gate_cache`, `period_table` and `dop853` in acquire-wide amortized-cost order. |
| Two constant tones | Compare acquire-wide DOP853 and Floquet costs before either guard. Try the cheaper family first, then the other if rejected. |
| Shaped two-tone or three-or-more-tone row | `dop853`, summing all drives. |

`t4` and `t6` are terminal in their domains. They must accept every dispatched
no-tone or single constant-tone row respectively. They have no DOP853 fallback.
The internal [registry](kernels/__init__.py) contains ten families. T7 is the
single-tone `dyson` family; the two-tone `floquet` family has its own modal
second-order Dyson implementation.

### Builders and variants

Dispatch also owns choices within a family. Kernels own numerical construction,
application and certificates that require their intermediate data.

| Family | Decision and certificate handoff |
| --- | --- |
| `floquet_period` | The period guard admits a single constant tone at `abs(f) >= 0.03 GHz` when its secular split bound passes. Dispatch selects dense Hilbert DOP853 at `abs(f) >= 1 GHz`. Below that, it compares Sambe work with CF4 coarse/fine work. A declined Sambe certificate uses standalone CF4, not gain anchors. For CF4 keys, dispatch fixes gain anchors from the complete gain group; the kernel checks interpolation and refines or builds independently when needed. |
| `gate_cache` | Dispatch sends a key-aligned builder mask. Positive durations up to 30 ns first try six phase nodes with two independent probes; failed convergence uses sixteen nodes. Other keys directly use sixteen nodes. Both builders produce four closed-system subsegments with secular Strang dissipation and the same phase-Fourier representation. |
| `period_table` and `shaped_sambe` | Ordinary period tables integrate frozen-envelope Hilbert periods; shaped Sambe uses finite-Hilbert-Fourier subperiods for real envelopes. Each family owns its guards and build. They share period playback and direct fractional tails. |
| `floquet` | The guard builds a Stark frame and returns row admissions. Passing k=0 omission/convergence bounds selects modal second-order Dyson when probe limits pass, otherwise periodic k=0 propagation. After k=0 omission rejection, a passing calibrated finite-Fourier certificate selects retained-sideband integration. Dispatch merges these admissions into one family plan. Apply does not repeat admission or construct sparse equations. |
| `dop853` and `flux_dop853` | Dispatch resolves complete chronological timeline routing, then assigns chain IDs to adjacent touching rows of the same direct family. Ordinary DOP853 uses the interaction picture only for independent, single-tone, nonzero-gain rows. Chains and other ordinary rows use the lab frame. Flux DOP853 has its own chain plan. |

The shaped and two-tone cost estimates are fixed before numerical guards.
Timeline workers borrow them without recounting subset reuse. Cache warmth,
worker width, chunking and apply-call composition do not change these choices.
Cost coefficients and numerical thresholds live in
[_shaped_dispatch.py](_shaped_dispatch.py),
[_two_tone_dispatch.py](_two_tone_dispatch.py) and the family modules.

[DOP853 planning](_dop853_dispatch.py) operates across prepared-payload
boundaries. The scheduler groups chain members by ID in chronological order;
it does not recognize direct-solver family names. A chain never crosses a
timeline or the `dop853`/`flux_dop853` boundary. Each row resets its adaptive
controller. Kernels own basis changes inside chains; the engine changes basis
between apply calls and returns the final density to the DC eigenbasis.

## Row plans and the kernel seam

[kernel.py](kernel.py) defines `SegmentTable`, `Kernel`, `Prepared`, `RowPlan`,
`ApplyContext` and guard handoff records. These are internal contracts, not
additional package integration entry points.

- `Prepared` is an opaque read-only numerical payload. `prepare` and guard
  handoffs build model-only data without reading density; `bind_input` may wrap
  that data in an evolve-local input-bound payload. `nbytes` charges owned arrays
  for the entire payload batch.
- `RowPlan` holds read-only decisions indexed by global `SegmentTable` row for
  one family in one evolve. Floquet stores admissions and retained equations;
  the direct solvers store frame or chain decisions. Plans never enter
  `PrepareCache` or persistent kernel state.
- `KernelContext(source, budget, work_plan)` supplies preparation, key and guard
  calls with one evolve owner. `ApplyContext(budget, plan, work_plan)` supplies the finite positive per-segment allowance
  and the family's plan. Floquet and both direct solvers require their own plan;
  a missing plan raises `ValueError`, and a wrong family type raises `TypeError`.
  Families without plans reject a non-`None` plan with `ValueError`.

Dispatch returns at most one plan per family and a separate global-row chain
array. [Scheduling](_schedule.py) keeps the originating plan through preparation
bindings and input-bound bindings, then supplies it unchanged in `ApplyContext`.
It neither slices nor interprets plans. A timeline worker resolves deferred
neighbors before building its local plan and chains. It does not mutate the
caller dispatch's shared decisions.

[WorkPlan](workplan.py) owns one evolve's lazy immutable facts, typed gain/carrier/
ramp group records, and single-flight resource handles. Prefix representatives
and timeline workers share the same plan. Table-indexed fact identities must
separate representative tables from the complete table. Builders publish owned
read-only copies and never retain densities or adaptive solver history.

Resource handles distinguish planned inventory from incrementally admitted
actual consumers. Open resources allow deferred timeline admission. Sealing
closes admission; the last actual handoff then releases the resource. Deferred
groups stay open until evolve exit. Engine drains its pool before closing the
plan on success, cancellation or failure. Cache ownership remains separate.

`admit_prepared` is the cache-first numerical admission hook. It returns current
row decisions and a row plan without construction when the payload domain covers
all required products. None means domain miss and retains the ordinary build
path. Production families currently decline this hook; cache lookup and routing
remain unchanged.

Guards are row-local: splitting or reordering a guard call must preserve its
acceptance mask. A guard may read the source initial density or an
engine-computed certified prefix input, never a worker's evolving density.
Guards do not evolve density. `guard_builds` identifies table-building guards;
`guard_prepared` hands accepted model-only products to apply without rebuilding.
Cross-timeline keys are guarded before shared preparation. Local builders run
inside their timeline worker and retain the same handoff.

`bind_input` can contract a common known input into an evolve-local read-only
payload. Dyson uses it for source or shared-prefix inputs. Prefix resolution
evolves one representative per identical prefix group in time order, including
nested and driven prefixes. Bound payloads never enter the prepare cache.
Returning `None` retains ordinary per-batch apply. Apply advances worker
densities in place and returns a row-aligned boolean exceedance mask.

## Algorithm ownership and reuse

Each family owns its build, apply and numerical certificates. Shared numerical
operations have one owner, rather than one kernel calling another kernel
class's methods.

| Shared operation | Owner and users |
| --- | --- |
| Rates → population generator Q and coherence decay | [kernels/_secular.py](kernels/_secular.py), used by static, split, modal and direct propagators. |
| Adaptive DOP853 tableau, tolerances and integration | [kernels/_dop853.py](kernels/_dop853.py), used by density solvers and Hilbert endpoint builders. Packing and chain traversal live in [kernels/_dop853_data.py](kernels/_dop853_data.py) and [kernels/_dop853_chain.py](kernels/_dop853_chain.py). |
| Ordinary density endpoint advancement | `advance_endpoints` in [kernels/dop853.py](kernels/dop853.py), used by ordinary DOP853 apply and period fractional tails with an explicitly supplied frame. |
| Period interpolation, secular playback and fractional tails | [kernels/_period_table.py](kernels/_period_table.py), shared by [period_table](kernels/period_table.py) and [shaped_sambe](kernels/shaped_sambe.py). |
| Gate Hilbert endpoint integration | [kernels/_gate_endpoints.py](kernels/_gate_endpoints.py), shared by sixteen-node and [adaptive phase](kernels/gate_phase.py) builders. |
| Constant-tone Hilbert period build and playback | [kernels/floquet_period.py](kernels/floquet_period.py), with [CF4 grid](kernels/_cf4_grid.py) and [Sambe](kernels/_period_sambe.py) helpers. Separate from two-tone Floquet frames. |
| Two-tone retained-sideband certificates and equations | [kernels/_floquet_sidebands.py](kernels/_floquet_sidebands.py), used by [floquet](kernels/floquet.py) guard admission and apply. |

The engine deduplicates prepare keys across an evolve. Keys used by distinct
timelines enter shared preparation. Local builders and repeated local keys stream
one timeline at a time; single-use ordinary keys prepare at their apply wave.
All classified keys, including singletons, can reuse the persistent LRU.
Accepted guard products are reused even when only a subset of timelines accepts.

The caller protects current cache hits and new shared payloads by their whole
allocation sizes. Workers submit model-only local products to a thread-safe
collector bounded by the remaining resident budget. After workers join, the
caller admits these products through the LRU. This can evict old workloads, but
streaming singletons cannot displace this evolve's protected allocations.
Oversize products stream without admission. Pending products and old residents
can coexist until the join; the resident limit is not a bound on that peak.

[PrepareCache](cache.py) namespaces whole model-only payload batches by family,
whole-source model identity and budget. The zero-level identity must describe the
entire source, including absolute DC flux and model parameters; row keys also
include quantized levels. A cache hit never replaces the row's guard. Eviction
releases cache ownership, not references already held by active work.

The resident-byte limit is not a process RAM cap. `live_bytes` counts only
registered evolve-scoped workspaces at their current size, not active payload
references or total live memory. Models, output densities, solver workspaces
and ADC preparations are separate allocations. Carrier
interpolation and gain-anchor workspaces belong to one evolve and are released
when it ends. Numba's compilation cache is independent of this prepare LRU.

## Accuracy, failures and lifecycle

`budget` is an absolute per-segment population and low-four allowance, not a
certified global error bound. Guards combine analytic bounds and calibrated
approximations. In particular, retained-sideband Floquet admission does not
certify aliasing beyond the Fourier grid, full finite-Fourier contractivity,
global ODE error or directed rounding of norms and eigensystems. Evolution does
not clip or renormalize density.

A post-apply exceedance keeps the uncorrected approximation. It does not trigger
a retry or reclassification. `PathDiagnostics.exceeded_by_kernel` counts those
segments by family, and one `RuntimeWarning` reports the evolve call's total.
All routing diagnostics use family names, not builder or frame variants. Shared
prefix counts and exceedances scale by the adopting timelines.
`max_trace_error` reports the maximum absolute complex trace-minus-one over final
readout densities, or zero for empty work. It never rejects output or applies
ADC's tolerance.

Invalid inputs fail at construction or entry. Missing required registry entries
raise with segment and family context. A rejected terminal guard raises
`RuntimeError` with `no valid kernel`. Preparation and apply failures propagate;
they do not restart candidate selection. Cancellation raises `Cancelled` without
partial results.

Calls on one Engine serialize. Joblib threading is the only parallel layer;
kernels are single-threaded and do not open pools. Multi-timeline calls with
multiple workers create and close a pool per evolve; empty work, one timeline
and one worker run inline. Cancellation is checked at chunk boundaries.
Native BLAS stays at one thread throughout evolve. Overlapping engines share a
process-level reference count and the last call restores the original limits;
external threadpoolctl users must not change them during active evolves.
`close()` waits for active work, clears the prepare cache and is idempotent.

`SEGMENT_NUMERIC_VERSION` is `segment-engine-v3`. `experiment.ftf` owns private
replay and checks backend, model, numerical, RNG and dependency versions before
recomputing a round. Evolution does not persist densities or offline tables.

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
