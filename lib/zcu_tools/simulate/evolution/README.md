# Segment evolution

**Last updated:** 2026-10-19, WorkPlan ownership, joint-basis multi-tone RWA including dual constant drives with recalibrated phase admission, simultaneous and single-tone covariant gate reuse with compiled builders and fact-backed admissions, unified resident/pending budget, long-period playback, guarded CF4 carrier anchors and shared coarse-node eigensystems, single-tone block RWA with iterative omission-driven retention, exact final-mask graph sharing and shared omission admission

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
prepare cache and a separate exact-result memo. `evolve(source, batches, *, budget=1e-3, cancel=event)` consumes a
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
Pulse support uses half-open absolute intervals. Tone columns hold only the
maximum simultaneously active count, packed in original track order, not the
number of historical pulses. Flat-top plateaus of at least
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
| One constant tone, unshared driven input, duration at most 10 us | `single_tone_rwa` → `floquet_period` → `dyson` → `t6`. |
| Other single constant tone | `dyson` → `single_tone_rwa` → `floquet_period` → `t6`. Dyson checks perturbative ratios and an analytic B2 bound, except that certified no-drive prefixes need no B2 check. Source and shared-prefix densities supply known inputs; other later rows use an arbitrary-input bound. |
| One shaped tone | Try `single_tone_gate` when complete-acquire cold build plus playback is at least 20% cheaper than Sambe. Rejected or unamortized rows try `shaped_sambe`, without a reuse threshold. Remaining rows try `gate_cache`, `period_table` and `dop853` in acquire-wide amortized-cost order. |
| Two constant tones | Try `multitone_rwa` first for distinct ports. Same-port and rejected rows retain the acquire-wide DOP853/Floquet cost order, cheaper family first. |
| Shaped two-tone or three-or-more-tone row | `multitone_gate` → `multitone_rwa` → `dop853`. RWA admits only fixed-flux, separated weak carriers with smooth real envelopes and at least one sampled envelope. Other rows retain the full-drive fallback. |

`t4` and `t6` are terminal in their domains. They must accept every dispatched
no-tone or single constant-tone row respectively. They have no DOP853 fallback.
The internal [registry](kernels/__init__.py) contains thirteen families. T7 is the
single-tone `dyson` family; the two-tone `floquet` family has its own modal
second-order Dyson implementation.

### Builders and variants

Dispatch also owns choices within a family. Kernels own numerical construction,
application and certificates that require their intermediate data.

| Family | Decision and certificate handoff |
| --- | --- |
| `floquet_period` | The period guard admits a single constant tone at `abs(f) >= 0.03 GHz` when its secular split bound passes. Dispatch selects dense Hilbert DOP853 at `abs(f) >= 1 GHz`. Below that, it compares Sambe work with CF4 coarse/fine work. A declined Sambe certificate uses standalone CF4, not gain anchors. For CF4 keys, dispatch fixes gain anchors from the complete gain group. Constant-CF4 inventories with at least sixteen same-sign carriers and a magnitude ratio at most 1.01 can instead use four real carrier anchors at fixed gain. Gain groups take precedence, so a key never expands both axes. Carrier prediction interpolates interaction-picture prefixes, restores the target carrier's lab frame and checks the original nine coarse-grid probes. Rejection refines the already-built coarse grid; incompatible anchor grids build independently. Prepare keys and row admission do not change. |
| `gate_cache` | Dispatch sends a key-aligned builder mask. Positive durations up to 30 ns first try six phase nodes with two independent probes; failed convergence uses sixteen nodes. Other keys directly use sixteen nodes. Both builders produce four closed-system subsegments with secular Strang dissipation and the same phase-Fourier representation. |
| `period_table` and `shaped_sambe` | Ordinary period tables integrate frozen-envelope Hilbert periods; shaped Sambe uses finite-Hilbert-Fourier subperiods for smooth real or complex envelopes. Complex playback interpolates magnitude and adds the envelope angle to the advancing carrier phase. It uses a tighter envelope-motion bound; unresolved ramps and failed certificates retain fallback. Each family owns its guards and build. They share period playback and direct fractional tails. |
| `floquet` | The guard builds a Stark frame and returns row admissions. Passing k=0 omission/convergence bounds selects modal second-order Dyson when probe limits pass, otherwise periodic k=0 propagation. After k=0 omission rejection, a passing calibrated finite-Fourier certificate selects retained-sideband integration. A cached sideband frame extends to missing probe bins without rebuilding the frame or existing certificates. Dispatch merges current-row admissions into one family plan. Apply does not repeat admission or construct sparse equations. |
| `single_tone_gate` | Dispatch compares summed canonical-key four-U builds, playback and guard/key work against carrier/shape-domain Sambe construction, period/envelope playback and fractional tails. Estimates use joint dimension, duration, retained drive strength, shape motion and carrier frequency. A 20% cold-cost margin selects this candidate before numerical admission. The kernel requires exactly one smooth real sampled tone and shares RWA, phase-graph pruning and secular split admission with simultaneous gates. Sliced support, nonzero envelope offsets and final sample hold are allowed. Rejection retains Sambe. |
| `multitone_gate` | Complete-inventory canonical reuse of at least 20 rows, shared RWA admission, coextensive real sampled support, compatible dressed phase charges (or explicitly admitted weak-cycle pruning), and a four-way secular split proxy at most 0.01. Builds four joint Hilbert propagators using the same `RWACoefficients.diagonal` window-excluded shifts and `interaction_hamiltonian` as ordinary RWA, plays D-U-D, and restores the lab/DC endpoint frame. Rejection retains ordinary RWA. |
| `single_tone_rwa` | After Dyson, single constant rows try a constant rotating-frame Lindbladian built from the shared retained graph, coefficient diagonal and full secular bath. All populations and within-component coherences form a real block of dimension at most 64. Cross-component coherences use small complex exponentials or scalar closed forms. Apply skips only exact-zero cross-block RHS, preserves arbitrary complex residuals and restores both phase endpoints. Admission shares the constant-only omission rule below, the 0.06 rad phase cap and explicit graph-pruning bound. Rejection keeps the original period/Dyson/T6 order. |
| `multitone_rwa` | Revision `joint-rwa-v3` retains joint-basis Fourier edges with `abs(Ei-Ej±f) < 0.05 GHz`, obtains window-excluded single-tone second-order shifts from `RWACoefficients.diagonal` and assembles them through `interaction_hamiltonian`, and integrates every tone in a row-local interaction picture with a 16 ns adaptive ceiling for dual constant rows and 4 ns otherwise. Dual constant admission shares the unified omitted one/two-photon estimate with single constant RWA. It restores Schrödinger coordinates at every endpoint. It keeps joint rates, dephasing and state ordering. This calibrated approximation does not certify exact RB fidelity. |
| `dop853` and `flux_dop853` | Dispatch resolves complete chronological timeline routing, then assigns chain IDs to adjacent touching rows of the same direct family. Ordinary DOP853 uses the interaction picture only for independent, single-tone, nonzero-gain rows. Chains and other ordinary rows use the lab frame. Flux DOP853 has its own chain plan. |

For the same whole source, budget and Stark key, moving a probe window reuses
cached Fourier candidates. Before shared/deferred guards split, the caller
computes only missing complete-bin certificates from the full table inventory.
WorkPlan single-flights the small new certificate publication. The immutable
union borrows the original frame arrays and existing certificates. Cache
replacement charges the union once instead of retaining two whole-frame charges;
active references to the old payload remain valid. Current phases, gains and
durations still receive fresh admission and sparse plans. A legacy-only frame
without Fourier candidates uses the ordinary rebuild if sidebands become needed,
so no FFT reconstruction or approximate frame reuse changes rounding.

Constant-tone period playback uses real-matrix binary powering of the complete
secular `D/2 → UρU† → D/2` channel for unsubdivided 16-level rows with at least
4096 complete periods. It preserves the existing splitting and complex density
residuals, but changes floating-point rounding. Shorter rows, other dimensions
and subdivided playback retain sequential updates with reusable buffers.
Fractional tails keep their existing endpoint and bath steps. Power matrices
are worker-local scratch, not retained prepare-cache products.

The shaped and two-tone cost estimates are fixed before numerical guards.
Timeline workers borrow them without recounting subset reuse. Cache warmth,
worker width, chunking and apply-call composition do not change these choices.
Cost coefficients and numerical thresholds live in
[_shaped_dispatch.py](_shaped_dispatch.py),
[_two_tone_dispatch.py](_two_tone_dispatch.py) and the family modules.

Multi-tone RWA admission is explicit in
[kernels/multitone_rwa.py](kernels/multitone_rwa.py). Every active carrier has
`abs(f) >= 0.05 GHz`, and pairwise carrier magnitudes differ by at least
0.1 GHz. The sum over tones of
`peak(abs(envelope))*abs(gain)*(norm(X,2)+norm(Y,2))/abs(f)` is at most 0.25.
Envelopes are real. Their piecewise-linear slopes divided by peak magnitude
are at most 0.025/ns. Differences between adjacent slopes divided by their
midpoint spacing and peak magnitude are at most 0.003/ns². Zero envelopes and
constant companion envelopes pass the smoothness check. Either at least one
envelope is sampled, or the row has exactly two constant envelopes on distinct
ports. Same-port constant Stark/probe spectra retain Floquet/DOP853 because
small population errors can still shift a narrow resonance peak. Continuous
flux must be constant over the complete row.
Budgets below 1e-3 decline this approximation. Admission also requires
`2*pi*duration*sum((peak*gain*norm(V_ret,2))**2/abs(f)) <= phase_limit`, where
`V_ret = (X-iY)/2` after the positive-frequency edge mask. Sampled rows and
single constant RWA retain `phase_limit = 0.06 rad`. Distinct-port dual constant rows use `0.16 rad`,
calibrated against strict lab populations with a twofold safety margin from
the smallest failing proxy in a 96-point flux/phase/gain/duration scan.
The kernel docstring records the grid and errors. This row-local
counter-rotating phase proxy accounts for long pulses even at small
strength/carrier ratios; it does not bound accumulated sequence error. In the
integer 256 ns sampled amplitude sweep, gain 0.4184 had proxy 0.063 rad and
population error above 1e-2. Those sampled rows still return to lab DOP853.
This proxy is calibrated, not a rigorous Stark-shift bound. The constant cap
enters only constant-row prepare keys; sampled and gate policy identities
remain unchanged.

Single and distinct-port dual constant rows share one omitted-coupling rule.
It checks window-excluded one-photon terms and every same-tone/cross-tone
two-step Fourier path, including signed sums and differences against full
dressed gaps. Both-retained paths already evolve in the RHS. Static
single-tone diagonal terms already live in `RWACoefficients.diagonal`.
Every other two-step path divides its coefficient product by the first omitted
detuning, or the reversed second detuning when only that edge is removed.
Distinct orders and intermediate states are summed; identical carriers are not
duplicated. With effective coupling `g` and detuning `delta`, every omission uses
`P = min(4*abs(g)**2/(4*abs(g)**2+delta**2), 4*abs(g)**2*duration_ns**2)`.
Zero coupling contributes zero. Distinct-port dual constants reject any
`P > 5.8e-3`, including when no edges are retained. Single constant rows instead
start at the 50 MHz window, restore over-threshold one-photon edges and the
omitted first-order edges in over-threshold two-photon paths, then recompute
until all remaining estimates pass. The resolved set determines Fourier terms,
virtual shifts, graph facts and prepare identities. Strength, phase, explicit
pruning and the 64-dimensional real-block cap still apply. Multitone and shaped
policies do not perform this iteration. This calibrated proxy is not an error
certificate.

The cap halves the smallest failing combination-channel proxy (0.0116496) in
the existing 25-point strict-lab scan. A 96-point single-constant scan covering
four flux positions, phase, gain, duration and window-crossing carrier offsets
has no failures below 0.045813; its 14 admitted points have population error
at most 0.002947. The same-tone counterexample has proxy 0.015629. The stricter
shared cap remains 0.0058. Rechecking the dual-constant calibration retains
24 of its 36 previously admitted points; twelve intermediate-flux points now
use the full-drive fallback. The 0.16 rad dual-constant phase cap is unchanged.
Shaped RWA and covariant gate admission do not use this constant-only rule.

Dual constant integration retains the adaptive solver and its error estimator,
with a 16 ns ceiling. Constant envelopes have no sampled internal features.
The larger ceiling reduces work after the fast carrier phase has been removed;
it is independent of phase admission and does not drop the joint dissipator.

The 50 MHz edge window and 4 ns ceiling come from simultaneous 128–512 ns
Gaussian gate measurements on a 16-level joint model. The prototype's maximum
population difference against lab DOP853 was 1.76e-3 through depth 16.
The calibrated half-flux pair has summed full-operator/carrier ratio 0.217.
The sampled 128 ns Gaussian with sigma 32 ns has normalized slope 0.0192/ns
and curvature 0.00231/ns², including DAC quantization. Admission uses 0.25,
0.025/ns and 0.003/ns² to leave margin around those measured pulses, rather
than treating arbitrary samples as smooth. The strength and carrier-separation
limits restrict use to weak, spectrally separated drives. They are admission limits, not rigorous omission-error
bounds. Preparation also computes one float64 coefficient per key, tone and
joint state: `sum(j,s, abs(C_s[i,j])**2/(Ei-Ej+s*f))`, restricted to
`abs(Ei-Ej+s*f) >= 0.05 GHz`, with `C_+=(X-iY)/2` and `C_-=C_+†`.
The Hamiltonian multiplies this diagonal shift by the actual
`gain**2 * abs(envelope(t))**2`, including ramps and envelope offsets.
Retained edges are excluded before division, so exact resonances are not
divided by zero or counted twice. The shift is phase independent and applies
to every joint state. It preserves phase covariance and adds no density,
rate or state-space reduction. Mixed-tone terms, micromotion and higher orders
remain omitted. The existing admission limits remain necessary; the diagonal
correction does not repair integer strong-drive plateaus.
Full policy revision, budget, level, timing, every active carrier,
port, gain, phase and envelope content enter RWA prepare keys. Tone phase
origins retain the ordinary SegmentTable contract.

### Covariant RWA gates

The named `single_tone_gate` and `multitone_gate` approximations share the
four-U builder and D-U-D playback in [kernels/_rwa_gate.py](kernels/_rwa_gate.py).
Their numerical guards share the calibrated RWA policy and the
`RWACoefficients` builder with `multitone_rwa`. All integrators call the same
`interaction_hamiltonian` assembly. All driven diagonal terms come from that
coefficient record; the gate cache does not assemble a separate drive RHS. It does not multiply independent qubit maps.
The model-only [phase graph](kernels/_rwa_graph.py) publishes integer dressed
charges satisfying `q[i]-q[j]=e[k]` for every retained edge of tone k. It orders
constraints by decreasing coefficient magnitude, with deterministic index ties.
Inconsistent cycles and coefficients at or below `1e-12 GHz/gain` are explicit
omissions. Their positive-operator spectral norms remain in the fact.

The named pruning policy admits only
`4*pi*duration*sum(abs(gain)*envelope_peak*omitted_norm) <= min(budget,1e-3)`.
This state-independent Duhamel/trace-distance bound controls the additional
edge removal relative to ordinary RWA for one row. It does not certify a whole
RB sequence or the RWA-vs-lab approximation. Strong incompatible cycles fall
back to ordinary RWA. Shared `phase_graph`, `pruning_bound` and
`pruning_admitted` are also available to constant-frame algorithms.

Simultaneous gates require every active envelope to be sampled and to
cover the same complete row support. Complex, partially overlapping and sliced
supports retain fallback. The four-way secular split uses the existing gate
proxy `duration**3*Gamma*V**2/16 <= 0.01`, with V the sum of retained joint
operator peak strengths. Diagonal shifts commute with the secular dissipator
and do not enter V. This is calibrated admission, not a strict splitting
certificate. Each quarter integrates a joint Hilbert propagator at
`rtol=1e-9`, `atol=1e-11`, with a 4 ns ceiling, in an interaction picture.
Population rates and pure dephasing remain joint and unchanged.

Keys include source/model identity, level, bath arrays, signed ordered carriers,
ports, gains, envelope identities, offsets, duration, graph policy, budget and
numerical controls. Only guard-admitted covariance permits reuse without theta
or absolute start. Playback uses actual row-start theta and both frame boundary
rotations. Four U, half-step secular maps and compact active-tone charges are read-only
owned arrays charged to the existing resident budget. There is no phase grid or global memo.
Whole-acquire choices are fixed before guards. Warm cache, worker width and
chunking do not change simultaneous gates' threshold of 20, above the measured
3–19-row break-even range. Single-tone gates instead compare cold costs in
`_shaped_dispatch.py`; a small inventory stays on Sambe even when a matching
gate is resident. No planner predicts later acquires or recognizes RB labels.
Single-tone covariance and the secular split are approximations relative to
full-drive Sambe, not exact fidelity certificates. Only cost-selected,
numerically admitted shaped rows change numerical results.

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
  Families without plans, including all three RWA families, reject a non-`None` plan with `ValueError`.

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
Cross-timeline keys are guarded before shared preparation. Local builders run
inside their timeline worker and retain the same handoff.

`bind_input` can contract a common known input into an evolve-local read-only
payload. Dyson uses it for source or shared-prefix inputs. Prefix resolution
evolves one representative per identical prefix group in time order, including
nested and driven prefixes. Bound payloads never enter the prepare cache.
Returning `None` retains ordinary per-batch apply. Apply advances worker
densities in place and returns a row-aligned boolean exceedance mask.

## WorkPlan and cache ownership

[Dispatch](_dispatch.py) creates one [WorkPlan](workplan.py) from the complete
evolve table before workers start. Prefix representatives, caller classification
and timeline workers share it through `KernelContext` and `ApplyContext`. Engine
drains its pool before closing the plan on success, cancellation or failure.
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
checks. Resident constant-RWA payloads still own and charge their array copies.
The shared facts include envelope peak/slope/real flags, period split rates, packed
DOP853 samples, covariant gate admissions and small Floquet probe-bin certificate
extensions. Gate admission facts use full canonical physics, budget and numerical
policies. Each row still checks its own continuous flux support before borrowing an
admission; only absolute start and covariant phase are omitted from the fact.
Floquet consumes full Fourier averages during frame construction. Its compact
frames and optional sideband candidates belong only to active preparations or
budget-admitted cache ownership, never evolve-long facts. Model facts include the full
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
actual key builds. Gain anchors use single-flight immutable resources. Cache
hits and rejected rows are not actual consumers. Caller-only groups seal before
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
Neither the kernel registry nor the resident cache retains this workspace.

### Cache-first and singleton admission

`admit_prepared` returns current row decisions and a row plan without construction
when the cached payload domain covers all required products. `None` means domain
miss and retains the ordinary build path. Dispatch looks up resident guard
products in the caller and passes strong references to deferred workers.
Floquet reuses frames only when all needed probe bins are present. Shaped Sambe
reuses key-covered convergence data. Both hooks recheck current rows without
building numerical products. Rejection of every row is not a domain miss.

The engine deduplicates prepare keys across an evolve. Keys used by distinct
timelines enter shared preparation. Local builders and repeated local keys stream
one timeline at a time; single-use ordinary keys prepare at their apply wave.
All classified keys, including singletons, can reuse the persistent LRU.
Accepted guard products are reused even when only a subset of timelines accepts.

Before prefix evaluation or numerical guards, dispatch inventories every
candidate family's prepare keys in the complete WorkPlan geometry. The caller
keeps intersecting whole resident batches protected, including hits used only
by a later prefix or suffix. Key and model/family indexes locate those hits
without scanning historical residents. Other residents stay in the reclaimable
LRU until capacity is needed. Array bytes and retention charges update only on
admission and eviction. Each immutable payload wrapper measures its array charge
once. Resident and pending allocations share one byte account across all prefix
and timeline schedules.

Workers offer model-only products through a locked collector. They never perform
caller lookups or LRU touches; the collector may remove reclaimable stale
ownership under its admission lock. An accepted offer reclaims pending failed guards and stale residents
only when capacity is needed. Failed stale residents yield before ordinary LRU
residents. An impossible or oversize offer leaves existing ownership unchanged
and streams without retention. Current whole-batch hits cannot be evicted by
worker offers. Capacity left idle preserves products for alternating experiments.
Caller shared builds and exact domain extensions use the same account. Exact
extensions replace their predecessor's whole charge; rejected extensions grow
only into idle capacity and retain their low priority.

After each schedule joins, the caller publishes pending products for later
prefix/suffix lookups. After the evolve pool drains, it releases current protection as newer than
surviving stale residents, even on failure or cancellation. This visits current
hits and products only, not the historical stale inventory. Fully
rejected guard handoffs stay separate from apply bindings and fill only idle
capacity. Schedule consumes Dispatch's rejected handoff queue on offer, so declined
or evicted products have no remaining Dispatch owner. Current row, probe-bin,
domain and budget checks still run on every
evolve. The unified account bounds retained arrays plus an estimated 4608 metadata bytes
per whole allocation. The estimate covers Python/NumPy headers, keys and cache
indexes, with headroom above the measured wide-scan mean of 3998 bytes. It does
not bound transient builders, active nonretained/oversize bindings, input-bound
data or process RSS. `PrepareCache.bytes` reports arrays only;
`PrepareCache.charged_bytes` includes the metadata estimate used for admission.

[PrepareCache](cache.py) namespaces whole model-only payload batches by family,
whole-source model identity and budget. The zero-level identity must describe the
entire source, including absolute DC flux and model parameters; row keys also
include quantized levels. A cache hit never replaces the row's guard. Eviction
releases cache ownership, not references already held by active work.

Exact-input repeats bypass geometry, planning and every kernel call. The
Engine-local result LRU retains owned densities, points and diagnostics, not
ADC or random samples. Its key covers the whole-source DC model identity,
initial density, optional flux expansion, budget and complete batch content,
including envelopes and continuous flux samples. Kernel collections and numerical
implementations stay fixed for the Engine's process-local lifetime. Custom and
production kernels must be deterministic and pure for identical inputs, as with
prepare reuse. Returned copies cannot mutate memo entries. Cancellation still
raises without a partial result, and cached exceedances still warn per call.

The result LRU charges arrays and metadata up to `min(64 MiB, cache_bytes)` in
addition to the prepare budget; zero disables both caches. `close()` clears both.
All Engines include the same per-allocation metadata estimate in retention
charges. FTF production uses a 2 GiB unified resident/pending budget for arrays
and estimated metadata, plus at most 64 MiB for results. The resident-byte limit is not a process RAM cap. Models, output densities, solver scratch, WorkPlan products,
active payload references and ADC preparations are separate allocations. The
process peak acceptance limit is 3.8 billion bytes, leaving 5% below the decimal
4 GB ceiling. Numba's compilation cache is independent of the prepare LRU.

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

`SEGMENT_NUMERIC_VERSION` is `segment-engine-v4`. `experiment.ftf` owns private
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
