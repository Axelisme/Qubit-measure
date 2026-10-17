# Segment evolution and prepared-data reuse

**Last updated:** 2026-10-17, compiled shared Dyson path pairing and pruning

## Segment engine contracts

The batch API uses ns, GHz and rad. `SecularModel` carries energy eigenvalues,
drive quadratures, directional rates and pure dephasing at one quantized flux
level. `ModelSource` caches each level under a lock. The model builder owns
secular applicability and device limits. Engine records own read-only array
copies and reject invalid shapes and numeric invariants at construction.

`EnvelopeSpec` describes unity or sampled complex amplitudes. Its shared
`values` method interpolates real and imaginary parts and holds the final
sample. `TimelineBatch` supplies all points of a template. `EvolutionResult`
represents densities at readout start in the DC eigenbasis. `Engine` segments
all templates with array event sorting, selects kernels by name, and runs
independent joblib-threading chunks. Missing required registry entries raise
with the segment and kernel name. T4 implements exact zero-drive secular
propagation, including quantized flux subsegments with basis changes. When a
flux waveform varies during a driven segment, `flux_dop853` interpolates
`TimelineBatch.flux_waveform` and evaluates the exact `FluxExpansion` in its
RHS, including tails that stay inside one grid level. The model owner truncates
tails at its absolute threshold and holds the terminal DC sample. The bath stays
at the row's grid level and switches at segment boundaries.
Other algorithm families plug into the internal registry.

The engine counts prepare keys with `np.unique`. Repeated keys are prepared
in the shared A stage. Single-use keys are prepared inside their timeline
chunk and discarded afterward. The LRU charges whole opaque prepared batches;
a live acquire can retain evicted payloads until its chunks finish. The cache
namespace includes the kernel name, zero-level model identity and budget.
The zero-level identity must describe the entire source, including all model
parameters and absolute DC flux. Row keys also include quantized flux levels.

Chunk-local waves call apply once per group of next rows sharing a kernel and
prepared batch. Apply advances densities in place and returns a bool accuracy
exceedance mask aligned with its rows. Kernels without a post-apply check return
all False. An exceedance is not a numerical failure: the engine keeps the
uncorrected approximation without retry. `PathDiagnostics.exceeded_by_kernel`
counts exceeded segments by kernel and omits kernels with no exceedances.
Each `evolve` call with exceedances emits one `RuntimeWarning`, with the total
segment count and each kernel's count. Consecutive DOP853 rows form chains only within the same kernel
and prepared batch, never across `dop853` and `flux_dop853`. Their kernels own
basis changes inside a chain; the engine changes basis between apply calls
and returns the final state to the DC eigenbasis. Cancellation is checked at
chunk boundaries and raises `Cancelled` without partial results. `close()`
waits for active work, clears the prepare cache and is safe to repeat. Each
`evolve` call opens and closes its own joblib threading pool. The prepare cache
persists across `evolve` calls, but no worker resources survive a call.
`PathDiagnostics.max_trace_error` reports the maximum absolute complex
trace-minus-one across final readout densities for every evolve call, or zero
for an empty call. This diagnostic never changes, rejects or retries kernel
output and does not compare against an ADC tolerance. The environment explicitly
converts densities to populations and calls ADC's `normalize_populations` once
before sampling. ADC owns its 1e-6 trace and negative-residual contract and
reports rejected row counts in `ValueError`; evolution does not import ADC.

`SEGMENT_NUMERIC_VERSION` is `segment-engine-v3` and identifies engine numerics
for checked replay.

The internal `kernel` module defines `SegmentTable`, `Kernel` and the opaque
`Prepared.nbytes` contract. The optional `guard_prepared` hook returns a
`GuardResult` containing the ordinary row-local acceptance mask and model-only
`GuardPreparation` batches. Floquet returns its guard's preparations; other
kernels return None and the engine calls their ordinary guard. The engine reuses
payloads only for accepted rows with the same kernel, prepare key, source and
budget within this evolve. Shared keys follow normal PrepareCache admission;
singletons bypass the cache. Neither the hook nor the handoff retains kernel
state or creates a separate cross-evolve store. Only apply and `bind_input` receive density arrays.
The hook receives the source initial density or an engine-computed no-drive
prefix output, never worker densities. For shared prepared batches, the engine
calls the hook once per prepare key, port and identical input group before
timeline chunks start. The engine converts that input into the row model basis. A returned opaque
payload replaces the prepared argument to apply, with key indices all zero.
None keeps the ordinary apply path. Input-bound payloads are read-only and
belong to one evolve call. They never enter PrepareCache or kernel state.
Dyson contracts its paths and groups first-order factors in this hook. Other
kernels return None. The shared
contract suite lives in `tests/simulate/evolution/contract/kernel_suite.py`.
Kernel tests inherit `KernelContract` and provide a `kernel_case` fixture with
accepted rows, a synthetic model and input densities. The suite compares against
an independent strict DOP853 reference. It checks guard row locality across
single-row, unequal and reversed-order split calls, apply row splitting for
both densities and bool exceedance masks, mask shape and dtype, and preparation
without density access. It also checks guard handoff acceptance across row
subsets and compares handed-off payloads with fresh preparation. The suite also checks that input binding leaves its
inputs unchanged for both source and pre-evolved prefix inputs. Bound and
ordinary apply agree within budget across repeated, reversed, non-contiguous
and empty subsets. Prefix-bound Dyson results agree within the budget/8
shared pruning allowance. The demo kernel stays in
tests.

## Dispatch

The engine classifies segments by flux motion, active tones, envelope shape and
prepare-key reuse. It does not know the experiment name. Flat-top plateaus of
at least 10 ns split into constant segments; shorter flat-tops stay shaped.

| Segment condition | Kernel |
| --- | --- |
| Readout start | Stop quantum evolution and hand populations to the ADC tools. FTF readout uses the DC model and ignores residual flux tails and overlapping controls. |
| Varying flux, no drive | The FTF model owner quantizes flux at DAC-interval midpoints to 1e-3 Phi0 subsegments. This T5 path uses T4 with basis changes, not a separate kernel. |
| Varying flux with drive | `flux_dop853` integrates continuous flux with the exact Hamiltonian expansion; baths switch at quantized grid boundaries. |
| Fixed flux, no drive | T4 uses analytic secular propagation. |
| One constant tone, known input density, r, r₂ and B2 guards pass | T7 second-order Dyson, batched over frequency, gain, duration and phase. |
| One constant tone after an identical shared no-drive prefix, r and r₂ pass | Dyson with the actual prefix output. B2 does not apply; post-apply population changes only report. |
| Same constant-tone branch, Dyson fails but r <= 0.2 | Reduced RWA if its omitted-sideband error bound fits the segment budget; otherwise try the Hilbert period kernel. |
| One constant tone not accepted above, known or unknown input, with abs(f) >= 1 GHz and duration*T²*Gamma*V² <= 1e-2 | `floquet_period` integrates one Hilbert-space U(T), translates the carrier phase in time, and steps each period with secular Strang dissipation. |
| Same constant-tone branch, Hilbert period guard fails | T6 periodic propagation accepts every single constant-tone row. |
| One shaped tone, prepare-key usage >= 10 | Gate cache with four Strang subsegments, closed-system U plus secular dissipation, and 16 carrier-phase Fourier samples. |
| One shaped tone, prepare-key usage < 10 | Period table with midpoint-frozen envelopes, eight amplitude Chebyshev nodes and 16 phase Fourier samples; integrate the fractional-period tail directly. This branch has no per-segment accuracy guard. |
| Two constant tones with a valid Floquet frame and passing harmonic omission and convergence bounds | Floquet k=0 preparation. Passing probe guards select modal second-order Dyson in H_F + D₀; other probes use periodic propagation in the same accepted frame. |
| Other two-tone segments, including failed Floquet guards | Sum both drives in batched numba DOP853. |
| Three or more tones | Sum all drives in batched numba DOP853. |

T4, T6, gate_cache and period_table are terminal families for their dispatched
rows. Their guards accept every row in those domains, so these branches have no
DOP853 fallback. Custom kernels replacing these families must cover the same
domains. A rejecting terminal guard raises `RuntimeError` with `no valid kernel`.
Preparation or apply failures also raise; they never trigger reclassification.
DOP853 remains the fallback for rejected two-tone Floquet frames and the direct
path for shaped two-tone and three-or-more-tone segments.

Dyson's B2 guard uses the full source density in the segment model basis for rows
starting at zero. It takes actual pair coherence magnitudes and the off-diagonal
trace norm, so diagonal mixed inputs contribute no coherence terms. Other later rows use an arbitrary-input density bound, except certified flux-tail
rows. A flux-tail group requires a nonempty prefix with no tones and at least two
identical prefix sequences in the same source/evolve. Identity includes every
segment's start, duration and level, the target basis and continuous flux samples.
Different tail lengths never share an input just because their final levels match.
The planner compares samples before probe start and the first sample at or after
it to retain the prefix's interpolation boundary. Later samples do not affect
prefix identity. Prefixes with earlier drives or no identical partner retain
ordinary dispatch. Source and initial density are
common to one evolve and no prefix identity persists across calls.

The model supplies the secular form. Missing secular data fails at entry rather
than selecting another evolution backend.

### Accuracy and truncation

Ordinary Dyson rows require all three guards. Certified flux-tail rows require
only r and r₂, because the long-pulse coherence term in B2 can reject even a
small actual population change:

- r <= min(0.1, 0.25 * budget^(1/3)).
- B2 + budget/8 <= min(0.02, (1.5 * budget)^(2/3)).
- r₂ <= 0.05.

B2 bounds driven-minus-free population changes in unpruned second-order Dyson.
It sums transition-wise scalar bounds from the secular model and pulse parameters,
without evolving a density or evaluating modal paths. For rows starting at zero,
source populations determine population differences, actual |rho_ij| bounds pair
coherence, and the actual off-diagonal trace norm bounds coherent contributions.
These values are taken after basis conversion, without zeroing small coherence.
Coherence phases are bounded, not evaluated. Diagonal inputs have c = L = 0.
Later rows use the arbitrary-input version, with population difference at most 1,
pair coherence at most 1/2 and off-diagonal trace norm at most 2. The budget/8 margin reserves the
omitted-path allowance.

After apply, Dyson also measures |Δp| from the output populations minus the
same input's free-evolved populations. Free evolution uses only the secular
population block, without transforming coherence modes. The threshold remains
min(0.02, (1.5 * budget)^(2/3)). This check only reports exceedances through
`PathDiagnostics.exceeded_by_kernel` and the per-evolve warning. It does not
reject, recompute or correct the output, and does not guarantee accuracy.

The r limit follows cubic off-resonant coherence-error scaling. The population
limit follows the two-level resonant estimate (2/3)|Δp|^(3/2). Synthetic-model
scans calibrate and check the coefficients 0.25 and 1.5; neither threshold is a
rigorous multilevel error bound. At the default budget of 1e-3, the limits are
r <= 0.025 and B2 + 0.000125 <= 0.0131. Classification uses this bound to avoid
large |Δp|. A post-apply |Δp| check only reports excess; it does not correct the
Dyson result or guarantee exact-dynamics accuracy. Rejected Dyson segments try
reduced RWA only in the r <= 0.2 branch. RWA segments whose error bound exceeds
their budget try the Hilbert period kernel, then T6.

Each segment receives a default budget of 1e-3. The end-to-end acceptance
threshold is a maximum absolute error of 1e-2 in both final readout-basis
populations and low-four density entries, for experiments up to about 10 us.
A segment budget is not a runtime guarantee of the total error, especially for
unguarded period tables. DOP853 defaults to rtol=1e-5, atol=1e-7 and
max_step=0.1 ns, tightening tolerances for smaller budgets.

Flux-tail truncation uses an absolute flux bound, not elapsed multiples of a
filter time constant. FTF sums absolute pulse-tail contributions and truncates
below 5e-4 Phi0 on every loop.

The Hilbert period kernel uses T=1/|f| and the density-free splitting estimate
`duration*T²*Gamma*V²`. Here `V=2*pi*|gain|*(||X||₂+||Y||₂)` is an angular
drive norm bound. `Gamma` is the larger of the population-generator spectral
norm and the maximum coherence decay rate. The 1 GHz cutoff separates measured
6.789 GHz builds of about 3 ms/key from 0.4-0.5 GHz builds of about 35 ms/key.
Prototype S3/S5 population errors against strict DOP853 were 1.1e-7 and 2.3e-6.
The 1e-2 estimate threshold is heuristic, not a rigorous segment-budget guarantee.
Preparation retains U(T) and finite degree-seven dense coefficients, with every
array charged to the prepare cache. Duration, start and theta do not enter keys.
Time translation handles signed frequency and phase; the fractional tail uses
the same dense solution with a periodic extension across the boundary. Each full
period and tail applies D/2, U, D/2 without constructing a Liouville channel.
Apply does not compare half-period results, normalize, retry or reroute.
Low-frequency flux-tail probes retain Dyson and two-tone rows retain Floquet.

T6 uses 1, 4 or 8 steps per carrier period according to r, powers the period
propagator and adds the fractional-period tail. Carrier phase uses Fourier
interpolation. Gate-cache Strang splitting and phase Fourier interpolation
avoid building a full dissipative propagator for every phase.

### Chunk scheduling and parallelism

All non-reps sweep points enter as template arrays. Prepare does not depend on
input density; apply does. The engine deduplicates prepare keys across the
acquire. Keys used at least twice are prepared in shared A-stage chunks before
timeline work starts. Floquet's guard hands its model-only preparations to
A-stage, avoiding another build for matching keys. Guard-built singleton
payloads also reach timeline chunks without cache admission or input binding. Other single-use
preparations stay inside their timeline chunk and are discarded afterward. The input-bound hook is an A-stage exception to
model-only preparation. It contracts the source initial density or a shared
flux-tail prefix output once per shared key, port and input group. A-stage evolves
one representative of each needed prefix with the ordinary T4 payloads and basis
changes. The contracted payload lives only until this evolve finishes. Single-use
keys, later rows without a certified shared prefix and kernels returning None
keep the per-batch path. Dyson uses the maximum absolute gain and maximum duration of all shared
rows to build its path superset. The shared table drops conjugate path pairs
by increasing summed entrywise bound, up to budget/8, matching B2's reserved
margin. Shared contraction preallocates candidate arrays from the nonzero
coupling/input topology instead of building an intermediate tuple list.
Conjugate lookup uses integer keys in a preallocated hash array. Pairing and the
sequential greedy discard loop run in compiled, single-threaded code. Stable
bound sorting and accumulation order stay unchanged. This reduces A-stage cost
for small shared groups without adding a density-dependent reuse threshold.
Workers reuse retained paths and factors in original order without further
pruning. Ordinary per-batch tables keep their original cutoffs.
Fixed-size modal reconstruction keeps shared Dyson results bitwise identical
across worker counts and chunk splits.

Timeline chunks interleave estimated costs, with six chunks per worker by
default. Within a chunk, local waves batch next segments by kernel and prepared
payload. A timeline preserves apply order, but different chunks do not
synchronize waves. A-stage completion and final collection are the global
waiting points. Cancellation is checked between chunks.

Joblib threading is the only parallel layer. Worker numba kernels are
single-threaded and use `nogil`; there is no process pool, GPU or Rust backend.
Each `evolve` call opens and closes its threading pool. Across acquires, the
engine retains only its prepare cache, not worker resources. Numba results need
not match NumPy bit for bit; numerical acceptance uses the observable-error
thresholds above, not a separate NumPy backend.

## Default execution and replay

FTF acquisition uses this engine's fast segment algorithms by default. Kernel
selection and preparation happen inside the engine, not in the caller. The
prepare cache holds deterministic numeric payloads in memory. It is not an
offline U-table library, and evolution does not persist exact density records.
There is no database builder or GPU dependency.

`experiment.ftf` owns private replay inputs and composes evolution with readout.
Replay recomputes a round from its archived model, templates, snapshot and
sampling identity. The Hilbert period dispatch and numerics use
`segment-engine-v3`; earlier numeric versions fail checked replay.
It requires matching `SEGMENT_NUMERIC_VERSION`, backend,
model, RNG and dependency versions. No record references pin old densities or
offline tables. Older exact-record manifests fail explicitly.

The prepare-cache budget charges prepared payloads, not process RSS. Models,
returned densities, live chunk references, solver workspaces and ADC preparations
are separate allocations. Clearing the cache or closing the engine releases
in-memory preparations without deleting replay inputs.
