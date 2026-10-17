# Segment evolution and prepared-data reuse

**Last updated:** 2026-10-17, Dyson-to-periodic constant-tone dispatch, terminal-family coverage, readout trace diagnostics, low-frequency Hilbert propagation, compiled shared Dyson tables, shared driven-prefix inputs, amortized shaped dispatch, model-only shaped guards, port-local period tables and low-carrier CF4 propagators

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
state or creates a separate cross-evolve store. Guard may read the source initial density or a certified, engine-computed
prefix input, never a worker's evolving density. Guard never evolves density.
Only apply and `bind_input` receive worker or binding density arguments.
The hook receives the source initial density or an engine-computed prefix output. For shared prepared batches, the engine
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
| One constant tone, known or unknown input, r, r₂ and B2 guards pass | T7 second-order Dyson, batched over frequency, gain, duration and phase. |
| One constant tone after an identical shared driven prefix, r, r₂ and actual-input B2 pass | Dyson with the actual prefix output and a shared contracted table. |
| One constant tone after an identical shared no-drive prefix, r and r₂ pass | Dyson with the actual prefix output. B2 does not apply; post-apply population changes only report. |
| One constant tone not accepted above, known or unknown input, with abs(f) >= 0.03 GHz and duration*T²*Gamma*V²/m² <= 1.0 for some m in 1..64 | `floquet_period` builds one Hilbert-space U(T), translates the carrier phase in time, and steps each period with secular Strang dissipation. |
| Same constant-tone branch, Hilbert period guard fails | T6 periodic propagation accepts every single constant-tone row. |
| One shaped tone | Order gate cache, period table and DOP853 by amortized preparation plus apply work. Try each numerical guard in that order. There is no fixed gate-key reuse threshold. |
| Shaped gate cache passes its split guard | Four Strang subsegments, closed-system U plus secular dissipation, and 16 carrier-phase Fourier samples. |
| Shaped period table passes its envelope-motion, bandwidth and split guards | Midpoint-frozen envelopes, eight amplitude Chebyshev nodes and 16 phase Fourier samples for the actual drive port. Integrate the fractional-period tail directly. |
| Shaped approximations rejected, or direct integration costs less | DOP853, with no input-state restriction. |
| Two constant tones with a valid Floquet frame and passing harmonic omission and convergence bounds | Floquet k=0 preparation. Passing probe guards select modal second-order Dyson in H_F + D₀; other probes use periodic propagation in the same accepted frame. |
| Other two-tone segments, including failed Floquet guards | Sum both drives in batched numba DOP853. |
| Three or more tones | Sum all drives in batched numba DOP853. |

T4 and T6 are terminal families for their dispatched rows. Their guards accept
every row in those domains, so these branches have no DOP853 fallback. Custom
kernels replacing these families must cover the same domains. A rejecting
terminal guard raises `RuntimeError` with `no valid kernel`. Shaped families can
reject rows before preparation and fall back to another shaped family or DOP853.
Preparation or apply failures still raise; they never trigger reclassification.
DOP853 remains the fallback for rejected two-tone Floquet frames and the direct
path for shaped two-tone and three-or-more-tone segments.

Shaped dispatch charges gate preparation per complete gate key and period-table
preparation per level, port, signed frequency and amplitude band. It counts keys
across the evolve call, including unknown-input rows. For duration tau in ns,
carrier magnitude f in GHz, and gate/table key counts qg and qp, its relative
work scores are:

- DOP853: `tau * (1 + 0.3*f)`.
- Gate cache: `tau * (60 + 3*f) / qg + 0.04`.
- Period table: `(150 + 550/f) / qp + c*tau*f + 5*tail*(1 + 0.3*f)`.

Here tail is the fractional-period duration. The coefficient c is 0.008 for real
envelopes and 0.064 for complex envelopes. DC skips the table score. These scores
come from single-threaded prepare/apply comparisons on 16-level FTF models over
50 MHz to 7 GHz, 30 ns to 10 us, and reuse counts 1, 2, 5, 9, 10 and 20. They are
cost estimates, not timing probes or numerical guards. Short-pulse scans use all
six reuse counts; long-pulse comparisons use one row. Port-local tables and
removal of the common energy zero reduce preparation work. Real-envelope apply
contracts the phase Fourier coefficients once per row for positive and negative
amplitudes, then evaluates only the amplitude polynomial each period. Complex
envelopes retain their time-varying phase. Fractional tails and DC reuse the
compiled DOP853 kernel with model-only prepared data. The tail shifts carrier
phase and envelope-local time together and retains strict rtol <= 1e-10 and
atol <= 1e-12. Its prepared arrays count toward the period-table cache charge.

Dyson's B2 guard uses the full known density in the segment model basis. For
rows starting at zero this is the source initial density. For shared prefixes
it is the engine-computed prefix output. Actual pair coherence magnitudes and
the off-diagonal trace norm enter the bound. Diagonal mixed inputs therefore
contribute no coherence terms. Later rows without a shared prefix use an
arbitrary-input density bound. Certified no-drive prefixes retain their B2
exemption; driven prefixes must pass actual-input B2.

One prefix mechanism covers driven and no-drive histories. A group requires at
least two identical nonempty segment sequences in one source/evolve. Identity
includes every segment's start, duration, level and active tone port, frequency,
gain, phase and envelope, plus the target basis and continuous flux samples.
Envelope identity uses content, not a batch-local index. The planner compares
samples before probe start and the first sample at or after it to retain the
interpolation boundary. Later samples do not affect prefix identity. Matching
final states or final levels alone never certifies a shared input.

Before dependent probes are classified, the engine resolves each group in time
order with ordinary dispatch, preparation, apply and basis changes. Nested
groups resume from their latest resolved prefix input. Known inputs belong to
this evolve alone. They never enter PrepareCache. Shared input binding consumes
the same known input as B2, in the row model basis. All constant single tones
still enter the ordinary family decision order.

The model supplies the secular form. Missing secular data fails at entry rather
than selecting another evolution backend.

### Accuracy and truncation

Ordinary Dyson rows, including segments after earlier drives, require all three
guards. Certified flux-tail rows require only r and r₂, because the long-pulse
coherence term in B2 can reject even a small actual population change:

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
Unshared later rows use the arbitrary-input version, with population difference at most 1,
pair coherence at most 1/2 and off-diagonal trace norm at most 2. These bounds
cover unknown inputs after non-certified prefixes without
reading worker densities. Such rows keep ordinary per-batch apply, not A-stage
input contraction. The budget/8 margin reserves the omitted-path allowance.

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
the Hilbert period kernel, then T6.

Each segment receives a default budget of 1e-3. The end-to-end acceptance
threshold is a maximum absolute error of 1e-2 in both final readout-basis
populations and low-four density entries, for experiments up to about 10 us.
A segment budget is not a runtime guarantee of the total error. Shaped guards
use model and envelope data, never initial or worker densities. Let V be the
largest spectral norm of the selected drive's two quadratures, Gamma the largest
outgoing or pure-dephasing rate, A the envelope peak, L its largest sampled
piecewise-linear slope, and g the signed gain. Period tables require:

- `2*pi*abs(g)*V*L/f**2 <= 0.05` for envelope motion within a period.
- `2**band*V/f <= 0.5` for amplitude/phase interpolation bandwidth.
- `tau*Gamma*(abs(g)*A*V)**2/f**2 <= 0.01` for secular splitting.

The band covers peak gain by a power of two. DC and zero gain use direct
integration inside the period family. Gate cache requires
`tau**3*Gamma*(abs(g)*A*V)**2/16 <= 0.01` for its four fixed subsegments.
These are calibrated rejection estimates, not multilevel error proofs. They
prevent unresolved fast ramps and strong dissipative long gates from silently
using fixed-order approximations. They can conservatively reject low carriers;
DOP853 then preserves accuracy at a higher apply cost. Neither shaped family
performs a density-dependent retry or post-apply correction.

DOP853 defaults to rtol=1e-5, atol=1e-7 and
max_step=0.1 ns, tightening tolerances for smaller budgets.

Flux-tail truncation uses an absolute flux bound, not elapsed multiples of a
filter time constant. FTF sums absolute pulse-tail contributions and truncates
below 5e-4 Phi0 on every loop.

The Hilbert period kernel uses T=1/|f| and the density-free splitting estimate
`duration*T²*Gamma*V²/m²`. Here `V=2*pi*|gain|*(||X||₂+||Y||₂)` is an angular
drive norm bound. `Gamma` is the larger of the population-generator spectral
norm and the maximum coherence decay rate. The guard requires abs(f) >= 0.03 GHz
and chooses the smallest integer m in 1..64 that keeps this estimate <= 1.0.
Each period is divided into m equal Strang steps. Their unitary endpoints come
from the same Hilbert trajectory. The tail uses steps no longer than T/m.
The count depends on the model and row duration, not the input density, and does
not enter preparation keys. This preserves the splitting allowance rather than
raising it to admit strong low-frequency drives. Rows below the carrier floor
or requiring more than 64 subdivisions still use T6. This covers long plateaus
without putting duration in a period key; the splitting allowance stays 1.0.

The original 16-level half-flux calibration covered 0.3-0.6 GHz and estimates
up to 0.871443. Its maximum population and low-four errors against strict DOP853
were <= 1.3e-5, including coherent unknown input. The low-frequency calibration
also covers a 49.991 MHz auxiliary model and the main-model half-flux sweet spot.
The 0.03 GHz floor includes the auxiliary spectrum's 0.038-0.062 GHz sweep.
The 21-point auxiliary strong spectrum and amplitude scans gave maximum
population/low-four errors of 8.34e-8/3.20e-6 and 3.68e-8/1.71e-6 against
DOP853 with rtol=1e-8, atol=1e-10 and max_step=0.1 ns. The amplitude scan reaches
an unsplit estimate of 14.04 and uses all four subdivision counts. Main-model
sweet-spot constant spectra at 0.294 and 0.219 GHz gave population errors below
6.1e-8. The 1.0 allowance remains a calibrated heuristic, not a universal error
bound or segment-budget guarantee. Small synthetic Engine tests cover mixed and
coherent inputs, signed carriers and phase-translated cross-period tails.

Below 1 GHz, preparation uses fourth-order commutator-free Magnus (CF4).
Each step multiplies two exact Hermitian matrix exponentials. Uniform grids
start below 2/3 of an instantaneous spectral turn. Preparation compares prefix
unitaries with a 1.5x finer grid at eight phase intervals, including U(T).
It retains the finer grid when the maximum Frobenius-norm difference is at most
min(1e-4, budget*0.1); otherwise it refines again. Accepted steps stay below
half a spectral turn. The comparison includes micromotion because errors at
U(T) alone can cancel. This is a calibrated convergence estimate, not a total
population-error bound. Prefix unitaries and the three Hamiltonian matrices
are immutable and charged to the cache. Arbitrary phase, subperiod and tail
endpoints use a fractional CF4 step from the preceding prefix, not interpolation.
No density enters preparation or the step selection.

At higher carriers, preparation keeps tolerance min(1e-10, budget*1e-4) and
max_step=T/4. The single-threaded nogil DOP853 core builds integration stages and the
degree-seven dense extension. It retains U(T) and the native nested polynomial
coefficients on accepted steps. There are no Python RHS callbacks, dense
resampling or per-key polynomial-basis solves. Stage sums reuse a state buffer
and skip zero tableau weights instead of allocating weighted-vector temporaries.
Every retained array counts against the prepare cache; no solver object is kept.
Polar projection restores unitarity of U(T) and dense fundamental solutions
before phase translation and fractional-tail composition. It does not normalize
density. Coherent-input trace drift in the calibration fell from 1.34e-10 to
1.68e-12. Prototype high-carrier S3/S5 population errors were 1.1e-7 and 2.3e-6.
Duration, start and theta do not enter keys.
Time translation handles signed frequency and phase; the fractional tail uses
the same Hilbert trajectory with a periodic extension across the boundary. Each full
subperiod and tail applies D/2, U, D/2 without constructing a Liouville channel.
Apply does not compare half-period results, normalize, retry or reroute.
Low-frequency flux-tail probes retain Dyson and two-tone rows retain Floquet.

T6 remains the terminal constant-tone route outside the Hilbert-period domain.
It uses 1, 4 or 8 steps per carrier period according to r, powers the period
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
preparations stay inside their timeline chunk and are discarded afterward.
Low-carrier Hilbert-period singleton keys, below 0.3 GHz, instead stream one
timeline at a time inside each worker chunk. This avoids retaining a sweep of
many distinct per-key trajectories. CF4 replaces the former roughly 80 MB
DOP853 trajectories at 50 MHz with compact unitary prefixes, but streaming
still prevents memory from growing with the number of sweep keys. Shared Hilbert-period keys
still use A-stage when there are at most four distinct low-carrier shared keys.
Larger low-carrier key sets prepare locally and stream timelines too. The LRU
limit alone cannot bound A-stage memory because live bindings retain evicted
payloads. This policy bounds sweep-wide low-carrier retention; a single timeline
can still retain its own distinct pulse keys. High-carrier batching is unchanged.
The input-bound hook is an A-stage exception to
model-only preparation. It contracts the source initial density or a shared
prefix output once per shared key, port and input group. Before final dispatch,
the engine evolves one representative of each group with ordinary kernel
payloads and basis changes. The contracted payload lives only until this evolve finishes. Single-use
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
For coherent inputs, shared construction first bounds all driven paths from
each input mode using absolute coupling sums. It drops conjugate mode groups
within budget/16 before allocating candidate paths. The remaining path-pruning
allowance is budget/8 minus that spent bound, so the total stays within budget/8.
Free evolution still uses the full input modes. Diagonal inputs skip this step
and retain the same table and numerical output as before.
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
