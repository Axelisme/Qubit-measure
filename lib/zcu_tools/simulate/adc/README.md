# ADC probability and readout contracts

**Last updated:** 2026-10-21, one- and two-cavity readout

**讀者：** framework-dev。ADC 機率、讀出訊號與 interval moment 契約。

`ReadoutModel` supports 1 or 2 cavities on a shared transmission line.
`cavity_count` comes from the frequency array's second axis. Other counts raise
`ValueError` at model construction. Frequencies have shape `(state,cavity)`;
internal/external decay rates and scattering phases have shape `(cavity,)`.
The two-cavity default `mismatch_phase_rad=zeros(2)` is unchanged. One-cavity
callers must supply a one-element phase array, including `np.zeros(1)` for zero
mismatch. Receiver I/Q offset and noise covariance remain `(2,)` and `(2,2)`.
The scalar path propagator is exact; two-cavity paths retain the existing 2x2
closed form. Constant, sampled and opaque drives share these model dimensions.

`cavity_fields` returns complex `(time,state,cavity)` fields in fixed joint-state
order, without population mixing or receiver calibration. Its optional complex `(cavity,)`
`initial_field` supplies the same field to every state at `initial_time_ns`.
`cavity_field_average` accepts the same pair of inputs. Both use sqrt(photons)
in the model's rotating reference at that absolute emitted time. The initial
field defaults to zero and the initial time to 0 ns. Omitting both equals
explicit vacuum at t=0, including mid-pulse queries. Propagation uses only later
pulse history without restarting pulse-local phase or envelope coordinates.
Queries before the initial time fail with both times in the error. `cavity_field_average`
returns `CavityFieldAverage`: photon occupations and same-time field products
averaged over an emitted-time interval, including coherent carriers and full
ringdown. Flat inputs use analytic constant-line propagation and quadratic
integrals. Shaped inputs retain the field ODE and converged quadrature, split at
pulse boundaries, envelope corners and the decay timescale. ADC owns these
field integrals; consumers project them into their own Stark or dephasing models.

ADC consumes explicitly ordered classical populations, not densities. The
probability tolerance is `POPULATION_TOLERANCE = 1e-6`. This is 1000 times below
the segment population budget of 1e-3 and gives about 38 times margin above the
measured shaped-kernel trace drift of 2.62e-8.

Environment authors explicitly call `normalize_populations` after converting
densities into readout-basis populations. This is the only probability repair
operation. It checks finite values, a nonempty state axis, row sums in the closed
float64 interval [1-1e-6, 1+1e-6], and entries at least -1e-6. It then sets tolerated
negative entries to zero and divides each row by its sum, returning an owned
array without changing its input. It never clips upper values or accepts rows
outside tolerance. Invalid shapes raise `ValueError`. Row validation raises
`PopulationValidationError`, a `ValueError` subclass whose `rejected_rows: int`
counts rows failing the first reported rule. It returns no partial results.
Callers can retain this count without parsing error text.

`PopulationInput`, `round_statistics`, `PreparedRawShots` and `sample_traces`
validate the same probability trace interval and require nonnegative entries.
They expect explicitly normalized distributions and do not normalize or clip them. Their tolerance accommodates
numeric residuals; it does not replace explicit preparation. State ordering must
match the readout model and conditional responses. No basis inference occurs in
ADC. Evolution retains its raw densities and reports trace drift separately.

`ConditionalReadout.response` and `responses` return microwave I+iQ. They
conjugate the demodulated physical field before receiver gain, phase, offset and
noise. Receiver calibration does not conjugate or rotate the added noise's I/Q
covariance. Constant and shaped pulses use the same convention.

`Receiver.time_of_flight_ns` is one finite nonnegative total line delay, default
zero. The solver evaluates the physical output at t−τ and restores the rotating
frame's exp(+2πi*f_ref*τ) before demodulation at ADC time t. Conjugation then gives
exp(−2πi*f_abs*τ) at the absolute carrier, without counting τ twice. Onset,
ringup, ringdown and envelopes all arrive late; offset/noise do not. Pulse phase
retains its documented rotating-coordinate convention.

`ReadoutModel.mismatch_phase_rad` is a finite array with one phase per cavity.
Each element multiplies only that resonator's scattered field by exp(−i*phi)
before conjugation, giving the hanger model's +phi sign in microwave IQ. The
reciprocal shared-line generator and transition rates are unchanged. This is a
scattering-phase model, not a full impedance network or a global signal rotation.

`ReadoutBatch` accepts one or more coherent incident pulses per content and
independent demodulation grids. Every grid observes the sum of all incident
carriers through the same physical line; receiver offset is applied once, not
once per tone. `ConditionalReadout` computes state-conditioned responses once
per distinct content. Integrated flat (`envelope=None`), common-start mux
contents sum each carrier's linear response through the same line. Different
pulse end times retain their own ringdown. Each carrier first tries guarded
direct discrete sums on uniform ADC grids. A rejected carrier uses the existing
constant-field solution, including full block exponentials when the coupling
guard rejects the projected solution. Guards are not relaxed and one rejection
does not send other carriers to the ODE. Calibration includes receiver offset
only once per sample after superposition. The workspace stays local to the
`ConditionalReadout` instance, not shared globally or across receivers.
Flat common-start trace/decimated contents also superpose vacuum-start carrier
fields through the same `ConstantLine`. Homogeneous propagators and sampled
scattering projections are shared across carriers, with the existing full block
exponential fallback for ill-conditioned coupling. Calibration adds receiver
offset once after field superposition. Different-start multi-pulse contents and opaque envelope callbacks retain the
field ODE. `PiecewiseLinearEnvelope(knots_ns, values)` owns read-only copies of
finite complex values and increasing pulse-local ns starting at zero. It linearly
interpolates and holds the final value through pulse end. Its lossless content
key allows independently constructed copies to reuse responses. Common-start
contents containing these envelopes superpose carrier-frame affine-input
solutions. An augmented exponential supplies population/moment propagators and
constant/linear forcing convolutions, including singular/defective generators.
The workspace retains only the current knot state, ADC outputs, and at most 128
step blocks per carrier. Nonfinite blocks retain the ODE fallback. Flat carriers
in mixed sampled/flat contents still use ConstantLine. Calibration occurs once
after superposition. The public single-grid `response` remains an ODE reference.

Integrated flat single-carrier contents sharing exact start/end and per-read
time coordinates use tiles of at most 64 carriers. Amplitude, carrier phase and
demodulation may differ. Uniform-grid validation and population evolution are
shared; separation, cancellation and residual guards remain per carrier. Rejected
contents retain their scalar sum/field fallback. No content-by-sample dense
moment tensor is constructed.
Integrated single flat pulses also use the guarded direct discrete sums. The constant moment solution
comes from `(M+iω)X−XQ=B`. In the eigenbases of M and Q, its coefficients are
`Y_ab=(Um⁻¹ B Uq)_ab/(μ_a+iω−λ_b)`. Each demodulated exponential sums to
`exp(z*a)*expm1(N*z*h)/expm1(z*h)`; the sampled-alias limit is `N*exp(z*a)`.
The incident active segment, causal on-step and opposite off-step are summed
separately. The off-step retains `exp(Q*T)` population transfer. Conjugation,
receiver gain/phase/offset, delay and mismatch follow the field convention.
There is no sample-period factor, noise draw or population normalization here.

Both eigenbasis condition numbers must be below 1e4, reconstruction residuals
below 1e-13 and spectral separation at least 1e-10. The original Sylvester
cancellation and residual guards also apply. Uniformity compares sample
coordinates to the endpoint-fitted lattice, within four coordinate ULPs plus
eight float64 epsilons times the grid span. Unlike a fixed ns tolerance on
adjacent differences, this accounts for quantization at large absolute starts.
Single samples are uniform. Direct sums reuse the exact-content cache across
contents and batches. Trace requests retain a separate field cache.

`ReadoutCache(max_bytes=8*1024**2)` optionally owns these solvers across caller
requests. Its key includes the full model, receiver and rtol/atol. Equivalent
models share one canonical immutable model. Content keys and numeric routes stay
unchanged. It incrementally measures the retained ADC object graph after each response,
including the accounting ledger, ConstantLine subcaches, keys, traces and full
backing allocations of views. Numeric content entries only append, so the ledger
visits new entries rather than rescanning earlier T1 starts. Opaque callbacks
receive a full recount because their captures can change. It evicts oldest solvers whole until the byte budget fits. Returned arrays
and active workspaces are outside that resident budget. Opaque callbacks remain
identity-keyed and referenced; their attributes, closure cells and defaults count,
but module globals and external resources do not. A failed response clears all
residents. `clear()` releases solvers and callback references. `builds`, `hits`
and `resident_bytes` report physical solver reuse and retained bytes. The owner
must serialize calls and clear on physical snapshot or numeric revision changes.
This cache performs no sampling and never retains noisy IQ.

Unsupported direct sums fall back individually, including nonuniform grids
and defective generators. Single-pulse trace requests retain their original
field route. Opaque callbacks retain the ODE. Flat fields share frequency-independent Schur bases and sampled output
projections when enough contents reuse the same relative grid to amortize their
construction. These grids evaluate scattered fields directly without advancing
a dense moment tensor at every sample. Other grids retain step propagation with
shared Schur bases. Ringdown subtracts the delayed step response;
ill-conditioned driven blocks retain the full matrix-exponential propagation.
The field routes do not assume diagonalizable rate matrices. No route changes
the evolution populations.
Its lossless numeric cache key includes delay and every cavity's mismatch phase.
`ADC.sample` and `sample_windows` draw continuous-time Markov paths
using the directional rates. The initial population origin is the earliest
incident pulse start, or zero with no pulses.
The path continues through ringdown to the latest emitted sample, including gaps
between windows. Jumps retain every resonator's field and reciprocal shared-line
damping. Vacuum baselines support flat, piecewise-linear and opaque envelopes;
no ODE is solved per shot. Delay, mismatch and receiver calibration follow the
same conventions as conditional responses. Windows share one path per shot and
add independent receiver noise. Zero rates retain deterministic conditional
traces. Multiple incident carriers superpose their vacuum fields and retain
one cavity memory on the same path. Integrated direct means retain conditional
moments and nearest-even shot rounding variance. Acquisition averaging belongs to program.

`PreparedRawShots(populations, batches, reps=...)` owns noiseless probabilities
and per-channel ReadoutBatch inputs. `bind(identity)` returns an `ADCRawShots`
source without doing shot work. `materialize(reps, avg_level)` alone draws full
CTMC paths shared across every carrier, ADC and read. Each receiver adds
independent noise. Zero jumps use direct vacuum sums. Jumped shots use compiled exact homogeneous propagation and discrete
sample sums. Uniform samples use binary geometric sums; irregular samples keep
their original coordinates. Flat baselines use singular-safe phi1 propagation.
Shaped and opaque envelopes retain the existing DOP853 vacuum-baseline tolerances.
There is no ODE per shot, jump-time LUT, path-count conditioning or tail cutoff.
Reads share one path through ringdown and gaps; offset is added once per sample.
All ADC models must declare identical state labels and directional rates.
Different receivers may deduplicate contents differently; point mappings select
physical inputs independently without changing the common path.

The source retains its inputs independently of later rounds and prepare-cache
eviction. Every call restarts its private PCG64 stream and returns fresh int64
arrays, keyed by ADC channel with shape `(*loop_dims, reads, 2)`. It does not cache
noisy results or adjust shots to independently drawn mean totals. Invalid reps,
layout, shapes or states raise ValueError, as do unsafe integer sums. Solver
failures raise RuntimeError. `simulate.rounding` owns shared numeric int64
conversion and shot-axis layout. Numeric sources do not import program.

`sample_traces(populations, batches, reps=..., identity=..., rtol=..., atol=...)`
requires physical `ReadoutBatch` inputs keyed by ADC channel in declaration order.
It returns `RoundSignals` with owned float64 arrays shaped
`(prod(point_dims)*reps, reads, samples, 2)`, in point-major C order then shot order.
Each ADC requires equal sample counts across reads and contents. The complete
`SamplingIdentity` seeds PCG64 without an extra suffix. FTF supplies
`(acquire_sequence, round_index, 0)` as its stream. Calls share the full CTMC path,
including times not observed by any window. Delay, mismatch and demodulation
remain per receiver/grid; noise is independent per ADC/read/sample/shot. There
is no integrated rounding term. Invalid probabilities, states, rates, layouts,
counts or tolerances raise `ValueError`; baseline/field failures raise
`RuntimeError`. There is no conditional-only trace sampler or fallback.

`PreparedStatistics(populations, channels, reps=...)` validates and owns immutable
integrated sampling inputs. It expands conditional means and factors each ADC's
noise covariance once. `draw(seed=..., acquire_sequence=..., round_index=...)`
validates a fresh identity, draws the same joint multinomial and ADC-ordered
Gaussian arrays as `round_statistics`, and retains no noisy results or identities.
The public `round_statistics` still validates arbitrary inputs by building a plan
for that call. FTF retains plans with noiseless preparations; no RNG draws or
deferred raw materialization move into preparation.
