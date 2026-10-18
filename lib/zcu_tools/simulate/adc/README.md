# ADC probability and readout contracts

**Last updated:** 2026-10-18, carrier-wise mux traces and discrete sums

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

`PopulationInput`, `round_statistics` and `sample_traces` validate the same trace
interval and require nonnegative entries. They expect explicitly normalized
distributions and do not normalize or clip them. Their tolerance accommodates
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

`ReadoutModel.mismatch_phase_rad` is a finite two-element array, default zeros.
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
offset once after field superposition. Shaped and different-start multi-pulse
contents retain the general field ODE. Single-pulse numeric paths are unchanged.
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

Unsupported direct sums fall back individually, including nonuniform grids
and defective generators. Single-pulse trace requests retain their original
field route. Shaped contents retain the ODE. Flat fields share frequency-independent Schur bases and sampled output
projections when enough contents reuse the same relative grid to amortize their
construction. These grids evaluate scattered fields directly without advancing
a dense moment tensor at every sample. Other grids retain step propagation with
shared Schur bases. Ringdown subtracts the delayed step response;
ill-conditioned driven blocks retain the full matrix-exponential propagation.
The field routes do not assume diagonalizable rate matrices. No route changes
the evolution populations.
Its lossless numeric cache key includes delay and both mismatch phases.
Round samplers share an initial-state draw across channels and windows,
then add independent receiver noise. The integrated sampler also accounts for
nearest-even shot rounding variance. These are conditional moments, not random
jump trajectories. QICK buffer layout and acquisition averaging belong to the
program adapter.
