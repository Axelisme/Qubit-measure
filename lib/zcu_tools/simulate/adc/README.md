# ADC probability and readout contracts

**Last updated:** 2026-10-18, microwave IQ, causal line delay and mismatch

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

`ConditionalReadout` computes state-conditioned responses once per distinct
content. Its lossless numeric cache key includes delay and both mismatch phases. Round samplers share an initial-state draw across channels and windows,
then add independent receiver noise. The integrated sampler also accounts for
nearest-even shot rounding variance. These are conditional moments, not random
jump trajectories. QICK buffer layout and acquisition averaging belong to the
program adapter.
