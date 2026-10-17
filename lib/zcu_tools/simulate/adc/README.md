# ADC probability and readout contracts

**Last updated:** 2026-10-17

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
outside tolerance. A failure raises `ValueError` with the number of rejected
rows, without returning partial results.

`PopulationInput`, `round_statistics` and `sample_traces` validate the same trace
interval and require nonnegative entries. They expect explicitly normalized
distributions and do not normalize or clip them. Their tolerance accommodates
numeric residuals; it does not replace explicit preparation. State ordering must
match the readout model and conditional responses. No basis inference occurs in
ADC. Evolution retains its raw densities and reports trace drift separately.

`ConditionalReadout` computes state-conditioned responses once per distinct
content. Round samplers share an initial-state draw across channels and windows,
then add independent receiver noise. The integrated sampler also accounts for
nearest-even shot rounding variance. These are conditional moments, not random
jump trajectories. QICK buffer layout and acquisition averaging belong to the
program adapter.
