# Heralded paired RB

**Readers:** operator, agent. Run heralded paired two-qubit RB and interpret herald acceptance alongside CZ-interleaved decay fits.

Select `two_qubit/rb_heralded`. Calibrate the X90 pulses, full mux readout,
dual-line CZ pulses and ordered local-Z corrections as described in the
[ordinary RB guide](guide.md). The same component roles, sequence seeds,
depths, bootstrap settings and d=4 error interpretation apply here.
Open or Reset captures sources. Refresh preserves authored inputs.

## Shot sequence

Each shot emits optional reset, herald, the paired Clifford sequence with
its own recovery, then terminal readout. Set `reset=None` to omit reset.
Herald and terminal emit the same full mux module with tags `herald` and
`terminal`. Readout `post_delay` is the wait after herald. The two reads
cannot use different gains or integration lengths.

Both F1 and F2 tones require circle classifiers with radius from GE Post.
Missing calibration, missing radius or linear classification fails before Run.
Repeat GE and Post after changing readout settings. Classification happens
during acquisition for both tagged reads, with g, e and other outcomes.

## Counts and analysis

Each depth/seed/arm compiles once. The host acquires `rounds` separate rounds,
each using `rounds=1` and `reps` shots. Result `sums` has float64 shape
`(Nd, Nseed, 2, Nround, 3, 3, 3, 3)`. The trailing axes are herald F1,
herald F2, terminal F1 and terminal F2. Each is ordered g, e, other.
Save retains all 81 unnormalized class counts. It does not save per-shot IQ
or grow with `reps`. Unfinished rounds remain NaN. Stop retains completed
slots; acquisition errors still need resolution. Native Load preserves the
saved source without changing the current Form. Old raw-IQ files lack
`sums` and are rejected.

Analysis selects the herald `(g,g)` slice, then uses ordinary RB's joint00
ratio and paired-round analysis. Terminal survival is accepted terminal
`(g,g)` counts divided by all accepted terminal counts, including other.
A round with no accepted shots has NaN survival and cannot enter the fit.
Only seed/round pairs complete across both arms and every depth enter.
Complete rounds average within each seed; seeds then receive equal weight.
The live plot instead shows the count-weighted accepted joint00 ratio
over seeds and rounds.

The herald plot shows all nine herald proportions for each arm and depth,
aggregated over seeds and rounds. The analysis result also retains per-round
herald proportions and pass fraction. Pass fraction is herald `(g,g)`
counts divided by all herald counts. It measures classified acceptance,
not ground-state purity. Readout errors can accept excited states and
reject ground states. Post-selection reduces the available shot count.

Read the fit curves, residual diagnostics and seed bootstrap interval with
those selection effects in mind. A low pass fraction can leave insufficient
paired curves for a fit. A finite CZ error remains a model-dependent
diagnostic, not a physical calibration acceptance.
