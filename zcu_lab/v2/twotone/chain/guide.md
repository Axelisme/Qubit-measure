# Two-tone readout-chain sweep

**Readers:** operator, agent. Run the readout-chain qubit frequency sweep from the GUI or MCP and read its sigma curve and peak fit.

**Last updated:** 2026-11-09

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Behavior

`twotone/chain` sweeps the qubit drive frequency like `twotone/freq`, but each dwell at one frequency is a chain of reads instead of one read.
A dwell is one shot: Reset, Init Pulse, then 48 cycles of the frequency-swept Probe Pulse followed by a readout, with no reset between cycles.
The experiment fixes the 48 cycles per dwell; the Form does not offer it.
The frequency loop runs outside the chain. One fast sweep visits every frequency once.
Reps sets the fast sweeps per round, and Rounds sets the number of rounds.

The host reduces each dwell to 11 numbers: the IQ means, the IQ second moments and IQ correlations between reads one to fifteen cycles apart.
Only running sums of these numbers are kept, so memory and the saved file do not grow with the number of sweeps.
From the sums, each frequency gets a distance from background in sigma.
The background is the median over frequencies.
The distance uses the IQ means, second moments and read-to-read correlations, so a line that changes these statistics without moving the mean IQ much, for example between nearly equally populated states, still raises it.

## Parameter sources

Calibrate the readout with `onetone/freq` first, as for `twotone/freq`. Enter the drive window directly for a first search.
The Form uses the same sources and starting values as `twotone/freq`: 301 points at `$qubit.freq` ± 20 * `$qubit.kappa` in MHz, a fresh `qub_probe` pulse with gain 0.1 and length 1.0 us, and the `$resonator.module.readout` slot or a fresh `readout` template.
Readout must be a pulse readout; a mux readout module is rejected by the Form.
Reset uses the optional `$qubit.module.reset` slot. Init Pulse starts disabled; when enabled it plays once per dwell, before the chain.
Missing or invalid sources leave the affected fields empty and Invalid; supply them before Run.

Reps starts at 100 and Rounds at 100, which is 10 000 fast sweeps in total.
Reps must be at least 2, because each point needs two dwells before its noise is defined.
Points × (Reps − 1) must be at least 11 so the first round can estimate the 11-feature noise covariance.
For 2 points, use at least 7 Reps; for 11 or more points, 2 Reps meets this minimum.
Each round acquires points × reps × 48 reads, so a smaller Reps gives more frequent live updates and less memory per round.
One dwell lasts about 48 × (probe length + readout length) plus Relax delay, about 100 us with the starting values.

## Live plot

After every round the `measurement` plot shows the distance from background in sigma at each frequency, computed from all completed rounds.
The dashed line is the 5% false-mark level: with no line anywhere in the sweep, the chance that any point crosses it is about 5%.
Points above the level are unlikely to be noise; points below it are compatible with noise.
Stop the run when the peaks are clear. Stop keeps the sums of completed rounds. A stop before the first round finishes keeps nothing and the Run fails with that message.

## Analysis

Analysis recomputes the sigma curve from the saved sums and fits one peak with the `twotone/freq` peak fit.
The options are those of `twotone/freq`: `expected_center_mhz` selects the nearest candidate, blank selects the most prominent, and `min_prominence`, `min_snr`, `min_width_samples` and `min_distance_mhz` gate the candidates.
Keep `model_type` at `lor`; the sigma curve is never negative, so it cannot show the opposite-signed sidelobes that the `sinc` model fits.
The `fit` plot shows the sigma curve, the dashed level, the fitted peak, the fitted centre in red and the other candidates in dotted gray.
The summary reports `freq`, `freq_err`, `fwhm`, `fwhm_err` in MHz, the `level` in sigma and the candidates in `peaks`.
The FWHM of a sigma peak approximates the line width. A weak line reads narrower, because points whose distance stays below the noise floor contribute zero sigma.
A sweep with no candidate above the thresholds produces an analysis failure; the live plot and the saved sums remain.

This experiment proposes no writeback. To calibrate `$qubit.freq` and `$qubit.kappa`, run `twotone/freq` around the centre found here.

## Saved data

The saved record contains the frequency axis, the dwell count and the 11 sums at each frequency, and one 11 × 11 product sum added over all frequencies.
Reads per dwell is not stored in the file and is not needed to restore the sums.
Loading the file and recomputing the curve gives the same sigma values as the live plot of the last completed round.

## MCP recipe

`twotone_chain` requires `center_mhz`, `span_mhz` and `points`.
Span must be positive and points must be at least 2; points maps to frequency Expts.
`reps` and `rounds` set the fast sweeps per round and the round count.
`expected_center_mhz` selects the peak to fit; null fits the most prominent.
The recipe runs, analyzes and saves. It never asks for writeback.
The report gives `frequency_mhz`, `frequency_err_mhz`, `width_mhz`, `width_err_mhz` and `level_sigma`.
Omitted inputs retain Form values; overrides apply last, then check_ready reports missing or invalid fields before Run.
