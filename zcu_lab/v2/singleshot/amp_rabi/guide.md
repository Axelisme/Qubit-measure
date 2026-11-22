## Behavior

**Readers:** operator, agent. How to run single-shot amplitude Rabi calibration from the GUI or MCP and interpret the results.

The hardware gain sweep preserves every raw IQ shot. Live view classifies Ground / Excited / Other. Analysis uses a joint IQ fit with constant oscillation amplitude and phase fixed at zero.

## Parameter sources

Open and Defaults take a snapshot of the current aliases, parameters and module choices. Relax delay starts at 5*T1. Gain sweep starts at -0.3. Defaults computes its stop from the stored π gain.

When the Live preview is enabled, its bars show one shot SEM of the raw class fractions from the same shots. Fewer than two retained shots give no bar.

Raw acquisition and joint fitting require one ADC: Pulse or a single-tone Mux. Live preview uses that readout's circle classifier under Readout → Advanced. Without a classifier, Run still collects raw IQ and Analyze still fits; only the population preview is skipped. A supplied linear classifier cannot show Other; a circle without radius prompts singleshot GE Post.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.time_of_fly` + 0.015 µs. Missing required sources leave the corresponding fields Invalid; supply them before Run.

The default gain stop is `min(1.0, 4.0 * $qubit.module.x180.gain)`. When the x180 module has a stored scalar waveform length, its editable pulse copy starts at length = 1.5*pi_len*pi_gain µs. pi_len must be a finite positive number; pi_gain must be finite and in (0, 1]. Both values must be numeric, with booleans treated as invalid. Missing, null or invalid values leave the required length empty. A finite invalid gain stays visible for correction; a nonfinite gain leaves both gain and length empty.

If the x180 module is absent, the qub_probe template starts at gain=1 and length=1 µs. An arbitrary-waveform x180 keeps the original module slot, waveform shape and data. Its duration stays with the selected waveform; change waveform first to edit duration.

T1-derived defaults use the Open/Defaults snapshot. A missing or null T1, or an unavailable snapshot, uses 30 µs, so the 5*T1 wait starts at 150 µs. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

The qubit pulse uses the x180 module when present and the qub_probe template when absent, with the starting values described above. Reset uses optional `$qubit.module.reset`; readout uses `$resonator.module.readout` or readout template. Pulse values remain editable, including those replaced by the sweep. You can select module slots by key. Run uses the Open/Defaults snapshot.

## Typical writeback

Reports pi/pi2 gain, frequency and amplitude. A backend-valid joint fit with finite centers, radius and confusion matrix proposes one complete readout-module replacement at `$resonator.module.readout`. Pulse updates `classifier`; a single-tone Mux updates `tones.0.classifier`. The proposal preserves library waveform keys and rejects differences in other tested settings. Missing saved cfg or parameter capture fails explicitly. Invalid or nonfinite fits produce no proposal.

## Analysis summary

Analysis returns fitted parameters, pre-pulse `initial_populations`, the fixed 3×3 `confusion_matrix` and circle-classification calibration. Backend diagnostics retain parameter names, fitted values, marginal standard deviations, validity and negative log-likelihood. Population curves, projected shots, histograms and the full covariance stay local to analysis for plotting and uncertainty calculations. They are not part of the returned summary.

## Recommended

Sweep a full Rabi period. Initial State describes the predominant state before the drive, at zero gain, even if the first gain is nonzero. Reps are shots per gain per round; rounds concatenate shots and update live populations. Joint fitting requires a file containing raw IQ shots.

## Reading the figures

In Run, check that all three classified outcomes change across the gain sweep. In Analyze, compare data against the joint fit and locate π and π/2 on the gain axis. If fewer than one full oscillation is visible, extend the range rather than applying an extrapolated gain. Use the projected-IQ fit and the confusion matrix to check whether the model separates g and e. The fit's negative log-likelihood is an objective value, not a population error or a pass threshold.

Population error bars are shot SEM from the retained IQ shots assigned to g, e or Other. They are not fit-parameter uncertainties or the width of the projected IQ histogram. For pointwise probability error bars, see Ding Fig. 5-13(a), p. 157; that plot varies CZ pulse width and shows fidelity, not oscillation versus drive gain.

**Last updated:** 2026-11-22
