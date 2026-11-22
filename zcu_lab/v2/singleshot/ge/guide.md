## Behavior

**Readers:** operator, agent. How to calibrate ground/excited single-shot readout from the GUI or MCP and interpret the results.

Single-shot ground/excited readout: measures with the probe pi-pulse off and on, takes 'shots' single-shot readouts of each, and fits the two IQ clusters to extract the assignment fidelity, rotation angle and threshold. Runs on real hardware with rounds=1 and reps=shots. Each run acquires readout shots twice, with the probe off and on.

After changing the parameter source, use Defaults to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Defaults can overwrite unsaved inputs.

## Parameter sources

Open and Defaults take a snapshot of the current aliases, parameters and module choices. Relax delay is 5 * `$qubit.t1`. The probe template uses `$qubit.freq` and `$qubit.wiring.ch`.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.time_of_fly` + 0.015 µs. Missing required sources leave the corresponding fields Invalid; supply them before Run.

T1-derived defaults use the Open/Defaults snapshot. A missing or null T1, or an unavailable snapshot, uses 30 µs, so the 5*T1 wait starts at 150 µs. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

Uses `$qubit.module.x180` for optional init and required probe, `$qubit.module.reset` for optional reset, and `$resonator.module.readout` for readout. Missing required slots use pi_pulse/readout templates; missing optional slots stay disabled. Run uses the Open/Defaults snapshot.

## Assignment and model quality

Fidelity is the empirical, equal-weight assignment of the two prepared labels at
the best projected-IQ histogram boundary. Probe off/on are preparation labels,
not guaranteed pure g/e states. Thermal mixing and imperfect pulses remain in
this score. A threshold can also exploit small differences inside one cluster;
a high prepared-label score does not establish pure-state readout fidelity.

Initial populations come from the fitted histogram model. Post-analysis instead
classifies shots by nearest-center regions and a radius, then solves the fitted
population matrix to remove preparation mixing. It does not reuse the primary
histogram threshold. These outputs need not equal the prepared-label score.
Low joint r2 means the histogram model does not describe the observations;
then fitted populations and population-corrected post results are not reliable
calibration evidence. Inspect raw clusters, residuals and uncertainty before
using centers, width or the confusion matrix.

## Typical writeback

Primary candidates are initially selected only when the joint histogram fit passes all four checks:

- r² is at least 0.9.
- Normalized residual RMS is at most 0.1.
- Relative width error (`s`) is at most 0.5.
- Relative center errors (`sg` for g and `se` for e) are each at most 0.5.
  Each is the center's standard error divided by the absolute value of the
  fitted center.

Post radius and confusion inherit the primary review and also require a finite
condition number of at most 10. A failed check leaves the candidate unselected
with a note. A required metric fails if it is missing, nonfinite, or itself
marked invalid. For a center error above the limit or itself marked invalid,
the note gives the reason, the value if finite, and the limit. If the error is
missing or nonfinite, the note reports it as unavailable. An issue on a
center's standard error alone does not block preselection. A relative center
error of 0.0 does not mean the center is precisely known. You may select a
candidate manually after inspecting the model and raw IQ.

Analysis and writeback use the fixed parameter source of the selected Run or Load result. Run keeps the source used for that measurement.

Load uses the parameter and library content saved in data.h5, without reading the historical point or live Form. Preview and Apply target the GUI's currently selected point. Select a point before applying.

Writeback resolves the Run's `$resonator.module.readout` slot to its library module. Primary proposes a circle classifier with g/e centers. Post proposes centers, radius and the 3x3 population-corrected `Classifier confusion` matrix from `post`. Centers and radius use raw ADC sums divided by integration length without IQ-offset subtraction. Fidelity and sigma are not stored. The proposal preserves waveform keys and rejects readout settings that differ from the captured library module, apart from classifier. Equal materialized settings are accepted even if the Run used an inline copy or another key.

In MCP, `singleshot_ge` analyzes Primary and asks about its candidates before starting Post. It applies the selected Primary candidates if accepted, then analyzes Post and asks about its candidates separately. You may skip Primary and still accept Post. The report lists confirmed writes in `primary_writeback_applied` and `post_writeback_applied`.
The report's `fit_quality` comes from the joint histogram fit.
`ground_fit_quality` and `excited_fit_quality` come from the individual fits.

A batch applies Primary before Post and leaves the complete classifier. Separate Apply calls use the last replacement. Applying Primary after Post clears radius and matrix; apply Post again before a measurement that needs radius. Old point calibration values are not read or migrated. Re-run GE after this update.

Changing readout settings does not clear or automatically invalidate the stored classifier.
After a readout change, re-run GE and Post-Analysis, then Apply the new calibration before using classification.
The proposal's settings comparison prevents writing new calibration to different settings; it does not check whether an existing classifier is still valid.

## Recommended

Set Initial State to the predominant state after reset/init and before the probe pi-pulse.
This labels the two acquisitions; each can contain a mixture of states.
Use a large 'shots' (~1e5) so the IQ histograms are well sampled.
The default analysis method is 'pca'.
Analysis also exposes histogram log scale, T1 alignment, and an optional shared length ratio.
The length ratio (a.u., readout length / T1) is inferred from the decay tail between the two clusters.
With Length Ratio left empty, two tight clusters without a visible tail make Analyze return `length_ratio: null` and title the IQ plot `GE IQ (length ratio: not identifiable)`.
This reporting decision does not change centers, fidelity or Post-Analysis.
If you set Length Ratio manually, Analyze returns the supplied value.
Run once the qubit pi-pulse and the readout are both calibrated.
Run and Analyze show raw IQ at equal I/Q scale. In each IQ bin, blue means
more probe-off shots and red means more probe-on shots on the live plot.
Analyze maps the acquisitions to g (blue) and e (red) using Initial State.
A tie is purple. Opacity rises with the log of the total shot count; empty
bins are transparent. Each successful Analyze also shows `diagnostic`.

The diagnostic has four panels:

- Top left uses the same raw IQ colors as the Analyze fit and marks the fitted
  centers.
- Bottom left overlays the projected preparation histograms and their fits.
  Each preparation is normalized separately. `Fraction per bin` is the
  fraction of that preparation's shots in the bin, not a count or a density.
  The prepared and ideal-model assignment scores appear between the rows.
- The right panels show the g preparation above and the e preparation below.
  Compare the fitted components with the bars and read the g, e and other
  fractions above each panel.

**Fitted SNR** divides the distance between the fitted peaks by their shared
Gaussian width. The number after ± is the standard error from the same joint
fit, not a confidence interval. `unavailable` replaces an estimate the fit
cannot supply. `not identifiable` on the SNR line means at least one fitted
center failed the relative-error check. Passing that check alone does not
make a candidate eligible for writeback preselection.

Histogram log scale changes the diagnostic histograms, not the IQ scatter.
Open the saved diagnostic in Data with **View figure** for a larger view.
## Read the Post figure

The left panel of `post` shows both preparations on the same raw, equal-scale
I/Q plane. Each bin is blue if it contains more g-preparation shots, red if it
contains more e-preparation shots, and purple if the counts tie. Opacity rises
with the log of the total count; empty bins are transparent. The colors do not
show purity or how the classifier labeled a shot. A bin's majority color
cannot tell you how individual shots from the minority preparation were classified.

The markers show the g and e centers. Dashed lines show the radius boundaries
and, where the disks overlap, the shared segment midway between the centers.
The classifier chooses the nearer center only inside its radius. Equidistant
shots and shots on or outside the radius boundary are Other. A purple bin is
a preparation count tie, not an Other classification.

Read `Observed %` across each preparation row for the shots classified as g,
e or Other. These percentages are measured before correcting for preparation
mixing. In **Initial Populations**, rows are prepared states and columns are
estimated actual states. In **Classifier confusion**, rows are actual states
and columns are measured labels after solving for preparation mixing.
The condition number indicates how sensitive inversion of the classifier
matrix is. It is not a fidelity score.

**Last updated:** 2026-11-22

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Obtain a calibrated pi pulse and verify readout timing. Primary fits the two clouds; Post derives classification diagnostics.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, choose Defaults to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, choose Defaults and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `singleshot_ge` after inspecting the same Form errors.
   shots sets each cloud size, not Expts. initial_state, backend, logscale, align_t1 and length_ratio update Primary analysis. Post has no independent controls.
   The Form chooses module sources. Omitted inputs retain Form values; null clears/disables fields. Overrides apply last, then check_ready reports missing or invalid fields before Run.
   Missing inputs return needs_parameters before acquisition. Defaults may replace unsaved inputs.
