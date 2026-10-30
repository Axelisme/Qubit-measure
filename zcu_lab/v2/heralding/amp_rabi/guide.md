# Heralded amplitude Rabi

**Readers:** operator, agent. Run a single-qubit gain sweep with ground-herald post-selection and review rotation-amplitude proposals.

## Before Run

Use `heralding/amp_rabi` after calibrating the qubit frequency and pulse readout at the intended operating point. Bind `qubit` and its `resonator`, then Open or Reset the tab to capture those sources. Later alias or library edits do not change an existing Run source.

Readout must be a pulse readout with a GE Post circle classifier: ground/excited centers and a positive radius in the same integrated-IQ coordinates as this readout. Linear and centroid3 classifiers are not supported. Missing calibration stops Run. Changing readout frequency, gain, duration, ADC window or receiver settings can invalidate the classifier; review or recalibrate it before proceeding.

Choose the Rabi pulse frequency, waveform and fixed duration. A prior pi amplitude is not required. Set Gain start/stop and Expts to cover visible oscillations within your approved drive limits. Confirm Reps, Rounds and Relax delay. The suggested form values are editable starting values, not hardware-safety approval.

## Run and analyze

1. Select an optional Reset module if needed. The experiment then emits a herald readout, the optional init pulse, the gain-swept drive and the terminal readout. The herald and signal readouts use separate instances with the same pulse/ADC settings and classifier. Heralding itself adds no active reset or joint-qubit classification.
2. Click Run. Each point averages only signal shots whose herald was classified as ground. Rounds are weighted by their accepted-shot counts. The live plot shows selected IQ magnitude versus the hardware-quantized gain.
3. Open Analysis and set Skip points if leading gain points should be excluded. Skip applies to the original grid. Analyze removes nonfinite gain/IQ points before calling the ordinary amplitude-Rabi fit. Pass fraction is not a fit weight. Too few finite points cannot produce a usable fit.
4. Inspect the rotated-IQ fit, pi/pi2 markers, uncertainties and fit quality. These estimates calibrate pulse amplitudes; they do not measure gate fidelity.

## Pass fraction and saving

Pass fraction is the herald-g count divided by the total class count at each gain. It is computed for analysis or live presentation, not stored. Zero accepted shots give fraction zero and NaN signal IQ. Unmeasured points remain NaN. Low acceptance reduces the effective sample count even when Reps is large.

The classified fraction is not an independent measurement of ground purity. It depends on classifier errors and on whether calibration and acquisition use matching IQ coordinates.

Save All writes native `sums` with shape (Ngain, 3, 3) and cfg, including the classifier. The final axes are herald_states (g/e/other) and components (count/sum I/sum Q). Old files without `sums` fail with a missing-variable error. Load Data restores the saved grid and classifier rather than rebuilding them from the current Form. Analyze can run again on the loaded record. Review each save output's status; successful native saving does not imply every export format succeeded.

## Review writeback

Successful analysis proposes the same five destinations as ordinary amplitude Rabi. Descriptions begin with Heralded:

- `$qubit.pulses.pi_amp` and `$qubit.pulses.pi2_amp` receive copies of the source Rabi pulse with fitted gains.
- `$qubit.module.x180` and `$qubit.module.x90` receive the corresponding resolved library keys.
- `$qubit.ext.pi_product` receives fitted pi gain times the source waveform length, in native gain × us.

Destinations come from the Run or Load capture. Inspect every proposed pulse and slot before Apply. Analysis alone does not write calibration. Missing saved cfg suppresses these proposals; missing captured parameters or a non-scalar source length prevents proposal generation.

The MCP recipe `heralding_amp_rabi` exposes readout/drive references, optional reset, fixed frequency/duration, gain endpoints, points, reps, rounds and analysis skip. `points` maps to the Form sweep count. Omitted inputs retain Form values; explicit null goes unchanged to the Form or analysis owner and may be rejected. Overrides are applied before readiness validation. The recipe runs, analyzes, asks about writeback and saves completed Run data even when analysis fails.
