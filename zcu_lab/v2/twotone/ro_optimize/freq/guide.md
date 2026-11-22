**Readers:** operator, agent. Choose the readout frequency that best separates the nominal g and e preparations.

# Readout frequency sweep

## What it measures

Use this sweep with a calibrated π pulse and fixed readout gain and duration. At each setting, run optional reset → omit or apply the calibrated qubit π pulse → readout. The two branches are nominal g and e. The SNR is the distance between their IQ centroids divided by the pooled single-shot spread along the line joining the centroids. A positive skew penalty reduces that SNR by a shape factor. Neither the raw SNR nor the penalized score is a state population or a measured assignment fidelity.

## Set the run parameters

- **Readout frequency (MHz):** enter absolute RF when a mixer is configured, or DDS frequency without a mixer. The initial range is the captured resonator frequency ± one linewidth, at 301 points. Run replaces the frequency in the readout module with this sweep.
- **Qubit π pulse:** select the captured, calibrated g-to-e pulse. When the slot is absent, the `pi_pulse` template takes frequency and wiring from the selected qubit; calibrate the pulse before using it.
- **Readout pulse and ADC window:** use the selected resonator's readout module or the `readout` template. The template uses the resonator frequency, wiring and flight time. It leaves 0.015 µs of pulse before and after the incoming ADC window. Start the readout post-delay at five photon decay times, `5/(2πκ)` µs for linewidth κ in MHz, and check that a longer wait does not change the signal.
- **Reset:** optionally select the reset used before each preparation. A blank field skips reset.
- **Qubit relaxation wait (µs):** wait after each read. The initial value is 5 × captured T1. If T1 is absent, the starting estimate is 30 µs and the wait starts at 150 µs. If captured T1 is invalid, supply the wait before Run.
- **Reps**, **Rounds:** shots per preparation and setting in each software round, and the number of rounds. The Form starts at 1000 and 10. The Analyze summary reports the settings from the analyzed run, not the current Form.
- **IQ skew penalty weight (a.u.):** leave at 0 to rank by SNR alone. Increase it to reduce the score of populations with mismatched widths or uncancelled skew along the separation axis.

Use **Defaults** after changing the qubit or resonator source to recapture these starting values. Fill any missing required fields before Run.

## Read the figures and choose a setting

Live points and the Analyze points are g/e IQ separation SNR versus readout frequency (MHz). Live shows how many settings have scored points. Analyze retains the measured points, overlays the selection SNR if it differs, and marks the selected frequency with a dashed line. No pointwise SEM is shown: this sweep locates the SNR peak, and the ordinate is a ratio of estimated IQ moments, not a mean.

**Smooth method** selects wavelet denoising or Gaussian smoothing. **Smooth strength** scales the wavelet threshold or sets the Gaussian width in sample steps, respectively. Analyze marks the maximum of the resulting selection data, not a fit or a sub-step estimate. The selection uses zero for any missing SNR before smoothing, so do not adopt a candidate from an incomplete sweep without checking the measured points.

If the selected frequency is at a boundary, extend the range there and rescan. Otherwise, refine around the peak using smaller steps and confirm it before writeback. The frequency correction in MHz is the selected frequency minus the captured readout frequency. Analyze proposes it as the new `$qubit.ext.best_ro_freq`. Review the writeback candidates before applying them. If the readout is a pulse module and the remaining calibration coordinates are available, Analyze also proposes an optimized resonator readout module and the captured resonator's readout slot.

There is no MCP recipe for this experiment. Use the experiment tab or its RPC entry point.
