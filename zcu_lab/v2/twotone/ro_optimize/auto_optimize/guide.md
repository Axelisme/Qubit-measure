**Readers:** operator, agent. Find a measured candidate for readout frequency, gain and RF pulse duration.

# Readout search

## What it measures

Run the search with a calibrated π pulse. The optimizer proposes candidates jointly and uses each measured score to guide later choices. At each setting, run optional reset → omit or apply the calibrated qubit π pulse → readout. The two branches are nominal g and e. The SNR is the distance between their IQ centroids divided by the pooled single-shot spread along the line joining the centroids. A positive skew penalty reduces that SNR by a shape factor. Neither the raw SNR nor the penalized score is a state population or a measured assignment fidelity.

## Set the run parameters

- **Readout frequency (MHz):** initial bounds are the resonator frequency ± 0.2 linewidths.
- **Readout gain (a.u.):** initial bounds are 0.1 to 0.25.
- **Readout pulse duration (µs):** initial bounds are 5 to 10 µs. The search changes the RF pulse duration, not the ADC window. Keep the integration window covered even at the minimum pulse duration.
- **Search candidate count (count):** set the run budget; the initial value is 1001. The three ranges start in Expts mode at 51 points each, but only the bounds constrain the continuous search; Expts does not divide its candidates into a grid.
- **Search random seed (count):** supply an integer to repeat the candidate sequence given the same scores; leave blank to start a new sequence.
- **Qubit π pulse:** select the captured, calibrated g-to-e pulse. When the slot is absent, the `pi_pulse` template takes frequency and wiring from the selected qubit; calibrate the pulse before using it.
- **Readout pulse and ADC window:** use the selected resonator's readout module or the `readout` template. The template uses the resonator frequency, wiring and flight time. It leaves 0.015 µs of pulse before and after the incoming ADC window. Start the readout post-delay at five photon decay times, `5/(2πκ)` µs for linewidth κ in MHz, and check that a longer wait does not change the signal.
- **Reset:** optionally select the reset used before each preparation. A blank field skips reset.
- **Qubit relaxation wait (µs):** wait after each read. The initial value is 5 × captured T1. If T1 is absent, the starting estimate is 30 µs and the wait starts at 150 µs. If captured T1 is invalid, supply the wait before Run.
- **Reps**, **Rounds:** shots per preparation and setting in each software round, and the number of rounds. The Form starts at 1000 and 10. The Analyze summary reports the settings from the analyzed run, not the current Form.
- **IQ skew penalty weight (a.u.):** leave at 0 to rank by SNR alone. Increase it to reduce the score of populations with mismatched widths or uncancelled skew along the separation axis.

Use **Defaults** after changing the qubit or resonator source to recapture these starting values. Fill any missing required fields before Run.

## Read the figures and choose a setting

The four panels plot each measured g/e SNR by candidate index, readout frequency (MHz), readout gain (a.u.) and RF pulse duration (µs). Live counts candidates with a score. Analyze places a red star on the best measured candidate in all four panels and shows its three settings. This is a search history, not three independent sweeps. Two points at the same frequency may use different gains or durations. The panels do not estimate uncertainty on the nonlinear SNR.

Compare the best candidate with other high-scoring points. If they cluster at a boundary, expand that bound if it remains safe to scan; if the search barely covers the range, use more candidates. Confirm the selected triple with a new measurement before using it. For each coordinate, its change is the displayed selection minus the old setting. Analyze proposes the triple in `$qubit.ext.best_ro_freq`, `$qubit.ext.best_ro_gain` and `$qubit.ext.best_ro_length`. Review the writeback candidates before applying them. If the readout is a pulse module and the remaining calibration coordinates are available, Analyze also proposes an optimized resonator readout module and the captured resonator's readout slot.

There is no MCP recipe for this experiment. Use the experiment tab or its RPC entry point.
