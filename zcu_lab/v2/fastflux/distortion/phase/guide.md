# Fast flux phase distortion

## Prepare the pulses

Use this experiment after readout and the qubit X90 pulse are calibrated at the
working flux. Bind the qubit and resonator aliases before opening the tab.
Open and Reset capture the current sources. Changing aliases later does not
replace the displayed inputs.

The form selects `$qubit.module.x90` and `$qubit.module.readout` when present.
Otherwise it opens editable pulse templates. Reset is optional and uses
`$qubit.module.reset`. Flux pulse uses the shared `flux_pulse` role. It starts
with a const waveform, gain 1.0, frequency 0 MHz, Nyquist zone 2 and channel from
`$qubit.wiring.lo_flux_ch`. This form starts its waveform duration at 1 us.
A missing channel stays invalid without a fallback. These values are not a
calibration. Set the gain and duration for the intended transient and check the
channel, frequency and delays against the actual wiring before Run. No safe
operating gain is implied.

## Choose the scan

- Length is the wait before the first X90, in us. The initial range is 0 to 1 us
  with 101 points. Set it to cover the transient and a background region.
- Phase is the second X90 projection phase, in degrees. The initial range is
  −180 to 180° with 61 points. Use enough distinct phases to resolve a sinusoid.
- Readout t is a minimum branch duration in us, initially 2 us. It is not a
  delay added after the pair. Set it long enough for the intended flux settling.
- Reps and Rounds control averaging. Relax delay starts at five times the
  captured T1, or 150 us if T1 is missing, and can be edited.

Review the resolved cfg before Run. The initial scan and timing values are
editing seeds, not hardware safety limits.

## Run and inspect

After reset, the flux pulse runs alongside a wait followed by two adjacent
X90 pulses, and alongside the Readout t wait. There is no free-evolution delay
between the X90 pulses. Readout starts after the longest branch finishes.

The measurement heatmap shows rotated real IQ against pair-start time and
projection phase. Analyze fits sine and cosine coefficients at each time and
plots unwrapped phase in degrees against the pair midpoint in us. The
rectangular overlay marks the programmed flux interval. It is a timing
reference, not a fitted transfer function or an automatic correction.

Inspect the heatmap for contrast and the phase trace for the transient before
changing pulse compensation. There are no automatic writeback proposals.

## Save and load

Save uses the core native format, including the cfg and complex IQ array.
In memory the signal shape is (wait, phase), time is in us and phase is in
degrees. Native files store the time axis in seconds. Load keeps the current
form unchanged. Analyze uses the loaded run's cfg for pulse timing; data
without a usable source cfg cannot be analyzed here.
