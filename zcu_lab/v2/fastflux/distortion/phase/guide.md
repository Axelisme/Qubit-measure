# Fast flux phase distortion

**Readers:** operator, agent. Set up a fast-flux phase scan from the GUI or MCP and inspect the transient response.

**Last updated:** 2026-10-21

## Prepare the pulses

Use this experiment after readout and the qubit X90 pulse are calibrated at the
working flux. Bind the qubit and resonator aliases before opening the tab.
Open and Reset capture the current sources. After changing aliases, use Reset
to refresh the displayed inputs.

The form selects `$qubit.module.x90` and `$resonator.module.readout` when present.
Otherwise it opens editable pulse templates. Reset is optional and uses
`$qubit.module.reset`. The Flux pulse module slot uses the `flux_pulse` template.
It starts with a const waveform, gain 1.0, frequency 0 MHz, Nyquist zone 2 and channel from
`$qubit.wiring.lo_flux_ch`. This form starts its waveform duration at 1 us.
A missing channel leaves the form Invalid until you supply a confirmed channel.
These values are editing seeds. Establish the flux calibration and safe operating
gain separately. Set the gain and duration for the intended transient and check
the channel, frequency and delays against the actual wiring before Run.

## Choose the scan

- Length is the wait before the first X90, in us. The initial range is 0 to 1 us
  with 101 points. Set it to cover the transient and a background region.
- Phase is the second X90 projection phase, in degrees. The initial range is
  −180 to 180° with 61 points. Use enough distinct phases to resolve a sinusoid.
- Readout t is a minimum parallel-branch duration in us, initially 2 us.
  Readout starts after the longest branch finishes. Set it long enough for the
  intended flux settling.
- Reps and Rounds control averaging. Relax delay starts at five times the
  captured T1, or 150 us if T1 is missing, and can be edited.

Review the resolved cfg before Run. The initial scan and timing values are
editing seeds. Confirm hardware safety limits separately.

## Run and inspect

After reset, the flux pulse runs alongside a wait followed by two adjacent
X90 pulses, and alongside the Readout t wait. The second X90 starts immediately
after the first. Readout starts after the longest branch finishes.

The measurement heatmap shows rotated real IQ against pair-start time and
projection phase. Analyze fits sine and cosine coefficients at each time and
plots unwrapped phase in degrees against the pair midpoint in us. The
rectangular overlay marks the programmed flux interval. It is a timing
reference. Transfer-function fitting and pulse correction require separate work.

Inspect the heatmap for contrast and the phase trace for the transient before
changing pulse compensation. Analysis produces plots only; calibration parameters
and modules retain their values.

## Save and load

Save preserves the cfg, scan axes and complex IQ array.
In memory the signal shape is (wait, phase), time is in us and phase is in
degrees. Native files store the time axis in seconds. Load keeps the current
form unchanged. Analyze uses the loaded run's cfg for pulse timing and reports
an error if that source cfg is missing or unusable.
