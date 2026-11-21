# STIRAP delay

**Readers:** operator, lab-author, agent. Sweep the relative start time of Pump and Stokes and inspect saved three-state counts.

## Before Run

Open `twotone/stirap/delay` in Measure GUI, or call MCP recipe `stirap_delay`.
Provide the measured Pump 0–3 and Stokes 3–2 carrier frequencies in MHz.
Choose two different charge-drive generators.
Pump and Stokes envelopes overlap, and one channel cannot play both arbitrary-waveform carrier frequencies at the same time.
Set each signed gain within [-1, 1]. The sign is preserved during playback.

Select a readout with saved `centroid3` centers and a 3×3 training confusion matrix.
The readout comes from the captured `$resonator.module.readout` when available.
A missing calibration blocks acquisition. This experiment does not train a classifier.

Pump and Stokes each accept Gaussian, cosine, DRAG or arbitrary envelopes.
Each envelope owns its duration. Gaussian and DRAG use their own length and sigma.
Arbitrary duration comes from the selected asset. Constant and flat-top envelopes are not offered.
The Form starts both envelopes as Gaussian with sigma 0.05 µs and length 0.3 µs.
The core cfg requires both envelopes explicitly.

After changing aliases, component parameters or the module library, use Defaults to seed the Form from a new capture.
Defaults may overwrite unsaved input. Refresh only updates sources and choices; it keeps the current input.

## Lead and acquisition

Set `stokes_lead.start`, `stokes_lead.stop` and `stokes_lead.n` in the Form.
Lead is in µs. Positive lead starts Stokes first. Negative lead starts Pump first.
Zero starts both together. For example, +0.1 µs starts Pump 0.1 µs after Stokes.
Lead describes pulse starts, not their centers or ends.

The host builds and acquires one program per lead coordinate.
Both drives retain their own carrier, signed gain, envelope and length throughout the scan.
Readout starts after the later drive end, even when the lengths differ.
`shots` controls repetitions per coordinate in one acquisition round.
`relax_delay` is the inter-shot wait in µs.
Physical simulation requires both selected drive generators to bind charge lines.
The zero-noise offline probe checks compiled playback without physical evolution.

The live figure shows measured g/e/f fractions against Stokes lead in µs.
The saved Result contains requested lead coordinates with shape `(N,)` and float64 g/e/f counts with shape `(N, 3)`.
It stores counts, not normalized fractions. Stopped scans leave unfinished counts as NaN.
Lead coordinates retain their requested values; hardware quantizes playback times to its clocks.

## Analyze and Save

Analyze uses only the Run's saved centroid3 readout, not current Form or library calibration.
It skips unfinished and zero-count points.
The single `populations` figure shows raw fractions as dashed curves and confusion-corrected fractions as solid curves.
Each state has the same color in both curves. The horizontal axis is Stokes lead in µs.
The vertical axis shows the preparation-label fractions for g, e, and f.
Corrected fractions are not clipped to [0, 1].

The summary contains the training confusion matrix, its condition number and its minimum singular value.
The summary and recipe report give `f_population_estimate`, the largest finite
corrected f population on this grid, and `stokes_lead_us`.
Ties go to the first acquired point. If no finite value is available, the peak
and its coordinates are null. The estimate is unclipped and does not identify
a global optimum.
A large condition number warns that correction can amplify count noise.
These diagnostics do not measure transfer fidelity or prove state purity.
Missing saved cfg, missing calibration, no completed points or a singular matrix makes analysis fail.

The recipe performs Run, Analyze, then Save. It does not ask for writeback.
After a successful Run, Save still runs if analysis fails or returns no result.
A failed Run does not trigger Save. Native data preserves the saved cfg and counts.
The new leaf does not load or migrate runs from the old combined STIRAP experiment.

## Recipe inputs

`start_us`, `stop_us` and `n` set the lead sweep. `n` maps to cfg `stokes_lead.expts`.
`pump_freq_mhz`, `stokes_freq_mhz`, `pump_gain` and `stokes_gain` are required.
Optional generator channels, readout reference, shots and relaxation retain Form values when omitted.
Explicit null goes to the field owner and may be rejected.

`pump_waveform_ref` and `stokes_waveform_ref` select a library key or an exact Form option such as `Inline: cosine`.
Use `overrides` to enter parameters for the chosen envelope, for example `pump_waveform.length`.
For an existing referenced envelope, issue the public Form `to_inline` command before editing its scalar parameters.
The recipe applies overrides last and checks readiness before Run.

Offline structural acceptance does not establish physical transfer accuracy or real-device calibration.
The full-Hamiltonian reference comparison is separate work and has not been run for this leaf.
