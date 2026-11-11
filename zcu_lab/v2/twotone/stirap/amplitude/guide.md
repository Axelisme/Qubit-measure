**Readers:** operator, agent. Run the STIRAP common-amplitude sweep and interpret saved-readout g/e/f fractions.

# STIRAP amplitude

Open `twotone/stirap/amplitude` in measure-gui, or call MCP recipe `stirap_amplitude`.
This experiment sweeps a dimensionless multiplier applied to both signed drive gains.
It compiles one program and acquires the whole scale grid in one hardware loop.
It does not compile or acquire separately at each coordinate.

## Before Run

- Measure the Pump 0–3 and Stokes 3–2 carrier frequencies at the intended operating point. Enter both in MHz. Do not substitute simulator parameters for measured carriers.
- Bind Pump and Stokes to different charge-drive generators. Check the physical wiring before using hardware.
- Select a pulse readout with a saved `centroid3` calibration. The calibration must contain distinct g/e/f IQ centers and a training confusion matrix. Obtain it separately before this experiment.
- Set the two base gains, including their signs. Choose scale endpoints so both scaled gains remain within [-1, 1]. Scale is not a probability or an absolute DAC gain.

## Controls and sequence

Pump and Stokes each accept gauss, cosine, drag or arb. They can use different families and lengths.
The Form initially selects Gaussian envelopes with sigma 0.05 us and length 0.3 us.
For arb, choose an available waveform asset. Its samples determine the duration.
Generator envelope capacity errors come from the existing upload check. Shorten the waveform or choose another generator if upload fails.

`stokes_lead` is in us. A positive value starts Stokes first and delays Pump.
A negative value starts Pump first. Zero starts both together.
The program joins both pulse ends before the terminal readout.
It uses the same two envelopes, carrier frequencies and lead at every scale.

`scale.start`, `scale.stop` and the Form's point count define the common multiplier grid.
Each multiplier scales both base gains, preserving their relative signs.
Each generator quantizes its resulting gain separately.
The saved coordinates are the requested multipliers, not a claim of identical gain quantization on the two generators.
`shots` sets repetitions at each coordinate in one acquisition round.
`relax_delay` is the wait in us after each shot.

## Analyze and Save

Run returns float64 g/e/f shot counts with shape `(number of scales, 3)`. Save writes those counts with their requested scale coordinates.
The live figure plots raw preparation-label fractions against dimensionless scale.
An interrupted acquisition without a complete round leaves counts as NaN.

Analyze uses only the readout calibration saved with this Run.
The left panel shows raw fractions. The right panel shows confusion-corrected fractions without clipping.
The title reports the training confusion matrix's condition number and minimum singular value.
Missing calibration, incomplete or empty counts, and singular confusion prevent analysis.
A large condition number warns that correction amplifies uncertainty. Corrected fractions can fall outside [0, 1].
These are preparation-label estimates, not transfer fidelity or a measurement of the intermediate state.

The MCP recipe performs Run, Analyze, then Save. It saves completed Run data even if Analyze fails.
A failed Run does not enter Analyze or Save. The recipe proposes no writeback.
For waveform selection, use `pump_waveform_ref` and `stokes_waveform_ref` with a library key or an inline Form option such as `Inline: cosine`.
Use `overrides` to set that waveform's parameters. Recipe `n` maps to the Form point count and core `expts`.

This offline-validated program structure does not establish physical transfer accuracy.
Full-Hamiltonian reference validation and hardware calibration are separate checks.
