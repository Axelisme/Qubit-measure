## Behavior

Reset Rabi check: sweeps the initialisation (rabi) pulse gain and acquires three branches simultaneously — without the tested reset, with the tested reset, and with the tested reset followed by another rabi pulse at the same swept gain. Fits the before-reset frequency and uses it for all branches, adding a second harmonic after reset. Reports half peak-to-peak amplitudes, relative contrast, phases, reset-only residual oscillation and residual RMS on one shared IQ axis. These are averaged-readout diagnostics, not reset fidelity. Runs on real hardware.

## Expects (parameter context)

Seeds 51 gain points from 0.0 to 1.0, with relax delay 1.0 us. Run overwrites the Rabi pulse gain. Open and Reset evaluate starting expressions once into direct input using captured sources. Inline qubit pulses read `$qubit.freq` and `$qubit.wiring.ch.` Inline readout reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Missing or invalid required sources leave the affected fields empty and Invalid; supply calibration or direct values before Run. No numeric calibration fallback is supplied.

## Expects (ModuleLibrary)

Rabi pulse uses `$qubit.module.x180` or a fresh pi_pulse template. Tested reset uses `$qubit.module.reset` or a fresh pulse-reset template; library reset families remain selectable. `$qubit.module.readout` selects a captured library link, otherwise a fresh inline readout template with a const waveform. Its required parameter sources must still be present. Optional `$qubit.module.reset` has no fallback: an unset/dangling slot or unbound alias leaves it disabled and Valid.

## Typical writeback

No writeback. Compare relative contrast, reset-only residual oscillation and fit residuals. Second-harmonic amplitude does not uniquely identify coherence or reset error; a flat reset-only trace does not establish ground-state preparation.

## Recommended

Use at least six finite points per branch, preferably 51 points spanning a full Rabi period with enough density to resolve 2f. Frequency is in cycles/gain, phases in degrees at gain zero; phase is unavailable when the fundamental is unresolved above residual noise. Partial sweeps spanning less than a period may give poorly constrained fits. relax_delay should be long enough for thermal equilibration (notebook: 5 × T1 for bath reset). No reset is optional — omit it if no upstream reset is calibrated yet.
