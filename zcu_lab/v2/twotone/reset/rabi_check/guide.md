## Behavior

Reset Rabi check: sweeps the initialisation (rabi) pulse gain and acquires three branches simultaneously — without the tested reset, with the tested reset, and with the tested reset followed by another rabi pulse at the same swept gain. Fits the before-reset frequency and uses it for all branches, adding a second harmonic after reset. Reports half peak-to-peak amplitudes, relative contrast, phases, reset-only residual oscillation and residual RMS on one shared IQ axis. These are averaged-readout diagnostics, not reset fidelity. Runs on real hardware.

## Expects (parameter context)

Reads from the parameter context (all optional): 'q_f' / 'qub_ch' — qubit frequency and channel, seeding rabi_pulse and tested_reset defaults; 'r_f' / 'res_ch' / 'ro_ch' / 'timeFly' — resonator / readout defaults.

## Expects (ModuleLibrary)

Needs rabi_pulse (used before and after tested_reset), tested_reset (any reset shape: none / pulse / two-pulse / bath), and a readout module. Optionally references a calibrated upstream reset (disabled when absent).

## Typical writeback

No writeback. Compare relative contrast, reset-only residual oscillation and fit residuals. Second-harmonic amplitude does not uniquely identify coherence or reset error; a flat reset-only trace does not establish ground-state preparation.

## Recommended

Use at least six finite points per branch, preferably 51 points spanning a full Rabi period with enough density to resolve 2f. Frequency is in cycles/gain, phases in degrees at gain zero; phase is unavailable when the fundamental is unresolved above residual noise. Partial sweeps spanning less than a period may give poorly constrained fits. relax_delay should be long enough for thermal equilibration (notebook: 5 × T1 for bath reset). No reset is optional — omit it if no upstream reset is calibrated yet.
