## Behavior

**Readers:** operator, agent. Reset Rabi check: sweeps the initialisation (rabi) pulse gain and acquires three branches at each gain: the Rabi pulse alone, Rabi followed by the tested reset, and Rabi followed by the tested reset and another Rabi pulse at the same swept gain. Fits the before-reset frequency and uses it for all branches, adding a second harmonic after reset. Reports half peak-to-peak amplitudes, relative contrast, phases, reset-only residual oscillation and residual RMS on one shared IQ axis. These averaged-readout diagnostics describe contrast and oscillation. Reset fidelity requires a separate state-preparation validation. Runs on real hardware.

**Last updated:** 2026-10-21

## Parameter sources

Seeds 51 gain points from 0.0 to 1.0, with relax delay starting at 5*T1. Run overwrites the Rabi pulse gain. Open and Reset fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Rabi pulse uses `$qubit.module.x180` or a fresh pi_pulse template. Tested reset uses `$qubit.module.reset` or a fresh pulse-reset template; saved reset families remain selectable. Readout uses the captured `$qubit.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Typical writeback

Analysis reports results for review; calibration stays unchanged. Compare relative contrast, reset-only residual oscillation and fit residuals. Second-harmonic amplitude can reflect coherence or reset error. Ground-state preparation requires independent validation even when the reset-only trace is flat.

## Recommended

Use at least six finite points per branch, preferably 51 points spanning a full Rabi period with enough density to resolve 2f. Frequency is in cycles/gain, phases in degrees at gain zero; phase is unavailable when the fundamental is unresolved above residual noise. Partial sweeps spanning less than a period may give poorly constrained fits. relax_delay should be long enough for thermal equilibration (notebook: 5 × T1 for bath reset). The upstream reset is optional; keep it disabled until calibrated.
