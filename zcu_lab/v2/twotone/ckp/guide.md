## Behavior

CKP ground/excited-resolved spectroscopy: prepares the qubit in g and e, drives resonator and qubit probe tones concurrently over a two-dimensional frequency sweep, and fits the two resonator branches to extract dispersive shift chi, linewidth kappa, and the readout frequency. Runs on real hardware.

## Expects (parameter context)

Inline templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us from the fixed capture. Missing required calibration remains Invalid; no numeric fallback. Resonator sweep is resonator.freq ± 1.5 * resonator.kappa. Qubit sweep runs qubit.freq - 10 to qubit.freq + 5 MHz. Probe timing scales inversely with resonator.kappa.

T1-derived defaults use the fixed Open/Reset capture. A missing/null T1 or absent capture uses 30 us, so the 5*T1 wait starts at 150 us. Stored invalid T1 leaves derived required fields empty while the Form opens. Alias/path/source failures propagate. Defaults remain editable, are not written back as calibration, and do not shorten automatically for active reset.

## Expects (ModuleLibrary)

Uses captured qubit.module.readout with a fresh readout template when unset or dangling. Optional qubit.module.reset stays disabled without a source. Library references remain selectable; sweeps overwrite their controlled fields only at Run. Pi pulse uses qubit.module.x180 with a pi_pulse template fallback; both probe pulses are fresh inline templates.

## Typical writeback

Proposes fitted chi into '`$qubit.ext.chi`', fitted kappa into '`$resonator.kappa`', and fitted readout frequency into '`$resonator.freq`'. No ModuleLibrary writeback.

## Recommended

Run only after qubit frequency, pi pulse, resonator frequency, and resonator linewidth are calibrated. Review both g/e maps and fitted branches before applying writeback; a branch-selection or fit failure can produce plausible-looking scalar values.
