## Behavior

**Readers:** operator, agent. CKP ground/excited-resolved spectroscopy: prepares the qubit in g and e, drives resonator and qubit probe tones concurrently over a two-dimensional frequency sweep, and fits the two resonator branches to extract dispersive shift chi, linewidth kappa, and the readout frequency. Runs on real hardware.

**Last updated:** 2026-10-21

## Parameter sources

Pulse templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.time_of_fly + 0.05 us from the sources captured at Open/Reset. Missing required calibration leaves the fields Invalid until you supply calibration or direct values. Resonator sweep is resonator.freq ± 1.5 * resonator.kappa. Qubit sweep runs qubit.freq - 10 to qubit.freq + 5 MHz. Probe timing scales inversely with resonator.kappa.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Readout uses the qubit.module.readout slot captured at Open/Reset, or a fresh readout template when the slot is unset or points to a missing module. Optional qubit.module.reset stays disabled when absent. Saved modules remain selectable; sweeps replace only the swept fields at Run. Pi pulse uses qubit.module.x180 with a pi_pulse template fallback; both probe pulses start from fresh templates.

## Typical writeback

Proposes fitted chi into '`$qubit.ext.chi`', fitted kappa into '`$resonator.kappa`', and fitted readout frequency into '`$resonator.freq`'. Proposals update scalar calibration only.

## Recommended

Run only after qubit frequency, pi pulse, resonator frequency, and resonator linewidth are calibrated. Review both g/e maps and fitted branches before applying writeback; a branch-selection or fit failure can produce plausible-looking scalar values.
