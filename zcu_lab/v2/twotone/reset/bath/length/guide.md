## Behavior

**Readers:** operator, agent. Bath-reset length sweep: holds the tested bath reset at its calibrated cavity frequency and gain and sweeps its duration (cavity and qubit tones together), with a 4-point pi/2 tomography phase cycle at each length, showing how the residual excitation decays with reset length. Runs on real hardware. Run after the cavity freq/gain are found, to pick the shortest fully-resetting length.

**Last updated:** 2026-10-21

## Parameter sources

Tested cavity frequency and gain read `$qubit.ext.reset_bath_freq` and `$qubit.ext.reset_bath_gain`. Seeds 201 length points from 0.05 to 15.0 us, with relax delay starting at 5*T1. Run replaces the pi/2 phase with its tomography cycle. Open and Reset fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Tested reset starts as an editable copy of the bath_reset template. Optional initialization uses `$qubit.module.x180`. An unset slot, missing module or unbound alias leaves it disabled and Valid. The pi/2 tone uses `$qubit.module.x90`, otherwise a fresh pi2_pulse template. Readout uses the captured `$qubit.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Typical writeback

Read the chosen reset length from the decay curve and enter it manually. Register the calibrated 'reset_bath' module in the final phase step.

## Recommended

A length sweep from ~0.05 us to a few times the expected reset time captures the full decay; shorten the span once the plateau is clear. Allow a long relax delay so the qubit fully relaxes before each reset.
