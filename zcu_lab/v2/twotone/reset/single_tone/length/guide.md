## Behavior

**Readers:** operator, agent. Single-tone sideband-reset length sweep: holds the tested pulse-reset at its calibrated sideband frequency and sweeps its duration, showing how the residual excitation decays with reset length. Runs on real hardware. Run after the reset frequency is found, to pick the shortest length that fully resets the qubit.

**Last updated:** 2026-10-21

## Parameter sources

The tested pulse frequency reads `$qubit.ext.reset_f`. Seeds 50 length points from 0.1 to 20.0 us, with relax delay starting at 5*T1. Open and Reset fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly` + 0.05 us for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Tested reset starts as an editable copy of the reset template. Optional initialization uses `$qubit.module.x180`. An unset slot, missing module or unbound alias leaves it disabled and Valid. Readout uses the captured `$resonator.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Typical writeback

Analysis plots the length response for manual selection. Calibration stays unchanged until you update it. The single-tone frequency experiment proposes the tested reset module.

## Recommended

A length sweep from ~0.1 us to a few times the expected reset time captures the full decay; shorten the span once the plateau is clear. Allow a long relax delay between shots so the qubit fully relaxes before each reset.
