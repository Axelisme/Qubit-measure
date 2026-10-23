## Behavior

**Readers:** operator, agent. Single-tone sideband-reset frequency sweep: drives the tested pulse-reset tone and sweeps its frequency, fitting the qubit response (Lorentzian) to find the reset sideband frequency and its linewidth. Runs on real hardware. Typically run after an init pulse excites the qubit, to locate the f01+f12 (`$resonator.freq` - `$qubit.freq`) sideband at which the reset pulse dumps the excitation.

**Last updated:** 2026-10-21

## Parameter sources

Seeds 201 frequency points at `$qubit.ext.reset_f` +/- 1.5 * `$qubit.ext.reset_fwhm`. The tested pulse has gain 0.3, length 5.0 us, and post-delay 5.0 / (2 * pi * `$resonator.kappa`) us. Relax delay starts at 5*T1. Open and Reset fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly` + 0.05 us for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Tested reset starts as an editable copy of the reset template. Optional initialization uses `$qubit.module.x180`. An unset slot, missing module or unbound alias leaves it disabled and Valid. Readout uses the captured `$qubit.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Typical writeback

Proposes fitted frequency and FWHM to `$qubit.ext.reset_f` and `$qubit.ext.reset_fwhm`. With acquired run settings and captured calibration, proposes `$qubit.reset.reset_10` from the tested reset, replacing only its frequency with this analysis. All other tested fields remain. An optional `$qubit.module.reset` row selects that saved reset module.

## Recommended

A sweep of ~201 points spanning a couple of linewidths around the expected sideband captures the dip cleanly; widen the span if `$qubit.ext.reset_f` is only a rough resonator-minus-qubit estimate. The reset-pulse length is held fixed here. Sweep it separately afterwards.
