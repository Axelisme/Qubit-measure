**Readers:** operator, agent. Find the sideband carrier of one reset tone. Each shot applies the optional preparation reset and preparation pulse, then the frequency-swept sideband drive at fixed gain and duration, then readout.

**Last updated:** 2026-11-22

## Parameter sources

Seeds 201 frequency points at `$qubit.ext.reset_f` +/- 1.5 * `$qubit.ext.reset_fwhm`. The tested pulse has gain 0.3, length 5.0 µs, and post-delay 5.0 / (2 * pi * `$resonator.kappa`) µs. Relax delay starts at 5*T1. Open and Defaults fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly` + 0.015 µs for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Defaults capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 µs, giving a 150 µs wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Tested reset starts as an editable copy of the reset template. Optional initialization uses `$qubit.module.x180`. An unset slot, missing module or unbound alias leaves it disabled and Valid. Readout uses the captured `$resonator.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Acquisition and figures

Reps starts at 1000 per sweep point and branch in each acquisition round; Rounds starts at 10. Check the requested Reps × Rounds in Analyze before comparing runs.

Live and Analyze plot mean complex IQ projected onto its principal variation axis, falling back to raw I when fewer than two valid means are available. This is a readout contrast in a.u., not an excited-state probability. The live title counts acquired RF points. This spectrum locates a resonance; it does not show shot SEM.

Analyze adds the Lorentzian and its center line. The legend reports the center and full width at half maximum (FWHM) with 1σ fit errors; these are not shot errors. Inspect the measured points on both sides of the feature before accepting the fit. If the feature is at an edge, recenter the next sweep at the fitted RF and capture both sides. Once resolved, a practical next window is center ± 1.5 × FWHM. If the width is comparable to one frequency step, reduce the step before treating the width as measured.

## Typical writeback

Proposes fitted frequency and FWHM to `$qubit.ext.reset_f` and `$qubit.ext.reset_fwhm`. With acquired run settings and captured calibration, proposes `$qubit.reset.reset_10` from the tested reset, replacing only its frequency with this analysis. All other tested fields remain. An optional `$qubit.module.reset` row selects that saved reset module.

The proposal uses only this Run's tested reset and captured parameters, not current aliases, ext values or an older library module. Missing saved cfg or parameter capture leaves only the scalar proposals. Source and Apply errors still propagate. Applying the library rows without the optional reset-slot row leaves the selected reset unchanged.

## Recommended

A sweep of ~201 points spanning a couple of linewidths around the expected sideband captures the dip cleanly; widen the span if `$qubit.ext.reset_f` is only a rough resonator-minus-qubit estimate. The reset-pulse length is held fixed here. Sweep it separately afterwards.
