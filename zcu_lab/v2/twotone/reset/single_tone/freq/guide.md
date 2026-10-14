## Behavior

Single-tone sideband-reset frequency sweep: drives the tested pulse-reset tone and sweeps its frequency, fitting the qubit response (Lorentzian) to find the reset sideband frequency and its linewidth. Runs on real hardware. Typically run after an init pulse excites the qubit, to locate the f01+f12 (r_f - q_f) sideband at which the reset pulse dumps the excitation.

## Expects (parameter context)

Requires $qubit.ext.reset_f and $qubit.ext.reset_fwhm for the sweep centre and half-span of 1.5 FWHM. Missing, null or invalid values fail cfg lowering. Drive defaults use the captured qubit frequency and wiring; post-delay uses $resonator.kappa.

## Expects (ModuleLibrary)

Builds the tested pulse reset inline and needs a readout module. Optionally references a calibrated upstream reset and an init pulse (a library pi pulse when present) — both disabled when no library entry exists.

## Typical writeback

Proposes fitted frequency and FWHM to $qubit.ext.reset_f and $qubit.ext.reset_fwhm. With Run cfg and capture, proposes $qubit.reset.reset_10 from tested_reset, replacing only its frequency with this analysis. All other tested fields remain. An optional $qubit.module.reset row selects the concrete root.

## Recommended

A sweep of ~201 points spanning a couple of linewidths around the expected sideband captures the dip cleanly; widen the span if reset_f is only a rough resonator-minus-qubit estimate. The reset-pulse length is held fixed here — sweep it separately afterwards.
