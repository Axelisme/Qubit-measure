## Behavior

Automatic readout optimization: a Bayesian optimizer (skopt) searches readout frequency, gain and length jointly to maximize the g/e signal-to-noise ratio (SNR), with the qubit toggled between g and e by a pi pulse. Runs on real hardware. Use it to find a good readout in one shot instead of tuning freq / power / length one axis at a time.

## Expects (parameter context)

Search bounds seed 51 frequency points at `$resonator.freq` +/- 0.2 * `$resonator.kappa`, 51 gain points from 0.1 to 0.25, and 51 length points from 5.0 to 10.0 us. These are optimizer bounds, not an acquisition grid. Relax delay reads 3.0 * `$qubit.t1`; optimizer points start at 1001. Open and Reset evaluate starting expressions once into direct input using captured sources. Inline qubit pulses read `$qubit.freq` and `$qubit.wiring.ch.` Inline readout reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Missing or invalid required sources leave the affected fields empty and Invalid; supply calibration or direct values before Run. No numeric calibration fallback is supplied.

## Expects (ModuleLibrary)

The qubit pulse uses `$qubit.module.x180` as a captured library link, otherwise a fresh pi_pulse template with a const waveform. `$qubit.module.readout` selects a captured library link, otherwise a fresh inline pulse readout template with a const waveform. Direct readout is not supported: frequency, gain and length optimization requires a pulse. Its required parameter sources must still be present. Optional `$qubit.module.reset` has no fallback: an unset/dangling slot or unbound alias leaves it disabled and Valid.

## Typical writeback

Proposes the optimizer's best readout frequency, gain and length into parameter context `$qubit.ext.best_ro_freq` (MHz), `$qubit.ext.best_ro_gain` (a.u.) and `$qubit.ext.best_ro_length` (us). When a cfg snapshot with pulse readout is available, also proposes `$qubit.readout.readout_dpm.`

## Recommended

The three sweep axes define the optimizer's SEARCH BOUNDS (min / max), not a grid — keep them reasonably tight (e.g. freq within a fraction of a linewidth, gain and length in a sensible band) so the search converges. 'Optimizer points' (~1000) sets how many evaluations to spend; raise it for wider bounds, lower it for a quick search. No analysis knobs — it reports the best evaluated point.
