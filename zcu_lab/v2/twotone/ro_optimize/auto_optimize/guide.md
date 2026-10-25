## Behavior

**Readers:** operator, agent. Automatic readout optimization: a Bayesian optimizer searches readout frequency, gain and length jointly to maximize the g/e signal-to-noise ratio (SNR), with the qubit toggled between g and e by a pi pulse. Runs on real hardware. Use it to find a good readout in one shot instead of tuning freq / power / length one axis at a time.

**Last updated:** 2026-10-21

## Parameter sources

Search bounds seed 51 frequency points at `$resonator.freq` +/- 0.2 * `$resonator.kappa`, 51 gain points from 0.1 to 0.25, and 51 length points from 5.0 to 10.0 us. The optimizer samples within these bounds. Relax delay starts at 5*T1; optimizer points start at 1001. Open and Reset fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly` + 0.05 us for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

The qubit pulse uses the `$qubit.module.x180` slot captured at Open/Reset, or a fresh pi_pulse template with a constant waveform. Readout uses the captured `$resonator.module.readout` slot, or a fresh pulse-readout template with a constant waveform. Frequency, gain and length optimization requires pulse readout. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Typical writeback

Proposes the optimizer's best readout frequency, gain and length to `$qubit.ext.best_ro_freq` (MHz), `$qubit.ext.best_ro_gain` (a.u.) and `$qubit.ext.best_ro_length` (us). When acquired run settings include pulse readout, also proposes `$resonator.readout.readout_dpm`. With a captured parameter source, a following `$resonator.module.readout` proposal points to that optimized module. Apply both updates to replace the selected readout for all later forms using this resonator. The qubit's `best_ro_*` scalar destinations stay unchanged.

## Recommended

The three sweep axes set the optimizer's search bounds. Keep frequency within a fraction of a linewidth and choose reviewed gain and length bounds so the search converges. 'Optimizer points' (~1000) sets how many evaluations to spend; raise it for wider bounds, lower it for a quick search. Analyze reports the best evaluated point directly.
