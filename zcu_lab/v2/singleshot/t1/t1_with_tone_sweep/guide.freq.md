## Behavior

**Readers:** operator, agent. How to map single-shot T1 transition rates versus probe frequency from the GUI or MCP and interpret the results.

Single-shot T1-with-tone, probe-frequency sweep: at each probe frequency it sweeps the wait-plus-probe length from both |g> and |e> initial states (Branch), classifies each shot during acquisition against the |g>/|e> centres, fits the dual transition rates per frequency, and renders the 3×3 rate landscape over (freq, time). Runs on real hardware.

## Parameter sources

Open and Reset take a snapshot of the current aliases, parameters and module choices. Relax delay starts at 5*T1; length stop uses 5 times the tone T1 time scale. Frequency form sweeps `$resonator.freq` +/- 1.5 * `$resonator.kappa`; gain form sweeps 0 to 1. Each form exposes only one outer axis.

Classification starts from `$qubit.ext.readout_cal.g_center`, `e_center` and `radius`. Missing calibration leaves these fields Invalid.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.time_of_fly` + 0.05 us. Missing required sources leave the corresponding fields Invalid; supply them before Run.

The tone time scale uses the stored `$qubit.ext.t1_with_tone`, then ordinary T1, then 30 us. Missing or null sources advance to the next choice. An invalid stored tone T1 leaves the required stop empty for correction. Natural relaxation always uses ordinary T1.

T1-derived defaults use the Open/Reset snapshot. A missing or null T1, or an unavailable snapshot, uses 30 us, so the 5*T1 wait starts at 150 us. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

Pi pulse uses `$qubit.module.x180` or pi_pulse. Probe starts from res_probe. Reset is optional; readout uses `$resonator.module.readout` or readout template. Pulse values remain editable, including those replaced by the sweep. You can select module slots by key. Run uses the Open/Reset snapshot.

## Typical writeback

This is a diagnostic only. Read the rate landscape from the grid; calibration parameters stay unchanged.

## Recommended

Set the classification calibration directly or seed it with 'singleshot/ge'. Keep the frequency sweep coarse (the inner length sweep multiplies the run time). 'uniform=True' (default) uses a linear length sweep; set False to cluster points along the expected exponential decay while preserving the configured window and count.

**Last updated:** 2026-10-21
