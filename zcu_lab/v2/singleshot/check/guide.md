## Behavior

**Readers:** operator, agent. How to check single-shot readout classification from the GUI or MCP and interpret the results.

Single-shot classification check: takes 'shots' single-shot readouts of the prepared state and draws the IQ scatter with the |g>/|e> discrimination circles overlaid, reporting the fraction classified ground / excited / other. Runs on real hardware. Acquisition records IQ shots directly; classification during analysis requires the calibrated centres.

## Parameter sources

Open and Reset take a snapshot of the current aliases, parameters and module choices. Relax delay is 5 * `$qubit.t1`. The fresh probe template uses `$qubit.freq` and `$qubit.wiring.ch`.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.ext.time_of_fly` + 0.05 us. Missing required sources leave the corresponding fields Invalid; supply them before Run.

T1-derived defaults use the Open/Reset snapshot. A missing or null T1, or an unavailable snapshot, uses 30 us, so the 5*T1 wait starts at 150 us. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

Optional init/reset use `$qubit.module.x180`/reset. Probe starts from qub_probe. Readout uses `$qubit.module.readout` or the readout template. Run uses the Open/Reset snapshot.

## Typical writeback

This is a visual diagnostic of the existing single-shot discrimination. Calibration parameters stay unchanged.

## Recommended

Run after 'singleshot/ge'. Use a large 'shots' (~5000+) for a well-sampled scatter; a tight cluster inside the matching circle indicates good discrimination.

**Last updated:** 2026-10-21
