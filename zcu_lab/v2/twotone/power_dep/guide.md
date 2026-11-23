**Readers:** operator, agent. Find a probe gain that reveals a qubit line without merging it with nearby features.

## Sequence and key settings

Optional **Reset** and preparation → qubit **Probe Pulse** → resonator **Readout** → **Relax delay**. Run sweeps **Probe RF frequency** at each **Probe gain**. Gain is a DAC full-scale fraction, not readout power or dBm.

Use calibrated readout, bind the qubit and resonator, and set RF endpoints in MHz around a known transition. Lower the top **Probe gain** if the line already broadens at small gain. **Reps** is per RF point per round; **Rounds** is per gain.

See [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md) for the shared GUI steps.

## Read the map and choose a gain

Run maps RF frequency versus probe gain. Color is the magnitude of mean IQ after subtracting the median complex IQ in that spectrum (a.u.), not probability. Blank cells lack data. The row count includes a gain still averaging. Adjacent axes compare the latest and previous spectra. This two-dimensional map does not report shot SEM.

Analyze redraws the selected run's map and spectra, including after Load; Load alone does not open a figure. **Reps × Rounds** is omitted if the loaded settings are unavailable.

Choose the smallest gain with a clear ridge across nearby gain rows. If the ridge is clear at 0.05 a.u. but merges by 0.15 a.u., start the follow-up at 0.05 a.u. Widen the RF window if the ridge meets an edge. Use `twotone/freq` for a finer scan around the observed center; this experiment does not fit or write back.
