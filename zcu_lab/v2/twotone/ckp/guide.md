**Readers:** operator, agent. Compare qubit frequency shifts with and without π preparation to estimate resonator linewidth, dispersive shift and readout frequency.

## Sequence and key settings

Optional **Reset** → skip or apply **Qubit π preparation pulse** → start both probe tones → wait for both ends → readout. The two branches label the preparation, not pure g and e states.

- Calibrate readout, qubit frequency, π pulse and resonator linewidth first. Bind the qubit and resonator correctly. Check independently the two drive gains and pulse timing; with resonator κ in MHz, initial **Resonator Drive Pulse** length is `1.5 + 5.1/(2πκ)` µs; **Qubit Probe Pulse** pre- and post-delay start at `5/(2πκ)` and `3.1/(2πκ)` µs.
- **Resonator drive RF frequency** starts at `$resonator.freq ± 1.5 × $resonator.kappa`. **Qubit probe RF frequency** starts at `$qubit.freq − 10 MHz … $qubit.freq + 5 MHz`. Set the windows to cover both branches and fill any required blanks before Run.
- **Relax delay** starts at `5 × qubit.t1` (150 µs if absent). **Reps** is per frequency pair and preparation per round.

See [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md) for the shared GUI steps.

## Read the maps and shared fit

Run shows qubit RF versus resonator RF for each preparation. Color is projected mean IQ scaled within each resonator-frequency slice to 0–1 a.u., not population. Equal colors across slices do not imply equal IQ. Blank cells lack data. These maps do not show pointwise shot SEM.

Analyze overlays qubit ridge centers and a shared model. It needs at least five usable centers in each preparation and valid covariance. Fit EDM is the optimizer's estimated distance from its objective minimum, not evidence that the model follows either ridge. The parameter errors come from the shared fit covariance. χ/2π is half the fitted resonator-center separation; κ/2π is the fitted resonator FWHM. A vertical line marks the proposed readout frequency, not necessarily either fitted center.

## Next step

Check both ridges against the shared curves. If a ridge meets a qubit RF edge, extend it past the visible center by at least roughly a linewidth and check both shoulders in the next map. If the fit is supported, writeback proposes `$qubit.ext.chi`, `$resonator.kappa`, and `$resonator.freq`; modules stay unchanged.
