**Readers:** operator, agent. Estimate spin-echo coherence time T₂ Echo from the decay of the mean IQ projection.

## Sequence and key settings

Optional **Reset** → π/2 → half delay → refocusing π → half delay → final π/2 → **Readout**. Sweep the sum of the two free-evolution intervals. The midpoint π pulse refocuses slow dephasing, so the echo envelope can outlast Ramsey T₂*. Check the Rabi-calibrated pulses and single-tone readout timing before Run.

- **Total echo delay** starts at `0…1.5 × qubit.t2e` (µs). Set the scan width to about **3 × the expected T₂ Echo** to measure most of the decay. A shorter scan leaves the tail unseen and makes the fit rely on extrapolation.
- **Detune ratio (fringes/step)** advances the final π/2 phase, not the drive frequency. Requested detune in MHz is the ratio divided by Step in µs. For fringe frequency *f* in MHz, take at least **10 points per fringe**: use Step ≤ `1/(10 × |f|)` µs and increase **Expts** as you extend the window. Sparse samples can hide or alias the fringe.
- **Fit method** defaults to `decay`; use ratio 0 for a standard echo. Choose `fringe` with a nonzero applied phase ramp to fit the modulation on the same envelope. **Fit phase offset (fringe only)** frees the phase instead of fixing it at 0°. A zero source detune forces decay. If neither the source cfg nor an in-memory actual detune is available, choose decay; a fringe request fails.
- **Relax delay** starts at `5 × qubit.t1` (150 µs when absent). It sets the wait after each shot.

See [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md) for the shared GUI steps.

## Read the plots

Check Live for a visible decay or fringe as the shots per point accumulate. The horizontal axis is total echo delay (µs). The vertical axis, **IQ projection (a.u.)**, shows the signed projection of mean IQ, not a state population.

Live estimates the principal axis from all available points. Analyze estimates it and fits after dropping the first delay by position, so the projected values can differ. Both use I if fewer than two finite points are available for the projection or if all available IQ values are equal. Compare the measured points with the model curve. T₂ Echo is the envelope decay time; fringe frequency is the fitted modulation, not the applied phase ramp. The Analyze summary separately lists the source Reps × Rounds and actual applied detune when available.

Each vertical bar is the projected shot sample standard deviation divided by the square root of its retained shot count. It estimates the uncertainty of that point's mean, not round-to-round scatter or fit error. A point needs at least two retained shots for a bar. Native Save stores the shot sums when available. After loading a Native file with shot sums, Analyze draws the same bars as it did for the original run. Older Native files without shot sums still load and fit, but show no bars. Labber export still contains only the mean IQ trace.

The uncertainty beside T₂ Echo is the fit standard error, not shot SEM. R² compares the residual variation with the measured variation. Normalized residual RMS compares the typical residual size with the signal span. Check for a visible envelope and systematic differences between data and model before using the estimate.

## Next step

Writeback proposes `qubit.t2e` and, when available, an independent sample at the captured bias. Neither changes RF frequency or module slots. Read the Writeback review note for any failed or unavailable preselection check, and inspect the decay and fit before applying the estimate.
