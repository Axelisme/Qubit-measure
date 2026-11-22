**Readers:** operator, agent. Set up a Ramsey delay sweep, read the fringe and its envelope, and decide whether to update the qubit frequency.

# Ramsey coherence and frequency

Ramsey measures loss of phase coherence during free evolution. One repetition applies optional Reset → π/2 → swept free-evolution delay → π/2 with a phase ramp → Readout. The phase ramp is applied to the **second pulse only**; both pulses keep the same RF carrier. Calibrate the π/2 pulse before running.

## Set up the sweep

| Form field | Setting |
| --- | --- |
| Reset | Choose an optional preparation module before the first π/2 pulse. None skips it. The starting source is `$qubit.module.reset`. |
| π/2 pulse | Use the calibrated `$qubit.module.x90` or fill the π/2 template. The pulse's Frequency is the RF carrier in MHz used for both pulses. It is not the phase-ramp frequency. |
| Readout | Use `$resonator.module.readout` or fill the readout template. The pulse excites the readout resonator; the ADC integrates IQ over the readout window. |
| Free-evolution delay | Set Start, Stop and Step in µs, or set Expts for the number of points including the endpoints. Open starts at 0 and stops at 1.5 × `qubit.t2r` with 101 points; fill a Stop if no T₂* is stored. |
| Detune ratio (fringes/step) | Default 0.05. Positive advances the second pulse phase as delay increases; negative reverses it. The requested ramp in MHz is ratio ÷ Step (µs). Zero disables the ramp. |
| Reps, Rounds | Reps repeats each delay per round. Rounds limits how many times the full sweep is averaged. Open starts at 1000 × 10. |
| Early-stop SNR | Leave empty to complete all rounds. Otherwise stop once the IQ trace reaches the requested signal-to-noise ratio at a round boundary. |
| Relax delay | Wait between repetitions for the qubit to relax. The start is 5 × `qubit.t1` in µs (150 µs if T₁ is absent). |

Templates take the qubit and resonator frequencies and wiring from the captured point. The readout timing starts at the resonator time of flight plus 0.015 µs. Fill any required blanks. Use Defaults after changing the point, bindings or calibration to recapture starting values; it can replace unsaved edits. Refresh updates choices without replacing inputs.

For a fringe fit, sample at least **10 points per fringe period**. If the observed fringe frequency is `f` MHz, use a Step no greater than `1/(10 f)` µs. A wider step can hide cycles or produce an apparently slower fringe by aliasing. Span about **3 × T₂*** so the envelope approaches baseline. A short window shows a fringe that has barely decayed and leaves the fit lifetime determined mostly by extrapolation. Extend Stop and increase Expts together so the step stays small. If the fringe is hard to resolve, set a known positive ramp large enough that the fitted frequency stays on the positive branch, then keep the sample density.

## Read the result

- Liveplot shows each delay against the mean IQ projected along the principal axis of the trace, in a.u. If that axis is undefined, the plot uses I. The title counts retained shots per point. IQ contrast is not a calibrated population.
- Each vertical bar extends one shot SEM around its mean: the sample standard deviation of the retained shots along the same IQ axis divided by the square root of their number. This is not the spread of individual shots, the scatter between rounds, or a fit-parameter error. A missing statistic or fewer than two shots leaves that point without a bar. Older mean-only files still load and fit without bars.
- Analyze overlays the fringe or decay model evaluated on a dense delay grid. The title reports T₂* and its fit standard error; a fringe fit also reports the observed **fringe frequency** and its fit standard error, both in MHz. R² compares the fit residual with the measured variation around its mean. Values closer to one mean the model explains more of the observed variation; there is no R² acceptance threshold. Check the fit through the full delay window and compare local deviations with the bars.
- **Fit fringe** and **Fit phase offset (fringe only)** default to on. Disabling the second fits a fixed 0° or 180° branch using the shortest-delay IQ point. On resonance, set the ratio to zero and switch off Fit fringe to measure the envelope directly.

## Update the captured calibration

The Analyze summary shows the same run's Reps × Rounds and **Applied detune**. Applied detune is the phase-ramp frequency realized after hardware rounding, not the fitted fringe frequency. The count product is the configured budget, not a measurement of which shots survived.

Only apply frequency correction for a resolved fringe on the positive-frequency branch with a known applied detune. Use the RF Frequency from the π/2 pulse used in **this** run, not a value changed since Run:

`suggested qubit.freq = source π/2 Frequency + Applied detune − Fringe frequency`

For example, 5800.000 MHz + 0.400 MHz − 0.402 MHz = 5799.998 MHz. Lower the carrier by 0.002 MHz. A fitted fringe 0.002 MHz below the applied detune calls for raising the carrier by 0.002 MHz. A fringe fit measures a magnitude, so if the branch is unclear, repeat with a different known positive ramp and verify that the fringe shifts in the same direction before applying the formula.

Analyze proposes `qubit.t2r` for the captured point and, when the source settings and applied detune are available, `qubit.freq` for a fringe fit. A reopened native file does not store the actual applied detune; it omits that summary line and does not propose a frequency correction. Do not substitute the current Form's detune. A decay-only fit also cannot update frequency. The lifetime can be reviewed independently. Frequency writeback does not retune a saved pulse. Recalibrate its gain or length at the new carrier before replacing the X90 module.

Use the `t2ramsey` recipe to Run, Analyze and Save with the same fields. It asks before applying writeback.
