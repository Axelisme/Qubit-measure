## Behavior

**Readers:** operator, agent. T2 echo (spin/Hahn echo): a pi/2 → delay → pi → delay → pi/2 sequence with the total free-evolution delay swept; the refocusing pi pulse cancels low-frequency dephasing, and the decay fit yields the echo coherence time T2E. Runs on real hardware. Run after both pi and pi/2 pulses are calibrated; T2E is typically longer than the Ramsey T2*.

**Last updated:** 2026-10-21

After changing the parameter source, use Defaults to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Defaults can overwrite unsaved inputs.

## Parameter sources

Total delay runs 0 to 1.5 * qubit.t2e µs over 101 points. Detune ratio is fringes per sweep step; native MHz detune is ratio / step. Relax delay is 5 * qubit.t1 µs. Other missing required calibration leaves the affected fields empty and Invalid.

Open and Defaults capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 µs, giving a 150 µs wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Readout uses the `$resonator.module.readout` slot captured at Open/Defaults, or a readout template. Reset uses qubit.module.reset and stays disabled when absent. Drive slots use qubit.module.x180/x90 with pi_pulse/pi2_pulse template fallbacks. Unset slots or slots pointing to missing modules use the named template. Saved modules remain selectable. Pulse templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.time_of_fly + 0.015 µs; missing required calibration remains Invalid.

## Typical writeback

Initial T2/sample selection requires positive finite T2, r2 at least 0.9,
normalized residual RMS at most 0.1 and relative T2 error at most 0.5.
Failed or unavailable checks leave candidates unselected with explanatory notes.
You may select them manually after reviewing the source and fit.

Proposes fitted qubit.t2e (µs) to the captured qubit and working point, including loaded data with IQ only. Frequency and module slots stay unchanged. Proposals keep the qubit and working point captured for this result even after focus changes. An independent t2e_us/t2e_err_us sample uses the captured Run bias when device evidence is available.

## Recommended

Enable Fit phase offset to fit a free fringe phase; it defaults to fixed phase and applies to fringe fits only. Set the 'Detune ratio (fringes/step)' input (default 0.1). This is the number of fringe periods per delay-sweep step; the absolute detune (MHz) applied to the final pi/2 pulse phase is detune_ratio / sweep step, leaving a residual oscillation on the echo decay. Analysis 'Fit method' defaults to 'decay'. Select 'fringe' explicitly only for nonzero source detune; choosing 'decay' remains valid. Zero source detune forces decay even when fringe is selected. Analysis uses the saved cfg detune, or finite saved actual detune without cfg. A fringe request without either source reports a missing-detune error; explicit decay still works. Set the ratio to 0 for a standard echo (pure exponential decay → T2E). Use a total-delay window about three times the expected echo decay time. A shorter window leaves a substantial envelope at the right edge and forces the fit to extrapolate. For fringe fits, use at least 10 samples per period (Step ≤ 1/(10 × fringe frequency in MHz) µs); sparse sampling can hide fringes or alias them into a slower pattern. Increase Expts when extending the window. Analyze drops the first delay point.

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Obtain calibrated pi and pi/2 pulses. Confirm the total-delay window, detune_ratio and readout timing.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, choose Defaults to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, choose Defaults and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `t2echo` after inspecting the same Form errors.
   points maps to total-delay Expts; max_delay_us sets the stop. fit_method and fit_phase change only analysis.
   The Form chooses module sources. Omitted inputs retain Form values; null clears/disables fields. Overrides apply last, then check_ready reports missing or invalid fields before Run.
   Missing inputs return needs_parameters before acquisition. Defaults may replace unsaved inputs.

## Reading the plots

Both plots show mean IQ projected along the trace's principal IQ axis (a.u.), not population. The live title counts retained shots per delay. Each bar is one shot SEM in the plotted direction: sample standard deviation of retained shots divided by the square root of their count. Legacy mean-only files load without bars. The fit is unweighted by those bars. Analyze overlays the fitted model on a dense grid. The title's uncertainty is a fit standard error, not shot SEM.

R² compares the residual with the measured variation; closer to one means the model explains more of the signal. Normalized residual RMS is the typical residual relative to the full signal range. Writeback uses the existing limits below; a candidate can be unselected even when a fit is displayed. The summary shows Reps × Rounds from the acquired cfg and Applied detune when the run returned the phase ramp realized after rounding. The fitted fringe frequency on the figure is not the applied ramp.
