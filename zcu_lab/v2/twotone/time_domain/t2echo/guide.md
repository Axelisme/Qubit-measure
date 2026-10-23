## Behavior

**Readers:** operator, agent. T2 echo (spin/Hahn echo): a pi/2 → delay → pi → delay → pi/2 sequence with the total free-evolution delay swept; the refocusing pi pulse cancels low-frequency dephasing, and the decay fit yields the echo coherence time T2E. Runs on real hardware. Run after both pi and pi/2 pulses are calibrated; T2E is typically longer than the Ramsey T2*.

**Last updated:** 2026-10-21

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Parameter sources

Total delay runs 0 to 1.5 * qubit.t2e us over 101 points. Detune ratio is fringes per sweep step; native MHz detune is ratio / step. Relax delay is 5 * qubit.t1 us. Other missing required calibration leaves the affected fields empty and Invalid.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Readout uses the qubit.module.readout slot captured at Open/Reset, or a readout template. Reset uses qubit.module.reset and stays disabled when absent. Drive slots use qubit.module.x180/x90 with pi_pulse/pi2_pulse template fallbacks. Unset slots or slots pointing to missing modules use the named template. Saved modules remain selectable. Pulse templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

Initial T2/sample selection requires positive finite T2, r2 at least 0.9,
normalized residual RMS at most 0.1 and relative T2 error at most 0.5.
Failed or unavailable checks leave candidates unselected with explanatory notes.
You may select them manually after reviewing the source and fit.

Proposes fitted qubit.t2e (us) to the captured qubit and working point, including loaded data with IQ only. Frequency and module slots stay unchanged. Proposals keep the qubit and working point captured for this result even after focus changes. An independent t2e_us/t2e_err_us sample uses the captured Run bias when device evidence is available.

## Recommended

Enable Fit phase offset to fit a free fringe phase; it defaults to fixed phase and applies to fringe fits only. Set the 'Detune ratio (fringes/step)' input (default 0.1). This is the number of fringe periods per delay-sweep step; the absolute detune (MHz) applied to the final pi/2 pulse phase is detune_ratio / sweep step, leaving a residual oscillation on the echo decay. Analysis 'Fit method' defaults to 'decay'. Select 'fringe' explicitly only for nonzero source detune; choosing 'decay' remains valid. Zero source detune forces decay even when fringe is selected. Analysis uses the saved cfg detune, or finite saved actual detune without cfg. A fringe request without either source reports a missing-detune error; explicit decay still works. Set the ratio to 0 for a standard echo (pure exponential decay → T2E). A total-delay sweep reaching a few T2E lets the echo decay; the first sweep point is dropped in the fit.

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Obtain calibrated pi and pi/2 pulses. Confirm the total-delay window, detune_ratio and readout timing.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, Reset to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, Reset and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `t2echo` after inspecting the same Form errors.
   points maps to total-delay Expts; max_delay_us sets the stop. fit_method and fit_phase change only analysis.
   Common library/inline inputs select sources; overrides apply last. Null optional inputs retain Form values.
   Missing inputs return needs_parameters before acquisition. Reset may replace unsaved inputs.
