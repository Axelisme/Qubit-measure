## Behavior

**Readers:** operator, agent. T2 Ramsey: two pi/2 pulses separated by a swept free-evolution delay; fits the decaying fringe to extract the Ramsey coherence time T2* and the qubit-drive detuning. Runs on real hardware. Run after a pi/2 pulse is calibrated; a small deliberate detuning produces the fringe whose frequency reads out the true qubit frequency offset.

**Last updated:** 2026-10-21

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Parameter sources

Delay runs 0 to 1.5 * qubit.t2r us over 101 points. Detune ratio is fringes per sweep step; native MHz detune is ratio / step. Relax delay is 5 * qubit.t1 us. Other missing required calibration leaves the affected fields empty and Invalid.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Readout uses the qubit.module.readout slot captured at Open/Reset, or a readout template. Reset uses qubit.module.reset and stays disabled when absent. The drive uses qubit.module.x90 with a pi2_pulse template fallback. Unset slots or slots pointing to missing modules use the named template. Saved modules remain selectable. Pulse templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

Proposes fitted qubit.t2r (us) to the captured qubit and working point. A fringe fit with acquired run settings and known realized detune also proposes qubit.freq (MHz): source pi/2 frequency + realized detune - fitted fringe detune. A pure decay fit, or loaded data with unknown acquisition detune, proposes only qubit.t2r. Module slots stay unchanged. Proposals keep the qubit and working point captured for this result even after focus changes. An independent t2r_us/t2r_err_us sample uses the captured Run bias when device evidence is available.

A frequency correction updates `qubit.freq`, not the existing pi/2 library
pulse. Recalibrate the pulse at the new frequency, for example with Rabi or
zigzag, and check its gain and length before saving and selecting a new module.
Changing only its frequency does not produce a calibrated pulse.
Calibration writeback, New from a template and `ml_edit` can create numeric
library modules. Template expressions read calibration when the module is
created; saved modules do not follow later frequency changes.
Reset downstream tabs after selecting the new module to capture it.
Historical Run/Load sources keep their original pulses.

## Recommended

Fit phase offset is enabled by default to fit a free fringe phase. Disable it to fix the phase at 0/180 degrees using the near-zero-delay signal. This option applies to fringe fits only. Set the 'Detune ratio (fringes/step)' input to a small deliberate offset (default 0.05). This is the number of fringe periods per delay-sweep step; the absolute detune (MHz) applied to the second pi/2 pulse phase is detune_ratio / sweep step, so the Ramsey signal oscillates at that detuning. The analysis 'Fit fringe' option (on by default, matching the nonzero default ratio) fits the fringe frequency to read back the true qubit-frequency offset and write qubit.freq. Set the ratio to 0 and turn 'Fit fringe' off for a pure decay fit when driving on resonance. A delay sweep reaching a few T2* captures enough fringe periods and decay.

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Obtain a calibrated pi/2 pulse. Confirm the delay window, detune_ratio and readout timing.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, Reset to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, Reset and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `t2ramsey` after inspecting the same Form errors.
   points maps to delay Expts; max_delay_us sets the stop. fit_fringe and fit_phase change only analysis.
   Common library/inline inputs select sources; overrides apply last. Null optional inputs retain Form values.
   Missing inputs return needs_parameters before acquisition. Reset may replace unsaved inputs.
