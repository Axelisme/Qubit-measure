## Behavior

T2 Ramsey: two pi/2 pulses separated by a swept free-evolution delay; fits the decaying fringe to extract the Ramsey coherence time T2* and the qubit-drive detuning. Runs on real hardware. Run after a pi/2 pulse is calibrated; a small deliberate detuning produces the fringe whose frequency reads out the true qubit frequency offset.

## Expects (parameter context)

Delay runs 0 to 1.5 * qubit.t2r us over 101 points. Detune ratio is fringes per sweep step; native MHz detune is ratio / step. Relax delay is 5 * qubit.t1 us. Other missing required calibration leaves the affected fields empty and Invalid.

T1-derived defaults use the fixed Open/Reset capture. A missing/null T1 or absent capture uses 30 us, so the 5*T1 wait starts at 150 us. Stored invalid T1 leaves derived required fields empty while the Form opens. Alias/path/source failures propagate. Defaults remain editable, are not written back as calibration, and do not shorten automatically for active reset.

## Expects (ModuleLibrary)

Uses captured qubit.module.readout with a readout template fallback, and qubit.module.reset with no fallback (disabled when absent). The drive uses qubit.module.x90 with a pi2_pulse template fallback. Unset or dangling slots use the named template, not a legacy blank. Key choices remain available. Inline templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

Proposes fitted qubit.t2r (us) to the captured qubit and working point. A fringe fit with source cfg and known realized detune also proposes qubit.freq (MHz): source pi/2 frequency + realized detune - fitted fringe detune. A pure decay fit or a loaded source without realized detune proposes only qubit.t2r. No module writeback; later focus changes do not retarget it. An independent t2r_us/t2r_err_us sample uses the captured Run bias when device evidence is available.

## Recommended

Enable Fit phase offset to fit a free fringe phase; it defaults to fixed phase and has no effect for decay-only analysis. Set the 'Detune ratio (fringes/step)' cfg knob to a small deliberate offset (default 0.05) — it is the number of fringe periods per delay-sweep step; the absolute detune (MHz) applied to the second pi/2 pulse phase is detune_ratio / sweep step, so the Ramsey signal oscillates at that detuning. The analysis 'Fit fringe' option (on by default, matching the nonzero default ratio) fits the fringe frequency to read back the true qubit-frequency offset and write qubit.freq. Set the ratio to 0 and turn 'Fit fringe' off for a pure decay fit when driving on resonance. A delay sweep reaching a few T2* captures enough fringe periods and decay.
