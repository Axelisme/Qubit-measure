## Behavior

T2 Ramsey: two pi/2 pulses separated by a swept free-evolution delay; fits the decaying fringe to extract the Ramsey coherence time T2* and the qubit-drive detuning. Runs on real hardware. Run after a pi/2 pulse is calibrated; a small deliberate detuning produces the fringe whose frequency reads out the true qubit frequency offset.

## Expects (parameter context)

Defaults use the tab's fixed qubit and referenced resonator. Delay ends at 1.5 * qubit.t2r (us, fallback 0.4 us); relax_delay is 5 * qubit.t1 (fallback 100 us). Inline pi/2 reads qubit.freq and qubit.wiring.L01.ch. Inline readout reads resonator.freq, resonator.wiring.ch/ro_ch, and resonator.ext.time_of_fly + 0.05 us (fallback 0.55 us). Missing or null calibration uses fallback; unbound roles remain Invalid until supplied.

## Expects (ModuleLibrary)

Uses captured qubit.module.x90 and qubit.module.readout. Missing or dangling slots seed fresh inline pulse/readout defaults. Optional qubit.module.reset stays disabled when absent. Verify the inline pulse values and readout timing before Run.

## Typical writeback

Proposes fitted qubit.t2r (us) to the captured qubit and working point. A fringe fit with source cfg and known realized detune also proposes qubit.freq (MHz): source pi/2 frequency + realized detune - fitted fringe detune. A pure decay fit or a loaded source without realized detune proposes only qubit.t2r. No module writeback; later focus changes do not retarget it. An independent t2r_us/t2r_err_us sample uses the captured Run bias when device evidence is available.

## Recommended

Enable Fit phase offset to fit a free fringe phase; it defaults to fixed phase and has no effect for decay-only analysis. Set the 'Detune ratio (fringes/step)' cfg knob to a small deliberate offset (default 0.05) — it is the number of fringe periods per delay-sweep step; the absolute detune (MHz) applied to the second pi/2 pulse phase is detune_ratio / sweep step, so the Ramsey signal oscillates at that detuning. The analysis 'Fit fringe' option (on by default, matching the nonzero default ratio) fits the fringe frequency to read back the true qubit-frequency offset and write qubit.freq. Set the ratio to 0 and turn 'Fit fringe' off for a pure decay fit when driving on resonance. A delay sweep reaching a few T2* captures enough fringe periods and decay.
