## Behavior

T2 echo (spin/Hahn echo): a pi/2 – delay – pi – delay – pi/2 sequence with the total free-evolution delay swept; the refocusing pi pulse cancels low-frequency dephasing, and the decay fit yields the echo coherence time T2E. Runs on real hardware. Run after both pi and pi/2 pulses are calibrated; T2E is typically longer than the Ramsey T2*.

## Expects (parameter context)

Defaults use the tab's fixed qubit and referenced resonator. Total delay ends at 1.5 * qubit.t2e (us, fallback 20 us); relax_delay is 5 * qubit.t1 (fallback 100 us). Inline pulses read qubit.freq and qubit.wiring.L01.ch. Inline readout reads resonator.freq, resonator.wiring.ch/ro_ch, and resonator.ext.time_of_fly + 0.05 us (fallback 0.55 us). Missing or null calibration uses fallback; unbound roles remain Invalid until supplied.

## Expects (ModuleLibrary)

Uses captured qubit.module.x180 for the pi pulse, qubit.module.x90 for pi/2, and qubit.module.readout. Missing or dangling slots seed fresh inline pulse/readout defaults. Optional qubit.module.reset stays disabled when absent. Verify the inline pulse values and readout timing before Run.

## Typical writeback

Proposes fitted qubit.t2e (us) to the captured qubit and working point, including analysis of a source without cfg. No frequency or module writeback; later focus changes do not retarget it. An independent t2e_us/t2e_err_us sample uses the captured Run bias when device evidence is available.

## Recommended

Enable Fit phase offset to fit a free fringe phase; it defaults to fixed phase and has no effect for decay-only analysis. Set the 'Detune ratio (fringes/step)' cfg knob (default 0.1) — it is the number of fringe periods per delay-sweep step; the absolute detune (MHz) applied to the final pi/2 pulse phase is detune_ratio / sweep step, leaving a residual oscillation on the echo decay. Analysis 'Fit method' defaults to 'fringe' (matching the nonzero default ratio) so the fringe frequency is fit; the detune is computed but NOT written back. Set the ratio to 0 and switch 'Fit method' to 'decay' for a standard echo (pure exponential decay → T2E). A total-delay sweep reaching a few T2E lets the echo decay; the first sweep point is dropped in the fit.
