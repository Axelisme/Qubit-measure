## Behavior

**Readers:** operator, agent. AllXY: runs the 21 standard pairs of I, X90, Y90, X180 and Y180 gates before readout. A calibrated qubit gives a staircase of ground, equator and excited levels; amplitude and detuning errors bend it in distinct patterns. Y gates are the X pulses with phase +90 deg. Runs on real hardware. Run after amplitude Rabi and Ramsey have calibrated the pi and pi/2 pulses and the frequency.

**Last updated:** 2026-11-13

## Parameter sources

AllXY uses the captured gate slots. Relax delay is 5 * qubit.t1 µs. Other missing required calibration leaves the affected fields empty and Invalid.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 µs, giving a 150 µs wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Readout uses the qubit.module.readout slot captured at Open/Reset, or a readout template. Reset uses qubit.module.reset and stays disabled when absent. Drive slots use qubit.module.x180/x90 with pi_pulse/pi2_pulse template fallbacks. Unset slots or slots pointing to missing modules use the named template. Saved modules remain selectable. Pulse templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.time_of_fly + 0.015 µs; missing required calibration remains Invalid.

## Typical writeback

Analysis reports results for review; calibration stays unchanged. The summary reports 'amplitude_error', the relative drive amplitude error (+0.02 means 2% too much rotation), and 'detune_param', the detuning parameter (a.u.) of the fit model. 'power_err' and 'detune_err' are the mean Bloch-z deviation (ground +1, excited -1) each error causes over the 21 pairs. Use a benchmarking experiment to estimate gate infidelity. 'residual_rms' is the fit residual in the same units; a residual comparable to those deviations means the model fits poorly, for example because T1/T2 decay matters. Fix power errors with amplitude Rabi or zig-zag and detuning errors with Ramsey.

## Recommended

Analysis always fits the ground and excited levels together with the rotation and detuning errors. The measured range sets their starting values, but the fitted levels can lie outside it. Center is bounded to [min - ptp, max + ptp], where ptp is the measured signal range. Contrast keeps its initial sign and its magnitude is bounded to [0.5 * ptp, 3 * ptp]. The model does not include T1/T2 decay; check the residual before using its error estimates. Use many reps and rounds: each point is a single gate pair.
