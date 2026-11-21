## Behavior

**Readers:** operator, agent. Zig-zag: applies an X90 pulse, then repeats the chosen pulse 0..n_times times before readout, amplifying small pulse-amplitude errors into a growing zig-zag of the signal versus repetition count. Runs on real hardware. Run after amplitude Rabi has calibrated the pi and pi/2 pulses.

**Last updated:** 2026-11-22

## Parameter sources

Maximum repetitions defaults to 10. Relax delay is 5 * qubit.t1 µs. Other missing required calibration leaves the affected fields empty and Invalid.

Open and Defaults capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 µs, giving a 150 µs wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Readout uses the qubit.module.readout slot captured at Open/Defaults, or a readout template. Reset uses qubit.module.reset and stays disabled when absent. Drive slots use qubit.module.x180/x90 with pi_pulse/pi2_pulse template fallbacks. Unset slots or slots pointing to missing modules use the named template. Saved modules remain selectable. Pulse templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.time_of_fly + 0.015 µs; missing required calibration remains Invalid.

## Typical writeback

Decide from the trace whether to adjust the pulse. Analyze does not fit the trace, estimate a correction, or propose a writeback. Preparation coherence, contrast, and drive history can also change the trace; compare controls before changing gate calibration.

## Recommended

Repeat on X180_pulse to check the pi pulse; repeat on X90_pulse (applied in pairs) to check the pi/2 pulse. About 10 repetitions usually show a mis-calibration clearly. Use the zig-zag gain or frequency scans to sweep the repeated pulse's gain or frequency.

## Reading the plots

After Run or a successful Load, click **Analyze** to redraw that trace. Load alone does not open the plot. The summary lists the source run's Reps × Rounds when its settings are available.

The x axis is repetition count: one count adds one X180 or a pair of X90 pulses. The title names the pulse when the source settings are available. The y axis is mean IQ projected along the trace's principal axis (a.u.), not a population. Each bar shows one shot SEM along the same axis: the sample standard deviation divided by the square root of that point's retained shot count. When shot statistics are available, the title reports the largest retained shot count across all points. Mean-only files show the trace without bars. Unmeasured counts appear as gaps.

Look for alternating contrast that grows with repetition count. If you need to locate a tuning minimum, use the zig-zag gain or frequency scan.
