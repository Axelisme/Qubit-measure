**Readers:** operator, agent. Measure how the coherence time changes with the number of refocusing pulses.

## Sequence and key settings

Optional reset → calibrated π/2 → train of N calibrated π refocusing pulses → phase-ramped π/2 → calibrated readout. Scan total free-evolution time at each count. Finite pulse durations matter when setting the time window.

- **Pi pulse counts** accepts **Log** or **List**. Log starts at 1, includes the maximum, rounds to integers and removes duplicates. Use two or more requested points if the maximum exceeds 1. Use List for any single count other than 1 or for an explicit sequence of positive integers, e.g. `[1, 2, 4, 8]`.
- **Minimum time** initially equals `16 × (π/2 duration + π duration)` for the initial maximum count of 16. It stays unchanged when you change counts. It starts blank if a pulse is missing or its duration comes from an arbitrary asset. Fill it and check spacing at the largest count before Run. If pulses do not fit, increase the minimum. **Maximum time** and **Time points** complete the window shared by all counts.
- **Detune ratio (fringes/step)** sets the final-pulse phase ramp; requested frequency is ratio ÷ delay step in µs. Use **Fit fringes** with a phase ramp; use a zero ratio and disable it for decay only. For fringe fits, take at least 10 points per period and span about three expected lifetimes to see the envelope fall. Short windows leave a weakly constrained decay; sparse sampling can hide or alias fringes.
- **Relax delay** starts at `5 × qubit.t1` (150 µs if absent). **Early-stop SNR** can complete averaging at each count separately.

See [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md) for the shared GUI steps.

## Read the plots

Both maps show normalized IQ contrast (not population) versus total delay (µs) and pulse count in acquisition order. Columns are equally spaced even for uneven counts. Repeated counts carry an acquisition index, e.g. `4 (#3)`. Blank cells lack data. Live compares the latest and previous counts on their own delay grids.

The Analyze right panel shows T₂ against numerical pulse count on log-log axes. The lifetime points connect only when there is no T₂(N) fit. Bars are lifetime fit errors, not shot SEM. Optional Ramsey T₂ and 2 × T₁ lines are references, not fit inputs. An incomplete row has no lifetime. A row must have resolved decay within its time window, fit error below half the lifetime, and contrast above four times the mean absolute smoothed residual. If no row passes or the fitter raises, Analyze fails.

## Next step

T₂(N) = T₂₁ × N^γ describes how refocusing changes the lifetime. The guide curve and `gamma` summary require at least three accepted rows with positive finite T₂ and fit error; zero-error points can plot but do not contribute. With fewer than three, the four fit values are null and `fit_rows` still counts usable rows. Widen the delay window for unresolved decay, or increase averaging when noise obscures it. Analyze does not write back.

For `cpmg`, null disables reset; null pulse or readout selectors fail during selection, before overrides. Null time, count, detune or acquisition inputs leave blanks. Supply only `times`, or both `max_n` and `points`; mixed or incomplete specifications fail before readiness. Null for the selected count input clears the count. Unresolved required fields after overrides fail readiness before Run. Null for **Fit fringes** fails at Analyze; null for either reference omits that line.
