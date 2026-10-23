**Readers:** operator, agent. How to measure readout time of flight from the GUI or MCP and use the result to calibrate the trigger offset.

**Last updated:** 2026-10-23

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Behavior

Lookback fires the readout pulse and captures the decimated ADC trace versus time. The trace shows when the signal arrives at the digitizer. Acquisition uses real hardware with `reps` fixed at 1. Averaging uses software rounds, initially 10. Run this first when bringing up a new readout chain or after changing cabling.

## Parameter sources

Defaults use the tab's fixed resonator selection, captured when the form opens or resets. The `$resonator` alias identifies that resonator.

The recipe retains the Form's initial frequency unless `frequency_mhz` is supplied. `check_ready` reports missing or invalid Form fields before Run. Invalid selected sources are not replaced with calibration fallbacks.

- Pulse and ADC frequency start at `resonator.freq + resonator.kappa` in MHz. Missing or null `kappa` uses 1 MHz. Both stored values must be positive, finite real numbers, with booleans excluded. Missing or null frequency, invalid frequency or linewidth, or an absent resonator capture leaves frequency required and blank.
- `resonator.wiring.ch` and `resonator.wiring.ro_ch` supply the generator and ADC channels through the readout slot.
- Missing or null `resonator.time_of_fly` starts the trigger offset at 0 us. A valid time of flight is a finite real number >= 0, with booleans excluded. The trigger offset starts at `time_of_fly - 0.1` us, including negative results. Invalid stored time of flight leaves the trigger offset required and blank.
- The ADC window starts at 1.5 us. Extend it when the arrival time plus pulse duration exceeds this window.

Alias, path, or parameter-source lookup failures appear as errors. Other missing required calibration leaves the form Invalid. Initial values remain editable; calibration changes require writeback.

## Module slots

The readout slot uses the captured resonator as its parameter source. Its initial gain is 1.0, with frequency, ADC window, and trigger offset set as described above. The form exposes only readout; Reset and Init pulse are hidden.

## Typical writeback

Analysis proposes the detected arrival time in us for `$resonator.time_of_fly`. The destination is the resonator captured for that analysis, even if the current selection changes. Writeback updates only this parameter; module and waveform definitions stay unchanged.

## Recommended

Analysis starts with `ratio=0.1`, `smooth=1.0`, and Plot fit enabled. The offset is the latest time before the peak where the magnitude remains below 10% of the maximum. If every pre-peak sample is at or above the threshold, analysis returns the first sample time. Check the trace before accepting this result as the arrival time.

Use a wide readout window to capture the full pulse arrival. Increase `smooth`, the Gaussian width in samples, if noise makes the offset jitter. Lower `ratio` if baseline ripple triggers the estimate too early.

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Check the ADC window and trigger offset. Reset and Init pulse stay hidden.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, Reset to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, Reset and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `lookback` after inspecting the same Form errors.
   No scan count is exposed; use frequency_mhz when no frequency source exists. Analysis controls are ratio, smooth and plot_fit.
   The Form chooses module sources. Omitted inputs retain Form values; null clears/disables fields. Overrides apply last, then check_ready reports missing or invalid fields before Run.
   Missing inputs return needs_parameters before acquisition. Reset may replace unsaved inputs.
