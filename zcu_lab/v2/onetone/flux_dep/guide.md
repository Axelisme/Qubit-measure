## Behavior

One-tone resonator flux dependence: a 2D sweep of an external flux-bias device value versus readout frequency, tracing how the resonator frequency moves with flux. The basis for locating a Fluxonium's flux sweet spots (half-flux and integer-flux points). Runs on real hardware; requires a SoC connection and a configured flux-bias device.

## Expects (parameter context)

Opening a tab or Reset evaluates Form starting expressions once against captured parameters into direct input. Range.around seeds 101 frequency points at `$resonator.freq` ± `$resonator.kappa`, in MHz. The inline readout reads `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch` for its tones and channels. `$resonator.ext.default_probe_length` - 0.1 us seeds the ADC window; `$resonator.ext.time_of_fly` + 0.05 us seeds trigger timing. Range.between seeds 101 flux points from 1.1 * `$qubit.flux_int` - 0.1 * `$qubit.flux_half` to 1.1 * `$qubit.flux_half` - 0.1 * `$qubit.flux_int`, in the selected device unit. Missing or invalid required sources leave the affected fields empty and Invalid, without a numeric sweep fallback. For first bring-up without calibration, enter safe, user-approved frequency and flux bounds. The picker separately reads captured `$qubit.flux_half`/`$qubit.flux_int`; absent/null leaves use its default line positions. Choose sweet spots from the measured map.

## Expects (ModuleLibrary)

Starts with a fresh inline pulse readout and const waveform. It does not adopt a library module or named waveform.

## Typical writeback

Interactive analysis (not a fit): after the run, the user drags two lines on the 2D map to mark the half-flux and integer-flux sweet spots, then clicks Done. The preview proposes qubit.flux_half, qubit.flux_int, and qubit.flux_period (= 2·|qubit.flux_int − qubit.flux_half|) through the captured role binding. The picking is user/agent judgement from the measured map, not simulator truth; apply writeback only after reviewing the figure.

## Recommended

Run after lookback and an initial onetone/freq, before any twotone flux mapping, to find `$qubit.flux_int` / period from the resonator map. Interactive flux-line pick (no fit). Review 'flux_dev': opening a tab or Reset resolves `$flux_dev` once from the alias table to a direct connected-device name, with no fallback. A missing/unbound alias leaves the required field empty and Invalid. Confirm the device and 'flux_unit', which starts as 'A' and does not convert values. The calibrated sweep uses 101 flux points across the padded interval between `$qubit.flux_int` and `$qubit.flux_half`, and 101 frequency points at `$resonator.freq` ± `$resonator.kappa` MHz. Start at a low readout gain (~0.005) to stay below punch-out so the dip tracks cleanly; if the dip is hard to see (poor SNR), raise the gain toward ~0.05 while keeping the frequency window tight. Survey wide, then narrow around a sweet spot — and if the resonator's flux dispersion (dispersive shift) looks small, narrow the frequency window early instead of keeping a broad default span. For consecutive flux sweeps, reverse the sweep direction (swap start/stop) on the next run so the current source need not ramp all the way back across the full range first. Inspect the 2D map with gui_tab_get_figure (subtab_id='run') to judge window / SNR / shift. After accepting `$qubit.flux_int`, move the flux device there and re-run onetone/freq at that flux before twotone/freq.
