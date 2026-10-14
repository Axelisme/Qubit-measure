## Behavior

One-tone resonator flux dependence: a 2D sweep of an external flux-bias device value versus readout frequency, tracing how the resonator frequency moves with flux. The basis for locating a Fluxonium's flux sweet spots (half-flux and integer-flux points). Runs on real hardware; requires a SoC connection and a configured flux-bias device.

## Expects (parameter context)

Reads fixed tab parameters: resonator.freq and resonator.kappa seed the frequency sweep; resonator.ext.default_probe_length minus 0.1 us seeds the ADC window; resonator.wiring.ch/ro_ch seed channels; resonator.ext.time_of_fly plus 0.05 us seeds trigger timing. Captured qubit.flux_half/flux_int seed the flux sweep and picker when calibrated. First bring-up must not assume these are known: absent/null flux values use a [-4e-3, 4e-3] device sweep that the operator must adjust to a safe, user-approved range.

## Expects (ModuleLibrary)

Starts with a fresh inline pulse readout and const waveform. It does not adopt a library module or named waveform.

## Typical writeback

Interactive analysis (not a fit): after the run, the user drags two lines on the 2D map to mark the half-flux and integer-flux sweet spots, then clicks Done. The preview proposes qubit.flux_half, qubit.flux_int, and qubit.flux_period (= 2·|flx_int − flx_half|) through the captured role binding. The picking is user/agent judgement from the measured map, not simulator truth; apply writeback only after reviewing the figure.

## Recommended

Run after lookback and an initial onetone/freq, before any twotone flux mapping, to find 'flx_int' / period from the resonator map. Interactive flux-line pick (no fit). Also set the 'flux_dev' field — the flux-bias device name ($flux_dev resolves from the alias table with no fallback) — and confirm it points at a connected device. Typical sweep: flux ~101 points across one period (driven by flx_half/flx_int when calibrated), frequency r_f ± one linewidth over ~101 points. Start at a low readout gain (~0.005) to stay below punch-out so the dip tracks cleanly; if the dip is hard to see (poor SNR), raise the gain toward ~0.05 while keeping the frequency window tight. Survey wide, then narrow around a sweet spot — and if the resonator's flux dispersion (dispersive shift) looks small, narrow the frequency window early instead of keeping a broad default span. For consecutive flux sweeps, reverse the sweep direction (swap start/stop) on the next run so the current source need not ramp all the way back across the full range first. Inspect the 2D map with gui_tab_get_figure (subtab_id='run') to judge window / SNR / shift. After accepting 'flx_int', move the flux device there and re-run onetone/freq at that flux before twotone/freq.
