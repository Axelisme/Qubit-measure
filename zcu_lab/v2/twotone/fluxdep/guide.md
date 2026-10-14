## Behavior

Two-tone qubit flux dependence: a 2D scan stepping an external flux device (outer) against the qubit-drive frequency (inner), reading out the resonator at each point, to map the qubit spectrum versus flux (the Fluxonium flux-dispersion arcs). Runs on real hardware and needs a connected flux device. No automated fit in the GUI — points are extracted interactively. This is not the early flux-finder; use it only after readout and qubit spectroscopy are already credible.

## Expects (parameter context)

Reads from the parameter context (all optional): 'q_f' — qubit frequency, centring the inner sweep (~2000–6000 MHz); 'qf_w' — qubit linewidth, half-span ±qf_w (~1–50 MHz); 'qub_ch' — qubit-drive channel; 'r_f' — resonator frequency for the readout tone (~4000–8000 MHz); 'res_ch' / 'ro_ch' — readout drive / ADC channels; 'timeFly' — readout trigger offset (~0–1 us); 'flx_half' / 'flx_int' — calibrated half-flux / integer-flux device values that set the flux bounds (device-specific units; absent → fixed [-4e-3, 4e-3]).

## Expects (ModuleLibrary)

Needs a qubit-probe pulse module and a pulse-readout module; references a ModuleLibrary waveform named 'ro_waveform' when present. Optionally references a calibrated reset module. Also needs a flux device name 'flux_dev' ($flux_dev resolves from the alias table with no fallback) — a device, not a library entry.

## Typical writeback

Interactive analysis (not a fit): after the run, the user drags two lines on the 2D map to mark the half-flux and integer-flux sweet spots, then clicks Done. The result writes back 'flx_half', 'flx_int' and 'flx_period' (= 2·|flx_int − flx_half|) to the parameter context as a preview. Picking is user/agent judgement from the measured map; do not use this preview to replace the earlier onetone/flux_dep flux calibration unless the map is clearly trustworthy.

## Recommended

No fit options. Use after onetone/flux_dep has established the flux period and twotone/freq has found 'q_f'. Do not run this early to find flux: with uncalibrated readout or an unknown qubit-drive window the arc is usually invisible. Default sweep: flux ~101 points across roughly one period (bracketed by 'flx_int'/'flx_half' when calibrated), frequency ~101 points spanning ±qf_w around 'q_f'. For a later model survey use a wide flux range and a broad frequency window at moderate drive gain to capture the full arc, then narrow both. For consecutive flux sweeps, reverse the sweep direction (swap start/stop) on the next run so the flux source need not ramp back across the full range first. Inspect the 2D map with gui_tab_get_figure (subtab_id='run'). Confirm 'flux_dev' points at a connected device before running.
