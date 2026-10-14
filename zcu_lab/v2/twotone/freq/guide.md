## Behavior

Two-tone qubit spectroscopy: drives a qubit probe tone while reading out the resonator, sweeps the drive frequency, and fits the qubit response (Lorentzian or sinc) to extract the qubit transition frequency and its linewidth. Runs on real hardware. In first bring-up, run it after onetone/flux_dep has found 'flx_int' and onetone/freq has re-calibrated the resonator at that flux.

## Expects (parameter context)

Reads from the parameter context (all optional): 'q_f' — qubit frequency, the sweep centre (~2000–6000 MHz); 'qf_w' — qubit linewidth, setting the half-span as 1.5*qf_w (~1–50 MHz); 'qub_ch' — qubit-drive channel; 'r_f' — resonator frequency for the readout tone (~4000–8000 MHz); 'res_ch' / 'ro_ch' — readout drive / ADC channels; 'timeFly' — readout trigger-offset cable delay (~0–1 us). Absent 'q_f'/'qf_w' → a broad survey fallback around 5000 MHz with ~GHz-scale span from the shared qubit-frequency helper; for a first qubit search, still review the hardware-safe passband and override the sweep if the line is expected elsewhere.

## Expects (ModuleLibrary)

Needs a qubit-probe pulse module and a pulse-readout module; references a ModuleLibrary waveform named 'ro_waveform' for the readout shape when present. Optionally references a calibrated reset module (disabled when none exists).

## Typical writeback

Proposes fitted qubit.freq and qubit.kappa (FWHM) in MHz. An independent freq_mhz/freq_err_mhz sample uses the captured Run bias when device evidence is available. No module writeback. Apply only after the agent/user verifies that the plotted feature is the intended qubit line.

## Recommended

Standard flow at a fresh 'flx_int': first run a wide survey over the user-approved passband to find a real qubit feature, without using predictor output or simulator truth as the answer. Then run a narrower scan around the observed feature and fit it. Analysis defaults to the Lorentzian fit ('lor') with plot-fit on; switch to 'sinc' for power-broadened or saturated lines. If the scan looks like noise, widen/re-centre and check readout/gain before concluding there is no qubit.
