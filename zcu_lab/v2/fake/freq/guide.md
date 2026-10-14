## Behavior

Simulated one-tone resonator frequency sweep — a HangerModel lineshape plus Gaussian noise, computed in software with no hardware or SoC. Mirrors the real onetone/freq run/analyze/writeback flow so you can rehearse the analysis offline.

## Expects (parameter context)

Reads from the parameter context (all optional): 'r_f' — resonator frequency, the sweep centre (~4000–8000 MHz); 'rf_w' — linewidth, used to set the sweep span and a loaded-Q guess (~0.1–5 MHz); 'res_ch' / 'ro_ch' — drive / readout channel indices; 'timeFly' — cable time-of-flight feeding the trigger offset (~0–1 us).

## Expects (ModuleLibrary)

Needs a readout module to shape the probe pulse, and references a ModuleLibrary waveform named 'ro_waveform' when one exists (optional).

## Typical writeback

Proposes the fitted resonator frequency and linewidth back into parameter context 'r_f' / 'rf_w'. The readout module / waveform are left to the user — a frequency fit alone does not justify rewriting the whole readout config.

## Recommended

Analysis defaults to the hanger-model fit ('hm'). Switch to the transmission model ('t') and enable amplitude-background fitting when the signal-to-noise is poor or the magnitude baseline is visibly tilted. Use this adapter to validate an analysis pipeline before taking it to real hardware.
