**Readers:** operator, agent. Map a qubit transition versus DC bias and decide whether the map supports flux calibration.

## What the scan measures

At each DC bias, Run steps the qubit probe RF frequency. One shot uses optional **Reset** → probe → resonator readout → **Relax delay**. A transition changes the mean resonator IQ; the resulting ridge is spectral contrast, not excited-state population.

## Key settings

- Bind the qubit to its resonator and use calibrated readout. Match **Flux Device** and **Flux unit** to the connected DC source; no unit conversion occurs. **DC bias** starts at the predicted half- and integer-flux positions plus 10% padding at each end, using the prior calibration and other bias values captured with the form. Fill blank endpoints if no prior calibration is available.
- **Probe RF frequency** endpoints start blank. Fill them in MHz around the expected line. Set **Probe Pulse › Gain** low enough to resolve nearby transitions without excessive broadening.
- For an inline pulse, **Gen ch** is the transmit generator (qubit for Probe Pulse, resonator for Readout); **RO ch** is the receiving ADC. **Waveform** sets the RF envelope (const is flat), **Length** its playback duration, **Gain** the DAC full-scale fraction, **Phase** the carrier phase. Readout **Frequency** is the RF target for transmit and demodulation, not the IF or probe sweep.
- **NQZ** selects the Nyquist zone; match it to the calibrated generator band. **Mixer freq** is the digital mixer setting, not the RF target or IF. Leave it blank only if that generator needs no mixer setting; blank does not disable an external LO.
- **Pre-delay** and readout **Trig offset** both start at the pulse segment start. They set RF playback and ADC opening respectively. **RO length** is integration duration. On both Probe Pulse and Readout, **Post-delay** is a wait after RF playback before advancing the sequence; it does not extend the ADC window. Terminal readout and flux picks use averaged IQ, not **Classifier** assignments.
- **Reps** is per RF point per round; **Rounds** is per bias. See [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md) for the shared GUI steps.

## Read the figures

- **Run:** x is DC bias; y is probe RF frequency in MHz. Each spectrum shows the magnitude of mean IQ after subtracting that spectrum's median complex IQ (a.u.). Blank cells lack data. The latest and previous bias traces help distinguish a ridge from an isolated bright cell.
- **Analyze picker:** dark features are stronger smoothed contrast on a GHz axis, with per-spectrum normalization in the phase-aware view. Red is the half-flux pick; blue is integer flux. Lower mirror loss means greater local symmetry, not correct physical assignment. Refine a nearby turning point with **Auto Align**; **Conjugate Line** inspects the other line, **Swap Lines** exchanges labels, and **Done** records positions.
- **Saved flux-line picks:** same MHz axis and contrast as Run. The lines mark selections, not extra spectra or a fit. The two-dimensional map locates a ridge and has no pointwise shot SEM. The Analyze summary identifies the run and its configured **Reps × Rounds**.

## Next step

Use only two adjacent sweet spots on the same continuous ridge; symmetry alone does not label half versus integer. If one is missing, extend **DC bias** toward its position in the prior validated flux frame. Widen **Probe RF frequency** if the ridge meets an RF edge; for isolated bright cells, verify readout and probe at one bias before adding averaging or sampling more finely.

The period magnitude is `2 × |x_integer − x_half|`. Writeback keeps the prior period sign (positive if unset). With signed period `P_s`, the offset is `0.5 − x_half/P_s − Σ x_o/P_o`, where the sum is over the other calibrated DC lines in the Run-start snapshot. Review those contributions; writeback offers the period, offset and record separately. Older maps/picks without usable captured cfg cannot establish the historical bias unit or propose calibration.

For `twotone_flux_dep`, `readout_ref` takes a library key or the exact **Readout** option label. Null fails during selection, before overrides. Null **Probe Pulse › Gain**, **Reps** or **Rounds** leaves a blank rather than keeping the form value. Required blanks remaining after overrides fail readiness before Run. A cached unit mismatch or equal flux endpoints fails setup before Run.
