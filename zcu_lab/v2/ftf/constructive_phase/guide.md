# Constructive relative phase

## Purpose

Scan F2 relative to F1 phase and common drive duration in the nominal B11 branch.
The sequence is Xπ on F1, Xπ on F2, simultaneous two-line drive, Xπ on F1,
Xπ on F2, then one full-tone mux readout. B11 is a thermal nominal preparation,
not a pure state. There is no heralding, shot rejection or matrix inversion.

## Before run

Bind scalar aliases `f1`, `f2`, `coupler` and `mux`. Open or Reset captures their
identities, the full ordered mux group and calibration sources. F1 and F2 tone
positions come from the group's qubits, not ADC numbers or tone labels.
Review both Xπ pulses, both GE centers, mux routes, all tones, repetitions and
rounds. Missing GE calibration or pulse inputs must be supplied before Run.

Use conditional spectroscopy to choose the transition and common frequency.
The drive pair starts from the coupler's ordinary `cz_f1` and `cz_f2` slots when
available. Without slots, fill the explicit cosine templates, including frequency,
signed gain, duration and Nyquist zone. Generator channels come from captured
role wiring, not hard-coded defaults. There is no coupler AC generator.

Enter relative-phase endpoints in degrees and positive duration endpoints in us.
Relative phase means F2.phase minus F1.phase; each scan point sets F2.phase to
F1.phase plus that coordinate. Keep the signed gains and their ratio fixed.
Both source pulse frequencies, envelope lengths and pre-delays must match.
Only common duration changes during each duration trace. The two generators start
together and the sequence waits for both, including their individual post-delays.

## Read the figures

The interference map and return curves show observed joint P00 after double-Xπ
contrast. They count same-shot outcomes, not a product of marginal populations.
They contain thermal population, preparation and readout errors. The GE partition
keeps every shot and does not identify leakage or physical gate fidelity.
Full-tone mean I/Q plots retain diagnostic signals beyond the selected pair.

Each complete duration trace is fitted to an undamped cosine. A rate below
1 divided by that trace's saved duration span is unresolved and excluded.
The remaining rates use the model
`rate² = A + B cos(relative phase) + C sin(relative phase)`.
The constructive relative phase is `atan2(C, B)` in degrees, the maximum-rate
branch. It is not a logical local-Z correction. More than three identifiable
phase slices and full-rank phase coverage are necessary for finite uncertainty.
Zero modulation, numerical failure or nonfinite uncertainty gives a reason and
no calibration proposal. Small nonzero contrast alone does not reject a fit.
Inspect the measured return curves against the model and its fit_quality.

Choose duration spacing so the expected Rabi rate stays below the Nyquist
limit `1/(2 × duration spacing)`. The adapter does not check this limit or
disambiguate aliased rates. It does not impose chi-square, contrast/sigma,
frequency/sigma or fidelity gates. Thermal mixtures can contain several rates;
a finite single-frequency fit does not establish a pure transition.

Incomplete duration points retain raw shots but are not classified. A missing
round is not a measured zero. Only complete duration traces can support fitting.
Run cancellation preserves completed round slots and all other positions remain
NaN. Transport errors and nonfinite acquired IQ are errors, not cancellation.

## Save, load and calibration

Native save preserves the source cfg, GE centers, full tone order, phase and
duration axes and round-major raw shots. Load does not fill the current Form.
Analysis must use the saved source cfg, never current aliases or library values.
A result without usable source cfg cannot provide phase calibration.

FIT produces proposals only. Inspect the destination, source waveform and phase
in preview. Select the F2 ordinary pulse and its independent slot row explicitly
before Apply. Partial selection is allowed; the rows are not a cross-file
transaction. Only F2 is proposed.
Its phase is saved F1.phase plus the fitted relative phase.
F2 frequency, signed gain, waveform and timing stay as measured.
F1 remains the saved phase reference. Review the complete pair before the next run.
A proposal that has not been applied is not calibration.

Proceed to the separate Rabi chevron experiment at the selected relative phase.
Do not search a parasitic line for cancellation or optimize the gain ratio here.
After amplitude and frequency selection, measure conditional phase and local Z.
Manual amplitude/frequency iteration can be repeated three times as a starting
workflow, not an automatic loop. Repeat calibration after changing pulse shapes,
bias settings or the pair's working point. Actual FTF control-chain validation is
separate from offline synthetic component validation.
