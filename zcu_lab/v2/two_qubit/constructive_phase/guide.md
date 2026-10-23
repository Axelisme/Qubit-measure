# Constructive relative phase

**Readers:** operator, agent. Choose the two charge drives' relative phase from thermal-ensemble interference measured through the GUI or MCP.

**Last updated:** 2026-10-21

Readout references `$mux.module.readout` when present, preserving the library's complete tones and scalar settings. Without a slot, fill every required readout field in the blank mux template.

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Purpose

Scan F2 relative to F1 phase and common drive duration in the nominal B11 branch.
The sequence is Xπ on F1, Xπ on F2, simultaneous two-line drive, Xπ on F1,
Xπ on F2, then one full-tone mux readout. B11 is a nominal thermal preparation.
Every shot and its original readout effects remain in the analysis.

## Before run

Bind scalar aliases `f1`, `f2`, `coupler` and `mux`. Open or Reset captures their
identities, the full ordered mux group and calibration sources. F1 and F2 tone
positions come from the group's `qubits`. Use those positions to identify F1/F2.
Review both Xπ pulses, both GE centers, mux routes, all tones, repetitions and
rounds. Missing GE calibration or pulse inputs must be supplied before Run.

Use conditional spectroscopy to choose the transition and common frequency.
The drive pair starts from the coupler's ordinary `cz_f1` and `cz_f2` slots when
available. Without slots, fill the explicit cosine templates, including frequency,
signed gain, duration and Nyquist zone. Generator channels come from captured
role wiring. Only the two charge lines are driven; the coupler stays DC-biased.

Enter relative-phase endpoints in degrees and positive duration endpoints in us.
Relative phase means F2.phase minus F1.phase; each scan point sets F2.phase to
F1.phase plus that coordinate. Keep the signed gains and their ratio fixed.
Both source pulse frequencies, envelope lengths and pre-delays must match.
Only common duration changes during each duration trace. The two generators start
together and the sequence waits for both, including their individual post-delays.

## Read the figures

The interference map and return curves show observed joint P00 after double-Xπ
contrast. They count the joint outcomes of each same-shot tone pair directly.
They contain thermal population, preparation and readout errors. The GE partition
keeps every shot. Leakage and physical gate fidelity need separate measurements.
Full-tone mean I/Q plots retain diagnostic signals beyond the selected pair.

Each complete duration trace is fitted to an undamped cosine. A rate below
1 divided by that trace's saved duration span is unresolved and excluded.
The remaining rates use the model
`rate² = A + B cos(relative phase) + C sin(relative phase)`.
The constructive relative phase is `atan2(C, B)` in degrees, the maximum-rate
branch. Logical local-Z correction is measured separately. More than three
identifiable phase slices and full-rank phase coverage are necessary for finite uncertainty.
Zero modulation, numerical failure or nonfinite uncertainty stops proposal
generation and reports the reason. Small nonzero contrast remains eligible for fitting.
Inspect the measured return curves against the model and its fit_quality.

Choose duration spacing so the expected Rabi rate stays below the Nyquist
limit `1/(2 × duration spacing)`. Check this limit and possible aliasing manually.
Statistical ratios and fidelity also require operator interpretation. Thermal
mixtures can contain several rates, so compare the single-frequency model with
the measured traces even when the fit is finite.

Incomplete duration points retain raw shots and appear as missing in classification.
Only complete duration traces can support fitting.
Run cancellation preserves completed round slots and all other positions remain
NaN. Transport errors and nonfinite acquired IQ report execution failures.

## Save, load and calibration

Native save preserves the source cfg, GE centers, full tone order, phase and
duration axes and round-major raw shots. Load leaves the current Form unchanged.
Analysis uses only the saved source cfg; unusable source cfg stops phase calibration.

FIT produces proposals only. Inspect the destination, source waveform and phase
in preview. Select the F2 ordinary pulse and its independent slot row explicitly
before Apply. These rows apply independently; review any partial selection.
Only F2 is proposed.
Its phase is saved F1.phase plus the fitted relative phase.
F2 frequency, signed gain, waveform and timing stay as measured.
F1 remains the saved phase reference. Review the complete pair before the next run.
Only an applied proposal is available as calibration for subsequent forms.

Proceed to the separate Rabi chevron experiment at the selected relative phase.
Keep the signed gain ratio fixed here. Parasitic-line cancellation requires a
separate experiment.
After amplitude and frequency selection, measure conditional phase and local Z.
Manual amplitude/frequency iteration can be repeated three times as a starting
workflow performed by the operator. Repeat calibration after changing pulse shapes,
bias settings or the pair's working point. Validate the actual FTF control chain
separately from offline synthetic data.
