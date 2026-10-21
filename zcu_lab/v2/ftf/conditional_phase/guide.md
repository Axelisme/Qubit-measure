# Conditional phase

**Readers:** operator, agent. Measure conditional ensemble phase through the GUI or MCP and review a common CZ frequency proposal.

**Last updated:** 2026-10-21

Readout references `$mux.module.readout` when present, preserving the library's complete tones and scalar settings. Without a slot, fill every required readout field in the blank mux template.

`ftf/conditional_phase` acquires the target's ensemble phase for two nominal spectator branches. Choose F1 or F2 as target. Each shot applies spectator I or Xπ, target X90, n repeated simultaneous CZ pulse pairs, one projection pulse, then the complete mux readout. X and Y projections are independent acquisitions. Spectator Xπ flips a thermal ensemble, with thermal admixture still present.

## Prepare the inputs

Bind `f1` and `f2` to distinct fluxoniums, `coupler` to their transmon, and `mux` to a MuxGroup containing both qubits. Calibrate both Xπ and X90 pulses and the complete mux readout first. Open or Reset captures these identities, module inputs and the original tone positions. Run uses these fixed sources throughout acquisition. Reset after changing sources, then check every input.

The Form copies `$coupler.module.cz_f1` and `cz_f2` into two ordinary editable pulses. Missing CZ slots remain required inputs. Use the same role generator for CZ, Xπ and X90. Both CZ lines must have the same finite frequency, envelope duration and pre-delay, on different generators. Gains, relative phase, waveforms and individual post-delays remain explicit. Playback starts both lines together and waits for both to finish. Only the two charge lines are driven; the coupler stays DC-biased.

Enter an increasing Common CZ frequency range in MHz. Set the positive integer `gate_counts`; the starting list is `[1, 3, 5]`. Every frequency point replaces only the two CZ frequencies in the acquisition pulses. Repetition preserves both source gains, relative phase and waveforms. Set reps, host rounds and waits for the authorized acquisition budget.

GE centers come from the fixed source's `f1.ext.readout_cal` and `f2.ext.readout_cal`. They must match the mux conditions used here. Missing centers allow raw acquisition but prevent phase analysis for that target. Invalid or coincident centers raise calibration errors. Supply centers from measured data under matching readout conditions.

## Read the phase plots

The target preparation pulse has phase φ. The X projection is another X90 at φ; the Y projection is X90 at φ + 90°. These names identify the axes of pulses in separate acquisitions.

For each independent projection, analysis linearly projects mean IQ along the saved GE center difference to obtain P1. The projection preserves the resulting population values and original readout effects. The quadratures are `qX = 2 P1X − 1 = 1 − 2 P0X` and `qY = 2 P1Y − 1 = 1 − 2 P0Y`. Contrast is the length of this vector. Logical Bloch Z phase is `atan2(qY, qX)` in degrees, relative to the preparation frame.

The phase figure shows contrast, logical phase for each spectator branch and n, and conditional phase `phase(Xπ) − phase(I)`. All angles wrap to `[-180°, 180°)`, so π appears at −180°. Each train's phase is its total accrued phase. Per-gate slopes require a separate unwrapped analysis across n. The per-tone figures retain mean I/Q for every spectator, projection and gate count, including tones other than the target.

A phase point requires both projections and all requested rounds at that frequency and n. Missing X or Y gives a reason rather than a phase estimate. A missing spectator branch also prevents conditional phase at that point. Exactly zero contrast gives an undefined-phase reason. Weak nonzero contrast remains available for manual inspection of quadrature curves before interpreting an angle. Every shot and its original readout effects remain in the thermal ensemble. Interpret the phase as an ensemble measurement with pulse and readout errors; pure-state phase and physical fidelity need separate evidence.

## Select a frequency and review proposals

Only n=1 selects frequency. Analysis unwraps adjacent frequency points, then uses piecewise-linear interpolation to find π modulo 2π within the scanned interval. This assumes the phase change between adjacent samples is smaller than 180° and that the local curve is adequately sampled for linear interpolation. Choose a fine enough grid by inspecting the quadratures and phase. Unobserved turns between samples remain ambiguous.

A unique crossing gives a frequency proposal. A sampled endpoint crossing counts once even when two segments share it. Missing n=1, an incomplete n=1 branch or projection, no crossing, multiple crossings, a flat π segment or an ambiguous half-turn step stops proposal generation and reports the reason. Missing frequency points split the interpolation intervals; multiple roots remain ambiguous regardless of their distance from the source frequency.

The figure also marks each n's own π crossings. The n=3/5 curves and their intersections are diagnostics for manual confirmation; n=1 remains the sole selection source. This is piecewise-linear interpolation with sampling assumptions, rather than a parametric fit with fitted uncertainty. Review consistency, statistical ratios and fidelity manually.

FIT proposes two pulses at `$coupler.pulses.cz_f1` and `cz_f2`. Both copy the saved source cfg and replace only their common frequency. Signed gains, relative phase, waveforms, generator identities and timing stay unchanged. Separate `$coupler.module.cz_f1` and `cz_f2` slot rows point to these concrete keys resolved in the same capture.

Inspect the plot and each preview row before Apply Selected. Pulse and slot rows apply independently, including partial selection of one line. Check the resulting complete cfg before the next run. Amplitude and local-Z calibration are separate manual steps. Manually alternate amplitude and frequency calibration, with three rounds as a starting workflow, then measure local Z after choosing both. Reset the next Form to capture accepted pulses.

## Save, cancel and reload

Raw complex IQ has shape `2 spectator branches × 2 projections × gate count × frequency × (rounds*reps) × full tones`. Branch indices 0 and 1 mean nominal I and Xπ. Projection indices 0 and 1 mean X90 and Y90. Shot indices are round-major. Raw keeps each shot's full ordered tones in the readout's ADC units/sample, preserving the acquired normalization.

Cancel preserves completed whole rounds. Unacquired round slots stay complex NaN. Partial rounds or nonfinite acquired IQ are errors. Analysis retains completed projection means; phase requires both projections.

Native Save retains the raw axes and source cfg, including roles, target, GE centers and both full CZ pulses. Load leaves the current Form unchanged. Analysis uses the loaded cfg only; missing historical cfg stops phase analysis with a source error.

Validate the actual FTF control chain and gate performance with experimental data. Record bias values externally and recalibrate after changing pulse shape or operating point.
