# Rabi chevron

Readout references `$mux.module.readout` when present, preserving the library's complete tones and scalar settings. Without a slot, fill every required readout field in the blank mux template.

## Purpose and sequence

Scan common drive frequency in MHz and common envelope duration in us at fixed
signed gains and relative phase. The sequence is nominal B11 preparation with
both calibrated Xπ pulses, simultaneous two-line drive, double Xπ contrast and
one full mux readout. Thermal population remains in every acquisition. This is
not pure-state preparation, heralding or a fidelity measurement.

## Prepare the Form

Bind `f1`, `f2`, `coupler` and `mux`, then Open or Reset. Review captured component
names and the original tone positions derived from the mux group's qubits.
Check both Xπ pulses, GE centers, complete mux tones and routing. Missing sources
remain required inputs. Existing coupler CZ pulse slots supply editable source
pulses. Without slots, fill both explicit cosine templates, including common
frequency, signed gains, length and Nyquist zones. Channels use captured role
wiring; no coupler AC generator is used.

Choose relative phase using the separate constructive-phase experiment. Keep
both signed gains and their ratio fixed in this scan. Both source frequencies,
envelope lengths and pre-delays must match. Enter frequency and positive duration
endpoints explicitly. Both pulse frequencies change together at each frequency
coordinate. Only common duration changes within that trace. Individual phases,
waveform styles and post-delays remain source values. Playback starts together
and waits for the longer line, including post-delay.

Review repetitions, rounds and timing before Run. Source values are not certified
safe settings. No live library or alias lookup occurs during acquisition.

## Interpret the result

Read the chevron and the return curves together to select a later CZ frequency
and width window. Joint P00 after contrast is a same-shot return proxy, not the
product of marginal probabilities. All GE-classified shots are retained, with
no radius exclusion or confusion-matrix inversion. Higher-level population is
not uniquely resolved by this binary partition. A single valley is not gate
fidelity and cannot select a calibrated CZ on its own.

Full-tone mean I/Q plots preserve diagnostic information outside the selected
pair. Incomplete points are shown as missing, not measured zero. Only complete
duration traces support rate fitting. Degenerate or failed fits report reasons,
not zero rates or model truth. Each complete duration slice uses the existing
undamped cosine fit. Rates below 1 divided by the saved duration span are
unresolved. The rate plot includes finite uncertainty; fit_quality reports
residuals and relative parameter errors. The adapter does not use chi-square,
contrast/sigma, frequency/sigma or fidelity acceptance gates.

Choose duration spacing to keep expected rates below the Nyquist limit
`1/(2 × duration spacing)`. There is no automatic Nyquist check or alias
selection. Judge the single-frequency model against the visible trace.
Thermal mixtures can violate its assumptions.

## Persistence and next steps

Native save retains complete source cfg, GE centers, coordinate axes, full tone
order and round-major raw shots. Cancellation preserves acquired rounds, while
unacquired slots remain NaN. Nonfinite acquired IQ is an error. Load leaves the
current Form unchanged; analysis uses only the loaded source cfg. Missing source
cfg prevents calibration-dependent analysis rather than borrowing current values.

This diagnostic experiment makes no automatic frequency, duration or module
writeback proposal. Select the next amplitude/frequency window after reviewing
the plots. Alternate amplitude and frequency manually, with three rounds as a
starting workflow rather than a required counter. Measure conditional phase and
local Z after selecting the pair. Changing pulse shapes or bias requires another
calibration. Offline synthetic evidence is not an actual FTF smoke run.
