# Rabi chevron

**Readers:** operator, agent. Select a CZ frequency and duration window from a Rabi chevron acquired through the GUI or MCP.

**Last updated:** 2026-10-21

Readout references `$mux.module.readout` when present, preserving the library's complete tones and scalar settings. Without a slot, fill every required readout field in the blank mux template.

## Purpose and sequence

Scan common drive frequency in MHz and common envelope duration in us at fixed
signed gains and relative phase. The sequence is nominal B11 preparation with
both calibrated Xπ pulses, simultaneous two-line drive, double Xπ contrast and
one full mux readout. Every shot of the thermal ensemble is retained. Interpret
the result as an ensemble diagnostic; pure-state preparation, heralding and
fidelity estimation require separate experiments.

## Prepare the Form

Bind `f1`, `f2`, `coupler` and `mux`, then Open or Reset. Review captured component
names and the original tone positions derived from the mux group's `qubits`.
Check both Xπ pulses, GE centers, complete mux tones and routing. Missing sources
remain required inputs. Existing coupler CZ pulse slots supply editable source
pulses. Without slots, fill both explicit cosine templates, including common
frequency, signed gains, length and Nyquist zones. Channels use captured role
wiring. The drive uses only the two charge lines; the coupler stays DC-biased.

Choose relative phase using the separate constructive-phase experiment. Keep
both signed gains and their ratio fixed in this scan. Both source frequencies,
envelope lengths and pre-delays must match. Enter frequency and positive duration
endpoints explicitly. Both pulse frequencies change together at each frequency
coordinate. Only common duration changes within that trace. Individual phases,
waveform styles and post-delays remain source values. Playback starts together
and waits for the longer line, including post-delay.

Review repetitions, rounds and timing against authorized limits before Run.
Treat source values as editable inputs requiring a safety review. Acquisition
uses the roles and pulses fixed at Open or Reset; Reset after changing sources.

## Interpret the result

Read the chevron and the return curves together to select a later CZ frequency
and width window. Joint P00 after contrast counts same-shot outcomes directly.
GE classification retains every shot and the original readout effects. This
binary partition leaves higher-level population unresolved. A valley selects
only a window for further calibration; gate fidelity needs separate evidence.

Full-tone mean I/Q plots preserve diagnostic information outside the selected
pair. Incomplete points are shown as missing. Only complete duration traces
support rate fitting. Degenerate or failed fits retain their failure reasons.
Each complete duration slice uses an undamped cosine fit. Rates below 1 divided by the saved duration span are
unresolved. The rate plot includes finite uncertainty; fit_quality reports
residuals and relative parameter errors. Review these diagnostics manually;
statistical ratios and fidelity are left to operator interpretation.

Choose duration spacing to keep expected rates below the Nyquist limit
`1/(2 × duration spacing)`. The operator must check Nyquist and possible aliasing.
Judge the single-frequency model against the visible trace.
Thermal mixtures can violate its assumptions.

## Persistence and next steps

Native save retains complete source cfg, GE centers, coordinate axes, full tone
order and round-major raw shots. Cancellation preserves acquired rounds, while
unacquired slots remain NaN. Nonfinite acquired IQ is an error. Load leaves the
current Form unchanged; analysis uses only the loaded source cfg. Missing source
cfg stops calibration-dependent analysis and reports the missing source.

This experiment provides diagnostic plots only. Frequency, duration and pulse
selection are manual. Select the next amplitude/frequency window after reviewing
the plots. Alternate amplitude and frequency manually, with three rounds as a
starting workflow rather than a required counter. Measure conditional phase and
local Z after selecting the pair. Changing pulse shapes or bias requires another
calibration. Validate the actual FTF control chain separately from offline data.
