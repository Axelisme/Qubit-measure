**Readers:** operator, agent. Run mux onetone maps versus the shared coil and write back one participant's flux calibration.

# Mux onetone flux global

## Before Run

Bind alias `mux` to a `group/mux` component. The group must name matching
ordered qubits and resonators. Tone index i always refers to that full group
position, even when participants are reordered or omitted.

The group's `flux_source` names the connected coil driver.
Choose `flux_unit` to match the driver's stored native unit. No unit conversion,
mode change or output enabling occurs. Confirm limits, wiring, power and output
before running on hardware.

Open or Reset captures identities, aliases and library values. Later edits to
the Library or aliases do not change the captured form. Reset discards form
edits and captures sources again.

## Form

- `participants` selects Q names in acquisition order. The tone picker only
  selects which full-list position to edit.
- `flux_dev` starts from the captured group `flux_source`.
- `flux` is one native-unit sweep shared by all participants, initially 101 points.
  Open/Reset starts with all Qs selected. It reads each Q's `coil_flux` half/int
  calibration from the captured group and takes the union of the expanded
  half-to-integer intervals. Any missing half/int leaves both endpoints blank.
  Changing participants later does not recompute these starting values.
- `settle` is the wait in seconds after all DC writes at each flux point.
- `modules` contains the optional common reset and full mux readout.
- `frequency` contains only frequency-scan inputs.
  `frequency.df` supplies a common nominal MHz offset for every selected
  readout tone. Each offset uses the existing host-side onetone run because
  the mux firmware cannot sweep tone frequencies in hardware. Inner
  `participants` supplies the original tone selection.

Readout uses the mux onetone ringup-copy policy. A captured slot retains its
mixer and ADC window; five ringup times are added to trigger coverage. Without
a slot, generator and ADCs come from group wiring, gain starts at 0.1, mixer
comes from group `mixer_freq`, and the ADC window starts at 1.0 µs. Trigger is
group `time_of_fly + 5/(2π min κ)`; pulse length covers trigger plus window.
Missing group flight time or any κ leaves trigger and length required and blank.

These initial values are editable candidates, not confirmed calibrations.
Averaging starts at 1000 reps and 10 rounds. Reduce both and the sweep point
counts for an offline smoke.

## Starting scenarios

1. **Blank kickstart.** Confirm group wiring, resonator nominal frequencies and
   the coil source. Missing DC calibration leaves both shared bounds
   required and blank. Enter deliberate native-unit bounds before Run.
   Fill any missing resonator frequency, mixer, trigger and pulse duration. Ringup timing needs every resonator linewidth and the group flight time; without them enter timing explicitly.
2. **Frequencies without a readout slot.** Open or Reset to capture updated
   frequencies. The readout template takes generator/ADCs from group wiring.
   The ADC window candidate is 1 us. Trigger and pulse duration remain blank when linewidth or group flight time is missing.
   Review gain 0.1 and the DC bounds rather than treating seeds as calibrations.
3. **Captured readout slot.** Open or Reset after assigning the group readout.
   The form copies it, rebuilds full group tones and adds ringup to its trigger while preserving its mixer and ADC window.
   Confirm frequency and DC scan bounds. Refresh preserves your edits and the
   original seed provenance while recapturing validation sources. Reset
   discards edits and captures sources again.
4. **Agent recipe.** Use the corresponding mux flux recipe with explicit
   missing DC bounds. Its inputs must describe the same full identities,
   native units and frequency ranges as this form. After acquisition, answer
   one picker handoff per participant and review the selected proposals.
   The recipe guide owns invocation and answer payload details.

## Typed cfg and form paths

Python cfg and saved cfg share a flat layout. There is no `spectrum` wrapper.
Set `participants: list[str]`, full `tones`, and `modules` at top level.
`modules.reset` is optional; `modules.readout` is the complete numeric mux
readout. In the form, module selectors expose inline fields below
`modules.readout.module`; this selector wrapper is not part of Python cfg.
`frequency.df` is a SweepCfg of common nominal MHz offsets.
`tones[i]` contains only its fixed name. `flux_dev` names the coil; `flux` is a SweepCfg in native DC units.
`flux_unit` is `A`, `V` or `native`; no conversion occurs.
`reps` and `rounds` are positive counts. `initial_delay` and `relax_delay`
are finite nonnegative microseconds. `settle` is finite nonnegative seconds.
Every SweepCfg supplies `start`, `stop`, positive `expts` and
`step = (stop-start)/(expts-1)`; a single point requires equal bounds and zero
step. Form/recipe range editors use `n` for that point count and derive step.

The form orders participants, tone identities, modules, frequency/DC scans,
averaging, then timing. Source descriptions distinguish captured settings,
derived candidates and missing required values. Generated candidates are
marked uncalibrated; a stored value alone is not evidence of calibration.
All full-list entries must validate even when not selected for acquisition.
Generated seeds never imply calibration approval. Inspect source descriptions
and required-field errors before Run.

## Run and Stop

Run sets the coil at each flux point, waits for settle, then runs
the existing mux onetone spectrum. Each `flux_i` figure shows absolute MHz
against that tone's DC coordinate. Sources retain their final written settings.
Run does not restore bias automatically.

Stop interrupts ramps, settle and acquisition. A flux point is published only
after the complete inner spectrum returns. Earlier completed points remain;
the interrupted point and later points stay NaN. Analyze needs at least five
completed flux points. Device and acquisition failures are errors, not empty
successful maps.

## Analyze and Writeback

The Analysis parameter `tone` is an original group tone index, not the
participant-list position. It initially selects the first acquired participant.
Choose another acquired index and click Analyze to calibrate that Q.
The picker starts in magnitude mode.
Move the half-flux and integer-flux lines, then click Done. The existing picker
also provides Swap Lines, Auto Align and Cancel. Period is twice the absolute
half-to-integer separation. The summary includes the original index and Q name.

Review the resulting proposals before applying Writeback. Destinations use the
Run/Load capture, not the current form or aliases.
Only that Q's calibration under the captured group's `coil_flux` mapping is proposed. Other Q records are preserved, including updates made by earlier analyses of this same Run.
Repeat Analyze and Writeback for each acquired participant. Cancel produces no
new calibration. Missing historical cfg prevents identity-dependent analysis;
missing parameter capture permits picking but yields no writeback proposals.

## Saved data

Native data keeps original tone indices, native DC coordinates per tone,
absolute MHz frequency coordinates per tone, and complex IQ shaped
`(participants, flux points, frequency points)`. Frequencies use Hz on disk and
MHz after loading. Stop gaps remain NaN after native round-trip.

Labber export currently does not support this three-dimensional independent
grid. Save reports that output as failed while native data and other successful
outputs remain saved. Inspect each output status before treating a Save as
complete. Load the native `data.h5` for analysis and writeback.
