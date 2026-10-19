# Mux qubit recipes

**Last updated:** 2026-10-19, readable experiment and recipe authoring

These recipes use the native mux Forms and numeric cores. Bind a complete MuxGroup,
open an entry and point, and connect an authorized acquisition environment first.
A recipe does not connect devices, repair bindings or infer a new scan range.
Small synthetic or truncated FTF runs check integration, not calibration quality.

## Authoring readable experiments and recipes

These four rules apply to all `zcu_lab` experiments and recipes. Keep the pulse
sequence in its core and the measurement procedure in its recipe. A reader
should not need to open a shared helper to discover either one.

### 1. Write the pulse sequence in run

Each core's `run()` lists its `Pulse`, `Delay`, `Join` and readout operations.
Share validation, reshaping and averaging mechanics, not a sequence builder that
switches experiments with a mode or flag.

For example, [mux Amp Rabi](../v2/mux/amp_rabi/core.py) writes reset,
`Join(Pulse(...), ...)`, then `MuxReadout` inside `run()`. Drives start together;
readout waits for the longest branch. Ramsey writes its own x90, delay, x90 and
readout stages instead of calling `acquire(mode="ramsey")`. The
[core authoring guide](../v2/experiment-authoring.md#1-在-run-直接寫出-pulse-序列)
shows a short sequence fragment.

### 2. Write each recipe step in the function body

Open the tab, choose library entries, edit fields, Run, Analyze and ask about
writeback in the recipe itself. A shared helper may perform one step, such as
one library selection. It must not hide setup plus Run, or Run plus Analyze
and writeback.

For example, the normal completion path of
[mux Amp Rabi](mux_amp_rabi.py) has the following shape. This fragment assumes
a valid bound group with Q1 at complete-list position 0 and the named library
pulse. It omits other input edits; keep cancellation checks when adding steps.

```python
# 1. Configure the selected qubit before acquisition.
tab = session.open_tab("mux/amp_rabi", aliases={"mux": target})
tab.use_library(("tones", 0, "drive"), "Q1.pulses.pi")
tab.set_fields([RecipeFieldEdit(("participants",), ["Q1"])])

# 2. Acquire and save IQ so the fit has a recorded source.
result = yield from tab.run_step()
run = result.run
if result.status == "cancelled" or run is None:
    return
run.save_raw()

# 3. Fit the gain sweep to locate pi and pi/2 rotations.
analysis = yield from run.analyze_step("primary")
if analysis.status == "cancelled":
    return

# 4. Ask before applying the proposed pulse calibration.
decision = yield from ask(run.propose_writeback())
if decision.status == "cancelled":
    return
if decision.decision == "accepted":
    tab.accept()
```

The recipe's input schema and reporting shape stay unchanged when moving these
steps into the body. GE also writes its Post analysis step explicitly.

### 3. Describe the measurement in docstrings

Say what is measured, which sequence is applied and what the results mean.
Keep units, input meanings and necessary failure conditions. Mention lifecycle,
rollback or caller duties in one sentence only when needed; link to framework
documentation for the details.

For example, a [mux Amp Rabi recipe](mux_amp_rabi.py) docstring can start with:

```python
"""Sweep drive gain on selected qubits with simultaneous mux readout.

Fit the resulting Rabi oscillations to estimate pi and pi/2 gains.
Ask whether to apply the proposed pulse calibrations; skipping writes nothing.
"""
```

Document the tool inputs after this measurement description. Do not replace it
with a list of transport, resource ownership and rollback clauses.

### 4. Separate stages with comments and blank lines

Use a blank line between setup, acquisition, analysis and writeback stages.
Explain each stage's purpose or a timing constraint, rather than narrating
individual assignments. Keep code comments in English.

For example, [mux GE](mux_singleshot_ge.py) needs distinct analysis stages:

```python
# Fit the paired probe-off/on IQ clouds before deriving discrimination settings.
analysis = yield from run.analyze_step("primary")
...

# Use the primary calibration to choose the Post discrimination radius.
analysis = yield from run.analyze_step("post")
...
```

This is a stage-layout sketch, not a complete recipe. Keep the cancellation
checks and writeback question visible in the actual function.

## Recipe names and inputs

| MCP recipe | GUI adapter | Scan fields | Estimates from successful analysis |
| --- | --- | --- | --- |
| `mux_freq` | `mux/freq` | `tones[i].sweep`, MHz | Frequency and FWHM, MHz |
| `mux_amp_rabi` | `mux/amp_rabi` | `tones[i].sweep`, gain | Pi and pi/2 gain |
| `mux_len_rabi` | `mux/len_rabi` | `tones[i].sweep`, us | Pi and pi/2 length, Rabi frequency |
| `mux_t1` | `mux/t1` | Common `delays`, us | T1 and background T1, us |
| `mux_t2ramsey` | `mux/t2ramsey` | Common `delay`, us | T2 and fringe rate |
| `mux_t2echo` | `mux/t2echo` | Common `delay`, us | T2 and fringe rate |
| `mux_singleshot_ge` | `mux/singleshot_ge` | `shots` | Primary fidelity, angle, threshold and width; Post radius |

Length Rabi accepts const and flat_top waveforms only. All pulse-scan elements
must have the same number of points. Delay scans keep one shared requested-delay
axis. GE keeps paired shots across tones; its Primary and Post use the existing
single-qubit analysis policy, not a four-state assignment calibration.

All seven tools take the following optional inputs:

- `target` binds the mux alias for this Reset. Omitted or null uses global aliases.
- `reuse_tab_id` names an idle tab of the matching adapter. It is reset before use.
  Omitted or null opens a new tab; recipes never choose a reusable tab themselves.
- `participants` is an ordered list of names. Omitted or null retains Reset defaults.
- `library` contains `{path, key}` selections. A null key retains and validates the
  current selection. A missing required selection fails; it does not create one.
- `inline` contains `{path}` copies from captured references. Copy ref ancestors
  before editing their children, outermost first.
- `overrides` contains `{path, value}` scalar or scalar-list edits. A null value
  retains the input. An expression uses `{"expr": "..."}` and stays authored.

The order is Reset, library selections, inline copies, then one override batch.
Each list preserves caller order. The Form owner checks paths, references and
fixed identities. A successful copy is not rolled back if a later edit fails.
Ordinary invalid drafts remain visible and block Run, including invalid settings
on nonparticipants. Recipes do not replace missing or invalid values with zero.

Use typed paths. A string is a field name; an integer is a complete-list position.
For complete tones Q1/Q2/Q3, participants Q3/Q1 run in original order `[2, 0]`.
An override for Q3 still uses position 2, never participants row 0 or the key `Q3`.
For example, this changes Q3's gain without changing the captured waveform:

```json
{
  "target": "MG3",
  "participants": ["Q3", "Q1"],
  "library": [{"path": ["tones", 2, "drive"], "key": "Q3.pulses.pi"}],
  "inline": [{"path": ["tones", 2, "drive"]}],
  "overrides": [{"path": ["tones", 2, "drive", "gain"], "value": 0.3}]
}
```

This example assumes those component and library names exist. Inspect the actual
Form before choosing paths and values. `tab.describe_cfg` describes fields,
positions, names and available commands. `tab.read_elements` reads the whole list
in one owner call. Frequency uses MHz, time uses us, and gain uses native units.
Range leaves use `start`, `stop` and `n`; no per-qubit override of shared
reps, rounds or recovery is provided. Editing one field replaces only that path.
Container writes do not merge with old content or run converters.

## Read actual conditions and estimates

Each definition declares its reported fields. `actual` contains those tone leaves
for the complete list, including nonparticipants. One owner `read_elements` call
captures the requested leaves before Run. Reset can change the complete list
length without changing the recipe definition.

Origins describe this recipe call, not the history of Form initialization:

- `explicit` means the caller supplied the value or library selection.
- `derived` means the recipe supplied an expression or computed the value.
- `default` means the recipe retained a Form tone leaf with no recorded origin.

Recorded origins take precedence over the tone fallback. A library selection can
be explicit while a child gain remains default. Default does not mean a literal
physical calibration, and the recipe does not perform extra per-leaf RPC reads to
reconstruct how the Form generated its initial values. Static shared fields use
their recorded recipe origins; they do not inherit the tone fallback.

For Q3/Q1, `actual["tones/2/gain"].element` is `{index: 2, name: "Q3"}`.
Estimates follow native analysis row order. `estimates["tones/0/freq"].element`
is therefore 2/Q3, while `tones/1/freq` belongs to 0/Q1. Only `element.index`
identifies the original complete-list position. The number in an estimate label
is a row offset, not a typed input path.

Failed fits keep their native reason and quality diagnostics. Missing estimate
values are absent, not invented zeros. An explicit null remains null. Without
source cfg, reporting does not guess a name from the current Form. Full replies
retain the native analysis summary as well as projected estimates.

## Save, review and Apply

A completed Run saves raw data, analyzes and saves Primary, and then asks for a
writeback decision. GE also analyzes and saves Post before asking. Inspect the
per-output artifact status as well as execution status. A finished recipe does
not imply every requested output format saved successfully.

Ramsey and Echo declare the Labber `independent` layout in their grouped axes
spec. Native `data.h5` and the Labber export preserve IQ on the delay/tone grid
and actual detune and source frequency on the tone-only grid. The writer does
not broadcast per-tone values or split them into separate files. It uses the
declared layout, not an automatic fallback for mismatched grids.
A successful save reports raw save `saved`, measurement data `Saved`, and both
native and Labber members `saved`. Native `data.h5` remains the authoritative
source for Load. If an output fails, inspect its error and status rather than
treating recipe completion as proof of a complete save.

Candidates use the Run/Load capture's original index, name and destination.
Rabi retains its pulse proposals and x180/x90 slots. Ramsey frequency correction
requires finite sourced carrier, actual detune and fitted fringe rate. A NaN
actual detune suppresses only that correction, not other usable coherence results.
GE keeps separate Primary and Post candidates. Missing capture produces no
candidates; editing current aliases or Form values does not supply historical data.

Review stable target names, values, uncertainty, quality and destination before
answering. `skipped` writes nothing. On `accepted`, the recipe calls `tab.accept()`
without an item selection. This writes all current draft items, Primary then
Post, ignoring GUI checkboxes. The draft is not frozen while waiting for an
answer. Changes made during that wait affect what is written. `accept()` does
not match the current draft to the question snapshot's version; that snapshot
is not a version lock or a bound item selection.
Preview, analysis and waiting do not Apply. Use manual single-item Apply when
only one calibration is authorized. Failed or partially acquired rows do not
become successful proposals.

Cancellation stops later steps. Finish-early keeps the native owner's partial-data
and save policy. Native, admission and transport failures propagate at the step
that failed; the recipe does not retry acquisition or silently refresh capture.

Summary, full and wait replies use detached snapshots. Later Reset, Form edits or
library changes do not rewrite previous actuals, estimates, candidates or native
analysis summaries. Native Load changes the analysis source, not current inputs.
For exact reporting and execution contracts, see the
[framework guide](../../lib/zcu_tools/mcp/measure/README.md#variable-element-reporting).
