# Fluxdep MCP control

**Last updated:** 2026-10-13, native spectrum stages and Filter exclusion counts

Fluxdep MCP composes native GUI RPCs into handwritten agent decision tools over
one shared bridge. It is an independent, optional L3 cross-experiment analysis
app, not part of Measure or Workflows.

## Design

Start from the agent's judgments: data usability (J1), symmetry alignment (J2),
sampling (J3), transition hypotheses (J4), fit acceptance (J5) and point rejection
(J6). Steps may repeat and their order is not fixed. RPC exposes basic controls and
full projections; MCP composes those controls into tools and bounded summaries.
Sending a stage tool expresses intent. The GUI opens/reuses the named draft and
owns prerequisites, lifetime, publication and downstream invalidation.
Commands and domain ranges live in tool descriptions, not reply schemas. Mechanical
edits return counts and changes; finish returns published evidence. Field changes
are [before, after], while point/mask changes are [added, removed]. Long operations
without an agent decision during computation wait until terminal. GUI and MCP
use the same renderer. Skill supplies the workflow and completion criteria;
measure_knowledge supplies detailed interpretation, examples and counterexamples.
These design principles are recorded in
[Agent interface layers](../../../../docs/adr/draft/agent-interface-layers.md#mcp-設計原則).
Fluxdep's fit dependency model is app-specific, not a rule for every MCP.

## Tool inventory

The server exposes 13 analysis tools, 3 control tools and 3 lifecycle tools, for
19 tools in total.

| Category | Tool | Decision or result |
| --- | --- | --- |
| Lifecycle | `fluxdep_connect` | Attach, initialize control identity and summarize project, spectra and fit. |
| Lifecycle | `fluxdep_launch` | Start an authorized GUI; attach by default. |
| Lifecycle | `fluxdep_disconnect` | Detach without closing the GUI. |
| Control | `fluxdep_control_status` | Read native control identity. |
| Control | `fluxdep_control_acquire` | Request control for the fixed owner connection. |
| Control | `fluxdep_control_release` | Return control to Local without cancelling work. |
| Analysis | `fluxdep_spectrum_list` | Spectrum stages/drafts, Filter counts and fit freshness. |
| Analysis | `fluxdep_spectrum_snapshot` | Revisit one spectrum with evidence PNG. |
| Analysis | `fluxdep_spectrum_load` | J1 raw-data usability, with summary and PNG. |
| Analysis | `fluxdep_spectrum_remove` | Remove one named source. |
| Analysis | `fluxdep_align` | J2 symmetry alignment; finish publishes and returns evidence. |
| Analysis | `fluxdep_sample` | J3 native picker sampling; finish publishes and returns evidence. |
| Analysis | `fluxdep_filter` | J6 joint-cloud filtering; finish publishes and returns evidence. |
| Analysis | `fluxdep_database_info` | Search database ranges, entries and levels. |
| Analysis | `fluxdep_fit` | J4 inputs/search and J5 recorded quality plus Show evidence. |
| Analysis | `fluxdep_fit_result` | Recorded params, inputs and freshness; images only for fresh quality-bearing results. |
| Analysis | `fluxdep_export_spectrums` | Sampled checkpoint, including empty completed spectra, without Filter masks. |
| Analysis | `fluxdep_spectrum_load_processed` | Restore checkpoint; reject all conflicting names before publication. |
| Analysis | `fluxdep_fit_export_params` | Fresh params.json handoff; native path, full-precision params and calibration source. |

Command parameters, numeric ranges and transition formulas are in each tool's
description. The agent does not manage open/cancel, context IDs, search tokens or
active display.

## Authority and lifecycle

The GUI owns control authority, interactive identities and search operations.
Business calls require holder unless explicitly readonly. Reads return current
projections, not per-connection mutation baselines. MCP does not maintain another
authority or operation registry. Domain validity, busy and owner-lifetime checks
remain independent. See [ADR-0070](../../../../docs/adr/0070-control-identity.md).

`assembly.py` combines explicit pipeline/control tools with Fluxdep lifecycle
initialization. `fluxdep_connect(label="agent", mode="owner")` initializes fixed
socket identity after authentication and wire checks. Owner auto-acquires unlocked
Local; a GUI lock or another online holder still permits connection. The reply
contains native `control` (`holder`, `locked`, `online`, `me`), project paths,
published spectrum stages/drafts and fit freshness. An optional project supplies
only chip/qubit names; GUI computes `result_dir` and `data_root`. Wire mismatch
still rejects the connection, but versions are not part of the agent receipt. `readonly` never acquires, even when its
label matches the holder. Omitted label/mode retain the prior choices on reconnect;
launch supports the same choices, including a later attach after auto_connect=false.

`fluxdep_control_status`, `fluxdep_control_acquire` and `fluxdep_control_release`
forward to the GUI authority. `not_holder`, `control_locked`, `control_held` and
`readonly_connection` become MCP isError replies with JSON `error` text preserving
native code/message/reason/data. MCP never reclassifies these refusals or retries.
Dispersive's shared readonly lifecycle does not initialize a control identity.

## Data, drafts and publication

`spectrum_tools.py` composes list with Filter/fit reads, and raw load with snapshot
and render. `projection.py` converts complete RPC point arrays into count/ranges,
rounds numerical spectrum summaries to three decimals and preserves GUI
alignment provenance. Published stages are loaded/aligned/sampled; pending is
align/sample/null independently of the published stage. Completed zero-point
sampling is still sampled. Spectrum lists directly project the GUI's stage and
per-spectrum filter_excluded_count; MCP does not reconstruct them from booleans
or the joint mask. Exclusion counts describe published Filter masks, not live
drafts or min_distance downsampling. List keeps the global Filter/fit summaries.
Internal stage counters are not agent observations.
Load and snapshot attach the native evidence PNG as image
content, not base64 text or an image file. PNG integrity uses the same core
producer as Measure. MCP retains no spectrum, draft, provenance or fit state.

`align_tools.py` implements J2 as `fluxdep_align(name, command, params?, image?)`.
It opens/reuses the named line draft, sends commands and summarizes half/integer
lines, period and changed fields as before/after pairs. Every move_line first
sets conjugate, defaulting to false, so the command does not inherit a hidden GUI
mode. auto_align polls interactive.read once per second with the starting
context_id until the GUI is not busy. A retired draft becomes an interrupted tool
error; MCP does not compare identities or keep a registry. finish accepts manual
or load-inherited lines without auto_align and attaches spectrum.render evidence.
image=true requests interactive.render for unfinished drafts. GUI open receipts
supply discarded stage facts; downstream invalidation belongs to app publication,
not MCP inference.

`sample_tools.py` implements J3 as `fluxdep_sample(name, command, params?, image?)`.
It opens `kind="sample"` without reading the spectrum first. The GUI chooses
OneTone or TwoTone and rejects unknown or unaligned sources. MCP projects the
returned native kind, detector settings and positional point differences.
Setting changes list only changed fields as before/after pairs. TwoTone replies
keep detected point counts separate from mask pixels. Initial and reopened
TwoTone drafts start empty in both GUI and RPC. finish reads published point
ranges and attaches spectrum.render evidence; image=true requests draft evidence
for other commands. MCP exposes no brush mode setter or full-mask command.

`filter_tools.py` implements J6 as `fluxdep_filter(command, params?, image?)`.
It opens/reuses the global GUI Filter and sends stroke, set_min_distance, undo
or finish. Counts and per-spectrum [selected, total] pairs come from the GUI.
Finish publishes and closes once, then attaches selection.render() evidence of
the published cloud and masks. image=true on unfinished commands requests
interactive.render(selection) draft evidence. MCP saves no mask or draft image.

Spectrum Export/Restore tools report sampled checkpoint summaries, including
zero-point completed spectra. Export does not save Filter masks. Restore refuses
all names before publication when any name conflicts and preserves GUI errors.

## Fit and recorded evidence

`fit_tools.py` composes `fluxdep_fit` as a current-input read, caller overlay,
complete replacement, search, bounded native awaits until terminal, final result
read and successful-result render. Omitted fields retain the GUI's current choices;
explicit null clears separate frequencies. It stores no choices and introduces no
business defaults or validators. Schemas reuse GUI declarations, including numeric
limits; the patch interface makes fields optional and permits frequency nulls.
`fluxdep_database_info(path)` returns native parameter/flux ranges and entry/level
counts without changing a fit.

Fit quality comes from the recorded GUI result, not a new pointcloud read.
Summaries round GHz residuals to three decimals and keep every compiled transition
in order, including duplicate labels and unused lines with null residuals. The
input point count sums recorded attribution counts. Unrecorded quality cannot be
represented as zero statistics. Successful fits attach the native Show overlay PNG;
failed/cancelled outcomes return status/error without an image or parameter result.
`fluxdep_fit_result(image?)` reads native availability and freshness without a
search. Recorded quality uses the same summary. Fresh quality-bearing results
attach Show evidence by default; stale results never attach it. No-result reads
return status=no_result and params=null. Params-only reads keep native inputs,
with points/residual/per_transition null, and attach no image. Neither state
creates zero statistics or reads the current cloud. Stage finish tools project
native invalidated labels without rebuilding the dependency graph.

`fluxdep_fit_export_params(savepath?)` calls only `fit.export_params`. Omission
uses the GUI project's default params.json; an explicit string names a GUI-local
destination. The GUI merges the fit while preserving independent sections. The
reply contains `savepath`, full-precision GHz `params` with EJ/EC/EL, and
`alignment_from`, the actual first aligned spectrum used for calibration, including
zero-point spectra. This is the export source, not alignment inheritance provenance.
MCP projects the native receipt without reading fit/spectrum state again, rounding
params, inferring the source or requesting an image. The GUI refuses stale fits
with `fit_stale` before I/O. Native readiness, control and I/O failures propagate
without retry, fallback or rollback.

## Failures and verification

The stdio entry supplies connection settings and instructions. Server exit only
disconnects; it does not kill or shut down the GUI. Tools preserve scalar JSON
types for GUI validation. Their descriptions identify composed reads; they do not retry, reconnect or rollback. Timeout/image failure may follow an accepted GUI
mutation. Failed search outcomes remain data, distinct from invocation errors.

Public tool/stdio behavior is tested through recording Transport in
`tests/mcp/fluxdep/`. Native GUI/domain behavior belongs to the app remote suite.
The agent workflow is `.agents/skills/run-fluxdep-gui/SKILL.md`.
