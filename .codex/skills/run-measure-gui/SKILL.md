---
name: run-measure-gui
description: Run, drive, screenshot, and smoke-test the measure-gui qubit-measurement GUI over its MCP control socket. Use when asked to launch/start/test the measure-gui app, drive a single-qubit measurement (lookback, onetone/twotone spectroscopy, Rabi, T1/T2, readout optimization) via the measure-gui MCP tools, take a GUI screenshot, or follow the recommended experiment flow.
skill_version: 41
---

# run-measure-gui

`measure-gui` is a Qt desktop GUI for superconducting-qubit (Fluxonium)
measurement on a ZCU216 FPGA. It is driven headlessly through an MCP server
(`measure-gui`, configured in `.mcp.json`) that launches the GUI subprocess and
relays a newline-delimited JSON RPC over a TCP control socket. **You drive it
with the `mcp__measure-gui__gui_*` tools** — there is no separate driver to
write; the tool sequence *is* the harness.

**You are an operator, not a developer.** Drive everything through these MCP
tools; do **not** read or edit the repo's source code. To resolve something
that looks wrong or contradictory, **re-measure** (widen the sweep, re-run, read
the figure) and trust the tool output — do not grep the implementation (it can
be stale relative to what you'd infer, and reading it is not your role).

A standalone smoke driver (`smoke.py`, next to this file) talks the same socket
directly (wire method names, no MCP client) and runs the whole experiment loop
against a **mock SoC** — use it to verify the GUI launches and the loop works on
a fresh checkout.

Paths below are relative to the repo root (`<repo>` = the directory with
`.mcp.json`; the GUI launcher lives at `script/run_measure_gui.py`).

## Before touching real hardware — READ THIS

> **Mock SoC / `FakeDevice` flow? Skip this section.** It is safe and offline —
> none of the hardware-safety rules below apply. They kick in only when a real
> `YOKOGS200` / `SGS100A` is involved.

You drive **software only**. You cannot see the cabling, the sample, or the
fridge. So on a real (non-mock) session you **must get hardware facts from the
user first**, and you **must respect device safety limits**.

- **⚠️ YOKOGS200 in current mode: keep the value within ±7e-3 A.** Exceeding it
  can **physically destroy the instrument**. `gui_device_setup(name,
  updates={"value": ...})` and any flux-sweep edge must stay in range. When in
  doubt, ask — do not guess a flux value.
- **Read the board first with `gui_soc_info`** (after connect). It returns the
  QICK soccfg: a compact human-readable `description` table (per-channel
  **generator/readout type, converter port, sample rate (`fs`), max pulse/buffer
  length**) plus a structured `cfg` carrying the full detail (DDS frequency
  ranges, tile layout, …). `gui_soc_connect` also folds
  this description into their reply. This is the hardware you *can* see in software.
- **Ask the user for what soccfg does NOT tell you** (the wiring/physics):
  - Which DAC/ADC **channels** are wired to the readout transmission line, and
    which to the qubit drive line (soccfg lists the channels and their rates,
    but not what each is cabled to).
  - Qubit type: **Transmon or Fluxonium** (changes the expected spectrum and the
    flux model).
  - Estimated **readout-resonator frequency range**.
  - The **filter / balun passband** on each channel — signal outside it is
    heavily attenuated, so sweeps must stay inside. (The notebook encodes this
    in channel names, e.g. `qub_1_4_ch` = a 1–4 GHz balun line — mirror that
    convention in your md attrs to remind yourself.)
- **`nqz` (Nyquist zone) rule of thumb:** use `nqz=2` for tones **> 2000 MHz**;
  only use `nqz=1` for tones **< 1500 MHz** (there `nqz=1` gives stronger signal
  than `nqz=2`).
- **There are many more details than fit here. When anything looks wrong or
  ambiguous, stop and ask the user** — a wrong device value or out-of-band sweep
  is expensive (instrument damage or hours of bad data), not just a retry.

None of this applies to the mock SoC / `FakeDevice` flow (safe, offline) — but
the moment a real `YOKOGS200`/`SGS100A` is involved, it does.

## Prerequisites

- The project venv (`.venv`) with deps installed; Python is pinned to 3.13.
- An X display. On a headless box: `sudo apt-get install -y xvfb` and prefix
  GUI launches with `xvfb-run -a`. On a desktop session `DISPLAY` is already
  set and no xvfb is needed.
- No hardware: the mock SoC + `FakeDevice` cover the whole flow offline.

## Run (agent path — MCP tools)

This is the path you use when asked to "drive measure-gui". The MCP server
auto-launches the GUI; you call tools.

### Lab notebook discipline (`agent-memory` MCP — no hook enforces it)

A separate `agent-memory` MCP server is your persistent, human-readable lab
notebook across sessions. It has three functions: **records** (one measurement per
folder — the verdict, the numbers, the figures; immutable), **troubleshooting**
(context-free symptom → fix, reusable across qubits), and **checklists** (one
acceptance list per experiment type). **This is your only notebook — keep all
measurement knowledge here, never in any other or built-in memory store, so it
stays recallable per chip/qub/experiment.** Two non-negotiable touch points:

- **BEFORE an experiment** call `memory_recall(chip, qub, exp_type)`. It returns
  three buckets: the acceptance `checklist` for this experiment, the `gotchas`
  (known fixes) for it, and the `recent` records for this chip/qub. Read all three
  before configuring — past you already learned things this session would repeat.
  Hit a symptom mid-run? `memory_search(query=<symptom>)` before improvising.
- **AFTER analyze** run the acceptance gate (below) and `memory_record_measurement`.

### Overview and project-identity tools

```
gui_overview         # standalone "current state" summary — returns:
                     #   {state, project:{chip_name,qub_name,res_name} or null, context, soc,
                     #    tabs:[{tab_id,adapter,is_running}],
                     #    running_tab, active_tab}
                     # Auto-connects if not already connected (pure read, no side effects).
                     # active_tab is the USER's current focus (collaboration only —
                     # not an operation target; agent ops always take an explicit tab_id).
                     # FIRST CALL when attaching to a running session: call gui_overview
                     # to load current state before assuming anything — the user may be
                     # actively operating the same GUI.
                     # gui_connect also folds the same overview into its reply ({note, overview}).

gui_tab_set_active(tab_id)   # push a tab to the user's foreground (user↔agent collaboration).
                             # Agent operations all take an explicit tab_id and are independent
                             # of which tab is active.  Use gui_tab_set_active only to guide the
                             # user's attention — e.g. ask them to drag flux lines on a specific
                             # tab — or to read back which tab they are currently looking at via
                             # gui_overview's active_tab field.

gui_project_info     # current project identity:
                     #   {chip_name, qub_name, res_name, result_dir, database_path}
                     # Fast-fails with precondition_failed: no_project when no project is applied.
                     # gui_overview.project also uses these long keys ({chip_name,qub_name,res_name},
                     # result_dir/database_path omitted for conciseness).
```

### Startup (fresh session)

```
gui_launch                                      # spawns the GUI, connects; banner shows the
                                                # handshake: "wire vN (mcp==gui); gui code vX, mcp code vY"
# Same startup path a GUI user takes (no mock shortcut):
gui_soc_connect(kind="mock")                    # mock SoC (or kind="remote", ip, port). SYNCHRONOUS:
                                                # returns the SoC summary directly, no wait/poll; a
                                                # remote connect fail-fasts at ~1s if unreachable.
gui_startup_apply(chip_name="Q1_Chip",          # apply the project; omit result_dir/database_path
                  qub_name="Q1", res_name="R1")  # to scope them under chip/qub (notebook layout)
gui_context_new(bind_device="fake_flux")        # create a context bound to a flux device (reads its
                                                # current value/unit; FakeDevice->none, YOKOGS200->A).
                                                # Mock sessions use "fake_flux" (FAKE_FLUX_DEVICE_NAME);
                                                # real hardware: use the device name you connected with.
                                                # Omit bind_device for an unbound context; clone_from=<label>
                                                # clones an existing one. Or gui_context_use(label) for an existing one.
gui_state_check                                 # global readiness: four flags (has_project/has_context/has_active_context/has_soc)
                                                # — all must be true before running; use gui_tab_snapshot for per-tab
                                                # progress (is_running/is_analyzing/has_run_result/…)
gui_soc_info                                    # the board: per-channel type, sample rate, port, max length (+ full cfg)
```

Then the experiment loop (per tab):

```
gui_adapter_list                                  # available experiments
gui_adapter_guide(adapter_name="onetone/flux_dep")# READ FIRST: per-experiment behavior, expected md/ml, recommended
                                                  # ranges + gotchas — live here, not in this skill. (gui_tab_stage1
                                                  # folds this guide in, so the recommended flow rarely calls it directly.)
gui_tab_new(adapter_name="fake/freq")             # PURE: just creates a tab -> {tab_id}. (gui_tab_stage1 creates + folds
                                                  # editor_id/tree/guide in one call.) id e.g. fake-freq-1a2b3c4d
gui_tab_snapshot(tab_id) -> editor_id             # per-tab progress + the cfg-editing session handle
gui_tab_get_cfg(tab_id)                           # returns a nested value tree (not a flat list). Leaf conventions:
                                                  #   scalar leaf    → bare value (null = unset)
                                                  #   enum leaf      → {"$value": v, "$choices": [...]}
                                                  #   sweep node     → {"start":..., "stop":..., "expts":..., "step":...}
                                                  #   ref node       → {"$ref":{"current":"key","options":[...]}, ...current-variant subtree...}
                                                  #                    options are bare names; set_field accepts a bare name
                                                  # $-prefixed keys mark leaf metadata; plain keys are subtree nodes.
                                                  # pass prefix="modules.readout" to return only that subtree (no match → {});
                                                  # pointing prefix at a sweep edge returns the whole sweep node.
                                                  # EDIT via dotted path (gui_editor_set_field) — path syntax unchanged.
gui_editor_set_field(tab_id, "rounds", 30)        # convenience: tab_id resolves the tab's cfg-editor automatically;
                                                  # explicit editor_id (from gui_tab_snapshot) also accepted
                                                  # Sweep edge fields (start/stop/expts/step) only accept plain numbers;
                                                  # eval/ref expressions are not accepted there (use scalar leaf fields
                                                  # for eval). If an adapter pre-wires an eval edge, override it by
                                                  # passing a numeric value directly.
# RECOMMENDED FLOW = the 4 bundle tools (each folds the NEXT decision's input, stops at a decision point):
#   gui_tab_stage1(adapter_name)            -> {tab_id, editor_id, tree, guide}   # ① new+guide
#                                              tree = nested value tree (same shape as gui_tab_get_cfg — see above).
#                                              guide = adapter guide (full prose) on FIRST call for this adapter in the
#                                              current MCP session; subsequent calls for the SAME adapter return
#                                              guide_omitted: true instead (guide already in session context, not resent).
#   gui_tab_stage2(tab_id, edits={path:v})  -> {..run.., figure, analyze_params}   # ② configure+run; STOPS before analyze
#   gui_tab_stage3(tab_id)                  -> {summary, figure, writeback_preview}   # ③ analyze
#   gui_tab_stage4(tab_id, save_data=False) -> {applied_ids[, data_path]}   # ④ writeback (+ optional save)
# The base tools below = ON-DEMAND (fine-grained control). DEV tools (gui_debug_screenshot/_versions/_operations —
#   debugging the GUI/MCP itself, NOT the measurement flow) are a separate tier in the server instructions:
gui_tab_run_start(tab_id)                         # waits ~1s; finished -> {status:finished, figure:<png>,...}, slow -> {status:pending}
gui_tab_run_wait(tab_id)                          # block until done (only after pending; blocks your turn — for a long run background it, see "Detecting completion")
gui_tab_get_current_figure(tab_id)                # RARELY NEEDED (run/analyze finished already fold the figure, incl 2D
                                                  # scans via run; use only for a re-render / mid-flight plot / chosen out_path).
                                                  # Writes the CURRENT plot (run's 2D map, analysis fit, or post-analysis
                                                  # figure — whatever is on the tab's plot stack) to a PNG FILE and
                                                  # replies {saved_to, bytes}. THE ONLY way to look at any plot, including
                                                  # non-analysis 2D scans (flux_dep / power_dep): Read the saved_to path.
                                                  # Always a file (fixed 640×480), never inline base64. Omit out_path to
                                                  # write a per-tab temp file; pass out_path="<abs path>" to choose where.
                                                  # gui_debug_screenshot(target, out_path?) [DEV tier] same contract: always
                                                  # writes a file, replies {saved_to, bytes}, never base64. target="window"
                                                  # grabs the main window; target=<dialog name> (setup/device/predictor/inspect/startup)
                                                  # grabs that dialog if open.
gui_tab_analyze(tab_id)                           # a FIT settles -> {status:finished, summary:{...}, figure:<png>} (its own
                                                  # fit result + plot; same summary as gui_tab_get_analyze_result); an
                                                  # INTERACTIVE pick (flux_dep) -> {status:pending} → see below
gui_tab_post_analyze(tab_id)                      # second analysis layer on top of the primary fit (e.g. single-shot ge);
                                                  # FIT-only, settles -> {status:finished, summary:{...}} inline;
                                                  # slow -> {pending} then gui_tab_post_analyze_wait/poll; needs primary analyze first
gui_tab_get_post_analyze_result(tab_id)           # re-fetch post-analysis summary (params: gui_tab_get_post_analyze_params)
gui_tab_save_data(tab_id) / gui_tab_save_image / gui_tab_save_result   # each returns the resolved written path ({data_path[, image_path]})
gui_tab_writeback_apply(tab_id)                   # PURE: apply the writeback -> {applied_ids}. (gui_tab_stage4 also folds an optional save.)
gui_tab_save_post_image(tab_id)                   # save the post-analysis figure (mirrors gui_tab_save_image)
```

Detecting completion — completion is observed synchronously or by waiting/polling
a handle, never by subscribing to a push stream:

| situation | what to do |
|---|---|
| fast run / fast fit | the call returns `{status:finished}` (`gui_tab_run_start` / `gui_tab_analyze` when it settles within `wait_seconds`, default 1.0) |
| `gui_tab_run_start` / `gui_tab_analyze` returned `{status:pending}` | wait (`gui_tab_run_wait` / `gui_tab_analyze_wait`, blocks) or poll (`gui_tab_run_poll` / `gui_tab_analyze_poll`, non-blocking) |
| want live progress bars | already in the `gui_tab_run_poll` reply while `status:running` (active + bars); no separate progress tool |
| a poll or wait says `status:cancelled` | a user/agent cancel (distinct from `failed`); **not a raise, not an error** — read optional `feedback` for the Stop reason (present when user clicked "Send & Stop"), then re-plan |

### Acceptance gate (after analyze, before writeback)

Once `gui_tab_analyze` has settled, before you write anything back, run the gate. It is
**self-grading, not a blocker** — an imperfect run is acceptable, but you must
record *why* honestly.

1. **Re-read the checklist** from the `memory_recall(chip, qub, exp_type)` you did
   before the run (the `checklist` bucket). If it is empty, grade against the
   experiment's `gui_adapter_guide` expectations and the figure instead, and
   consider *proposing* one — apply `memory_checklist_set` only with the user's
   agreement (see *Checklist is user-owned* below).
2. **Grade each item with evidence.** Look at the analysis figure (the finished
   `gui_tab_analyze` reply folds it) and the summary; for every checklist item write a
   one-line pass/fail with the number or the visual fact that justifies it. The
   figure is the evidence — a small fit error bar alone does not pass an item.
3. **`memory_record_measurement`** the run: `decision=accept|reject`, a one-line
   `reason` (the verdict), the per-item pass/fail in `body`, `figure_paths` to copy
   the plot(s) into the record folder (pass the PNG path the run/analyze reply
   gave you; omit `figure_paths` if there is no figure — a record with no figure
   is still valid), and `data_ref` if you saved the data.
4. **Then writeback as usual** — the gate does not block it (`gui_tab_stage4` /
   `gui_tab_writeback_apply`). If an item failed, prefer the partial-writeback rules in
   "Gotchas" (write the safe subset, leave the dubious one unset) and say so in the
   record's `reason`.
5. **If you learned a reusable rule** (a symptom→fix that generalizes beyond this
   qubit): `memory_search` the symptom; update the matching solution
   (`memory_update_solution`, add this record id to `seen_in`) or add a new one
   (`memory_add_solution`). Keep records for the instance, solutions for the rule.

**Checklist is user-owned.** Records (`memory_record_measurement`) and troubleshooting
solutions (`memory_add_solution`/`memory_update_solution`) are yours to write freely.
The acceptance checklist is the user's curated rubric: when a run or user feedback
suggests a new or changed acceptance criterion, *propose* it and call
`memory_checklist_set` **only after the user explicitly agrees** — never edit a
checklist on your own.

**Interactive analysis (e.g. `onetone`/`twotone flux_dep`).** Some adapters have no
automatic fit — the analysis is a 2D map the **user picks on**. `gui_adapter_guide`
flags these (its writeback describes a hand-pick). The flow:

1. Run as usual, then call `gui_tab_analyze(tab_id)`. It returns **`{status:pending}`** —
   that is expected, *not* an error: a live picker is now mounted in the tab.
2. **Tell the user what to do** — e.g. "drag the red (half-flux) and blue
   (integer-flux) lines to the sweet spots on the 2D map and click **Done**" — then
   `gui_tab_analyze_poll(tab_id)` until `{status:finished}` (it stays `running` until the
   user clicks Done). Don't `gui_tab_analyze_wait` inline — it blocks your turn on a
   human; poll and check back (or wait from a background agent).
3. On finished, the `summary` in the `{status:finished}` reply (or
   `gui_tab_get_analyze_result(tab_id)` to re-fetch) contains the picked values
   (`flx_half` / `flx_int` / `flx_period`); apply them with `gui_tab_writeback_apply`.
   **The picking is the user's judgement — you set up, prompt, observe, and write
   back; you never decide the line positions.**

To **abort** a mounted interactive picker without a result — you set it up but no
human will pick, or you need to re-plan — call `gui_tab_analyze_cancel(tab_id)`. It
settles the interactive handle (`{ok:true, cancelled:true}` when one was in
flight; `cancelled:false` is a graceful no-op when nothing is mounted) and
unmounts the picker so the tab is free again. The cancel outcome rides
`gui_tab_analyze_cancel`'s own reply — **after a cancel, `gui_tab_analyze_poll` reports
`status:finished`, not `cancelled`** (only `gui_tab_run_*` surfaces a cancelled status
through poll/wait).

**Post-analysis (second layer, e.g. single-shot `ge` discrimination).** Some
adapters offer a second analysis on top of the primary fit. After a primary
`gui_tab_analyze` settles, `gui_tab_post_analyze(tab_id)` runs it — FIT-only, so it
degrades exactly like `gui_tab_analyze` (settles → `{status:finished}`, slow →
`{status:pending}` then `gui_tab_post_analyze_wait` / `gui_tab_post_analyze_poll`). It
fast-fails with `precondition_failed` until a primary analyze result exists, so
run `gui_tab_analyze` first. Read its summary with `gui_tab_get_post_analyze_result`
(its params come from `gui_tab_get_post_analyze_params`). The post figure shares
the tab's plot container — view it with `gui_tab_get_current_figure` and persist
it with `gui_tab_save_post_image`.

**`gui_tab_run_wait` blocks your whole turn until the run ends** (a big sweep is
minutes), and nothing pushes a completion event — the MCP server cannot wake
you. So for a long run, pick by who should wait:

- **Free your main loop, auto-continue when done** → call `gui_tab_run_wait` from a
  *background agent*. The block lives in the sub-agent; your main loop stays free
  and the harness re-invokes you with the run's result when it finishes.
- **Just don't block, you'll check back yourself** → `gui_tab_run_start`, then
  `gui_tab_run_poll(tab_id)` when you choose (the `running` reply carries live
  progress bars).

Reserve inline `gui_tab_run_wait` for runs you expect to finish quickly. The same
choice applies to `gui_device_wait_operation`, though those
ops are usually short.

### User feedback wakeup (cooperative interrupt — ADR-0023)

While you are blocked inside any `*_wait` tool, the user can type into the
GUI's feedback bar (a text field above the status bar) and click Send. The
wait returns early with:

```json
{"status": "user_feedback", "feedback": "<user's text>"}
```

The operation is **not cancelled** — it keeps running. You should:
1. Read and act on `feedback` (re-plan, adjust parameters, ask a follow-up).
2. Re-call the same `*_wait` with a fresh timeout to keep observing.

If you never re-await, `gui_tab_run_poll` / `gui_tab_list_all` still work for
non-blocking status checks.

A `diagnostic{severity}` push (errors / info the GUI would show in a dialog) rides
along in the *next* tool reply's notifications — you get it without asking. Don't
busy-poll `gui_tab_list_all` in a sleep loop.

### Agent-to-user prompt (`gui_notify_user` — BLOCKS your turn)

When you need the user to make a decision mid-workflow, call:

```
gui_notify_user(message, timeout=600)
```

This opens a **non-modal dialog** in the GUI and **BLOCKS your entire MCP turn**
until the user replies, dismisses, or the dialog times out. Plan accordingly —
do not call it casually from inside a larger automated sequence.

The call returns one of three outcomes:

| `reason` | meaning | `reply` field |
|---|---|---|
| `"reply"` | user typed a response and clicked Reply | the text they entered |
| `"dismiss"` | user clicked Dismiss (or closed the dialog) | absent |
| `"timeout"` | dialog auto-closed after `timeout` seconds | absent |

`gui_notify_user` **never raises** on dismiss or timeout — those are normal
outcomes. Check `reason` before reading `reply`.

**When to use proactively** (representative cases):

- A coarse scan shows multiple candidate features and you cannot pick without
  physics context: display the figure, describe what you see, and ask which
  feature to pursue.
- A critical fit result is borderline or visually suspicious: show the figure
  and ask whether to write it back or re-run.
- A writeback would overwrite a key value (`r_f`, `q_f`, `pi_gain`, …) with
  something significantly different from the current value.
- Before ramping a real YOKOGS200 to a new flux point if you are uncertain the
  value is within safe range.

The full, authoritative tool reference is the **MCP server instructions block**
(shown by the client when the server connects, defined in
`lib/zcu_tools/mcp/measure/server.py`). Read it for the call
contract (failed calls raise — never fire duplicates), preconditions, and the
diagnostic-push model.

## Run (smoke harness — verify the loop without an MCP client)

`smoke.py` launches the GUI itself and drives the mock pipeline over the raw
socket. Use it to prove a fresh checkout works:

```bash
# desktop session (DISPLAY set):
.venv/bin/python .claude/skills/run-measure-gui/smoke.py
# headless:
xvfb-run -a .venv/bin/python .claude/skills/run-measure-gui/smoke.py
```

Expected tail (≈30–60s):

```
[smoke] ready: {'project': True, 'context': True, 'active_context': True, 'soc': True}
[smoke] tab: fake-freq-09dbdc70
[smoke] edited reps/rounds
[smoke] progress: rounds %v/%m [0:00<0:04] (3.3%)
[smoke] run finished (saw_live_progress=True)
[smoke] analyzed + saved data
[smoke] screenshot -> /tmp/measure_gui_smoke.png (137535 bytes)
[smoke] tab closed
[smoke] SMOKE OK
```

It writes the screenshot to the OS temp dir (`$TMPDIR/measure_gui_smoke.png` —
the Analysis tab with the fitted resonator dip + writeback panel). It uses
control port **8799** so a live MCP session on the default 8765 is undisturbed.

## Recommended single-qubit flow (notebook → adapters)

`notebook_md/single_qubit.md` is the canonical bring-up procedure. Each notebook
section maps to a GUI **adapter** (`gui_tab_new(adapter_name=...)`); run →
analyze → write results back (the Writeback panel, or just feed the next tab).
The flow threads calibrated values through the MetaDict context (`r_f`, `q_f`,
`pi_gain`, …) — the GUI persists these per flux-context. **`magic_names.md`
(next to this file) is the glossary for these names**: meaning, unit, which
experiment produces each, and the cross-experiment relations (e.g.
`reset_f = r_f − q_f`, `post_delay = 5/(2π·rf_w)`). Per-experiment recommended
sweep ranges stay in `gui_adapter_guide`, not the glossary.

The **standard order** to characterize one bias point (→ = "then"; `(opt)` =
optional). Each stage's fitted result is written back into the context and
consumed by the next:

1. `lookback` → `timeFly` (readout time-of-flight)
2. `onetone/freq` → `r_f`, `rf_w` (resonator freq + linewidth)
3. `onetone/power_dep` **(opt)** → readout saturation power
4. `onetone/flux_dep` → flux period / sweet spot — **then move the flux bias to
   the integer (half-flux) point** before continuing
5. `twotone/freq` → `q_f`, `qf_w` (qubit freq)
6. `twotone/flux_dep` **(opt)** → fit qubit model parameters
7. `twotone/rabi/len_rabi` → `pi_len`, `pi2_len`, `rabi_f`
8. `twotone/rabi/amp_rabi` → `pi_gain`, `pi2_gain`
9. optimize readout **(opt)** (`twotone/power_dep` etc. → readout freq/gain/length)
10. `twotone/t2ramsey` → `t2*` — **use it to calibrate `q_f`** (Ramsey detuning)
11. **re-run `twotone/rabi/amp_rabi`** with the corrected `q_f`
12. `twotone/t1` → `t1`

Steps 7–12 (Rabi/T2/recalibrate/Rabi/T1) are the per-point qubit
characterization. Reset characterization (single/dual/bath, in the notebook) is
a separate sub-procedure layered on top once a π pulse exists.

**Estimate `q_f` before `twotone/freq` — sweep the right window or you see only
noise.** At a fresh bias point `q_f` is unknown, and a narrow or mis-centered
twotone window returns pure noise that *looks* like "no qubit". After step 4 (at
the integer / sweet-spot bias) predict `q_f` from the fluxonium model first:
- Set the model in the predictor — `gui_predictor_set_model_params(EJ, EC, EL,
  flux_half, flux_period)` (energies in GHz; a typical fluxonium is roughly
  `EJ:EC:EL ≈ 4:1:1`), or `gui_predictor_load(path)` from a `params.json`
  `fluxdep_fit`. Then `gui_predictor_predict(value)` at the current flux returns
  the predicted `q_f` (`gui_predictor_info` reads back the active EJ/EC/EL + flux
  alignment).
- Then run `twotone/freq` with a **wide** sweep bracketing that estimate (e.g. a
  ~2 GHz span such as 4000–6000 MHz) to actually catch the peak; narrow only once
  you see it. **A twotone scan that looks like noise almost always means the
  window missed `q_f`, not that there is no qubit** — widen and re-centre before
  giving up. `q_f` also moves strongly with flux (a fluxonium sits low at the
  half-flux sweet spot and much higher at the integer point), so confirm which
  bias you are at before trusting a window.

## Decision boundaries — act vs. ask

Separate what you may infer automatically from what needs the user. The
dividing line is **physical interpretation**: safe *procedure* is yours, *which
feature is real / is the target* is the user's.

**You may decide automatically** (safe procedure): which adapter implements a
notebook step; sweep ranges within the guide's recommended bounds; narrowing or
widening a window; raising readout gain to improve SNR within safe limits;
reversing a flux sweep direction; re-running after a bad fit.

**Stop and ask the user** (physical interpretation):

- A coarse one-tone scan shows **multiple plausible dips/peaks** — ask which
  feature is the target resonator **before** narrow-fitting or writeback. Do not
  pick the strongest/first by default.
- A **weak or unusual spectral structure** might be the intended mode or an
  unrelated one (or a DAC mirror image) — confirm it is the target before
  building on it.
- Data stays **ambiguous or poor after a few reasonable parameter changes** —
  stop and consult rather than keep guessing; the next choice likely depends on
  physics you cannot see.

When you stop, say what you see (with a `gui_tab_get_current_figure` PNG), name
the options, and let the user choose.

**Readout bring-up rules that generalize beyond one notebook run:**

- **Do not start `onetone/freq` on real hardware until `lookback` has set a
  believable `timeFly`.** A guessed trigger offset can shift the readout window
  off the arriving signal and distort every downstream resonator fit.
- **Run `lookback` off-resonance when calibrating `timeFly` for an `hm`
  resonator.** On resonance the resonator can absorb the probe and hide the
  clean square-step edge you actually need for timing.
- **Tune `lookback` to the rising edge, not the saturated plateau.** Leave
  enough pre-trigger baseline to see the step start cleanly, keep the pulse /
  readout window short enough that the plot focuses on the edge, and only then
  trust the predicted offset.
- **If `lookback` is using a trigger expression derived from an old `timeFly`,
  break that feedback loop before re-measuring.** Use a direct fallback
  `trig_offset` (or remove the old md value) when you need an unbiased timing
  re-calibration.
- **Probe gain is part of the timing calibration.** If the user tells you the
  line only works at a stronger gain, re-run `lookback` at that gain before
  trusting the earlier `timeFly`.
- **Do not confuse GUI pulse gain with external device power/current values.**
  The readout-pulse `gain` fields in these adapters are normalized waveform
  amplitudes, typically safe in the range `-1` to `+1`, and do **not** mean the
  same thing as a source's physical output setting.
- **For power-insensitive readout experiments such as `lookback` and
  `onetone/freq`, `gain=1.0` is often the right default on a weak line.** If the
  user tells you the line is not power-sensitive for the task at hand, prefer
  the stronger readout drive so the signal-to-noise ratio is good enough to
  judge timing and resonator features cleanly.
- **After every important `run`, look at the figure before trusting any number.**
  Finished `gui_tab_run_start`/`gui_tab_run_wait`/`gui_tab_run_poll`/`gui_tab_analyze` replies FOLD a
  `figure` (PNG path; `None` if the render failed) — Read that. `gui_tab_get_current_figure`
  is rarely needed (a re-render, a mid-flight plot, or a chosen `out_path`). It returns the current plot
  whether or not the adapter does analysis, so for a **2D scan with no fit**
  (`onetone/twotone flux_dep`, `power_dep`) it is the 2D map itself.
  Judge: is the feature clean, the window right (too wide / too narrow), the SNR
  acceptable, the dispersive shift actually small? See the figure first, trust the
  number second. A plausible-looking fit value can still come from a visibly bad
  fit.
- **Before choosing a `lookback` frequency, ask whether the readout resonator is
  hanger-like (`hm`) or transmission-like (`t`).** For a hanger, the resonant
  point absorbs and `lookback` should be done off-resonance; for a transmission
  resonator the resonant point transmits more strongly and `lookback` should be
  done on-resonance.
- **When the user does not specify the resonator family, ask instead of
  guessing.** A common lab convention is that multi-qubit chip-integrated
  readout resonators are hanger-type, while single-qubit sample-box readout
  modes are often transmission-type, but this is only a heuristic.
- **For narrow windows and overlapping peaks, treat `fit_bg_slope` as a
  potential overfit knob.** If the background line is doing obvious work the
  physics should be doing, re-run with `fit_bg_slope=false` and compare the
  figure, not just the summary.
- **Do not force single-peak `hm` fits onto obviously overlapped or truncated
  features.** In those cases it is better to save the trace, report a manually
  verified dip frequency, and leave linewidth / derived module writeback unset
  than to write back bad fit parameters.
- **When characterizing multiple resonators, rename writeback targets per peak
  before applying them.** Avoid clobbering shared keys like `r_f`, `rf_w`,
  `readout_rf`, or `ro_waveform` unless the user explicitly wants the latest
  result to become the canonical default.

Flux/RF sources (YOKOGS200, SGS100A) are driven as **devices**:
`gui_device_connect(type_name, name, address)` → `gui_device_setup(name,
updates={"value": ...})` ramps an output (cancellable; a slow setup degrades to
a handle, and its progress bars ride the `gui_device_poll` reply while running).
`gui_device_setup_spec(name)` lists the settable fields. Sweeping a device across an experiment is done in the adapter cfg's
`dev` / sweep section, not by manual per-point setup. Different devices set up
**concurrently**: `gui_device_active_operations` lists every in-flight device op
in one call (each entry has `device_name` + `kind`, where kind is
device_connect / device_disconnect / device_setup) — then poll/wait each device
by name.

**Stash reusable constants in the context (md/ml), then reference them by name
in cfg.** Channel numbers, `res_probe_len`, probe-pulse lengths etc. go into the
MetaDict (`gui_context_md_set_attr` for a single key, `gui_context_md_set_attrs`
for a batch); to read multiple keys at once use `gui_context_md_get_attrs([keys])`
which returns `{values: {key: value}}`. Named waveforms/modules go into the
ModuleLibrary (`gui_context_ml_create_from_role`/role tools). A cfg field can then reference
`md.<attr>` (e.g. a pulse `freq: r_f`) or a module/waveform by its library key,
instead of hard-coding — the notebook does exactly this (`md.res_ch`,
`ro_waveform`, `readout_rf`, `pi_amp`).

For mock/offline practice use `fake/freq` (resonator-spectroscopy fake, no
hardware) — the smoke harness uses it.

## Gotchas

- **`gui_launch` banner is the reload check.** It prints `wire vN (mcp==gui);
  gui code vX, mcp code vY`. After editing GUI code you must `/mcp reconnect
  measure-gui` **then** `gui_launch` — the MCP server is a separate process and
  caches the old code until reconnected. `wire` mismatch = hard error; `gui
  code` / `mcp code` are reported, you eyeball them to confirm the reload.
- **`/mcp reconnect` does NOT stop the running GUI — it keeps its port.** A
  reconnect drops the bridge's socket but leaves the old `run_measure_gui.py` listening
  on 8765. So always `gui_stop` (or kill it) before relaunching. `gui_launch`
  fails fast if the port is occupied ("Port 8765 is already in use …")
  rather than silently attaching to the stale process — but if you ignore that
  error you stay on old code. **The handshake alone won't save you:** a stale
  process reports whatever version it was built at, so if it happens to match,
  `gui code vN` looks fine while you're on old behaviour. Confirm a relaunch by
  an *observable effect of your change*, not just the version banner.
- **`gui_launch` vs `gui_connect` (both default port 8765, opposite
  expectations).** launch starts a NEW GUI and needs the port FREE; connect
  attaches to an EXISTING GUI and needs the port LISTENING (errors "No GUI is
  listening on 127.0.0.1:8765" otherwise). Use launch to start, connect only to
  re-attach to one already up.
- **A run starts by clearing its tab's prior run/analyze/writeback result.** So
  while a run is in flight — and after it fails — the tab has no result:
  `gui_tab_analyze` / `gui_tab_save_*` fail-fast with `no_run_result` (the true reason:
  this run hasn't produced one yet), not a "busy" message. `gui_editor_set_field`
  while running is the one that returns `precondition_failed: ... is currently running`.
  Wait for the run to settle (`tab.snapshot.interaction.is_running` false, or
  `gui_tab_run_wait` / `gui_tab_run_poll`) before analyzing/saving. **Cancelled runs are
  the exception:** if the worker produced a partial result before observing the
  stop signal, the tab intentionally keeps that partial result (`has_run_result`
  true) and analysis/save may proceed; if no partial result exists, analyze/save
  still fail with `no_run_result`. (The smoke harness waits on `is_analyzing`
  before `tab.save_data`.)
- **`run` success is not `analyze` success.** A completed acquisition can still
  produce a bad or misleading fit. On real data, open the figure and verify the
  model visually before you trust `gui_tab_get_analyze_result`, especially for
  lookback timing, narrow resonator windows, and overlapping dips. **A small fit
  error bar does not prove the fit is right** — a noisy Rabi/freq fit can converge
  to a confident-looking value with a tiny uncertainty; only the figure tells you
  whether the model actually matches the data.
- **Minimum writeback bar: inspect the analysis figure first.** Do not write fit
  results into the context or module library unless the plotted fit matches the
  feature you intended to measure.
- **Saved data is always `.hdf5`, with a uniqueness suffix.** `gui_tab_save_data` /
  `gui_tab_save_result` force the `.hdf5` extension and append `_N` (e.g. a
  `data_path` of `foo` or `foo.h5` lands as `foo_1.hdf5`). **The save reply
  returns the resolved path directly** (`{data_path}` / `{image_path}` /
  `{data_path, image_path}`) — read the file back by that, not by the path you
  passed in. (The tab's `save_paths` and the diagnostic also carry it.)
- **cfg paths have no `value` segment.** Module sub-fields are
  `modules.qub_pulse.freq`, not `...qub_pulse.value.freq`; an unknown path
  fails `invalid_params` rather than silently no-op'ing. Get editable paths from
  `gui_tab_get_cfg` (the nested tree — `$`-prefixed keys mark leaf metadata;
  plain keys are subtree nodes). `stage1`'s `tree` field is the same tree as
  `get_cfg` — use it directly.
- **`gui_editor_set_field` / `gui_editor_set_fields` accept either a `tab_id`
  (convenience — the server resolves that tab's cfg-editor automatically) or an
  explicit `editor_id` from `gui_tab_snapshot`.** Both edit the same live draft
  the form shows (WYSIWYG — no separate commit step is needed to run). Switching
  a ModuleRef key (`<path>.ref`) returns `removed`/`added` settable paths so you
  needn't re-list.
- **Tab cfg edits are live — no commit needed before `gui_tab_run_start`.** Changes
  made via `gui_editor_set_field` / `gui_editor_set_fields` take effect on the
  tab immediately (WYSIWYG). `gui_editor_save(name=...)` is a separate
  operation that saves the current draft as a *named ModuleLibrary module/waveform*
  — it has nothing to do with applying the tab's cfg edits, and you never need it
  just to run the experiment.
- **Real VISA devices (YOKOGS200/SGS100A) need a VISA backend.** Without
  `pyvisa-py` or the IVI binary, `gui_device_connect` fails
  `Could not locate a VISA implementation`. Use `FakeDevice` for offline device
  flows.
- **`tab.run_start` is fire-and-forget.** A duplicate call starts a *second* run.
  Issue mutating tools exactly once and read the reply.
- **If a writeback item is only partially trustworthy, apply only the safe
  subset.** Frequency-only writeback is often safer than writing a dubious
  linewidth or derived readout module after a marginal fit.
- **Mirror/image peaks in spectra (ZCU DAC artifact).** The Xilinx ZCU board's
  DAC has a finite sampling rate and side-band leakage, so a real transition can
  show up *mirrored* around `sample_f/2` in a twotone spectrum. A peak at the
  "wrong" frequency may be an alias, not a real transition — sanity-check
  against the predicted qubit/resonator frequency before trusting it, and ask
  the user if unsure.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `gui_launch` → `Port 8765 is already in use` | A previous GUI is still running on the port; `gui_stop` it (or kill the stale `run_measure_gui.py`), then relaunch — or launch on another port. |
| `gui_connect` → `No GUI is listening on 127.0.0.1:8765` | Nothing running there; `gui_launch` first (connect only re-attaches to a running GUI). |
| `precondition_failed: ... is currently running` on `gui_editor_set_field` | The tab is running; wait for it to finish (run clears prior results, so editing mid-run is blocked). |
| `no_run_result` on `gui_tab_analyze` / `gui_tab_save_*` | No result for *this* run yet — the run is still in flight, failed, or was cancelled before producing a partial result (a run clears the previous result on start). Wait for it to finish, or re-run. |
| `precondition_failed: no_project` on `gui_context_new` | No project applied; `gui_startup_apply` first. |
| `precondition_failed` on run/save with no busy tab | Missing active file-backed context — `gui_state_check`, then `gui_startup_apply` + `gui_context_new`/`gui_context_use` if a context is missing. |
| `invalid_params` on `gui_editor_set_field` | Path wrong (often a stray `value` segment); re-check `gui_tab_get_cfg`. |
| `Could not locate a VISA implementation` | Real device driver with no VISA backend; use `FakeDevice` or install `pyvisa-py`. |
| GUI never renders / launch times out | No X display; set `DISPLAY` or run under `xvfb-run -a`. |
| Stale GUI behaviour after a code change | `gui_stop`, `/mcp reconnect measure-gui`, then `gui_launch`; confirm the change via an observable effect (the version banner can match a stale process). |
