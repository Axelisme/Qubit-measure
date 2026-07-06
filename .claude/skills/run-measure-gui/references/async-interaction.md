# Async Operations and User Interaction

Read this before interactive analysis, long waits, post-analysis, or GUI user prompts.

**Interactive analysis (e.g. `onetone`/`twotone flux_dep`).** Some adapters have no
automatic fit — the analysis is a 2D map the **user picks on**. `gui_adapter_guide`
flags these (its writeback describes a hand-pick). The flow:

1. Run as usual, then call `gui_tab_analyze_start(tab_id)`. It returns **`{status:pending, handle}`** —
   that is expected, *not* an error: a live picker is now mounted in the tab.
2. **Tell the user what to do** — e.g. "drag the red (half-flux) and blue
   (integer-flux) lines to the sweet spots on the 2D map and click **Done**" — then
   `gui_op_poll(handle)` until `{status:finished}` (it stays `running` until the
   user clicks Done). Don't `gui_op_wait` inline — it blocks your turn on a
   human; poll and check back (or wait from a background agent).
3. On finished, read the picked values with `gui_tab_get_analyze_result(tab_id)`
   (`flx_half` / `flx_int` / `flx_period`) — after a `pending`->`finished` analyze the
   generic poll/wait reports only status, so fetch the summary explicitly; apply them
   with `gui_tab_writeback_apply`.
   **The picking is the user's judgement — you set up, prompt, observe, and write
   back; you never decide the line positions.**

To **abort** a mounted interactive picker without a result — you set it up but no
human will pick, or you need to re-plan — call `gui_tab_analyze_cancel(tab_id)`. It
settles the interactive handle (`{ok:true, cancelled:true}` when one was in
flight; `cancelled:false` is a graceful no-op when nothing is mounted) and
unmounts the picker so the tab is free again. Cancel stays op-specific (the generic
`gui_op_*` family has no cancel). The cancel outcome rides `gui_tab_analyze_cancel`'s
own reply, and `gui_op_poll(handle)` on that settled handle reports `status:cancelled`
(the true terminal — poll never mislabels a cancelled or still-running op as finished).

**Post-analysis (second layer, e.g. single-shot `ge` discrimination).** Some
adapters offer a second analysis on top of the primary fit. After a primary
`gui_tab_analyze_start` settles, `gui_tab_post_analyze_start(tab_id)` runs it — FIT-only, so it
degrades exactly like `gui_tab_analyze_start` (settles → `{status:finished}`, slow →
`{status:pending, handle}` then `gui_op_wait` / `gui_op_poll`). It
fast-fails with `precondition_failed` until a primary analyze result exists, so
run `gui_tab_analyze_start` first. There is NO cancel for post-analysis (pure CPU
recompute). Read its summary with `gui_tab_get_post_analyze_result` (its params come
from `gui_tab_get_post_analyze_params`). The post figure shares the tab's plot
container — view it with `gui_tab_get_current_figure` and persist it with
`gui_tab_save(tab_id, figure="post")`.

**`gui_op_wait` blocks your whole turn until the op ends** (a big sweep is
minutes), and nothing pushes a completion event — the MCP server cannot wake
you. So for a long run, pick by who should wait:

- **Free your main loop, auto-continue when done** → call `gui_op_wait(handle)` from a
  *background agent*. The block lives in the sub-agent; your main loop stays free
  and the harness re-invokes you with the run's result when it finishes.
- **Just don't block, you'll check back yourself** → `gui_tab_run_start`, then
  `gui_op_poll(handle)` when you choose (the `running` reply carries live
  progress bars).

Reserve inline `gui_op_wait` for ops you expect to finish quickly. The same generic
handle drains drive device ops too (`gui_device_*` START → `gui_op_wait` /
`gui_op_poll`), though those ops are usually short.

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

If you never re-await, `gui_op_poll(handle)` / `gui_tab_list` still work for
non-blocking status checks. `gui_op_poll` additionally DRAINS every feedback
message buffered since your last drain and returns them as `feedback:[...]` (in
arrival order) while still reporting the TRUE status — so a non-blocking check
never loses a nudge and never falsely reads `finished` on a still-running op.
Draining consumes those messages: a subsequent `gui_op_wait` will not re-deliver
them (only the sticky terminal outcome is re-readable by wait).

A `diagnostic{severity}` push (errors / info the GUI would show in a dialog) rides
along in the *next* tool reply's notifications — you get it without asking. Don't
busy-poll `gui_tab_list` in a sleep loop.

### Agent-to-user prompt (`gui_prompt_user` — BLOCKS your turn)

When you need the user to make a decision mid-workflow, call:

```
gui_prompt_user(message, timeout=600)
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

`gui_prompt_user` **never raises** on dismiss or timeout — those are normal
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
(shown by the client when the server connects). Read it for the call
contract (failed calls raise — never fire duplicates), preconditions, and the
diagnostic-push model.
