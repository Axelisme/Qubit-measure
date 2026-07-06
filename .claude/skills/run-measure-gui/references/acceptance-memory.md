# Acceptance and Agent Memory

Read this before the first measurement and again before writeback.

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

### Acceptance gate (after analyze, before writeback)

Once `gui_tab_analyze_start` has settled, before you write anything back, run the
gate. It is **self-grading, not a blocker** — an imperfect run is acceptable,
but you must record *why* honestly. The gate is evidence for the agent/human
writeback decision; it is not an automatic fidelity threshold and does not
replace looking at the figure.

1. **Re-read the checklist** from the `memory_recall(chip, qub, exp_type)` you did
   before the run (the `checklist` bucket). If it is empty, grade against the
   experiment's `gui_adapter_guide` expectations and the figure instead, and
   consider *proposing* one — apply `memory_checklist_set` only with the user's
   agreement (see *Checklist is user-owned* below).
2. **Grade each item with evidence.** Look at the analysis figure (the finished
   `gui_tab_analyze_start` reply folds it) and the summary; for every checklist item write a
   one-line pass/fail with the number or the visual fact that justifies it. The
   figure is the evidence — a small fit error bar alone does not pass an item.
3. **`memory_record_measurement`** the run: `decision=accept|reject`, a one-line
   `reason` (the verdict), the per-item pass/fail in `body`, `figure_paths` to copy
   the plot(s) into the record folder (pass the PNG path the run/analyze reply
   gave you; omit `figure_paths` if there is no figure — a record with no figure
   is still valid), and `data_ref` if you saved the data.
4. **Then decide writeback deliberately** — the gate does not block it
   (`gui_tab_commit` / `gui_tab_writeback_apply`), but the responsibility stays
   with the agent/human reviewing the figure and preview. If an item failed,
   prefer the partial-writeback rules in "Gotchas" (write the safe subset, leave
   the dubious one unset) and say so in the record's `reason`.
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
