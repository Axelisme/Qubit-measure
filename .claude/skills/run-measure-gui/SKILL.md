---
name: run-measure-gui
description: Run, drive, screenshot, and smoke-test the measure-gui qubit-measurement GUI over its MCP control socket. Use when asked to launch/start/test the measure-gui app, drive a single-qubit measurement (lookback, onetone/twotone spectroscopy, Rabi, T1/T2, readout optimization) via the measure-gui MCP tools, take a GUI screenshot, or follow the recommended experiment flow.
---

# run-measure-gui

`measure-gui` is a Qt desktop GUI for superconducting-qubit (Fluxonium)
measurement on a ZCU216 FPGA. It is driven headlessly through an MCP server
(`measure-gui`, configured in `.mcp.json`) that launches the GUI subprocess and
relays a newline-delimited JSON RPC over a TCP control socket. **You drive it
with the `mcp__measure-gui__gui_*` tools** — there is no separate driver to
write; the tool sequence *is* the harness.

A standalone smoke driver (`smoke.py`, next to this file) talks the same socket
directly (wire method names, no MCP client) and runs the whole experiment loop
against a **mock SoC** — use it to verify the GUI launches and the loop works on
a fresh checkout.

Paths below are relative to the repo root (`<repo>` = the directory with
`run_gui.py` and `.mcp.json`).

## Before touching real hardware — READ THIS

You drive **software only**. You cannot see the cabling, the sample, or the
fridge. So on a real (non-mock) session you **must get hardware facts from the
user first**, and you **must respect device safety limits**.

- **⚠️ YOKOGS200 in current mode: keep the value within ±7e-3 A.** Exceeding it
  can **physically destroy the instrument**. `gui_device_setup(name,
  updates={"value": ...})` and any flux-sweep edge must stay in range. When in
  doubt, ask — do not guess a flux value.
- **Read the board first with `gui_soc_info`** (after connect). It returns the
  QICK soccfg: a human-readable `description` plus structured `cfg` — DAC/ADC
  **channel count, sample rates (`fs`), DDS frequency ranges, tile layout**.
  `gui_connect_mock` / `gui_connect_start` also fold this description into their
  reply. This is the hardware you *can* see in software.
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

- The project venv (`.venv`) with deps installed; Python is pinned to 3.9.
- An X display. On a headless box: `sudo apt-get install -y xvfb` and prefix
  GUI launches with `xvfb-run -a`. On a desktop session `DISPLAY` is already
  set and no xvfb is needed.
- No hardware: the mock SoC + `FakeDevice` cover the whole flow offline.

## Run (agent path — MCP tools)

This is the path you use when asked to "drive measure-gui". The MCP server
auto-launches the GUI; you call tools.

```
gui_launch                       # spawns the GUI, connects; banner shows the
                                 # handshake: "wire vN (mcp==gui); gui code vX, mcp code vY"
gui_connect_mock                 # one-shot: mock SoC + project + active context
gui_state_check                  # all four flags must be true before running
gui_soc_info                     # the board: channels, sample rates, freq ranges
```

Then the experiment loop (per tab):

```
gui_adapter_list                                  # available experiments
gui_tab_new(adapter_name="fake/freq") -> tab_id   # readable id, e.g. fake-freq-1a2b3c4d
gui_tab_snapshot(tab_id) -> editor_id             # the cfg-editing session handle
gui_tab_list_paths(tab_id)                        # dotted cfg paths + current values + choices
gui_editor_set_field(editor_id, "rounds", 30)     # WYSIWYG edit of the form's draft
gui_run_start(tab_id)                             # waits ~1s; finished -> {tab}, slow -> {status:pending}
gui_run_progress                                  # live bars while running (fallback to events)
gui_run_wait(tab_id)                              # block until done (after pending)
gui_analyze_start(tab_id)                         # fit; then read gui_tab_get_analyze_result
gui_save_data(tab_id) / gui_save_image / gui_save_both
gui_view_screenshot(tab_id)                       # base64 PNG of the window/tab
```

Detecting completion — **prefer events over polling**: `run_started` /
`run_finished{outcome}` and `device_setup_started/finished` are auto-subscribed;
`gui_events_poll` drains them. A `diagnostic{severity}` push carries the same
error/info the GUI would show in a dialog. `gui_run_progress` is a fallback —
don't busy-poll `gui_run_running_tab` in a sleep loop.

The full, authoritative tool reference is the **MCP server instructions block**
(shown by the client when the server connects, defined in
`lib/zcu_tools/gui/services/remote/mcp_server.py`). Read it for the call
contract (failed calls raise — never fire duplicates), preconditions, and the
event/diagnostic model.

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
`pi_gain`, …) — the GUI persists these per flux-context.

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

Flux/RF sources (YOKOGS200, SGS100A) are driven as **devices**:
`gui_device_connect(type_name, name, address)` → `gui_device_setup(name,
updates={"value": ...})` ramps an output (cancellable, with progress via
`gui_device_setup_progress`). `gui_device_setup_spec(name)` lists the settable
fields. Sweeping a device across an experiment is done in the adapter cfg's
`dev` / sweep section, not by manual per-point setup.

**Stash reusable constants in the context (md/ml), then reference them by name
in cfg.** Channel numbers, `res_probe_len`, probe-pulse lengths etc. go into the
MetaDict (`gui_context_set_md_attr`); named waveforms/modules go into the
ModuleLibrary (`gui_context_new`/role tools). A cfg field can then reference
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
- **Reconnect leaves the old GUI on the port.** Port release has a TIME_WAIT
  delay; if `gui_launch` reports `Address already in use`, the previous GUI
  subprocess hasn't exited yet — wait or kill it.
- **Analyze and run are separate operations; both make the tab busy.** Saving
  or editing while a tab is running/analyzing returns
  `precondition_failed: ... is busy`. Wait for the operation to settle
  (`tab.snapshot.interaction.is_analyzing` / `is_running` false, or the
  `run_finished` event) before the next mutating call. (The smoke harness hits
  this — it waits on `is_analyzing` before `save.data`.)
- **cfg paths have no `value` segment.** Module sub-fields are
  `modules.qub_pulse.freq`, not `...qub_pulse.value.freq`; an unknown path
  fails `invalid_params` rather than silently no-op'ing. Always confirm against
  `gui_tab_list_paths`.
- **Edit through the tab's `editor_id`, not the tab directly.** Take `editor_id`
  from `gui_tab_snapshot`; `gui_editor_set_field` edits the same draft the form
  shows. Switching a ModuleRef key (`<path>.ref`) returns `removed`/`added`
  settable paths so you needn't re-list.
- **Real VISA devices (YOKOGS200/SGS100A) need a VISA backend.** Without
  `pyvisa-py` or the IVI binary, `gui_device_connect` fails
  `Could not locate a VISA implementation`. Use `FakeDevice` for offline device
  flows.
- **`run.start` is fire-and-forget.** A duplicate call starts a *second* run.
  Issue mutating tools exactly once and read the reply.
- **Mirror/image peaks in spectra (ZCU DAC artifact).** The Xilinx ZCU board's
  DAC has a finite sampling rate and side-band leakage, so a real transition can
  show up *mirrored* around `sample_f/2` in a twotone spectrum. A peak at the
  "wrong" frequency may be an alias, not a real transition — sanity-check
  against the predicted qubit/resonator frequency before trusting it, and ask
  the user if unsure.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `gui_launch` → `Address already in use` | Old GUI still on the port after a reconnect; wait for TIME_WAIT or kill the stale `run_gui.py`. |
| `precondition_failed: ... is busy` on save/edit | The tab is still running/analyzing; wait for the operation to finish first. |
| `precondition_failed` on run/save with no busy tab | Missing active file-backed context or no run result yet — `gui_state_check`, then `gui_connect_mock` (or set up a context). |
| `invalid_params` on `gui_editor_set_field` | Path wrong (often a stray `value` segment); re-check `gui_tab_list_paths`. |
| `Could not locate a VISA implementation` | Real device driver with no VISA backend; use `FakeDevice` or install `pyvisa-py`. |
| GUI never renders / launch times out | No X display; set `DISPLAY` or run under `xvfb-run -a`. |
| Stale GUI behaviour after a code change | `/mcp reconnect measure-gui` then `gui_launch`; confirm `gui code vX` bumped in the banner. |
