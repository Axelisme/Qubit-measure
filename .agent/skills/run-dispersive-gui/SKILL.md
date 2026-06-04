---
name: run-dispersive-gui
description: Launch the dispersive-fit-gui (a standalone Qt GUI for fluxonium dispersive-shift fitting) for the user, and observe its state over MCP. The USER drives the analysis in the GUI (load the fluxonium fit from params.json → load a one-tone spectrum → preprocess → tune the coupling g and resonator frequency bare_rf by slider or auto-fit → export the dispersive section back to params.json); the agent is READ-ONLY and reports current state. Use when asked to open/launch the dispersive-fit-gui app or check what state it is in.
skill_version: 1
---

# run-dispersive-gui

`dispersive-fit-gui` is a Qt desktop GUI for fluxonium **dispersive-shift fitting** —
an optional analysis tool, a sibling of `fluxdep-gui` and `measure-gui` (its own
state / services / MCP server / skill). It is launched and observed through an MCP
server (`dispersive-gui`, configured in `.mcp.json`) that starts the GUI subprocess
and relays a newline-delimited JSON RPC over a TCP control socket.

**The agent is READ-ONLY.** The user performs the analysis in the GUI; the agent
launches it and reports current state with the `mcp__dispersive-gui__dispersive_*`
read tools. There are deliberately no load / preprocess / tune / fit / export tools:
the slider tuning of `g` / `bare_rf` and the judgement of when the predicted lines
match the spectrum need the user's eye on the GUI plot, which the agent does not have.

Paths below are relative to the repo root (the directory with `.mcp.json`).

## The analysis pipeline (what the USER does in the GUI)

A single linear flow (each step enables the next; the agent observes progress via
`dispersive_state_check`'s five flags):

1. **Load fit inputs** — read the `fluxdep_fit` section of `params.json` (the
   `(EJ, EC, EL)` fluxonium params + flux alignment + a `bare_rf` seed). This is the
   hard prerequisite: it comes from **fluxdep-gui**, which must have run first.
2. **Load a one-tone spectrum** — a resonator-flux hdf5 (the GUI's "Transpose axes"
   toggle handles legacy x=frequency files; the user judges this from the preview).
3. **Preprocess** — fit + remove the electronic delay, smooth, fit a common circle
   centre, take the phase, differentiate / normalize → the normalized phase image
   the tuning works against (a 3-panel diagnostic shows signal / edelay / phases).
4. **Tune g / r_f** — sliders drive a live prediction of the ground/excited
   dispersive frequencies over the spectrum; the user matches them by eye. Or
   **auto-fit** — a scipy optimizer fits `g` (and optionally `bare_rf` within ±2 MHz)
   by maximizing the overlap with the signal.
5. **Render** the result figures (dispersive-with-onetone + chi-shift).
6. **Export** — write the `dispersive = {g, bare_rf}` section back to `params.json`,
   preserving the `fluxdep_fit` section it read.

## Cross-app data flow

`params.json` is shared with fluxdep-gui: fluxdep writes `fluxdep_fit`, dispersive
reads it and writes `dispersive`. The typical order is **fluxdep-gui first** (fit
EJ/EC/EL + flux alignment), **then dispersive-fit-gui** (fit g + bare_rf). If
`dispersive_fit_inputs_info` shows `has_inputs=false`, the user needs to run
fluxdep-gui (or browse to a params.json that already has a fluxdep_fit section).

## Prerequisites

- The project venv (`.venv`) with deps installed; Python is pinned to 3.9.
- An X display. On a headless box: `xvfb-run -a` prefix (or rely on the MCP
  server's offscreen launch). On a desktop session `DISPLAY` is already set.
- No hardware — this is a pure offline analysis tool.

## Run (agent path — MCP tools)

```
dispersive_launch              # opens the GUI for the user, connects (control port 8767)
                               # or dispersive_connect to attach to a GUI the user already started
dispersive_state_check         # {has_project, has_fit_inputs, has_onetone, has_preprocess, has_result}
dispersive_project_info        # {chip_name, qub_name, result_dir, database_path}
dispersive_fit_inputs_info     # {has_inputs, params:{EJ,EC,EL} or null, flux_half, flux_int, flux_period, bare_rf_seed}
dispersive_preprocess_status   # {has_preprocess, n_flux, n_freq, edelay}
dispersive_fit_result          # {has_result, g, bare_rf, g_bound, fit_bare_rf, qub_dim, qub_cutoff, res_dim, auto_fit_done}
dispersive_disconnect          # detach the bridge; does NOT stop the user's GUI
```

That is the whole agent surface: launch/connect/disconnect plus the five read tools.
The agent never mutates the analysis and never stops the GUI.

## Gotchas

- **Read-only over RPC.** If asked to "run the analysis", the agent cannot — it
  launches the GUI and the user drives it. Report the state you can read; do not
  claim to have performed steps you cannot perform.
- **Frequencies are in GHz** in the state and the read tools (`g`, `bare_rf`,
  `bare_rf_seed`). The GUI's sliders display MHz, but the wire/state values are GHz.
- **fluxdep_fit is the hard input.** `dispersive_state_check.has_fit_inputs` /
  `dispersive_fit_inputs_info.has_inputs` is false until the user loads a params.json
  that has a `fluxdep_fit` section — produced by fluxdep-gui. Point the user there.
- **Launch vs connect** (both default port 8767): `dispersive_launch` starts a NEW
  GUI for the user (needs the port free); `dispersive_connect` attaches to one the
  user already started. There is no stop tool — the agent never closes the user's
  GUI; `dispersive_disconnect` only detaches the bridge.
- **After editing GUI code**, `/mcp reconnect dispersive-gui` then `dispersive_launch`
  — the MCP server is a separate process and caches old code until reconnected.
