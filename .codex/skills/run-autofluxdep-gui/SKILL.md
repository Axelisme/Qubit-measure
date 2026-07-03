---
name: run-autofluxdep-gui
description: Launch the autofluxdep-gui (a standalone Qt GUI for an automated fluxonium flux-dependence workflow) for the user, and observe its state over MCP. The USER drives the workflow in the GUI (set up project + SoC → assemble the node list → set the flux sweep + flux device → Run/Stop the sweep → watch each node's live fit); the agent is READ-ONLY and reports current state. Use when asked to open/launch the autofluxdep-gui app or check what state it is in.
skill_version: 1
---

# run-autofluxdep-gui

`autofluxdep-gui` is a Qt desktop GUI for an **automated flux-dependence
workflow** — it sweeps a flux device and, at each flux point, runs an ordered
chain of single-qubit experiment nodes (qubit_freq → lenrabi → ro_optimize / t1 /
t2 / mist), each feeding the next via an adaptive predictor. It is fully
independent of `measure-gui` / `fluxdep-gui` / `dispersive-gui` (its own state /
services / MCP server / skill). It is launched and observed through an MCP server
(`autofluxdep-gui`, configured in `.mcp.json`) that starts the GUI subprocess and
relays a newline-delimited JSON RPC over a TCP control socket.

**The agent is READ-ONLY.** The user assembles and runs the workflow in the GUI;
the agent launches it and reports current state with the
`mcp__autofluxdep-gui__autofluxdep_*` read tools. There are deliberately no setup
/ edit-node / set-flux / run / stop tools: building the node graph, picking the
flux sweep, and judging whether each node's live fit is good need the user's eye
on the GUI, which the agent does not have.

Paths below are relative to the repo root (the directory with `.mcp.json`).

## The workflow (what the USER does in the GUI)

The agent can observe how far the user has progressed via `autofluxdep_state_check`
(readiness flags), `autofluxdep_workflow_list` (the node graph) and
`autofluxdep_result_summary` (per-node sweep progress):

1. **Set up** the project (chip/qubit names) and connect a SoC — a flux-aware
   `MockSoc` offline, or real hardware. Optionally load a fluxonium predictor.
2. **Assemble the node list** — add / remove / reorder / rename experiment nodes.
   A node provides info keys (e.g. `qubit_freq`, `pi_pulse`) that later nodes
   require; a missing dependency skips a node for that flux point.
3. **Set the flux sweep** — the list of flux values to step through — and pick the
   **flux device** the sweep is applied through (its unit labels the flux axis).
4. **Run / Stop** the sweep. At each flux point the predictor service runs first,
   then each user node in order; each fills its own sweep Result in place and the
   GUI redraws.

## Prerequisites

- The project venv (`.venv`) with the `gui` extra installed; Python is pinned to
  3.13.
- An X display. On a headless box: the MCP server launches offscreen; otherwise a
  desktop session's `DISPLAY` is used.
- Offline runs use a flux-aware `MockSoc` (no hardware); a real run needs a
  connected board + flux source.

## Run (agent path — MCP tools)

```
autofluxdep_launch             # opens the GUI for the user, connects (control port 8768)
                               # or autofluxdep_connect to attach to a GUI the user already started
autofluxdep_state_check        # {has_project, has_soc, node_count, flux_count,
                               #  has_flux_device, is_running, has_results, ...predictor flags}
autofluxdep_project_info       # {chip_name, qub_name, result_dir, database_path, params_path}
autofluxdep_workflow_list      # each node's {name, type, enabled, provides, requires, has_result}
autofluxdep_node_cfg name=...  # one placed node's {name, type, knobs:{...}}
autofluxdep_result_summary     # per node-with-result {name, kind, n_flux, n_measured, fit_summary}
autofluxdep_disconnect         # detach the bridge; does NOT stop the user's GUI
```

That is the whole agent surface: launch/connect/disconnect plus the five read
tools. The agent never mutates the workflow and never stops the GUI or the sweep.

## Gotchas

- **Read-only over RPC.** If asked to "build the workflow" or "run the sweep", the
  agent cannot — it launches the GUI and the user drives it. Report the state you
  can read; do not claim to have performed steps you cannot perform.
- **The node list excludes the predictor service.** `autofluxdep_workflow_list`
  shows only the user-placed nodes; disabled nodes remain listed with
  `enabled=false` but are omitted from future runs. The predictor service is
  prepended only while a run is in progress and never appears as a list row.
- **`result_summary` is a progress summary, not the raw data.** It reports how many
  flux rows are measured and a tiny fit summary per node, never the raw 2D signal
  arrays (the agent does not plot).
- **Launch vs connect** (both default port 8768): `autofluxdep_launch` starts a NEW
  GUI for the user (needs the port free); `autofluxdep_connect` attaches to one the
  user already started. There is no stop tool — the agent never closes the user's
  GUI nor stops a running sweep; `autofluxdep_disconnect` only detaches the bridge.
- **After editing GUI code**, `/mcp reconnect autofluxdep-gui` then
  `autofluxdep_launch` — the MCP server is a separate process and caches old code
  until reconnected.
