---
name: run-fluxdep-gui
description: Launch the fluxdep-gui (a standalone Qt GUI for fluxonium flux-dependence fitting) for the user, and observe its state over MCP. The USER drives the analysis in the GUI (load spectrum hdf5 → pick half/integer flux lines → select spectral points → cross-spectrum filter → export spectrums.hdf5 → search a database for EJ/EC/EL → export params.json); the agent is READ-ONLY and reports current state. Use when asked to open/launch the fluxdep-gui app, check what state it is in, or build its search database.
skill_version: 7
---

# run-fluxdep-gui

`fluxdep-gui` is a Qt desktop GUI for fluxonium **flux-dependence fitting** — an
optional analysis tool, fully independent of `measure-gui` (its own state /
services / MCP server / skill). It is launched and observed through an MCP server
(`fluxdep-gui`, configured in `.mcp.json`) that starts the GUI subprocess and
relays a newline-delimited JSON RPC over a TCP control socket.

**The agent is READ-ONLY.** The user performs the analysis in the GUI; the agent
launches it and reports current state with the `mcp__fluxdep-gui__fluxdep_*` read
tools. There are deliberately no load / align / point-pick / select / fit / export
tools: point-picking (dragging lines, painting the brush mask) and judging whether
a spectrum's axes are oriented correctly need the user's eye on the GUI preview,
which the agent does not have.

Paths below are relative to the repo root (the directory with `.mcp.json`).

## The analysis pipeline (what the USER does in the GUI)

Each step feeds the next; the agent can observe how far the user has progressed
via `fluxdep_spectrum_list` (per-spectrum `aligned` / `points_selected` flags) and
`fluxdep_fit_result`:

1. **Load** a raw spectrum hdf5 (a OneTone resonator-flux or TwoTone qubit-flux
   sweep). The GUI's "Transpose axes" toggle handles files whose Labber step
   channels are ordered freq-then-flux (common for OneTone) — the user judges this
   from the preview.
2. **Align** — pick the *half-flux* and *integer-flux* lines; this fixes
   `flux_period = 2·|int − half|` and the flux coordinate.
3. **Select points** — mark the spectral feature's points (an automatic threshold
   for OneTone; a hand-painted brush mask for TwoTone).
4. Repeat 1–3 to **accumulate several spectra** (a new one can inherit an existing
   one's alignment as its starting guess).
5. **Cross-spectrum filter** — select/downsample the joint point cloud.
6. **Export** `spectrums.hdf5`.
7. **Database search (v2)** — search a precomputed fluxonium database for the best
   `(EJ, EC, EL)` matching the selected joint point cloud, then **export
   `params.json`**.

## Prerequisites

- The project venv (`.venv`) with deps installed; Python is pinned to 3.9.
- An X display. On a headless box: `xvfb-run -a` prefix (or rely on the MCP
  server's offscreen launch). On a desktop session `DISPLAY` is already set.
- No hardware — this is a pure offline analysis tool.

## Run (agent path — MCP tools)

```
fluxdep_launch                 # opens the GUI for the user, connects (control port 8766)
                               # or fluxdep_connect to attach to a GUI the user already started
fluxdep_state_check            # {has_project, spectrum_count, has_active}
fluxdep_project_info           # {chip_name, qub_name, result_dir, database_path}
fluxdep_spectrum_list          # each spectrum's {name, spec_type, aligned, points_selected}
fluxdep_selection_pointcloud   # the joint {fluxs, freqs} cloud (freqs in GHz)
fluxdep_fit_result             # {has_result, params:{EJ,EC,EL} or null, inputs…}
fluxdep_disconnect             # detach the bridge; does NOT stop the user's GUI
```

That is the whole agent surface: launch/connect/disconnect plus the five read
tools. The agent never mutates the analysis and never stops the GUI.

## Generating the search database

`script/generate_fluxonium_sample.py` builds a database: it samples `(EJ, EC, EL)`
points (rays through the `EJb × ECb × ELb` box) and computes each one's energy
levels vs flux. It uses the fast `calculate_energy_vs_flux` (~100x over the stock
scqubits sweep), pins BLAS to one thread, and parallelises across cores, so a 10k
"all" database runs in a **few minutes**, not hours.

```bash
# real run (back up any existing DB first — see WARNING below):
.venv/bin/python script/generate_fluxonium_sample.py \
    --output Database/simulation/fluxonium_all.h5 \
    --preset all --num-samples 10000 --overwrite

# tiny dry run (random energies, no scqubits) to check plumbing — fast:
.venv/bin/python script/generate_fluxonium_sample.py \
    --output /tmp/db.h5 --preset all --num-samples 8 --dry-run
```

Key flags: `--preset normal|integer|all` (named `(EJb, ECb, ELb)` boxes), or set
`--EJb/--ECb/--ELb min,max` per axis; `--num-samples`, `--cutoff`,
`--evals-count`, `--num-flux` (flux points over [0, 0.5], mirrored to [0, 1]);
`--plot` for a 3D scatter of the sampled directions (off by default); `--dry-run`
(writes a `*_dryrun.h5` sibling, never the real path); `--overwrite` (required to
replace an existing output). `--n-jobs` defaults to -1 (all cores) — BLAS is
pinned to 1 thread so per-row process workers parallelise cleanly (~5x on 8
cores); set `--n-jobs 1` for serial.

The shipped databases live in `Database/simulation/` (`fluxonium_all.h5`,
`fluxonium_int.h5`, `fluxonium_1.h5`); each has `fluxs` / `params` / `energies`
datasets.

> **WARNING — don't clobber a database you can't cheaply rebuild.** The output
> path must not exist unless `--overwrite` is passed. Before regenerating an
> existing DB, back it up first: `cp Database/simulation/fluxonium_all.h5 ~/backup/`.

## Gotchas

- **Read-only over RPC.** If asked to "run the analysis", the agent cannot — it
  launches the GUI and the user drives it. Report the state you can read; do not
  claim to have performed steps you cannot perform.
- **Spectrum frequencies are in GHz** everywhere (the loader converts the raw Hz
  axis to GHz, and `fluxdep_selection_pointcloud` returns GHz).
- **Launch vs connect** (both default port 8766): `fluxdep_launch` starts a NEW
  GUI for the user (needs the port free); `fluxdep_connect` attaches to one the
  user already started. There is no stop tool — the agent never closes the user's
  GUI; `fluxdep_disconnect` only detaches the bridge.
- **After editing GUI code**, `/mcp reconnect fluxdep-gui` then `fluxdep_launch`
  — the MCP server is a separate process and caches old code until reconnected.
