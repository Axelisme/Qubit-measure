---
name: run-fluxdep-gui
description: Run, drive, and smoke-test the fluxdep-gui — a standalone Qt GUI for fluxonium flux-dependence fitting (load spectrum hdf5 → pick half/integer flux lines → select spectral points → cross-spectrum filter → export spectrums.hdf5). Use when asked to launch/start/test the fluxdep-gui app, drive the flux-dependence analysis pipeline via its MCP tools, or follow the recommended analysis flow.
skill_version: 1
---

# run-fluxdep-gui

`fluxdep-gui` is a Qt desktop GUI for fluxonium **flux-dependence fitting** — an
optional analysis tool, fully independent of `measure-gui` (its own state /
services / MCP server / skill). It is driven headlessly through an MCP server
(`fluxdep-gui`, configured in `.mcp.json`) that launches the GUI subprocess and
relays a newline-delimited JSON RPC over a TCP control socket. **You drive it
with the `mcp__fluxdep-gui__fluxdep_*` tools** — the tool sequence *is* the
harness.

A standalone smoke driver (`smoke.py`, next to this file) talks the same socket
directly (wire method names, no MCP client) and runs the whole pipeline against
the converted real fixtures — use it to verify the GUI launches and the loop
works on a fresh checkout.

Paths below are relative to the repo root (the directory with `.mcp.json`).

## What this tool does (v1)

The pipeline, in order — each step feeds the next:

1. **Load** a raw spectrum hdf5 (a OneTone resonator-flux or TwoTone qubit-flux
   sweep) into the spectrum collection.
2. **Align** — pick the *half-flux* and *integer-flux* lines on the spectrum;
   this fixes `flux_period = 2·|int − half|` and the flux coordinate.
3. **Select points** — mark the spectral feature's (device-value, frequency)
   points. OneTone uses an automatic threshold; TwoTone uses a hand-painted brush
   mask (in the GUI) — over RPC you feed the points directly.
4. Repeat 1–3 to **accumulate several spectra**; a new spectrum can **inherit**
   an existing one's alignment as its starting guess.
5. **Cross-spectrum filter** — select/downsample the joint point cloud across all
   spectra.
6. **Export** `spectrums.hdf5`.

> Database search + scipy fit + result visualisation (EJ/EC/EL → params.json) are
> **deferred to v2** and not in this build.

## Prerequisites

- The project venv (`.venv`) with deps installed; Python is pinned to 3.9.
- An X display. On a headless box: `xvfb-run -a` prefix (or rely on the MCP
  server's offscreen launch). On a desktop session `DISPLAY` is already set.
- No hardware — this is a pure offline analysis tool.

## Run (agent path — MCP tools)

The MCP server auto-launches the GUI; you call tools.

```
fluxdep_launch                                  # spawns the GUI, connects (control port 8766)
fluxdep_project_setup(chip_name="Q3_2D",        # locate files (no hardware connection)
                      qub_name="Q2")
fluxdep_state_check                             # {has_project, spectrum_count, has_active}
```

Then the analysis loop:

```
fluxdep_spectrum_load(filepath="…/onetone_flux.hdf5", spec_type="OneTone") -> {name}
fluxdep_alignment_set(name, flux_half=0.0, flux_int=10.0)   # half/integer flux device values
fluxdep_points_set(name, dev_values=[...], freqs=[...])     # paired selected points (sorted by dev)
fluxdep_spectrum_load(filepath="…/twotone_flux.hdf5",
                      spec_type="TwoTone", inherit_from=name)  # inherit the first's alignment
fluxdep_alignment_set(name2, ...)
fluxdep_points_set(name2, ...)
fluxdep_spectrum_list                           # each spectrum's aligned / points_selected stage
fluxdep_selection_pointcloud                    # {fluxs, freqs} of the joint cloud
fluxdep_selection_set(selected=[true, ...])     # mask over the joint cloud (len = cloud size)
fluxdep_export_spectrums(filepath="out.hdf5") -> {path}
fluxdep_resources_versions                      # optimistic-concurrency version table
```

## Run (smoke harness — verify the loop without an MCP client)

```bash
# desktop session (DISPLAY set):
.venv/bin/python .claude/skills/run-fluxdep-gui/smoke.py
# headless:
xvfb-run -a .venv/bin/python .claude/skills/run-fluxdep-gui/smoke.py
```

Expected tail:

```
[smoke] connected on 8788
[smoke] loaded OneTone: onetone_flux_Q2_1.hdf5
[smoke] loaded TwoTone: twotone_flux_Q1_1.hdf5
[smoke] joint cloud: 5 points
[smoke] exported -> …/fluxdep_smoke_out.hdf5
[smoke] SMOKE OK
```

It uses control port **8788** so a live MCP session on the default 8766 is
undisturbed.

## Fixtures

The smoke fixtures (`fixtures/onetone_flux_Q2_1.hdf5`, `fixtures/twotone_flux_Q1_1.hdf5`)
are real Q3_2D spectra converted to the canonical axis layout. **They are
gitignored** (large `.hdf5` + under `Database/`), so on a fresh checkout that has
the raw `Database/Q3_2D/` files, regenerate them once:

```bash
.venv/bin/python .claude/skills/run-fluxdep-gui/make_fixtures.py
```

The raw files store their axes **transposed** (x=freq(Hz), y=flux) versus what
the loader expects (x=flux, y=freq(Hz)); `make_fixtures.py` re-saves them in the
canonical layout. Any raw hdf5 with that transpose needs the same fix first.

## Gotchas

- **`fluxdep_points_set` takes *paired* points**, not separate axes: `dev_values`
  and `freqs` must be equal length (one (dev, freq) point per index). A
  length mismatch returns `invalid_params`.
- **Alignment before points.** `points.set` derives each point's flux coordinate
  from the spectrum's alignment, so set the lines first.
- **The in-figure picking (drag lines / brush points) is a GUI-only human
  action.** Over RPC you feed `alignment.set` / `points.set` / `selection.set`
  numerically — that drives the same state, but it is *not* a simulation of the
  user dragging. Say so honestly when reporting a smoke run.
- **`spec_type` is not persisted** by `export_spectrums` (a known limitation of
  the underlying `dump_spectrums`): re-loading the exported file loses OneTone/
  TwoTone, which must be re-supplied.
- **Launch vs connect** (both default port 8766): `fluxdep_launch` starts a NEW
  GUI (needs the port free); `fluxdep_connect` attaches to an existing one.
  `fluxdep_stop` before relaunching.
- **After editing GUI code**, `/mcp reconnect fluxdep-gui` then `fluxdep_launch`
  — the MCP server is a separate process and caches old code until reconnected.
