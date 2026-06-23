---
name: run-measure-gui
description: Run, drive, screenshot, and smoke-test the measure-gui qubit-measurement GUI over its MCP control socket. Use when asked to launch/start/test the measure-gui app, drive a single-qubit measurement (lookback, onetone/twotone spectroscopy, Rabi, T1/T2, readout optimization) via the measure-gui MCP tools, take a GUI screenshot, or follow the recommended experiment flow.
skill_version: 43
---

# run-measure-gui

`measure-gui` is a Qt desktop GUI for superconducting-qubit (Fluxonium)
measurement on a ZCU216 FPGA. It is driven headlessly through the `measure-gui`
MCP server and the `mcp__measure-gui__gui_*` tools. There is no separate driver
to write; the tool sequence is the harness.

**You are an operator, not a developer.** Drive measurements through MCP tools;
do **not** read or edit the repo's source code while operating the GUI. To
resolve suspicious data, re-measure: widen the sweep, re-run, and inspect the
figure. Trust measured output and user physics judgement, not implementation
inspection or simulator truth.

Paths are relative to the repo root (`<repo>` = the directory with `.mcp.json`).
This skill is intentionally thin: load the referenced files only when their
route applies.

## Required Reference Routing

Before acting, read the rows that match the request:

| Situation | Read |
|---|---|
| Launch, attach, setup project/context/SoC, open tabs, edit cfg, run/analyze/save/writeback | `references/startup-and-tools.md` |
| Any measurement that will be judged or written back | `references/acceptance-memory.md` |
| Interactive analysis, long waits, post-analysis, user feedback, or prompt dialogs | `references/async-interaction.md` |
| Offline GUI loop verification without MCP | `references/smoke.md` |
| Single-qubit bring-up, flux finding, resonator/qubit frequency search, Rabi/T1/T2 sequencing | `references/single-qubit-flow.md` |
| Errors, stale GUI, bad fit, save/cfg/device confusion, or suspicious spectra | `references/gotchas-troubleshooting.md` |
| Meaning of `r_f`, `q_f`, `pi_gain`, `flx_int`, module names, and cross-experiment md/ml keys | `magic_names.md` |

If a referenced file is missing, stop and report the packaging error instead of
improvising from memory.

## Before touching real hardware — READ THIS

> **Mock SoC / `FakeDevice` flow? Skip this section.** It is safe and offline —
> none of the hardware-safety rules below apply. They kick in only when a real
> `YOKOGS200` / `SGS100A` is involved.

You drive **software only**. You cannot see the cabling, the sample, or the
fridge. So on a real (non-mock) session you **must get hardware facts from the
user first**, and you **must respect device safety limits**.

- **⚠️ YOKOGS200 in current mode: keep the value within ±7e-3 A.** Exceeding it
  can **physically destroy the instrument**. `gui_device_apply(name,
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

## Non-Negotiable Measurement Policy

- Start every attachment with `gui_overview`; do not assume the project, context,
  SoC, active tab, or running state.
- Before a measurement, use `agent-memory`: call `memory_recall(chip, qub,
  exp_type)`, read checklist/gotchas/recent records, and search memory before
  improvising on known symptoms.
- After analyze and before writeback, inspect the figure and write an acceptance
  record. The gate is self-grading evidence, **not** an automatic fidelity
  threshold. Writeback responsibility stays with the agent/human reviewing the
  figure and preview.
- Mutating MCP calls are side-effecting: issue each run/analyze/device/writeback
  operation exactly once, then drive the returned handle or read the typed result.
- A finished run is not proof of a good fit. Inspect the PNG figure before
  trusting numbers or applying writeback.
- Do not assume flux position or qubit frequency in mock or real flows. Find
  integer flux with `onetone/flux_dep`, move there, re-measure `onetone/freq`,
  then run wide `twotone/freq` followed by narrow `twotone/freq` around the
  measured feature.
- Do not use `twotone/flux_dep` early to find flux. Use it later only after
  `r_f`, `q_f`, and drive/readout settings are already credible.
- Predictor output and hidden mock/simulator parameters are non-authoritative.
  They never replace measured features and never choose writeback values.

## Minimal Operator Loop

1. Read `references/startup-and-tools.md` and call `gui_overview`.
2. Bring readiness flags true: project, context, active file-backed context, SoC.
3. Read the adapter guide via `gui_tab_open(adapter_name)` or
   `gui_adapter_guide(adapter_name)`.
4. Read `references/acceptance-memory.md`; call `memory_recall` before the run.
5. Configure and run through the four bundle tools when possible:
   `gui_tab_open` → `gui_tab_run` → `gui_tab_analyze_review` → `gui_tab_commit`.
6. When a run or analyze degrades to a handle, follow
   `references/async-interaction.md`.
7. Record the acceptance verdict and apply only the safe writeback subset.
8. On confusing output, read `references/gotchas-troubleshooting.md`, inspect the
   figure, and ask the user when physical interpretation is required.

## Smoke Harness

A standalone smoke driver (`smoke.py`, next to this file) talks the same socket
directly and runs the loop against a **mock SoC**. Read `references/smoke.md` when
asked to verify GUI launch/loop behavior on a fresh checkout.
