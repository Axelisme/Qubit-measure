---
name: measure-setup
description: Collect the stable, per-rig measurement setup (board connection, project identity, channel wiring, device addresses) into a single user-filled measure_setup.yaml at the repo root, so the measuring agent reads it instead of re-asking the user every session. Use before driving a measure-gui measurement when setup info (channel wiring, device addresses, chip/qub/res names, board IP/port) is needed and not already in the file.
skill_version: 3
---

# measure-setup

A single file, `measure_setup.yaml` at the repo root, holds the setup that only the
user can provide and that stays stable across sessions. You generate it once, the user
fills it, and from then on you read it instead of asking the same questions every
session.

## The format

This is the canonical template. When you create the file, write exactly this shape —
drop device blocks the user does not use, add channel/device entries the rig needs:

```yaml
# measure_setup.yaml — fill in the blanks, then tell the agent to continue.
# Filled values are reused next session; the agent reads this instead of re-asking.

connection:
  kind: remote          # remote | mock
  ip: 192.168.10.179    # ZCU216 board IP (ignored when kind=mock)
  port: 8887

project:
  chip_name: Q5_2D
  qub_name: Q1
  res_name: R1
  # result_dir / database_path: leave empty to use the default chip/qub layout

channels:               # role -> DAC/ADC channel index (your rig wiring)
  res_ch: 0             # resonator drive DAC
  ro_ch: 0              # readout ADC
  qub_4_5_ch: 1         # qubit 4->5 drive DAC (add/rename to match wiring)
  # lo_flux_ch: 15
  # qub_1_4_ch: 2

devices:                # instruments to connect; delete any block you do not use
  flux_yoko:
    type: YOKOGS200
    address: "USB0::0x0B21::0x0039::91WB18859::INSTR"
    mode: current       # current | voltage
    bind: flux_dev      # context-binding label (the flux source)
  jpa_yoko:
    type: YOKOGS200
    address: "USB0::0x0B21::0x0039::91T810992::INSTR"
    mode: current
  jpa_sgs:
    type: RohdeSchwarzSGS100A
    address: "TCPIP0::192.168.10.89::inst0::INSTR"

# OPTIONAL, task-specific. Add a block like this only when a task needs it
# (e.g. a flux_dep sweep). These are SUGGESTIONS / starting defaults only — NOT
# authoritative: the adapter's gui_adapter_guide recommended ranges and live
# tuning take precedence. Omit the whole block if you have nothing to suggest.
# flux_scan:
#   start: -4.0e-3        # flux device value (A for YOKOGS200 current mode)
#   stop:  4.0e-3
#   expts: 101            # points
#   freq_half_span_mhz: 5 # resonator window half-span around r_f
#   readout_gain: 0.005   # raise toward ~0.05 if SNR is poor (see the guide)
#   reverse_on_consecutive_runs: true   # swap start/stop each run to skip ramp-back
#   flx_half:             # (optional) calibrated half-flux sweet-spot device value
#   flx_int:              # (optional) calibrated integer-flux sweet-spot device value
```

## Workflow

1. **Look for `measure_setup.yaml` first.** If it exists and is filled, `Read` it and go
   straight to *Apply* — do not regenerate. This is the whole point: stop re-asking for
   setup you already have.

2. **If it is missing, create it.** `Write` the template above. Pre-fill anything you
   already know from the conversation (e.g. the chip/qub/res the user just named); leave
   physical wiring and instrument addresses as the example values or `<FILL>`
   placeholders. Tell the user to fill or correct it and say when done, then **stop and
   wait** — never guess wiring or addresses.

3. **⚠ When a new requirement means the file needs UPDATING (an extra channel, a new
   device), READ IT FIRST — never blind-overwrite.** The user may have already filled in
   part of it. Read the current content, keep every value the user entered, add only the
   new keys the requirement needs, then ask the user to fill just those new blanks.
   Overwriting with a fresh template would erase their work.

4. **Read back and check before applying.** `Read` the file. If any required field is
   still a `<FILL>` placeholder or empty, ask the user only for that field — fail fast,
   do not proceed with a hole.

5. **Apply, in this order** (order matters: a context binds a flux device, so the device
   must exist first):
   - `gui_soc_connect(kind, ip, port)` with `kind` from `connection` (a synchronous call — returns once the board is connected).
   - `gui_startup_apply(chip_name, qub_name, res_name)` — add result_dir/database_path only if the user set them.
   - For each `devices` block: configure it (`gui_device_setup_spec` shows the fields, then `gui_device_setup`).
   - `gui_context_new(bind_device=<the flux device name>)`, or `gui_context_use(label)` to reuse an existing context.
   - `gui_context_md_set_attrs(<the channels mapping>)`.
   - Confirm with `gui_state_check`: has_project / has_context / has_active_context / has_soc all true.

## Notes

- The file is the single source of truth for the setup the user provides. When wiring or
  instruments change, **edit the file** — do not go back to asking field by field.
- The `devices` section lists the instruments currently in use; add a block (with `type`
  plus its address/fields) when a new instrument joins the rig.
- **Task-driven expansion.** When a task needs information the file does not yet hold
  (e.g. switching from a resonator search to `flux_dep`), **edit the file to add the new
  keys first, then ask the user only for those new blanks** — never re-ask for fields the
  file already has. This is the whole point of the file: each task grows it once.
- **Optional task blocks are suggestions, not authority.** A block like `flux_scan` only
  records sensible starting defaults the user wants to keep around. The adapter's
  `gui_adapter_guide` recommended ranges and live tuning (inspecting the figure, adjusting
  window / gain / direction) always take precedence — treat these values as a starting
  point, not a constraint. Stable per-qubit calibration values (e.g. `flx_half`/`flx_int`)
  live in the GUI context (MetaDict); duplicating them here is optional convenience.
