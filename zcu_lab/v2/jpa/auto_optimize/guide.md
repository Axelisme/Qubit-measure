## Behavior

JPA joint optimization: with the qubit prepared in g and e (a pi pulse toggles it), a multi-phase optimizer searches the JPA flux device value, pump frequency and pump power jointly to maximize the g/e signal difference. Runs on real hardware and commands the selected RF and flux devices at every evaluated point, leaving them at the last evaluated point. WARNING: review the selected devices and the search bounds before running — the seeded bounds are bring-up defaults, not certified safety limits.

## Expects (parameter context)

Uses current aliases: qubit (qubit/*), resonator (resonator), and jpa (amplifier/jpa). Add a jpa alias row in the global dialog, for example `$resonator.amplifier`. Scalars use `$jpa.pump_freq`, `$jpa.pump_power`, `$jpa.flux` and canonical qubit/resonator wiring. Missing aliases or calibration leave cfg Invalid; no static fallback.

## Expects (ModuleLibrary)

SMART modules use `$qubit.module.x180`, `$qubit.module.readout` and optional `$qubit.module.reset`. Inline readout uses resonator scalar and wiring calibration; module waveforms follow their module source.

## Typical writeback

Proposes the signal-maximizing flux device value, pump frequency and pump power into three separate parameter context drafts — 'best_jpa_flux', 'best_jpa_freq' (MHz) and 'best_jpa_power' (dBm) — so you can accept, cancel or retarget each one independently. Nothing is written without your acceptance, writeback never commands a device, and it never touches 'cur_jpa_A'.

## Recommended

For each sweep, start/stop are SEARCH BOUNDS and expts is a RELATIVE RESOLUTION HINT — expts is not a Cartesian or exact sample count. The optimizer compares all three hints when allocating its phase-1 flux slices and points per slice. 'Global optimizer points' (num_points, at least 4, default 1001) remains the total evaluation budget: raise it for wider bounds, lower it for a quick search. Analysis reports the best evaluated point with no tuning knobs.
