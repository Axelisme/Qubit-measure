## Behavior

JPA joint optimization: with the qubit prepared in g and e (a pi pulse toggles it), a multi-phase optimizer searches the JPA flux device value, pump frequency and pump power jointly to maximize the g/e signal difference. Runs on real hardware and commands the selected RF and flux devices at every evaluated point, leaving them at the last evaluated point. WARNING: review the selected devices and the search bounds before running — the seeded bounds are bring-up defaults, not certified safety limits.

## Expects (parameter context)

Uses the qubit (qubit/*), resonator and jpa (amplifier/jpa) aliases. Search bounds use `$jpa.flux`, `$jpa.pump_freq` and `$jpa.pump_power.` The required JPA flux device resolves flux_dev once at Open or Reset; select the connected RF device in the form. RF defaults capture jpa_rf_dev at Open or Reset; an unbound alias leaves the optional field empty. flux_unit starts as A with no conversion. Missing required aliases or calibration leave the form Invalid.

## Expects (ModuleLibrary)

Module slots use `$qubit.module.x180`, `$qubit.module.readout` and optional `$qubit.module.reset.` Missing pi/readout slots fall back to the pi_pulse/readout templates. Inline pulse calibration uses `$qubit.freq` and `$qubit.wiring.L01.ch`; inline readout uses resonator frequency/wiring and `$resonator.ext.time_of_fly` + 0.05. An unset reset slot disables reset. Missing required fallback calibration leaves the form Invalid.

## Typical writeback

Proposes the signal-maximizing flux device value, pump frequency and pump power into three separate drafts: `$jpa.flux` in `$jpa.flux_unit` coordinates, `$jpa.pump_freq` (MHz) and `$jpa.pump_power` (dBm). You can accept, cancel or retarget each independently. Nothing is written without your acceptance. Writeback never commands a device or changes the current-source value.

## Recommended

For each sweep, start/stop are SEARCH BOUNDS and expts is a RELATIVE RESOLUTION HINT — expts is not a Cartesian or exact sample count. The optimizer compares all three hints when allocating its phase-1 flux slices and points per slice. 'Global optimizer points' (num_points, at least 4, default 1001) remains the total evaluation budget: raise it for wider bounds, lower it for a quick search. Analysis reports the best evaluated point with no tuning knobs.
