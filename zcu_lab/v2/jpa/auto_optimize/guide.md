## Behavior

**Readers:** operator, agent. Run joint JPA optimization from the GUI or MCP and interpret the results. A pi pulse switches the qubit preparation between g and e. The optimizer searches the JPA flux device value, pump frequency and pump power jointly to maximize the g/e signal difference. Each evaluated point commands the selected RF and flux devices on real hardware. The devices remain at the last evaluated point. WARNING: review the selected devices and search bounds before running. Treat the seeded bounds as bring-up defaults only, and establish safe limits for the connected hardware.

## Parameter sources

Uses the qubit (`qubit/*`), resonator and jpa (`amplifier/jpa`) aliases. Search bounds use `$jpa.flux`, `$jpa.pump_freq` and `$jpa.pump_power`. Opening or resetting the form reads the required JPA flux device from `$jpa.flux_dev` and the RF default from `$jpa.rf_source`. Both direct names remain editable. A missing flux or RF source leaves its field empty and marks the form Invalid. Select the connected RF device in the form. `flux_unit` starts as A, and the device receives the entered values unchanged. Missing required aliases or calibration marks the form Invalid.

Relax delay starts at 1*T1 using the parameters captured when you open or reset the form. T1 must be a finite positive real number; Boolean values count as invalid. Missing/null T1 or an unavailable parameter snapshot gives a 30 us initial delay. An invalid stored T1 leaves the required wait field empty, and parameter-source failures report an error. Edit the delay in the form as needed; the stored T1 stays unchanged.

## Module slots

Module slots use `$qubit.module.x180`, `$resonator.module.readout` and optional `$qubit.module.reset`. Empty pi/readout slots use the pi_pulse/readout templates. The pi pulse uses `$qubit.freq` and `$qubit.wiring.ch`; readout uses resonator frequency/wiring and `$resonator.time_of_fly` + 0.05. An empty reset slot disables reset. Missing calibration required by these templates marks the form Invalid.

## Typical writeback

Analysis proposes the signal-maximizing settings as three separate drafts: `$jpa.flux` in `$jpa.flux_unit` coordinates, `$jpa.pump_freq` in MHz and `$jpa.pump_power` in dBm. You can accept, cancel or retarget each draft independently. Accept a draft to update its stored parameter. Writeback changes stored parameters only; device settings and the current-source value stay unchanged.

## Recommended

For each sweep, start/stop set the search bounds and expts gives a relative resolution hint. The optimizer compares all three hints to allocate its phase-1 flux slices and points per slice; the actual sample counts follow that allocation. Global optimizer points (`num_points`, at least 4, default 1001) sets the total evaluation budget. Raise it for wider bounds or lower it for a quick search. Analysis reports the best evaluated point using fixed settings.

**Last updated:** 2026-10-21
