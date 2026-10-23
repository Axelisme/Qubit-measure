## Behavior

**Readers:** operator, agent. Calibrate the JPA pump power from the GUI or MCP using g/e readout contrast. A pi pulse switches the qubit preparation between g and e. The run sweeps pump power in dBm and measures the g/e signal difference to find the power that maximizes readout contrast. WARNING: this run commands the selected JPA RF device on real hardware. Review that device and the power sweep before running. Treat the seeded bounds as bring-up defaults only, and establish safe limits for the connected hardware.

## Parameter sources

Uses the qubit (`qubit/*`), resonator and jpa (`amplifier/jpa`) aliases. The pump-power bounds use `$jpa.pump_power`. Opening or resetting the form reads the RF default from `$jpa.rf_source`; a missing source leaves the field empty and Invalid. The direct device name remains editable. Select the connected JPA RF device in the form. Missing required aliases or calibration marks the form Invalid.

Relax delay starts at 1*T1 using the parameters captured when you open or reset the form. T1 must be a finite positive real number; Boolean values count as invalid. Missing/null T1 or an unavailable parameter snapshot gives a 30 us initial delay. An invalid stored T1 leaves the required wait field empty, and parameter-source failures report an error. Edit the delay in the form as needed; the stored T1 stays unchanged.

## Module slots

Module slots use `$qubit.module.x180`, `$qubit.module.readout` and optional `$qubit.module.reset`. Empty pi/readout slots use the pi_pulse/readout templates. The pi pulse uses `$qubit.freq` and `$qubit.wiring.ch`; readout uses resonator frequency/wiring and `$resonator.time_of_fly` + 0.05. An empty reset slot disables reset. Missing calibration required by these templates marks the form Invalid.

## Typical writeback

Analysis proposes the signal-maximizing JPA pump power as a draft for `$jpa.pump_power` in dBm. Accept the draft to update the stored parameter. Writeback changes stored calibration only; device settings and the current-source value stay unchanged.

## Recommended

Review the selected RF device and power sweep bounds before every run. The seeded range is a conservative low-power survey and only a starting point. Analysis picks the peak of the absolute signal difference.

**Last updated:** 2026-10-21
