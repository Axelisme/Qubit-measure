## Behavior

**Readers:** operator, agent. Calibrate the JPA flux device value from the GUI or MCP using g/e readout contrast. A pi pulse switches the qubit preparation between g and e. The run sweeps the JPA flux device value and measures the g/e signal difference to find the value that maximizes readout contrast. WARNING: this run commands the selected flux device on real hardware. Review that device and the sweep before running. Treat the seeded bounds as bring-up defaults only, and establish safe limits for the connected hardware.

## Parameter sources

Uses the qubit (`qubit/*`), resonator and jpa (`amplifier/jpa`) aliases. The flux bounds use `$jpa.flux`. Opening or resetting the form selects the optional JPA flux device from `flux_dev`; an unbound alias leaves bias control disabled. `flux_unit` starts as A, and the device receives the entered values unchanged. Missing required calibration marks the form Invalid.

Relax delay starts at 1*T1 using the parameters captured when you open or reset the form. T1 must be a finite positive real number; Boolean values count as invalid. Missing/null T1 or an unavailable parameter snapshot gives a 30 us initial delay. An invalid stored T1 leaves the required wait field empty, and parameter-source failures report an error. Edit the delay in the form as needed; the stored T1 stays unchanged.

## Module slots

Module slots use `$qubit.module.x180`, `$qubit.module.readout` and optional `$qubit.module.reset`. Empty pi/readout slots use the pi_pulse/readout templates. The pi pulse uses `$qubit.freq` and `$qubit.wiring.ch`; readout uses resonator frequency/wiring and `$resonator.time_of_fly` + 0.05. An empty reset slot disables reset. Missing calibration required by these templates marks the form Invalid.

## Typical writeback

Analysis proposes the signal-maximizing JPA flux device value as a draft for `$jpa.flux`, in `$jpa.flux_unit` coordinates. Accept the draft to update the stored parameter. Writeback changes stored calibration only; device settings and the current-source value stay unchanged.

## Recommended

Review the selected flux device and sweep bounds before every run. The seeded range around the current centre is only a starting point. Analysis picks the peak of the absolute signal difference.

**Last updated:** 2026-10-21
