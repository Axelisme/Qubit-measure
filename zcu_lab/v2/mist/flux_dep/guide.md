# MIST over flux

## Prepare and open

Open `mist/flux_dep` from the measure catalog. Calibrate the resonator and
readout first. Bind `qubit`, `resonator` and `flux_dev` before Open or Reset.
Open/Reset copies the flux device alias into a direct device name. Later alias
changes do not change this form.

Reset and the initial x180 pulse are optional library selections. Missing slots
disable them. The probe starts from the resonator-probe template; readout uses
the captured readout slot when present, otherwise its template. Review probe
frequency, generator, duration and readout settings before Run.

## Review and run

Flux starts at the qubit's integer-flux device value and ends at its half-flux
value, with 101 points. Missing calibration leaves the sweep invalid. Gain starts
at 0 and ends at 0.5, with 101 points. These are editable starting values, not
certified safety limits. Relaxation starts at five T1, using the shared 30 us
scale only when T1 is missing. Reps/rounds start at 1000/1.

Check the direct flux device name and set flux_unit to its actual A, V or native
coordinate. No unit conversion occurs. Review device limits and both sweep
directions before Run. The host sets each bias, then the board resets, applies
the optional init pulse, applies the gain-swept probe and reads IQ.
The device remains at the last acquired bias; Run does not return it to zero.

## Analyze and save

The live map subtracts each flux row's mean IQ over its first 5% of gains
(at least one point), takes the magnitude and clips strong outliers for display.
The saved data retains raw complex IQ, native flux values and hardware-rounded
gains, not the display transform.

Primary analysis creates the native `fit` map. By default it shows device values
and probe gain. Enter both half-flux device value and flux period to convert the
horizontal axis. Enable the secondary device-value axis only with both values.
Photon coefficient optionally maps gain squared to photon number.
These analysis-only values are explicit inputs, not readings of current aliases.
Analysis also works on loaded data without a usable historical cfg.

There is no fitted scalar or writeback proposal. Inspect the map as a response,
not a calibrated state-population measurement. Save raw data and the analysis
figure through the normal Data/Save controls. Native Load retains its recorded
source and does not replace the current form. The existing mist_fluxdep workflow
is a separate process and is unchanged.

The `mist_flux_dep` recipe runs one map with explicit native flux/gain ranges
and the captured probe/readout selections. It saves raw data and Primary analysis
without proposing calibration writes.
