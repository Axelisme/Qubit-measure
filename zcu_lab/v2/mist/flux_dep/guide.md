# MIST over flux

**Readers:** operator, agent. Measure the MIST response over flux and probe gain from the GUI or MCP.

**Last updated:** 2026-10-21

## Prepare and open

Open `mist/flux_dep` from the measure catalog. Calibrate the resonator and
readout first. Bind `qubit`, `resonator` and `flux_dev` before Open or Reset.
Open/Reset copies the flux device alias into a direct device name. The form
keeps that name until the next Open or Reset.

Reset and the initial x180 pulse use optional module slots. Missing slots
leave those pulses disabled. The probe starts from the resonator-probe template.
Readout uses the captured readout slot when present, otherwise its template.
Review probe frequency, generator, duration and readout settings before Run.

## Review and run

Flux starts at the qubit's integer-flux device value and ends at its half-flux
value, with 101 points. Missing calibration leaves the sweep invalid. Gain starts
at 0 and ends at 0.5, with 101 points. These are editable starting values.
Set safe ranges using the limits of your hardware. Relaxation starts at five T1,
using a 30 us scale only when T1 is missing. Reps/rounds start at 1000/10.

Check the direct flux device name and set flux_unit to its actual A, V or native
coordinate. The sweep sends values directly in that coordinate. Review device
limits and both sweep directions before Run. The host sets each bias, then the
board resets, applies the optional init pulse, applies the gain-swept probe and
reads IQ.
The device remains at the last acquired bias after Run. Set a safe bias
explicitly when the measurement finishes.

## Analyze and save

The live map subtracts each flux row's mean IQ over its first 5% of gains
(at least one point), takes the magnitude and clips strong outliers for display.
The saved data retains raw complex IQ, native flux values and hardware-rounded
gains. The transform affects only the display.

Primary analysis creates the native `fit` map. By default it shows device values
and probe gain. Enter both half-flux device value and flux period to convert the
horizontal axis. Enable the secondary device-value axis only with both values.
Photon coefficient optionally maps gain squared to photon number.
Enter these analysis-only values explicitly. Analysis uses those inputs and
the acquired or loaded data, including records with missing historical
configuration.

Analysis produces a response map, with calibration parameters left unchanged.
Use a classified-population experiment for calibrated state populations. Save
raw data and the analysis figure through the normal Data/Save controls. Native
Load keeps the saved record as the data source and preserves the current form.
The `mist_fluxdep` workflow is a separate measurement procedure.

The `mist_flux_dep` recipe runs one map with explicit native flux/gain ranges
and the captured probe/readout selections. It saves raw data and Primary analysis
and leaves calibration parameters unchanged.
