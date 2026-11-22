# MIST over flux

**Readers:** operator, agent. Measure the MIST response over flux and probe gain from the GUI or MCP.

**Last updated:** 2026-11-12

## Prepare and open

Open `mist/flux_dep` from the measure catalog. Calibrate the resonator and
readout first. Bind `qubit` and `resonator` before opening the tab or choosing Defaults.
Open/Defaults reads `$qubit.flux_dev` into an editable direct device name. The form
keeps that name until the next Open or Defaults.

Reset and the initial x180 pulse use optional module slots. Missing slots
leave those pulses disabled. The probe starts from the resonator-probe template.
Readout uses the captured readout slot when present, otherwise its template.
Review probe frequency, generator, duration and readout settings before Run.

## Review and run

Flux starts at the qubit's integer-flux device value and ends at its half-flux
value, with 101 points. Missing calibration leaves the sweep invalid. Gain starts
at 0 and ends at 0.5, with 101 points. These are editable starting values.
Set safe ranges using the limits of your hardware. Relaxation starts at five T1,
using a 30 µs scale only when T1 is missing. Reps/rounds start at 1000/10.

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

Primary analysis creates the native `fit` map. The analysis controls are
half-flux device value, flux period, photon coefficient and full acquired range.
Half-flux and period start from the Run's captured qubit calibration for the
scanned device. Missing calibration or historical configuration leaves both blank.
You can edit or clear either value. With both values, the bottom axis shows φ
and the top ticks show device values. If either is blank, only the bottom
device-value axis is shown.
Photon coefficient optionally maps gain squared to photon number; otherwise
the vertical axis shows probe gain. Full acquired range uses the acquired
endpoints. These controls affect only analysis of the acquired or loaded data.

Analysis produces a response map, with calibration parameters left unchanged.
Use a classified-population experiment for calibrated state populations. Save
raw data and the analysis figure through the normal Data/Save controls. Native
Load keeps the saved record as the data source and preserves the current form.
The `mist_fluxdep` workflow is a separate measurement procedure.

The `mist_flux_dep` recipe runs one map with explicit native flux/gain ranges
and the captured probe/readout selections. It saves raw data and Primary analysis
and leaves calibration parameters unchanged.

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Enter native flux and gain endpoints. Confirm the cached flux unit and supply GE calibration where requested.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, choose Defaults to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, choose Defaults and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `mist_flux_dep` after inspecting the same Form errors.
   flux_device, flux_unit, flux_range and gain_range are required. points maps to Expts on both axes. flux_half, flux_period, ac_coeff and auto_range update only analysis.
   The Form chooses module sources. Omitted inputs retain Form values; null clears/disables fields. Overrides apply last, then check_ready reports missing or invalid fields before Run.
   Missing inputs return needs_parameters before acquisition. Defaults may replace unsaved inputs.
