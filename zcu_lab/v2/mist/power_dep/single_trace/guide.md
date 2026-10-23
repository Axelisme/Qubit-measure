# MIST probe-gain trace

**Readers:** operator, agent. Measure an averaged-IQ MIST trace over probe gain from the GUI or MCP.

**Last updated:** 2026-10-21

## Measurement

Open `mist/power_dep/single_trace` for an averaged-IQ MIST trace. Each sequence
applies optional reset and initialization, then a gain-swept probe and readout.
The board scans probe gain; probe frequency and duration stay fixed. Use
`singleshot/mist/power` for classified-population measurements.

## Starting configuration

Open and Reset capture aliases, parameter sources and module selections.
Reset and initialization use optional `$qubit.module.reset` and
`$qubit.module.x180` slots.
Missing optional slots stay disabled. Disable initialization for ground-state
preparation, or select a calibrated pulse to prepare another state.

Probe starts from the `res_probe` template using `$resonator.freq` and
`$resonator.wiring.ch`, with a 1 us constant pulse. Readout uses
`$qubit.module.readout` when present, otherwise the `readout` template. The
readout template needs resonator frequency, generator/ADC wiring and
`$resonator.time_of_fly`. Missing required sources stay Invalid; select a
valid module or fill the inline fields before Run.

The initial gain range is 0 to 1 with 151 points. Reps starts at 1000 and Rounds
at 100. Relax delay starts at 5 times the captured `$qubit.t1`. Missing/null T1
uses 30 us, giving a 150 us wait; invalid T1 leaves the required wait empty.
These are editable starting values. Set safe ranges using the limits of your
hardware. Review gain, frequency, pulse length, readout and state preparation
before running.

## Plot and analysis

The live `measurement` plot subtracts the mean IQ over the first 5% of gain
points and shows the displacement magnitude. Analyze publishes a response
figure named `fit`. This figure shows IQ displacement; use a
classified-population experiment for population estimates. Its default ground
reference is the first measured IQ, so it can differ from the live plot.

Optional Ground IQ and Excited IQ accept complex strings such as `1+2j`. Ground
IQ sets the displacement reference. Excited IQ only sets the vertical range to
1.1 times the reference separation. Leave both empty to use the first point and
automatic y limits. Photon number / gain squared is an optional coefficient for
`n = coefficient * gain**2`; leave it empty for a gain axis. Enter a coefficient
from an independently checked AC-Stark calibration when needed. Analysis uses
the acquired or loaded data. Calibration parameters remain unchanged.

## Save and repeat

Data → Save All saves the native complex IQ and gain coordinates with the Run
configuration, plus available figures and analysis. Native data uses the
`mist/gain` tag and `Drive Power` axis in a.u. Load preserves the current form
and uses the saved record for analysis. Analyze also accepts records with
missing historical configuration.

The `mist_power_trace` recipe opens or resets this measurement, optionally
selects probe/readout modules, runs, saves raw IQ and analyzes. It uses the
reset form's reset/init choices and gain defaults, including when reusing a tab.
Reuse discards previous configuration edits. Run directly in the GUI for custom
acquisition settings. The recipe leaves calibration parameters unchanged.

Offline `white_noise` exercises the pulse sequence, sweep, readout and saving
with a simulated device. Its output is noise only. Establish MIST onset and
calibrated photon number with a suitable physical model or hardware
measurement.

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Set probe gain endpoints and verify probe/readout timing. Supply GE calibration where the Form requests it.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, Reset to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, Reset and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `mist_power_trace` after inspecting the same Form errors.
   The captured Form supplies gain Expts. probe_ref/readout_ref select modules. g0, e0 and ac_coeff update the existing display analysis.
   The Form chooses module sources. Omitted inputs retain Form values; null clears/disables fields. Overrides apply last, then check_ready reports missing or invalid fields before Run.
   Missing inputs return needs_parameters before acquisition. Reset may replace unsaved inputs.
