# MIST probe-gain trace

## Measurement

Open `mist/power_dep/single_trace` for an averaged-IQ MIST trace. Each sequence
applies optional reset and initialization, then a gain-swept probe and readout.
The board scans probe gain; probe frequency and duration stay fixed. This is not
the classified-population experiment `singleshot/mist/power`.

## Starting configuration

Open and Reset capture aliases, parameters and library modules. Reset and
initialization use optional `$qubit.module.reset` and `$qubit.module.x180` slots.
Missing optional slots stay disabled. Disable initialization for ground-state
preparation, or select a calibrated pulse to prepare another state.

Probe starts from the `res_probe` template using `$resonator.freq` and
`$resonator.wiring.ch`, with a 1 us constant pulse. Readout uses
`$qubit.module.readout` when present, otherwise the `readout` template. The
readout template needs resonator frequency, generator/ADC wiring and
`$resonator.ext.time_of_fly`. Missing required sources stay Invalid; select a
valid library module or fill the inline fields before Run.

The initial gain range is 0 to 1 with 151 points. Reps starts at 1000 and Rounds
at 100. Relax delay starts at 5 times the captured `$qubit.t1`. Missing/null T1
uses 30 us, giving a 150 us wait; invalid T1 leaves the required wait empty.
These are editable starting values, not hardware-safety limits. Review gain,
frequency, pulse length, readout and state preparation before running.

## Plot and analysis

The live `measurement` plot subtracts the mean IQ over the first 5% of gain
points and shows the displacement magnitude. Analyze publishes a `fit` figure,
but does not fit a model or estimate populations. Its default ground reference
is the first measured IQ, so it can differ from the live plot.

Optional Ground IQ and Excited IQ accept complex strings such as `1+2j`. Ground
IQ sets the displacement reference. Excited IQ only sets the vertical range to
1.1 times the reference separation. Leave both empty to use the first point and
automatic y limits. Photon number / gain squared is an optional coefficient for
`n = coefficient * gain**2`; leave it empty for a gain axis. Enter a coefficient
from an independently checked AC-Stark calibration when needed. Analysis uses
the acquired or loaded source, never the current form. There is no writeback.

## Save and repeat

Data → Save All saves the native complex IQ and gain coordinates with the Run
cfg, plus available figures and analysis. Native data keeps the existing
`mist/gain` tag and `Drive Power` axis in a.u. Load retains the current form;
it does not replace it with historical cfg. Analyze also works without a usable
historical cfg.

The `mist_power_trace` recipe opens or resets this adapter, optionally selects
probe/readout libraries, runs, saves raw IQ and analyzes. It uses fresh Form
reset/init choices and gain defaults, including when reusing a tab. Reuse
discards previous cfg edits. Run directly in the GUI for custom acquisition
settings. The recipe makes no calibration proposals.

Offline `white_noise` supports the pulse/sweep/readout pipeline, but has no MIST
or pulse-response model. A successful noise run checks plumbing, not MIST onset
or calibrated photon number. No hardware is needed for that check.
