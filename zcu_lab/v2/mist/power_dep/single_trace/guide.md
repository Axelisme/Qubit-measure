**Readers:** operator, agent. Run an averaged-IQ MIST gain sweep and check where the readout response changes.

# MIST probe-gain trace

## What the sweep measures

Open `mist/power_dep/single_trace`. MIST stands for measurement-induced
state transitions. This sweep looks for a change in the readout response
after a resonator drive. Each shot runs **optional reset → optional qubit
preparation → resonator probe → readout → relaxation wait**. Only the probe
gain changes across the sweep. The readout pulse stays fixed and supplies one
integrated complex IQ value per shot.

A change in mean IQ does not identify a state, a transition probability or a
leakage rate. Residual resonator signal and readout drift can also move mean
IQ. Use `singleshot/mist/power` with calibrated state discrimination to check
whether a candidate change in IQ tracks population.

## Set up and run

Open or choose **Defaults** after binding the qubit and its resonator.
Defaults capture the current values and replace draft edits. Review these
fields before Run:

| Field | Starting value and what to check |
| --- | --- |
| Reset | Uses the qubit's reset module if assigned. Turn it off to omit the reset before each shot. |
| Preparation pulse | Uses the qubit's x180 module if assigned. Turn it off to omit driven preparation, or select a calibrated pulse for the intended starting state. |
| Resonator probe | Starts from the resonator frequency and generator and a 1 µs constant waveform. Match the carrier to the resonance you want to drive. Length and gain control how strongly the probe excites the resonator. The sweep sets gain, so there is no separate probe gain field. |
| Readout | Uses the resonator's assigned readout, or starts from a pulse readout template with its frequency, wiring and time of flight. Use a calibrated pulse, trigger delay and integration window. The readout gain is distinct from the swept probe gain. |
| Probe gain sweep | Start and Stop are 0 and 1 a.u., with 151 points at first. Start at a verified low probe gain to get a reference. Check the pulse settings and restrict the range for the sample before running. Expts sets how many points, including the endpoints. |
| Reps / Rounds | Start at 1000 shots per gain point and 10 sweeps. Their product is the planned shots per point. |
| Relax delay | Starts at five times qubit T1. A missing T1 uses 30 µs, so the wait starts at 150 µs. A bad stored T1 can leave the field blank. Enter a wait in µs that lets the qubit relax before the next shot. |

Use one measured tone and one integrated ADC read per shot. A pulse
readout satisfies this when its ADC is set. A multi-tone readout does not.
In the pulse settings, Gen ch and RO ch route the DAC and ADC. Frequency
is the programmed carrier in MHz, not a detuning. Waveform sets the drive
envelope; its Length sets the pulse duration. RO length sets the ADC
integration window, and Trig offset places that window after the pulse
starts. For the pulse readout template, the trigger uses the time of flight
plus 0.015 µs. The 1 µs pulse leaves 0.015 µs on each side of a 0.97 µs
window at the ADC. NQZ selects the generator's Nyquist zone. Set Mixer freq
to the generator mixer frequency, or leave it blank without a mixer. Phase
sets the carrier phase.
Pre-delay places the pulse later, and Post-delay waits after the pulse.
The classifier does not affect this averaged-IQ trace.

Fill any remaining required blanks. Run shows the IQ displacement as shots
accumulate. The number in the live title is the number of shots acquired per
gain point, not the number of gain points. The joined dots show the progress
of the sweep; the line between them is not a fit.

## Read the trace

Both Run and the default Analyze figure use the first measured gain point as
the complex IQ reference. Each dot shows the magnitude of the difference
between that reference and the mean IQ at the dot's gain, in a.u. The first
point is zero by definition. Analyze shows just the measured points and
error bars. Its summary identifies the source run and that run's configured
Reps × Rounds, not the current Form settings.

Each bar is one shot standard error of the mean (SEM). The analysis projects
the individual IQ fluctuations along the direction from the reference to
that point's mean IQ, then uses their sample standard deviation divided
by the square root of the shot count. This first-order estimate treats the
reference as fixed. It does not include uncertainty in a measured reference,
and it is less useful when the bar is large compared with the displacement.
There is no bar at zero displacement, where the direction is undefined,
at points with fewer than two shots, or on old mean-only files. A zero or
very small SEM can be hidden by its dot. A symmetric bar can extend below
zero, even though the measured magnitude cannot.

Find a rise that spans adjacent gains and repeats when you rerun the sweep.
Do not treat one point with a large bar as an onset. If a rise starts between
two gains, repeat with those gains as Start and Stop and a smaller step.
For example, 21 points between 0.12 and 0.14 a.u. give a nominal 0.001 a.u.
step. If the whole trace is flat, the sweep has not located an onset. If
the reference looks suspect, repeat with a lower Start or use an independently
measured reference. Repeat with more shots if the bars obscure the change.
This analysis does not fit or write back an onset.

## Change the Analyze display

- **Reference IQ (a.u.)** sets the complex reference. Enter a value such as
  `1+2j` measured with the same readout, or leave it blank to use the first point.
- **Comparison IQ (a.u.)** sets the y-axis upper limit to 1.1 times its distance
  from Reference IQ. It does not change the dots or bar lengths and may hide
  points above the limit. Leave it blank for automatic limits.
- **Photon number / gain squared** replaces the gain axis with the coefficient
  times gain squared. Leave it blank without an independent calibration.
  Use one sign of gain for this view, since opposite signs give the same
  photon number.

Save All stores mean complex IQ, gain coordinates and any available shot
moments with the run. Load leaves the current Form alone. Analyze uses the
loaded record and omits the acquisition line if the loaded configuration was
unavailable.

The `mist_power_trace` recipe opens or resets the same Form, can select
Resonator probe and Readout modules, and uses that Form's gain grid and averaging.
Use overrides to set a different grid or shot count. The recipe runs the
sweep, analyzes it and saves the result without updating calibration.

An FTF simulator trace can show probe-dependent IQ without modeling MIST
transitions. Do not use it to choose a hardware onset.
