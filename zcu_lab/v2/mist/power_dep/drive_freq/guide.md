**Readers:** operator, agent. Map how the mean IQ response after a resonator probe depends on its carrier frequency and gain.

# MIST probe frequency and gain

Open `mist/power_dep/drive_freq`. Each shot runs optional reset, optional
qubit preparation, the resonator probe, readout, then the relaxation wait.
The two sweep axes change the probe only. The readout carrier and gain stay
fixed. The map locates frequency and gain regions where the probe changes
the mean readout IQ. It does not determine which state was occupied.

## Set up and run

Bind the qubit and resonator, then open the experiment or choose **Defaults**
to capture current sources. Defaults discards your draft edits.

- **Reset** and **Preparation pulse** start from the qubit's reset and x180
  modules, respectively. Turn either off to skip that step. Select a
  preparation pulse for the initial state you want to test.
- **Resonator probe** starts from the resonator generator and a 1 µs constant
  envelope. The sweep replaces both the carrier and gain. Keep the probe
  length fixed when comparing maps.
- **Readout** uses the resonator's assigned module, or a pulse readout template
  with its frequency, wiring and time of flight. Confirm the pulse, trigger
  delay and integration window. Fill any required blanks before Run.
- **Probe carrier frequency sweep** starts at the resonator frequency
  ±1.5 linewidths, with 51 points. Start and Stop are carrier frequencies
  in MHz, not offsets. Set Expts to the number of frequency points.
- **Probe gain sweep** starts at 0–1 a.u. with 51 points. Pick a verified
  low-gain starting region and set the end and point count for the sample.
- **Reps** starts at 1000 shots per frequency and gain pair in each round.
  **Rounds** starts at 10 complete sweeps. **Relax delay** starts at five
  times the qubit T1. Missing T1 uses 30 µs, giving 150 µs. An invalid
  stored T1 leaves the required wait blank.

## Read the maps

The live title counts how many frequency/gain pairs have IQ data. Completed
rounds appear in acquisition progress. Blank cells have no displayed data.
Analyze uses the same axes and color scale convention. Each map scales its
colors to its own data, so compare the colorbar values rather than colors
between runs. The summary gives Reps × Rounds for the source run.

The color shows the magnitude of mean IQ shift in a.u. Both maps subtract
the same type of complex reference before taking the magnitude: the mean over
all frequencies and the first 5% of probe gain points, with at least one gain
point. Start at low gain so the reference samples the weak-drive response.
This is a two-dimensional map, with no per-point bars or fit.

Look for a band of increased displacement that persists over adjacent
frequencies and gains. If it reaches a frequency edge, extend that edge to
see the other side. To resolve a feature between 7189 and 7191 MHz, for
example, rescan that interval with 41 points for a nominal 0.05 MHz spacing.
Use the corresponding gain range for a finer map or run a single gain trace
at a chosen carrier. Check repeated changes with classified readout before
attributing them to MIST populations.

Layout reference: Ding Fig. 5-8(b), p. 151, uses a heatmap to locate features
in a two-axis scan. It sweeps frequency and pulse length and plots population.
This map sweeps probe frequency and gain and shows IQ displacement.

Save retains complex IQ and both sweep coordinates. Load does not change the
current Form. The `mist_drive_frequency` recipe runs and saves the map using
explicit carrier and gain endpoints. Its Analyze step has no display
parameters and writes back no calibration.
