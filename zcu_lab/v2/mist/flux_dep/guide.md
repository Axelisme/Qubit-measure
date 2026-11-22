**Readers:** operator, agent. Map resonator probe response over DC flux bias and probe gain.

# MIST over flux

Open `mist/flux_dep` to find where a gain-dependent readout response changes
with flux bias. At each bias the pulse sequence is optional reset, optional
qubit preparation, the gain-swept resonator probe, readout, then a relaxation
wait. The bias holds still during each gain sweep. The readout sees mean IQ,
not individually classified qubit states.

## Set up and run

Bind the qubit and resonator. Open the experiment or choose **Defaults**
to recapture their values and discard draft edits.

- **Flux device** starts from the chosen qubit's DC source. Confirm its name
  and **Flux-bias unit**. The sweep sends values directly in A, V or native
  device units, with no conversion.
- **Flux-bias sweep** starts from calibrated integer flux to half flux with
  101 points, including endpoints. If the calibration is missing, fill
  Start and Stop in the selected device unit. Confirm both endpoints.
- **Reset** and **Preparation pulse** use the qubit's reset and x180 modules
  if assigned. Turn off either one to omit it, or select a calibrated module
  for the intended preparation.
- **Resonator probe** starts from the resonator's frequency and generator
  and a 1 µs constant envelope. **Probe gain sweep** replaces its gain. It
  starts at 0–0.5 a.u. with 101 points. Choose a low-gain starting region
  for the reference and confirm the stop gain before Run.
- **Readout** uses the resonator's assigned module or the pulse readout
  template. Check its frequency, gain, timing and integration window.
- **Reps** starts at 1000 shots per gain and bias in one round. **Rounds**
  starts at 10 sweeps per bias. **Relax delay** starts at five times the qubit
  T1. Missing T1 uses 30 µs, giving 150 µs. An invalid T1 leaves the wait blank.

The device stays at the last applied bias. Restore your working bias explicitly
after the scan.

## Read the figures

In Run, the left map shows how many bias rows have IQ. Completed averaging
rounds appear in acquisition progress. The right panel shows the most recent
row in blue with its bias above the axis, and up to three earlier rows in
gray. Compare their rises to see whether the response changes smoothly as
the bias moves. Blank map cells have no displayed data.

Each bias row has its own complex IQ reference, the mean of its first 5% of
gain points with at least one point. The map and trace show the magnitude of
mean IQ shift from that reference, in a.u. To keep outliers from setting the
entire color scale, the display clips values at ten times the median of the
lower half of the displacement values. Equal colors at the upper limit can
therefore hide stronger responses. Save retains unmodified mean complex IQ.
The map does not use per-cell error bars and there is no fit.

Analyze shows the same map with optional axis calibrations. The summary
records the source run's Reps × Rounds. **Half-flux bias (device units)** and
**Flux period (device units / Φ₀)** start from the qubit calibration captured
for that Run. With both present, the bottom axis shows calibrated flux in
Φ₀ and the top shows the corresponding device values. Clear either to keep
only device units. **Photon number / gain squared** changes the vertical
axis to coefficient × gain². Leave it blank without a gain-to-photon
calibration. **Full acquired range** limits the view to acquired endpoints.

Find a band of changing response that repeats across nearby biases and gains.
Rescan a candidate window rather than choosing a single bright cell. If
the change spans 0.20–0.24 a.u., 41 gain points across that window give a
nominal 0.001 a.u. step. Compare the IQ map to a classified measurement before
assigning a state transition or population.

Layout reference: Ding Fig. 5-8(b), p. 151, uses a heatmap for frequency
and pulse length versus population. This map uses flux bias and probe gain
versus an IQ displacement.

Load uses the saved run, not today's settings. The `mist_flux_dep` recipe
runs one map using explicit native flux and gain ranges, saves and analyzes
it, and leaves calibration unchanged. The adaptive `mist_fluxdep` workflow
is a separate procedure.
