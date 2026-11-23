**Readers:** operator, agent. Map the mean IQ response versus resonator probe carrier frequency and gain.

# MIST probe frequency and gain

For opening, refreshing and saving, see [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).

Each shot applies `optional reset → optional preparation → resonator probe → readout → relaxation wait`. The sweeps change the probe frequency and gain, not the readout settings. The map measures IQ displacement, not state population.

## Set the scan rectangle

- Bind the qubit and resonator. Select reset and preparation for the intended initial state. They start from the qubit's assigned reset and x180 modules; disable either to skip it.
- Verify the readout pulse, trigger and integration window. Confirm the probe generator and keep probe duration fixed when comparing maps.
- **Probe carrier frequency sweep** starts at resonator frequency ± 1.5 × linewidth, in MHz. Set actual carrier endpoints, not offsets.
- Begin the gain sweep in the weak-drive region. Choose stop gain and point spacing to cover and resolve the feature of interest.
- **Relax delay** starts at 5 × captured T1, or 150 µs if T1 is missing. Invalid T1 leaves the required wait blank. Choose a wait that limits history from earlier shots.

## Read the maps

Blank cells have no displayed data.

Run and Analyze subtract the complex reference averaged over all frequencies and the first 5% of gain points (at least one), then show the magnitude in a.u. Each map scales to its own data; compare colorbar values rather than colors across runs. There is no fit or per-cell error bar.

## Use the response map

Look for a band that persists across adjacent frequencies and gains. Extend the window if the band meets an edge. Rescan the feature at smaller steps, or use a single gain trace at a selected carrier. Confirm changes with classified readout before assigning populations. There is no writeback.

The `mist_drive_frequency` recipe rejects null or missing `freq_range` and `gain_range` at input validation. Omit `points` to keep both sweep counts. Null clears both Expts values; readiness stops before acquisition if overrides leave them blank. Analyze has no display parameters.
