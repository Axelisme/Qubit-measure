**Readers:** operator, agent. Map resonator probe response over DC flux bias and probe gain.

# MIST over flux

For opening, refreshing and saving, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).

At each fixed DC bias, a shot applies `optional reset → optional preparation → gain-swept resonator probe → readout → relaxation wait`. The map measures changes in mean IQ, not state populations.

## Set the bias and gain ranges

- Bind the qubit and resonator. Verify **Flux device** and its native unit. The sweep applies values in A, V or native device units without conversion. The initial bias range is the captured integer-to-half flux interval. Supply both endpoints if calibration is missing.
- Select reset and preparation for the state you want to probe. They start from assigned reset and x180 modules, if present. Disable either to skip it.
- Verify the readout pulse and integration timing. Check the resonator probe frequency and duration; set the gain sweep to begin in the weak-drive region, which supplies the IQ reference.
- **Relax delay** starts at 5 × captured T1, or 150 µs if T1 is missing. Invalid T1 leaves the required wait blank. Choose a wait that limits history from earlier shots.

The device stays at the last applied bias. Restore the working bias explicitly after the scan.

## Read the maps

Run shows bias rows with data and the latest gain trace in blue, with up to three earlier traces in gray. Blank cells have no displayed data.
Each row subtracts its own mean IQ over the first 5% of gain points (at least one) before taking the magnitude. The display caps displacement at ten times the median of its lower half, so the same top color can hide different responses. The maps have no per-cell error bars or fit.

Analyze shows the same displacement with optional axis calibrations. The two flux values start from the source Run's qubit calibration for the scanned device. Set both **Half-flux bias** and **Flux period** for Φ₀ below and device units above; clear either for native units only. Set **Photon number / gain squared** only with an independent calibration: `n = coefficient × gain²`. **Full acquired range** uses the acquired endpoints.

## Use the response map

Look for a band that persists across nearby gains and biases, then rescan that window. Compare against classified readout before assigning a transition or population. There is no writeback.

The `mist_flux_dep` recipe scans one map, not the adaptive `mist_fluxdep` workflow. Required `flux_device`, `flux_unit`, `flux_range` and `gain_range` reject null at input validation. After opening the tab and before Run, the recipe rejects a unit that does not match the device and equal endpoints on either axis. Omit `points` to keep the counts. Null clears both Expts values; readiness stops before acquisition if overrides leave them blank.
Omit `flux_half`, `flux_period`, `ac_coeff` or `auto_range` to keep their Analyze values. Null for either flux calibration or for `ac_coeff` leaves the corresponding axis uncalibrated. Null for `auto_range` is rejected after Run, at Analyze settings validation before analysis starts. The recipe still attempts Save.
