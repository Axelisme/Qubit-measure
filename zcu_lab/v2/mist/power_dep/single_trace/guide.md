**Readers:** operator, agent. Measure how mean readout IQ changes with resonator probe gain.

# MIST probe-gain trace

For opening, refreshing and saving, see [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).

Each shot applies `optional reset → optional preparation → gain-swept resonator probe → readout → relaxation wait`. The probe carrier and duration stay fixed. This trace measures IQ displacement and does not classify states.

## Set the gain window

- Bind the qubit and resonator. Reset and preparation start from assigned reset and x180 modules. Missing optional slots remain disabled; select calibrated modules for the preparation you want to test.
- Confirm the resonator probe frequency, duration and gain endpoints. Start in a weak-drive region so the first points provide a reference.
- Verify the readout module, or supply Inline frequency, wiring and flight time for the fallback template. Check the readout window and fill required blanks before Run.
- **Relax delay** starts at 5 × captured T1, or 150 µs if T1 is missing. Invalid T1 leaves the required wait blank. Choose a wait that limits history from earlier shots.

## Read the trace

Run shows the magnitude of mean IQ displacement from the mean of the first 5% of gain points, with at least one point. Analyze shows the `fit` figure, but it has no model fit. Its default reference is the first measured IQ, so its displacement can differ from Run. Neither plot shows shot SEM.

**Ground IQ** sets the reference and **Excited IQ** sets only the top of the vertical display, at 1.1 × their separation. Enter references as complex values such as `1+2j`. Leave them blank for the first-point reference and automatic y limits. **Photon number / gain squared** sets `n = coefficient × gain²`; use an independently checked AC-Stark calibration, or leave it blank for gain units.

## Use the response

Check for a response across adjacent gain points before narrowing the window. Use [`singleshot/mist/power`](../../../singleshot/mist/power/guide.md) to measure classified populations. This trace proposes no writeback.

The `mist_power_trace` recipe uses the fresh Form's gain sweep and optional reset/preparation, including when reusing a tab. Omit `probe_ref` and `readout_ref` to keep their sources. Null passes input validation but fails at either required Form selector before Run.
Omit `g0`, `e0` or `ac_coeff` to keep the corresponding Analyze setting. Null selects first-point IQ, automatic y limits, or gain units, respectively. Invalid complex text fails in Analyze after Run. The recipe still attempts Save.
