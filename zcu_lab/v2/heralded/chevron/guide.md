# Heralded chevron

**Readers:** operator, agent. Compare the target CZ chevron with three spectator branches classified before the drive.

## Prepare the run

Select `heralded/chevron`. Bind Qubit1, Qubit2, mux and coupler as described in the [two-qubit setup](../../two_qubit/guide.md#綁定位置與讀出-tones).

- Choose calibrated Qubit1 and Qubit2 Xπ pulses and a full mux readout with GE Post circles and radii on both tones. Both reads use the same mux module. Check that its `post_delay` leaves enough time after herald before the drive, including the ADC integration window. The ADC window does not extend the programmed wait.
- Leave **Reset** blank or select an ordinary reset. This experiment acquires its own herald read and does not accept a herald reset mask.
- Use the coupler's `cz_qubit1` and `cz_qubit2` slots or fill the two drives in the form. Keep their signed gains and relative phase fixed. The common CZ frequency sweep replaces both carriers. The duration sweep changes the full envelope on both lines, not just the plateau. Use positive durations. See [ordinary chevron setup](../../two_qubit/chevron/guide.md#prepare-the-form) for choosing the sweep span and spacing.
- **Reps × Rounds** gives the attempted shots **per preparation** at each frequency and duration. A full scan attempts four times that many shots per point. Check retained counts if you stop early or a branch is blank.

## Read the result

Each shot applies optional reset, one nominal I/Xπ preparation, mux herald, simultaneous two-line drive, simultaneous double Xπ contrast, and terminal mux readout. The four nominal preparations get equal attempts, not equal retained counts. The measured herald class selects the analysis branch. It does not change any pulse.

| Measured herald (Qubit1, Qubit2) | Inferred drive input | Terminal bin counted as return |
| --- | --- | --- |
| gg | \|00⟩ | ee |
| ge | \|01⟩ | eg |
| eg | \|10⟩ | ge |
| ee (target) | \|11⟩ | gg |

Live and Analyze put all four chevrons on the same 0–1 color scale. For each cell, first pool counts across nominal preparations. Divide the return count by all terminal counts for that herald class, including terminal `other`. Gray cells have no measured probability. A cell with shots but no return hits has probability zero. Native Save retains all four preparation branches and all 81 herald × terminal bins per branch and sweep point, including herald `other`.

The cross selects the lowest measured target return. The target duration slices let you inspect the measured points and shot SEM. The rate plot gives cosine-fit rates and fit SE only where the target trace supports them. Missing spectator data does not block the target fit. Readout labels do not prove state purity, and return probability is not gate fidelity.

Use the target valley and duration traces to choose a frequency and length window for a follow-up CZ measurement. This experiment does not write back. See the [common operator workflow](../../operator-workflow.md) for saving and loading runs.
