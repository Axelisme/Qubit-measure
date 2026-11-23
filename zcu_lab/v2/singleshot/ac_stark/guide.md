# Single-shot AC-Stark calibration

**Readers:** operator, agent. Measure the qubit resonance shift versus cavity drive gain and calibrate mean photon number.

## What is measured

Optional **Reset** and **Initial preparation pulse** → overlapping **Cavity Stark tone** and **Qubit probe tone** → **Readout**. Sweep cavity gain and qubit probe RF frequency; classify the shots as g, e or Other. Other means outside both accepted circles or tied, not an identified higher level.

## Key settings

- Before Run, select Pulse **Readout** (not Mux) with a circle classifier and positive radius under **Advanced**. Obtain the radius from [GE Post-Analysis](../ge/guide.md). Measure dispersive shift `$qubit.ext.chi` and resonator linewidth `$resonator.kappa` before Analyze.
- **Qubit probe RF frequency** starts at stored qubit frequency − 700 to + 100 MHz. Center it on the expected line and leave room for the shift across the **Cavity Stark gain** scan. Probe length and mixer start from the stored π pulse. For κ in MHz, the cavity tone starts from the π length plus 1/κ µs, with pre/post delays of 1/κ and 0.5/κ µs.
- **Relax delay** starts at 5 × stored ordinary T1 (150 µs if absent; blank if invalid). Shorten it only when independently justified; adding reset does not do so automatically.

## How to read the results

Compare where g/e/Other features occur. Each colorbar spans the values in its own panel; compare fractions on the bars, not by color. Gray cells have no data. Analyze applies readout correction only with the confusion matrix in the selected result's saved cfg. Without a usable matrix, the maps and resonance fit use classified fractions. A singular matrix or a nonfinite solve at a valid point logs one warning. Old files without class counts cannot be converted on Load; acquire again.

In the fit, points mark resonance frequencies extracted from the baseline-subtracted g/e contrast, and the line plots their shift versus gain². R² is the fraction of resonance variation explained by the line. Neither map colors nor resonance locations carry shot SEM bars. Check for a single ridge fully within the span; extend the frequency limit if the selected ridge hits it.

## Next step

A well-resolved linear shift yields the coefficient `c` in photons per gain². Writeback proposes `$qubit.ext.ac_stark_coeff`; MIST uses `mean photon number = c × gain²` when that conversion is available.

For general operations, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
