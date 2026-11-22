# Single-shot classification check

**Readers:** operator, agent. Check an existing g/e readout calibration against the IQ cloud of a chosen preparation.

## What is measured

Optional **Reset** → optional **Initial preparation pulse** → **Qubit probe pulse** → **Readout**. Live shows the raw IQ shots. Analyze overlays the selected Run or Load result's decision circles and reports the fractions within each class.

## Key settings

Choose the preparation you want to test and set **Shots** high enough to see the cloud and its tails (start around 5,000). **Relax delay** starts at 5 × stored ordinary T1 (150 µs if absent; blank if invalid). Reset does not shorten it automatically.

Run can acquire without discrimination calibration. For Analyze, use a readout with one ADC and a circle classifier with a positive radius. A missing saved cfg, missing classifier, linear classifier or missing radius fails at Analyze. Use [GE Post-Analysis](../ge/guide.md) to obtain the radius before collecting shots for this check.

## How to read the results

A tight cloud within the expected I/Q decision circle supports discrimination for that preparation. Other counts shots outside both accepted circles or tied between them, not identified leakage. The scatter shows individual shots, not a mean with shot SEM bars.

## Next step

If the cloud misses the expected circle, check preparation first, then recalibrate readout with GE. This check does not write back calibration.

For general operations, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
