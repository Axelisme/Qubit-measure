# Single-qubit randomized benchmarking

**Readers:** operator, agent. Measure Clifford decay, and optionally estimate a target gate's error by comparing reference and interleaved decay.

## Sequence and key settings

For each seed and depth: optional **Reset** → random Cliffords → inverse of the complete sequence → **Readout**. With a **Target gate** (X90, X180, Y90 or Y180), Run also measures a paired arm inserting that gate after every random Clifford. Both arms share the random sequence; each has its own inverse. Depth counts random Cliffords, not physical pulses.

- Calibrate frequency, X90, X180 and readout. Check repeatable initialization and readout timing. **Relax delay** starts at `5 × qubit.t1` (150 µs if absent) even with active reset.
- **Clifford depths** in Log mode include zero and max; rounding can merge points. Check **Effective points** or enter distinct nonnegative depths in List order. Use at least four distinct depths that span the decay. Aim for 20 or more independent sequences and use **Random seed** to reproduce them. Start with a short run to verify execution.

See [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md) for the shared GUI steps.

## Read the results

Live shows mean projected IQ (a.u.) per depth for each arm. Stop keeps completed rounds; empty cells stay blank. Run uses finite seeds per depth and arm; Analyze needs at least two seeds finite at every depth in all arms. Each bar is one sample SD across contributing seeds in the same projection, not shot SEM, fit error or round spread. There is no bar with fewer than two seeds. Increasing spread alone does not identify coherent error.

Analyze fits `A × p^depth + B`. Reference error per Clifford = `(1 − p_ref)/2`. Target error estimate = `(1 − p_interleaved/p_ref)/2`; target fidelity is one minus that error. The 95% intervals resample seeds, paired where applicable. They do not account for gate-dependent/coherent noise or model bias. Negative target errors or fidelities above one are not clipped.

## Next step

Check that the deepest points approach the fitted offset. Paired Analyze reports R² when available; the reference-only figure does not. A high R² alone cannot constrain a decay whose tail is outside the scan. Extend depth and recheck with independent seeds before adopting a calibration. Analyze does not write back.

Load rejects older files with a round axis and does not convert old standalone RB/IRB files.
