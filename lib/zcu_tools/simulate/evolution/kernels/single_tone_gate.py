"""Named single-tone domain of the shared covariant RWA gate algorithm."""

from ._rwa_gate import CovariantGateKernel


class SingleToneGateKernel(CovariantGateKernel):
    """Adaptively split joint U with dressed-phase playback for smooth tones.

    Dispatch compares admitted cold gate/Sambe mixed plans from the complete
    acquire inventory. Four-step keys retain the fixed-four compiled fast path. Real sampled
    slices, nonzero offsets and final sample hold are supported. Rates, dressed
    ordering and coefficient/interaction Hamiltonian owners remain unchanged.
    Rejection falls back to shaped Sambe, never ordinary multi-tone RWA.
    """

    name = "single_tone_gate"
    single_tone = True
