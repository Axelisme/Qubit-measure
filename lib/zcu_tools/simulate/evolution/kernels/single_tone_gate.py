"""Named single-tone domain of the shared covariant RWA gate algorithm."""

from ._rwa_gate import CovariantGateKernel


class SingleToneGateKernel(CovariantGateKernel):
    """Four joint U with dressed-charge phase playback for smooth single tones.

    Dispatch compares cold build plus playback against Sambe from the complete
    acquire inventory before invoking the shared numerical guard. Real sampled
    slices, nonzero offsets and final sample hold are supported. Rates, dressed
    ordering and coefficient/interaction Hamiltonian owners remain unchanged.
    Rejection falls back to shaped Sambe, never ordinary multi-tone RWA.
    """

    name = "single_tone_gate"
    single_tone = True
