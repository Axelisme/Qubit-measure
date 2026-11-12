"""Simultaneous covariant RWA gates; dispatch owns preparation amortization."""

from ._rwa_gate import CovariantGateKernel


class MultitoneGateKernel(CovariantGateKernel):
    """Joint RWA gates with shared numerical admission and adaptive splitting.

    Requires shared real-envelope RWA admission, pruning bound <=min(budget,1e-3),
    and adaptive n-way secular split proxy <=.01. Row support bounds playback;
    sampled slices, constant companions and final sample hold are supported.
    Rejection retains ordinary row RWA. Products live only within one evolve.
    The 20-use default cost preference belongs to multitone dispatch; pinning
    this family permits smaller inventories without changing numerical guards.
    """

    name = "multitone_gate"
