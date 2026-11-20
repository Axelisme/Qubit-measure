"""Pure two-qubit Clifford sequences for FTF reference and CZ-interleaved RB.

Qubit 0 is the left tensor factor: basis |q0 q1> = |00>, |01>, |10>,
|11>. X90/Z90 mean exp(-i*pi*X/4) / exp(-i*pi*Z/4) on the named
qubit; CZ = diag(1, 1, 1, -1). Gate tuples are in playback order, so
(g0, g1) has matrix U_g1 @ U_g0. Equality ignores global phase.

The full group is generated lazily using exact signed-Pauli actions.
Breadth-first traversal over the original X90/Z90/CZ generators fixes the
Clifford indices. Decomposition minimizes CZ count first (1.5 per Clifford on
average), then counts X90 and X180 as one physical pulse each. Virtual Z90s
are free. The search visits (CZ count, physical X count, Clifford index) in
heap order and keeps the first equal-cost word it finds in the original
X90/Z90/CZ generator order, then X180_Q0 and X180_Q1. Composition and inverse
use the faithful Pauli action.
No pulses, virtual frames, device bindings or acquisition state live here.

Call make_rb_arms(seed=7, depths=[0, 1, 50]) for both arms. For depth d,
play the arm's first d Clifford blocks, followed by its recovery at d.
"""

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache
from heapq import heappop, heappush
from itertools import pairwise
from random import Random
from types import MappingProxyType

NUM_CLIFFORDS = 11_520


class TwoQubitGate(Enum):
    """Synthetic generators; Q0/Q1 select the left/right tensor factor.

    X90_Q0/X90_Q1 are positive quarter-turn X rotations. Z90_Q0/Z90_Q1
    are positive quarter-turn Z rotations (logical gates, not frame state).
    X180_Q0/X180_Q1 are half-turn X rotations. CZ acts on both qubits.
    Identity is an empty gate tuple.
    """

    X90_Q0 = "X90_q0"
    Z90_Q0 = "Z90_q0"
    X90_Q1 = "X90_q1"
    Z90_Q1 = "Z90_q1"
    CZ = "CZ"
    X180_Q0 = "X180_q0"
    X180_Q1 = "X180_q1"


@dataclass(frozen=True)
class Clifford:
    """One projective two-qubit Clifford, identified by index in [0, 11520).

    Index 0 is identity. Other indices follow breadth-first traversal in
    the original X90/Z90/CZ order. Use decompose_clifford for playback.
    Construction rejects non-integer (including bool) or out-of-range indices.
    """

    index: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.index, "Clifford index")
        if self.index >= NUM_CLIFFORDS:
            raise ValueError(f"Clifford index must be less than {NUM_CLIFFORDS}")


@dataclass(frozen=True)
class RBArm:
    """One seed's reference or CZ-interleaved arm at the requested depths.

    seed identifies the random.Random draw. interleaved says whether CZ
    follows each random Clifford. depths is nonempty and strictly increasing.
    cliffords contains one playback word per draw, with the trailing CZ
    included for interleaved arms. At depth d, play cliffords[:d], then
    recoveries[d]: a single Clifford word that inverts that prefix.
    """

    seed: int
    interleaved: bool
    depths: tuple[int, ...]
    cliffords: tuple[tuple[TwoQubitGate, ...], ...]
    recoveries: Mapping[int, tuple[TwoQubitGate, ...]]


# A Pauli label 4*a+b denotes P_a tensor P_b, with (I, X, Y, Z).
# An action contains the signed images of labels 1..15 under U P U†.
# This action is faithful modulo global phase, including Pauli sign flips.
_PauliAction = tuple[int, ...]
_IDENTITY: _PauliAction = tuple(range(1, 16))
# CZ sends XI to XZ and IX to ZX. Products fix the signs.
_CZ_ACTION: _PauliAction = (13, 14, 3, 7, 10, -9, 4, 11, -6, 5, 8, 12, 1, 2, 15)


@dataclass(frozen=True)
class _CliffordGroup:
    actions: tuple[_PauliAction, ...]
    decompositions: tuple[tuple[TwoQubitGate, ...], ...]
    index_by_action: Mapping[_PauliAction, int]


def _require_nonnegative_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a Python int (not bool)")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _require_clifford(value: object) -> None:
    if not isinstance(value, Clifford):
        raise TypeError("Expected a Clifford")


def _compose_actions(after: _PauliAction, before: _PauliAction) -> _PauliAction:
    """Return conjugation by U_after @ U_before, using signed labels."""
    return tuple(
        after[label - 1] if label > 0 else -after[-label - 1] for label in before
    )


def _generator_actions() -> tuple[_PauliAction, ...]:
    """Exact images in X90/Z90/CZ BFS order; no floating-point matching."""
    actions: list[_PauliAction] = []
    # Local conjugation: Rx maps (X,Y,Z) to (X,Z,-Y), Rz to (Y,-X,Z).
    for qubit, images in (
        (0, (0, 1, 3, -2)),
        (0, (0, 2, -1, 3)),
        (1, (0, 1, 3, -2)),
        (1, (0, 2, -1, 3)),
    ):
        action: list[int] = []
        for label in range(1, 16):
            left, right = divmod(label, 4)
            image = images[left if qubit == 0 else right]
            sign = -1 if image < 0 else 1
            target = 4 * abs(image) + right if qubit == 0 else 4 * left + abs(image)
            action.append(sign * target)
        actions.append(tuple(action))
    return (*actions, _CZ_ACTION)


def _minimal_words(
    successors: list[tuple[int, ...]],
) -> tuple[tuple[TwoQubitGate, ...], ...]:
    """Find minimum (CZ, physical X) words from identity using generator edges.

    successors[index] lists the targets of appending each gate in declaration
    order. Dijkstra's search compares total costs lexicographically.
    """
    gate_costs = tuple(
        (
            gate,
            int(gate is TwoQubitGate.CZ),
            int(
                gate not in (TwoQubitGate.Z90_Q0, TwoQubitGate.Z90_Q1, TwoQubitGate.CZ)
            ),
        )
        for gate in TwoQubitGate
    )
    costs = {0: (0, 0)}
    words: list[tuple[TwoQubitGate, ...]] = [()] * len(successors)
    pending = [(0, 0, 0)]
    while pending:
        cz, x_pulses, index = heappop(pending)
        if (cz, x_pulses) != costs[index]:
            continue
        for target, (gate, gate_cz, gate_x) in zip(
            successors[index], gate_costs, strict=True
        ):
            cost = (cz + gate_cz, x_pulses + gate_x)
            previous = costs.get(target)
            # Strict improvement also prevents revisiting zero-cost cycles.
            if previous is None or cost < previous:
                costs[target] = cost
                words[target] = (*words[index], gate)
                heappush(pending, (*cost, target))
    return tuple(words)


@cache
def _group() -> _CliffordGroup:
    """Assign indices in BFS order, then cache their minimum-cost words."""
    actions = [_IDENTITY]
    indices = {_IDENTITY: 0}
    successors: list[tuple[int, ...]] = []
    pending = deque([0])
    generators = _generator_actions()
    while pending:
        index = pending.popleft()
        targets: list[int] = []
        for action in generators:
            product = _compose_actions(action, actions[index])
            target = indices.get(product)
            if target is None:
                target = len(actions)
                indices[product] = target
                pending.append(target)
                actions.append(product)
            targets.append(target)
        successors.append(tuple(targets))
    if len(actions) != NUM_CLIFFORDS:
        raise RuntimeError(f"Expected {NUM_CLIFFORDS} Cliffords, got {len(actions)}")
    # Add decomposition edges only after the original generators assign indices.
    x180s = tuple(_compose_actions(generators[i], generators[i]) for i in (0, 2))
    for i, action in enumerate(actions):
        successors[i] += tuple(
            indices[_compose_actions(x180, action)] for x180 in x180s
        )
    return _CliffordGroup(
        tuple(actions), _minimal_words(successors), MappingProxyType(indices)
    )


def clifford_group() -> tuple[Clifford, ...]:
    """Return all 11,520 distinct projective Cliffords, identity first.

    Initializes the immutable group on first use. No caller-owned random state
    or sequence state is read or changed.
    """
    return tuple(Clifford(index) for index in range(len(_group().actions)))


def decompose_clifford(clifford: Clifford) -> tuple[TwoQubitGate, ...]:
    """Return a minimum-(CZ, physical X) playback word (identity: ()).

    Minimize CZ count first, then X90 and X180 pulse count. Z90 adds no cost.
    Word length is bounded independently of RB depth. Raises TypeError for a
    non-Clifford.
    """
    _require_clifford(clifford)
    return _group().decompositions[clifford.index]


def compose_cliffords(after: Clifford, before: Clifford) -> Clifford:
    """Return U_after @ U_before modulo global phase (before plays first).

    Raises TypeError if either argument is not a Clifford.
    """
    _require_clifford(after)
    _require_clifford(before)
    group = _group()
    product = _compose_actions(group.actions[after.index], group.actions[before.index])
    return Clifford(group.index_by_action[product])


def inverse_clifford(clifford: Clifford) -> Clifford:
    """Return the two-sided projective inverse; reject non-Cliffords with TypeError.

    Decomposing the result selects a bounded group word. It never reverses a
    sequence's full history.
    """
    _require_clifford(clifford)
    group = _group()
    inverse = [0] * 15
    for source, target in enumerate(group.actions[clifford.index], start=1):
        inverse[abs(target) - 1] = source if target > 0 else -source
    return Clifford(group.index_by_action[tuple(inverse)])


def make_rb_arms(seed: int, depths: Sequence[int]) -> tuple[RBArm, RBArm]:
    """Draw one deepest sequence and return (reference, interleaved) arms.

    A private random.Random(seed) draws max(depths) uniform Clifford indices
    with replacement. Both arms use the same draws, and each requested depth
    uses a prefix and its own recovery. Only the interleaved arm appends a CZ
    to each random Clifford. The global random state is unchanged.

    seed and all depths must be nonnegative Python ints (TypeError for other
    types, including bool). Empty, negative, repeated or unsorted depths and
    negative seeds raise ValueError.
    """
    _require_nonnegative_int(seed, "seed")
    for depth in depths:
        _require_nonnegative_int(depth, "depth")
    if not depths or any(a >= b for a, b in pairwise(depths)):
        raise ValueError("RB depths must be nonempty and strictly increasing")
    rng = Random(seed)
    drawn = tuple(Clifford(rng.randrange(NUM_CLIFFORDS)) for _ in range(depths[-1]))
    return (
        _build_arm(seed, drawn, tuple(depths), interleaved=False),
        _build_arm(seed, drawn, tuple(depths), interleaved=True),
    )


def _build_arm(
    seed: int,
    drawn: tuple[Clifford, ...],
    depths: tuple[int, ...],
    *,
    interleaved: bool,
) -> RBArm:
    cz = Clifford(_group().index_by_action[_CZ_ACTION])
    cliffords: list[tuple[TwoQubitGate, ...]] = []
    products = [
        Clifford(0)
    ]  # products[d] combines the first d Cliffords (and inserted CZs).
    for clifford in drawn:
        gates, step = decompose_clifford(clifford), clifford
        if interleaved:
            gates += (TwoQubitGate.CZ,)
            step = compose_cliffords(cz, clifford)
        cliffords.append(gates)
        products.append(compose_cliffords(step, products[-1]))
    recoveries = {d: decompose_clifford(inverse_clifford(products[d])) for d in depths}
    return RBArm(
        seed, interleaved, depths, tuple(cliffords), MappingProxyType(recoveries)
    )
