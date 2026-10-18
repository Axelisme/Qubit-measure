"""Pure two-qubit Clifford sequences for FTF reference and CZ-interleaved RB.

Qubit 0 is the left tensor factor: basis |q0 q1> = |00>, |01>, |10>,
|11>. X90/Z90 mean exp(-i*pi*X/4) / exp(-i*pi*Z/4) on the named
qubit; CZ = diag(1, 1, 1, -1). Gate tuples are in playback order, so
(g0, g1) has matrix U_g1 @ U_g0. Equality ignores global phase.

The full group is generated lazily using exact signed-Pauli actions. A
breadth-first search stores one shortest word in these five generators per
Clifford, not a dense multiplication table. Composition and inverse use the
faithful Pauli action, then select that Clifford's bounded decomposition.
No pulses, virtual frames, device bindings or acquisition state live here.

Call make_paired_sequences(depth=50, seed=7), then translate each arm's
``gates`` tuple to pulses. The shared ``random_cliffords`` retain block
boundaries; each arm also exposes ``body``, ``recovery`` and ``recovery_gates``.
"""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import cache
from random import Random
from types import MappingProxyType

NUM_CLIFFORDS = 11_520


class TwoQubitGate(Enum):
    """Synthetic generators; Q0/Q1 select the left/right tensor factor.

    X90_Q0/X90_Q1 are positive quarter-turn X rotations. Z90_Q0/Z90_Q1
    are positive quarter-turn Z rotations (logical gates, not frame state).
    CZ acts on both qubits. Identity is an empty gate tuple.
    """

    X90_Q0 = "X90_q0"
    Z90_Q0 = "Z90_q0"
    X90_Q1 = "X90_q1"
    Z90_Q1 = "Z90_q1"
    CZ = "CZ"


@dataclass(frozen=True)
class Clifford:
    """One projective two-qubit Clifford, identified by index in [0, 11520).

    Index 0 is identity. Other indices follow breadth-first traversal in
    TwoQubitGate declaration order. Use decompose_clifford for playback.
    Construction rejects non-integer (including bool) or out-of-range indices.
    """

    index: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.index, "Clifford index")
        if self.index >= NUM_CLIFFORDS:
            raise ValueError(f"Clifford index must be less than {NUM_CLIFFORDS}")


@dataclass(frozen=True)
class RBSequence:
    """One arm, with immutable playback body and its full-product inverse.

    body contains primitive gates before recovery, including inserted CZs for
    the interleaved arm. recovery is a single Clifford, not a reversed body.
    This record describes logical gates only; every pulse translation must
    start its own frame at zero and preserve the complete logical product.
    """

    body: tuple[TwoQubitGate, ...]
    recovery: Clifford

    @property
    def recovery_gates(self) -> tuple[TwoQubitGate, ...]:
        """Bounded decomposition of recovery, independent of sequence depth."""
        return decompose_clifford(self.recovery)

    @property
    def gates(self) -> tuple[TwoQubitGate, ...]:
        """Complete playback tuple: body followed by recovery gates."""
        return self.body + self.recovery_gates


@dataclass(frozen=True)
class PairedRBSequences:
    """Immutable reference/CZ-interleaved pair for one depth and seed.

    random_cliffords is the shared uniformly drawn group-element tuple, one
    entry per depth step. reference and interleaved each carry their own
    recovery; only interleaved inserts CZ after every random Clifford.
    """

    random_cliffords: tuple[Clifford, ...]
    reference: RBSequence
    interleaved: RBSequence


# A Pauli label 4*a+b denotes P_a tensor P_b, with (I, X, Y, Z).
# An action contains the signed images of labels 1..15 under U P U†.
# This action is faithful modulo global phase, including Pauli sign flips.
_PauliAction = tuple[int, ...]
_IDENTITY: _PauliAction = tuple(range(1, 16))


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
    """Exact images in TwoQubitGate order; no floating-point group matching."""
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
    # CZ: XI -> XZ, IX -> ZX, ZI -> ZI, IZ -> IZ. Products fix
    # the signs, notably XX -> YY, XY -> -YX, YX -> -XY, YY -> XX.
    actions.append((13, 14, 3, 7, 10, -9, 4, 11, -6, 5, 8, 12, 1, 2, 15))
    return tuple(actions)


@cache
def _group() -> _CliffordGroup:
    """Build once, then share only immutable group data across calls."""
    actions = [_IDENTITY]
    words: list[tuple[TwoQubitGate, ...]] = [()]
    indices = {_IDENTITY: 0}
    pending = deque([0])
    generators = tuple(zip(TwoQubitGate, _generator_actions(), strict=True))
    while pending:
        index = pending.popleft()
        for gate, action in generators:
            product = _compose_actions(action, actions[index])
            if product not in indices:
                indices[product] = len(actions)
                pending.append(len(actions))
                actions.append(product)
                words.append((*words[index], gate))
    if len(actions) != NUM_CLIFFORDS:
        raise RuntimeError(f"Expected {NUM_CLIFFORDS} Cliffords, got {len(actions)}")
    return _CliffordGroup(tuple(actions), tuple(words), MappingProxyType(indices))


def clifford_group() -> tuple[Clifford, ...]:
    """Return all 11,520 distinct projective Cliffords, identity first.

    Initializes the immutable group on first use. No caller-owned random state
    or sequence state is read or changed.
    """
    return tuple(Clifford(index) for index in range(len(_group().actions)))


def decompose_clifford(clifford: Clifford) -> tuple[TwoQubitGate, ...]:
    """Return a shortest generator word in playback order (identity: ()).

    Raises TypeError for a non-Clifford argument. Words contain at most 14
    generators, fixed by this finite group, not by the depth of an RB sequence.
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


def make_paired_sequences(depth: int, seed: int) -> PairedRBSequences:
    """Draw depth uniform group elements and build paired, fully recovered arms.

    depth and seed must be nonnegative Python ints, excluding bool. Wrong
    types raise TypeError; negative values raise ValueError, before generation.
    depth=0 returns empty bodies and identity recoveries for both arms.

    A private random.Random(seed) draws each index with randrange(11520),
    independently and with replacement, not via a primitive random walk.
    Identical depth/seed reproduce identical pairs; shorter depths are prefixes
    of longer ones with the same seed. Reproduction assumes this generator
    ordering and Python's Random implementation, not a cross-version archive
    format. Global random state is untouched.

    Every shared Clifford is decomposed without merging boundaries. Only the
    interleaved body appends CZ after each such block, including identity blocks.
    Each arm's recovery inverts its own complete product, including those CZs.
    """
    _require_nonnegative_int(depth, "depth")
    _require_nonnegative_int(seed, "seed")
    rng = Random(seed)
    random_cliffords = tuple(
        Clifford(rng.randrange(NUM_CLIFFORDS)) for _ in range(depth)
    )
    reference_body: list[TwoQubitGate] = []
    interleaved_body: list[TwoQubitGate] = []
    reference_product = Clifford(0)
    interleaved_product = Clifford(0)
    group = _group()
    cz = Clifford(group.index_by_action[_generator_actions()[-1]])
    for clifford in random_cliffords:
        gates = decompose_clifford(clifford)
        reference_body.extend(gates)
        interleaved_body.extend((*gates, TwoQubitGate.CZ))
        reference_product = compose_cliffords(clifford, reference_product)
        interleaved_product = compose_cliffords(
            cz, compose_cliffords(clifford, interleaved_product)
        )
    return PairedRBSequences(
        random_cliffords,
        RBSequence(tuple(reference_body), inverse_clifford(reference_product)),
        RBSequence(tuple(interleaved_body), inverse_clifford(interleaved_product)),
    )
